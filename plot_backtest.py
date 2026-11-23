"""Run a historical trading experiment and write its results, charts, and allocation heatmaps."""

import argparse
import csv
import hashlib
import io
import math
import shlex
import sys
from bisect import insort
from calendar import monthrange
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, fields, replace
from datetime import date, datetime, timedelta, timezone
from functools import partial
from itertools import groupby
from math import ceil
from pathlib import Path
from statistics import fmean, harmonic_mean, median, stdev
from tempfile import mkdtemp

import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

from cushion_selection import HYBRID, MULTIPLIERS, SMOOTH_SCALES, TIME_AWARE, select_multiplier, select_smooth_scale
from plot_allocation_heatmaps import ReferencePath, write_allocation_heatmaps
from trader import (
    AggressiveTieredCushionTrader, ChanceConstrainedDPTrader, DiscountedFloorDPTrader, SplitDPTrader,
    FLOOR, FixedSplitTrader, FloorAwareGlideTrader, FloorAwareVolatilityTargetTrader,
    GentleGlideTrader, HighVolatilityReductionTrader, SmoothCushionTrader, TimeAwareSmoothCushionTrader, Trader,
    StochasticDPTrader, TradingState, VolCappedSmoothCushionTrader, VolatilityScaledCushionTrader, VolatilityTargetTrader,
)

DP_VARIANTS = {
    "split-cushion-dp": SplitDPTrader,
    "chance-constrained-dp": ChanceConstrainedDPTrader,
    "discounted-floor-dp": DiscountedFloorDPTrader,
}

VOL_CAPPED = "vol-capped-smooth-cushion"
FIXED_SMOOTH = "fixed-time-aware-smooth-cushion"
STRATEGIES = {
    **DP_VARIANTS,
    TIME_AWARE: TimeAwareSmoothCushionTrader,
    FIXED_SMOOTH: TimeAwareSmoothCushionTrader,
    VOL_CAPPED: VolCappedSmoothCushionTrader,
    HYBRID: partial(StochasticDPTrader, multiplier=MULTIPLIERS[0]),
    "stochastic-dp": StochasticDPTrader,
    "fixed-split": FixedSplitTrader,
    "glide-65-35": GentleGlideTrader,
    "floor-aware-glide": FloorAwareGlideTrader,
    "vol-target": VolatilityTargetTrader,
    "floor-aware-vol-target": FloorAwareVolatilityTargetTrader,
    "high-vol-reduction": HighVolatilityReductionTrader,
    "vol-scaled-cushion": VolatilityScaledCushionTrader,
    "tiered-cushion-3x": AggressiveTieredCushionTrader,
    "smooth-cushion-4x": SmoothCushionTrader,
}
INITIAL = 100_000_000.0
BREACH_LIMIT = 0.02
DEFAULT_VOL_LOOKBACK = 30
LAUNCH_CONDITION_LOOKBACK = 63
VOLATILITY_GROUPS = ("lower-volatility", "higher-volatility")
CONDITIONS = (*VOLATILITY_GROUPS, "crisis")


@dataclass(frozen=True)
class Session:
    date: date
    equity_return: float
    cash_return: float


@dataclass(frozen=True)
class AnnualResult:
    strategy: str
    year: int
    start_month: str
    first_session: date
    last_session: date
    sessions: int
    starting_volatility: float
    median_volatility: float
    condition: str
    terminal_wealth: float
    total_return: float
    max_drawdown: float
    floor_breach: bool
    starting_drawdown: float | None = None
    horizon_months: int = 12


def read_sessions(data: bytes) -> list[Session]:
    reader = csv.reader(io.StringIO(data.decode("utf-8-sig")))
    for row in reader:
        if "Mkt-RF" in row and "RF" in row:
            market, rf = row.index("Mkt-RF"), row.index("RF")
            break
    else:
        raise ValueError("Missing Mkt-RF and RF header")

    sessions = []
    for row in reader:
        if not row or not any(cell.strip() for cell in row):
            continue
        if row[0].startswith("Copyright "):
            break
        try:
            if len(row[0]) != 8 or not row[0].isdigit():
                raise ValueError("expected YYYYMMDD")
            day = datetime.strptime(row[0], "%Y%m%d").date()
            excess, interest = float(row[market]), float(row[rf])
            if any(x in (-99.99, -999) for x in (excess, interest)):
                raise ValueError("missing return")
            equity, cash = (excess + interest) / 100, interest / 100
            if any(not math.isfinite(x) or x < -1 for x in (equity, cash)):
                raise ValueError("invalid simple return")
            if sessions and day <= sessions[-1].date:
                raise ValueError("dates must be unique and increasing")
        except (ValueError, IndexError) as exc:
            raise ValueError(f"Invalid data on CSV line {reader.line_num}: {exc}") from exc
        sessions.append(Session(day, equity, cash))
    if not sessions:
        raise ValueError("No daily returns found")
    return sessions


def rolling_daily_volatilities(returns, lookback):
    """Estimate j uses only the returns before session j + lookback."""
    if type(lookback) is not int or lookback < 2:
        raise ValueError("volatility lookback must be an integer of at least 2 sessions")
    return [stdev(returns[i:i + lookback]) for i in range(len(returns) - lookback)]


def annual_volatility_signals(daily_vols: list[float]) -> list[tuple[float, float | None]]:
    """Pair each annualized forecast with the nearest-rank percentile of earlier forecasts."""
    history = []
    signals = []
    annualizer = math.sqrt(252)
    for daily_vol in daily_vols:
        forecast = daily_vol * annualizer
        p80 = history[math.ceil(0.8 * len(history)) - 1] if history else None
        signals.append((forecast, p80))
        insort(history, forecast)
    return signals


def closing_drawdowns(sessions):
    """Equity drawdown over (one calendar year ago, today's close]."""
    peaks = deque()
    level = 1.0
    values = []
    for session in sessions:
        day = session.date
        cutoff = day.replace(year=day.year - 1,
                             day=min(day.day, monthrange(day.year - 1, day.month)[1]))
        level *= 1 + session.equity_return
        while peaks and peaks[0][0] <= cutoff:
            peaks.popleft()
        while peaks and peaks[-1][1] <= level:
            peaks.pop()
        peaks.append((day, level))
        peak = peaks[0][1]
        # A total loss leaves the index permanently at zero.
        drawdown = (peak - level) / peak if peak else 1.0
        values.append(drawdown if sessions[0].date <= cutoff else None)
    return values


def is_crisis(drawdown):
    return drawdown is not None and (drawdown >= .20 or math.isclose(drawdown, .20, rel_tol=1e-14))


def crisis_episodes(sessions, drawdowns):
    start = None
    for session, drawdown in zip(sessions, drawdowns):
        if is_crisis(drawdown):
            if start is None:
                start = session.date
            last = session.date
        elif start is not None:
            yield start, last
            start = None
    if start is not None:
        yield start, last


def replay(name: str, trader: Trader, days: list[Session], vol: float, threshold: float,
           signals: list[tuple[float, float | None]] | None = None, observe_trade=None) -> AnnualResult:
    if signals is not None and len(signals) != len(days):
        raise ValueError("Expected one volatility signal per trading session")
    equity, cash = 0.0, INITIAL
    peak, drawdown = INITIAL, 0.0
    for i, session in enumerate(days):
        forecast, p80 = signals[i] if signals is not None else (None, None)
        order = trader.trade(TradingState(i, len(days), equity, cash, forecast, p80))
        if not math.isfinite(order) or not -equity <= order <= cash:
            raise ValueError(f"{name} on {session.date}: infeasible order {order}")
        if observe_trade is not None:
            observe_trade(equity + cash, equity + order)
        equity = (equity + order) * (1 + session.equity_return)
        cash = (cash - order) * (1 + session.cash_return)
        wealth = equity + cash
        if not math.isfinite(wealth):
            raise ValueError(f"{name} on {session.date}: nonfinite wealth")
        peak = max(peak, wealth)
        drawdown = max(drawdown, (peak - wealth) / peak)
    return AnnualResult(name, days[0].date.year, days[0].date.strftime("%Y-%m"), days[0].date, days[-1].date, len(days),
                        vol, threshold, VOLATILITY_GROUPS[vol > threshold], wealth,
                        wealth / INITIAL - 1, drawdown, wealth < FLOOR)


def launch_trader(name, multiplier, scale, vol_capped_scale_source="incumbent"):
    if name == HYBRID or name in DP_VARIANTS:
        return STRATEGIES[name](multiplier=multiplier)
    if name == TIME_AWARE:
        return SmoothCushionTrader() if scale is None else TimeAwareSmoothCushionTrader(scale)
    if name == VOL_CAPPED and vol_capped_scale_source == "incumbent":
        if scale is None:
            return VolCappedSmoothCushionTrader(annual_funding_rate=0)
        return VolCappedSmoothCushionTrader(scale)
    return STRATEGIES[name]()


def run_years(sessions, names, start, end, *, horizon_months=12, selection_history=None,
              episodes=None, smooth_selection_history=None, reference_paths=None,
              vol_capped_scale_source="incumbent", smooth_launch_settings=None,
              vol_lookback=DEFAULT_VOL_LOOKBACK):
    if type(horizon_months) is not int or horizon_months < 1:
        raise ValueError("horizon_months must be a positive integer")
    if vol_capped_scale_source not in ("incumbent", "fixed-1"):
        raise ValueError("vol_capped_scale_source must be incumbent or fixed-1")
    drawdowns = closing_drawdowns(sessions)
    if episodes is not None:
        episodes.extend(crisis_episodes(sessions, drawdowns))
    offsets = {}
    stops = {}
    for i, session in enumerate(sessions):
        month = session.date.year * 12 + session.date.month - 1
        offsets.setdefault(month, i)
        stops[month] = i + 1

    # Only the final observed month lacks a following session to establish its end.
    last = sessions[-1].date
    month_end = last.replace(day=monthrange(last.year, last.month)[1])
    while month_end.weekday() >= 5:
        month_end -= timedelta(days=1)
    last_month = last.year * 12 + last.month - 1
    returns = [s.equity_return for s in sessions]
    vols = rolling_daily_volatilities(returns, vol_lookback)
    signals = annual_volatility_signals(vols)
    launch_vols = (vols if vol_lookback == LAUNCH_CONDITION_LOOKBACK
                   else rolling_daily_volatilities(returns, LAUNCH_CONDITION_LOOKBACK))

    results, excluded = [], {}
    adaptive = HYBRID in names or any(name in DP_VARIANTS for name in names)
    history = {m: [] for m in MULTIPLIERS}
    match_incumbent = VOL_CAPPED in names and vol_capped_scale_source == "incumbent"
    smooth_adaptive = TIME_AWARE in names or match_incumbent
    smooth_history = {scale: [] for scale in SMOOTH_SCALES}
    smooth_year = None
    first_year = min(start, sessions[0].date.year) if adaptive or smooth_adaptive else start
    first_session = {}
    if adaptive or smooth_adaptive:
        for session in sessions:
            first_session.setdefault(session.date.year, session.date)
    latest = None
    selected = scale = None
    dp_reason = smooth_mode = ""
    selection_year = None
    for month in range(first_year * 12, (end + 1) * 12):
        year, zero_based_month = divmod(month, 12)
        label = f"{year:04d}-{zero_based_month + 1:02d}"
        if month not in offsets:
            excluded[label] = "no start-month data"
            continue
        index = offsets[month] - vol_lookback
        if index < 252:
            excluded[label] = "insufficient pre-start volatility history"
            continue
        final_month = month + horizon_months - 1
        if (final_month > last_month or any(m not in offsets for m in range(month + 1, final_month + 1))
                or (final_month == last_month and last < month_end)):
            excluded[label] = "incomplete one-year window" if horizon_months == 12 else f"incomplete {horizon_months}-month window"
            continue
        # Include the last session of the final calendar month.
        days = sessions[offsets[month]:stops[final_month]]
        daily_signals = signals[index:index + len(days)]
        if adaptive:
            if year >= start and year != selection_year:
                selected, audit = select_multiplier(history, first_session[year], VOLATILITY_GROUPS, BREACH_LIMIT)
                if selection_history is not None:
                    selection_history.extend(dict(strategy=HYBRID, **row) for row in audit)
                dp_reason = audit[0]["reason"]
                selection_year = year
            calibration_median = median(vols[:index])
            for m in MULTIPLIERS:
                trader = STRATEGIES[HYBRID](multiplier=m)
                history[m].append(replay(HYBRID, trader, days, vols[index], calibration_median, daily_signals))
        # All strategies are scored in the same launch groups across trading lookbacks.
        launch_index = offsets[month] - LAUNCH_CONDITION_LOOKBACK
        if launch_index < 252:
            excluded[label] = "insufficient pre-start volatility history"
            continue
        launch_vol = launch_vols[launch_index]
        threshold = median(launch_vols[:launch_index])
        starting_drawdown = drawdowns[offsets[month] - 1]
        if starting_drawdown is None:
            excluded[label] = "insufficient pre-start drawdown history"
            continue
        if smooth_adaptive:
            if year >= start and year != smooth_year:
                scale, audit = select_smooth_scale(smooth_history, first_session[year], CONDITIONS, BREACH_LIMIT)
                if smooth_selection_history is not None:
                    smooth_selection_history.extend(audit)
                smooth_mode = audit[0]["mode"]
                smooth_year = year
            for candidate in SMOOTH_SCALES:
                outcome = replay(TIME_AWARE, TimeAwareSmoothCushionTrader(candidate), days,
                                 launch_vol, threshold, daily_signals)
                condition = "crisis" if is_crisis(starting_drawdown) else outcome.condition
                smooth_history[candidate].append(replace(outcome, condition=condition))
        if year < start:
            continue
        if reference_paths is not None:
            latest = (days, daily_signals, launch_vol, threshold, selected, scale, dp_reason, smooth_mode)
        for name in names:
            if name == HYBRID:
                outcome = replace(history[selected][-1], starting_volatility=launch_vol, median_volatility=threshold,
                                  condition=VOLATILITY_GROUPS[launch_vol > threshold])
            elif name == TIME_AWARE:
                if scale is None:
                    outcome = replay(name, SmoothCushionTrader(), days, launch_vol, threshold, daily_signals)
                else:
                    outcome = smooth_history[scale][-1]
            else:
                trader = launch_trader(name, selected, scale, vol_capped_scale_source)
                outcome = replay(name, trader, days, launch_vol, threshold, daily_signals)
            if smooth_launch_settings is not None and name in (VOL_CAPPED, TIME_AWARE, FIXED_SMOOTH):
                matched = name == TIME_AWARE or (name == VOL_CAPPED and match_incumbent)
                smooth_launch_settings.append(dict(
                    start_month=label, strategy=name, scale_source="incumbent" if matched else "fixed-1",
                    mode=smooth_mode if matched else "fixed", risk_scale=scale if matched and scale is not None else 1.,
                ))
            # Selection history keeps the trader's original two volatility groups.
            results.append(replace(outcome, starting_drawdown=starting_drawdown, horizon_months=horizon_months,
                                   condition="crisis" if is_crisis(starting_drawdown) else outcome.condition))
    if latest is not None:
        days, daily_signals, vol, threshold, selected, scale, dp_reason, smooth_mode = latest
        for name in names:
            trader = launch_trader(name, selected, scale, vol_capped_scale_source)
            settings = ""
            if name == HYBRID or name in DP_VARIANTS:
                settings = f"{selected}× multiplier"
                if dp_reason != "qualified":
                    settings += f", {dp_reason}"
            elif name == TIME_AWARE or (name == VOL_CAPPED and match_incumbent):
                settings = smooth_mode if scale is None else f"{smooth_mode}, scale {scale:g}"
            elif isinstance(trader, TimeAwareSmoothCushionTrader):
                settings = f"fixed scale {trader.risk_scale:g}"
            path = ReferencePath(trader, days[0].date, len(days), daily_signals[0][1], settings)
            replay(name, path, days, vol, threshold, daily_signals)
            reference_paths[name] = path
    return results, {month: reason for month, reason in excluded.items() if int(month[:4]) >= start}


def write_csv(path, rows, fieldnames=None):
    """Write records, omitting optional tables with no rows."""
    if fieldnames is None:
        if not rows:
            return
        fieldnames = rows[0].keys()
    with path.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def schedule_sensitivity(rows, horizon):
    counts = defaultdict(lambda: [0, 0])
    for row in rows:
        day = row.first_session
        offset = (day.year * 12 + day.month - 1) % horizon
        counts[offset][0] += row.floor_breach
        counts[offset][1] += 1

    def rate(offset):
        breaches, starts = counts[offset]
        return breaches / starts

    offsets = sorted(counts)
    stats = dict(schedules_represented=len(offsets), horizon_months=horizon)
    for label, select in (("lowest", min), ("highest", max)):
        offset = select(offsets, key=rate, default=None)
        breaches, starts = counts[offset] if offset is not None else (None, None)
        stats[f"{label}_schedule_offset"] = offset
        stats[f"{label}_schedule_breaches"] = breaches
        stats[f"{label}_schedule_starts"] = starts
        stats[f"{label}_schedule_rate"] = breaches / starts if starts else None
    return stats


def episode_report(rows, episodes):
    groups = defaultdict(list)
    for row in rows:
        groups[row.strategy, "all"].append(row)
        groups[row.strategy, row.condition].append(row)
    records = []
    for first, last in episodes:
        if not any(r.first_session <= last and r.last_session >= first for r in rows):
            continue
        for name in dict.fromkeys(r.strategy for r in rows):
            for condition in ("all", *CONDITIONS):
                group = groups[name, condition]
                exposed = [r for r in group if r.first_session <= last and r.last_session >= first]
                hits = sum(r.floor_breach for r in exposed)
                total_hits = sum(r.floor_breach for r in group)
                remaining = len(group) - len(exposed)
                remaining_hits = total_hits - hits
                delta = 100 * (remaining_hits / remaining - total_hits / len(group)) if remaining else None
                records.append(dict(first_close=first, last_close=last, strategy=name, condition=condition,
                                    exposed_breaches=hits, exposed_starts=len(exposed), remaining_breaches=remaining_hits,
                                    remaining_starts=remaining, change_percentage_points=delta))
    return records


def summary_row(name, condition, rows, horizon=12):
    breaches = sum(r.floor_breach for r in rows)
    rate = breaches / len(rows) if rows else None
    returns = [r.total_return for r in rows]
    return dict(strategy=name, condition=condition, starts=len(rows),
                ce_wealth=harmonic_mean(r.terminal_wealth for r in rows) if rows else None,
                floor_breaches=breaches, breach_rate=rate, above_breach_limit=rate > BREACH_LIMIT if rows else None,
                mean_return=fmean(returns) if rows else None, worst_return=min(returns, default=None),
                max_drawdown=max((r.max_drawdown for r in rows), default=None),
                **schedule_sensitivity(rows, horizon))


def smooth_comparison(results, names):
    launches = {r.start_month: r for r in results if r.strategy == TIME_AWARE}
    records = []
    for name in names:
        group = [r for r in results if r.strategy == name]
        if {r.start_month for r in group} != launches.keys():
            raise ValueError("Smooth comparisons require identical monthly starts")
        feasible = all(sum(r.floor_breach for r in group if r.condition == c)
                       < BREACH_LIMIT * sum(r.condition == c for r in group) for c in CONDITIONS)
        values = [r.terminal_wealth for r in group]
        gain = fmean(r.terminal_wealth - launches[r.start_month].terminal_wealth for r in group)
        records.append(dict(strategy=name, meets_group_limit=feasible, mean_wealth=fmean(values),
                            mean_gain_over_time_aware=gain, ce_wealth=harmonic_mean(values), worst_wealth=min(values)))
    return sorted(records, key=lambda row: (row["meets_group_limit"], row["mean_wealth"]), reverse=True)


def vol_capped_comparison(results, names):
    """Compare terminal dollars and CE on exactly the same launches in each group."""
    outcomes = {name: {row.start_month: row for row in results if row.strategy == name} for name in names}
    capped = outcomes[VOL_CAPPED]
    records = []
    for name, group in outcomes.items():
        if group.keys() != capped.keys() or any(row.condition != capped[m].condition for m, row in group.items()):
            raise ValueError("Volatility-capped comparisons require identical starts and launch conditions")
        for condition in ("all", *CONDITIONS):
            months = [m for m, row in group.items() if condition == "all" or row.condition == condition]
            if not months:
                continue
            values = [group[m].terminal_wealth for m in months]
            paired = [capped[m].terminal_wealth for m in months]
            ce = harmonic_mean(values)
            records.append(dict(strategy=name, condition=condition, starts=len(months), mean_wealth=fmean(values),
                                worst_wealth=min(values), ce_wealth=ce,
                                capped_minus_strategy_mean=fmean(a - b for a, b in zip(paired, values)),
                                capped_minus_strategy_ce=harmonic_mean(paired) - ce))
    return records


STRATEGY_COLORS = {
    "smooth-cushion-4x": "#3E5495",
    "tiered-cushion-3x": "#C52577",
    "vol-scaled-cushion": "#B14356",
    "vol-target": "#755094",
    "floor-aware-vol-target": "#337D8C",
    "high-vol-reduction": "#A85B12",
    "fixed-split": "#0072B2",
    "glide-65-35": "#D55E00",
    "floor-aware-glide": "#00805E",
}


def write_terminal_wealth_ecdf(folder, results, names, conditions, floor, breach_limit, horizon=12):
    floor_m = floor / 1_000_000
    wealth = [floor_m, *(r.terminal_wealth / 1_000_000 for r in results)]
    pad = max(1, (max(wealth) - min(wealth)) * .05)
    limits = (min(wealth) - pad, max(wealth) + pad)

    fig = Figure(figsize=(13, 7.2), layout="constrained")
    axes = fig.subplot_mosaic([["full", "zoom"], ["states", "states"]],
                              gridspec_kw={"height_ratios": [1.6, 1]})
    ax, zoom, score = axes["full"], axes["zoom"], axes["states"]
    ax.set(title=f"Distribution of {horizon}-month terminal wealth",
           xlim=limits, ylim=(-.025, 1.025), ylabel="Cumulative share of monthly starts")
    zoom.set(title=f"Detail near the ${floor_m:g}M floor", xlim=(floor_m - 2, floor_m + 2),
             ylim=(-.002, 2.5 * breach_limit))
    for panel in (ax, zoom):
        panel.set_xlabel("Terminal wealth threshold ($ millions)")
        panel.spines[["top", "right"]].set_visible(False)
        panel.grid(axis="y", alpha=.17)
        panel.set_axisbelow(True)
        panel.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        panel.axvspan(panel.get_xlim()[0], floor_m, color="#fbe9e6", zorder=0)
        panel.axhline(breach_limit, color="#414141", linestyle="--", linewidth=1.2,
                      label=f"{breach_limit:.0%} limit")
        panel.axvline(floor_m, color="#666666", linewidth=1, label=f"${floor_m:g}M floor", zorder=1)
    for index, name in enumerate(names):
        group = [r for r in results if r.strategy == name]
        if not group:
            continue
        color = STRATEGY_COLORS.get(name, f"C{index % 10}")
        values = sorted(r.terminal_wealth / 1_000_000 for r in group)
        xs, ys = [limits[0]], [0.]
        count = 0
        for value, ties in groupby(values):
            count += sum(1 for _ in ties)
            xs.append(value)
            ys.append(count / len(values))
        rate = sum(r.floor_breach for r in group) / len(group)
        for panel in (ax, zoom):
            panel.step(xs + [max(limits[1], panel.get_xlim()[1])], ys + [1.],
                       where="post", color=color, linewidth=1.6, label=name)
            panel.scatter(floor_m, rate, color=color, edgecolor="white", zorder=4)
    zoom.legend(loc="center left", bbox_to_anchor=(1, .5), frameon=False, handlelength=3)
    if not results:
        ax.text(.5, .5, "No eligible monthly starts", transform=ax.transAxes, ha="center")
    score.set_title(f"Share finishing below ${floor_m:g}M by starting condition", loc="left")
    score.axhline(breach_limit, color="#414141", linestyle="--", linewidth=1.4,
                  label=f"{breach_limit:.0%} limit")
    for offset, marker, condition in zip((-.18, 0, .18), ("o", "^", "D"), conditions):
        xs, rates = [], []
        for index, name in enumerate(names):
            group = [r for r in results if r.strategy == name and r.condition == condition]
            if group:
                xs.append(index + offset)
                rates.append(sum(r.floor_breach for r in group) / len(group))
        score.scatter(xs, rates, marker=marker, s=48, label=condition.capitalize(), zorder=3)
    score.set_xticks(range(len(names)), [name.replace("-", "-\n", 1) for name in names])
    score.set_xlim(-.5, len(names) - .5)
    score.set_ylim(bottom=-.055 * score.get_ylim()[1])
    score.yaxis.set_major_formatter(PercentFormatter(1, decimals=1))
    score.spines[["top", "right"]].set_visible(False)
    score.legend(loc="center left", bbox_to_anchor=(1, .5), frameon=False)
    fig.savefig(folder / "terminal_wealth_distribution.png", dpi=160)


def ce(values, axis=None):
    # Zero terminal wealth makes the harmonic mean zero.
    with np.errstate(divide="ignore"):
        return 1 / np.mean(1 / values, axis=axis)


def paired_intervals(months, hybrid, other):
    """Moving-block bootstrap, drawing the same contiguous months for both policies."""
    n = len(months)
    size = 12
    starts = [i for i in range(n - size + 1) if months[i + size - 1] - months[i] == size - 1]
    if not starts:
        return None, None, None, None
    rng = np.random.default_rng(1950)
    estimates = []
    for _ in range(2000):
        blocks = rng.choice(starts, size=ceil(n / size))
        index = (blocks[:, None] + np.arange(size)).ravel()[:n]
        a, b = hybrid[index], other[index]
        estimates.append([ce(a) - ce(b), (a - b).mean()])
    lo, hi = np.percentile(estimates, [2.5, 97.5], axis=0)
    return lo[0], hi[0], lo[1], hi[1]


def comparison_report(folder, results, names, floor):
    rows = {name: sorted((r for r in results if r.strategy == name), key=lambda r: r.start_month)
            for name in names}
    starts = [r.start_month for r in rows[HYBRID]]
    months = np.array([12 * int(m[:4]) + int(m[5:]) for m in starts])
    wealth = {}
    distribution = []
    fig = Figure(figsize=(10, 4.5), layout="constrained")
    ax, zoom = fig.subplots(1, 2)
    floor_m = floor / 1e6
    for name, group in rows.items():
        if [r.start_month for r in group] != starts:
            raise ValueError("Paired comparison requires identical monthly starts")
        wealth[name] = np.array([r.terminal_wealth for r in group])
        dollars = np.sort(wealth[name])
        distribution.append(dict(strategy=name, mean_wealth=dollars.mean(), median_wealth=np.median(dollars),
                                 p05_wealth=np.percentile(dollars, 5),
                                 worst_1pct_mean_wealth=dollars[:ceil(.01 * len(dollars))].mean(),
                                 largest_shortfall=max(0, floor - dollars[0])))
        values = dollars / 1e6
        for axis in (ax, zoom):
            axis.step(np.r_[min(values[0], floor_m - 2), values, max(values[-1], floor_m + 3)],
                      np.r_[0, np.arange(1, len(values) + 1) / len(values), 1],
                      where="post", label=name)
    for axis in (ax, zoom):
        axis.axvline(floor_m, color="gray", linestyle="--")
        axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.set_xlabel("Terminal wealth ($ millions)")
    ax.set(title="All monthly starts", ylabel="Cumulative share")
    ax.legend(fontsize="small")
    zoom.set(title="Outcomes near the floor", xlim=(floor_m - 2, floor_m + 3), ylim=(-.002, .15))
    fig.savefig(folder / "cushion_distribution.png", dpi=160)
    write_csv(folder / "cushion_distribution.csv", distribution)
    gains = []
    for name in names:
        if name == HYBRID:
            continue
        a, b = wealth[HYBRID], wealth[name]
        ce_low, ce_high, mean_low, mean_high = paired_intervals(months, a, b)
        retained = None
        if name == "stochastic-dp" and "smooth-cushion-4x" in wealth:
            baseline = ce(wealth["smooth-cushion-4x"])
            advantage = ce(b) - baseline
            if advantage > 0:
                retained = (ce(a) - baseline) / advantage
        gains.append(dict(comparator=name, combined_minus_comparator_ce=ce(a) - ce(b),
                          combined_minus_comparator_mean=(a - b).mean(), ce_gain_ci_low=ce_low, ce_gain_ci_high=ce_high,
                          mean_gain_ci_low=mean_low, mean_gain_ci_high=mean_high, share_of_ce_advantage_retained=retained))
    write_csv(folder / "cushion_paired_gains.csv", gains)

    decades = []
    buckets = sorted({(r.year // 10 * 10, r.condition) for r in rows[HYBRID]})
    for decade, condition in buckets:
        for name, group in rows.items():
            subset = [r for r in group if r.year // 10 * 10 == decade and r.condition == condition]
            values = np.array([r.terminal_wealth for r in subset])
            decades.append(dict(decade=decade, condition=condition, strategy=name, starts=len(subset),
                                ce_wealth=ce(values), mean_wealth=values.mean(), floor_breaches=sum(r.floor_breach for r in subset)))
    write_csv(folder / "cushion_by_decade.csv", decades)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", action="append", choices=STRATEGIES)
    parser.add_argument("--horizon-months", type=int, default=12)
    parser.add_argument("--vol-lookback", type=int, default=DEFAULT_VOL_LOOKBACK,
                        help="Trading sessions in the volatility estimate (default: %(default)s)")
    parser.add_argument("--vol-capped-scale-source", choices=("incumbent", "fixed-1"), default="incumbent")
    parser.add_argument("--start-year", type=int)
    parser.add_argument("--end-year", type=int)
    parser.add_argument("--data", type=Path, default=Path("F-F_Research_Data_Factors_daily.csv"))
    args = parser.parse_args(argv)
    names = list(dict.fromkeys(args.strategy or ["fixed-split"]))
    try:
        data = args.data.read_bytes()
        sessions = read_sessions(data)
        start = args.start_year if args.start_year is not None else sessions[0].date.year
        end = args.end_year if args.end_year is not None else sessions[-1].date.year
        if not 1 <= start <= end <= 9999:
            raise ValueError("Expected 1 <= start-year <= end-year <= 9999")
        reference_paths = {}
        smooth_selections = []
        launch_settings = []
        selections = []
        episodes = []
        horizon = args.horizon_months
        results, excluded = run_years(sessions, names, start, end, horizon_months=horizon, episodes=episodes,
                                      selection_history=selections, smooth_selection_history=smooth_selections,
                                      reference_paths=reference_paths, vol_capped_scale_source=args.vol_capped_scale_source,
                                      smooth_launch_settings=launch_settings if VOL_CAPPED in names else None,
                                      vol_lookback=args.vol_lookback)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))

    Path("runs").mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    folder = Path(mkdtemp(prefix=stamp + "-", dir="runs"))
    included = sorted({r.start_month for r in results})
    write_csv(folder / "run_metadata.csv", [dict(
        command=shlex.join(["python3", Path(__file__).name, *(sys.argv[1:] if argv is None else argv)]),
        data_path=args.data, data_sha256=hashlib.sha256(data).hexdigest(), start_year=start, end_year=end,
        horizon_months=horizon, vol_lookback=args.vol_lookback, eligible_starts=len(included),
        first_start=min(included, default=None), last_start=max(included, default=None))])
    write_csv(folder / "excluded_starts.csv", [dict(start_month=m, reason=why) for m, why in excluded.items()])
    write_csv(folder / "annual_results.csv", (asdict(row) for row in results), [f.name for f in fields(AnnualResult)])
    summary = []
    for name in names:
        rows = [r for r in results if r.strategy == name]
        summary.append(summary_row(name, "all", rows, horizon))
        for condition in CONDITIONS:
            summary.append(summary_row(name, condition, [r for r in rows if r.condition == condition], horizon))
    write_csv(folder / "summary.csv", summary)
    write_csv(folder / "crisis_sensitivity.csv", episode_report(results, episodes))
    write_csv(folder / "selection_history.csv", selections)
    write_csv(folder / "smooth_selection_history.csv", smooth_selections)
    write_csv(folder / "smooth_launch_settings.csv", launch_settings)
    if HYBRID in names and results:
        comparison_report(folder, results, names, FLOOR)
    if TIME_AWARE in names and results:
        write_csv(folder / "smooth_comparison.csv", smooth_comparison(results, names))
        modes = {r["selection_date"].year: r["mode"] for r in smooth_selections}
        counts = dict.fromkeys(("nominal warm-up", "calibrated", "cash fallback"), 0)
        for row in results:
            if row.strategy == TIME_AWARE:
                counts[modes[row.year]] += 1
        write_csv(folder / "smooth_mode_counts.csv", [dict(mode=mode, starts=n) for mode, n in counts.items()])
    if VOL_CAPPED in names:
        write_csv(folder / "vol_capped_comparison.csv", vol_capped_comparison(results, names))
    write_terminal_wealth_ecdf(folder, results, names, CONDITIONS, FLOOR, BREACH_LIMIT, horizon)
    write_allocation_heatmaps(folder, names, reference_paths, FLOOR, expanded_grid=VOL_CAPPED in names)
    print(folder)


if __name__ == "__main__":
    main()
