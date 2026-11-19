"""Replay overlapping fixed-horizon investments starting each month against the common daily trader interface."""

import argparse
import csv
import hashlib
import io
import math
import shlex
import sys
from bisect import insort
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, fields, replace
from datetime import date, datetime, timedelta, timezone
from calendar import monthrange
from functools import partial
from pathlib import Path
from statistics import fmean, harmonic_mean, median, stdev
from tempfile import mkdtemp

from cushion_selection import HYBRID, MULTIPLIERS, select_multiplier
from trader import (
    AggressiveTieredCushionTrader,
    FLOOR, FixedSplitTrader, FloorAwareGlideTrader, FloorAwareVolatilityTargetTrader,
    GentleGlideTrader, HighVolatilityReductionTrader, SmoothCushionTrader, Trader,
    StochasticDPTrader, TradingState, VolatilityScaledCushionTrader, VolatilityTargetTrader,
)

STRATEGIES = {
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


def run_years(sessions, names, start, end, *, horizon_months=12, selection_history=None,
              episodes=None):
    if type(horizon_months) is not int or horizon_months < 1:
        raise ValueError("horizon_months must be a positive integer")
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
    lookback = 63
    # Forecast j is available before session j + lookback.
    vols = [stdev(returns[i:i + lookback]) for i in range(len(returns) - lookback)]
    signals = annual_volatility_signals(vols)

    results, excluded = [], {}
    adaptive = HYBRID in names
    history = {m: [] for m in MULTIPLIERS}
    first_year = min(start, sessions[0].date.year) if adaptive else start
    first_session = {}
    if adaptive:
        for session in sessions:
            first_session.setdefault(session.date.year, session.date)
    selection_year = None
    for month in range(first_year * 12, (end + 1) * 12):
        year, zero_based_month = divmod(month, 12)
        label = f"{year:04d}-{zero_based_month + 1:02d}"
        if month not in offsets:
            excluded[label] = "no start-month data"
            continue
        index = offsets[month] - lookback
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
        threshold = median(vols[:index])
        if adaptive:
            if year >= start and year != selection_year:
                selected, audit = select_multiplier(history, first_session[year], VOLATILITY_GROUPS, BREACH_LIMIT)
                if selection_history is not None:
                    selection_history.extend(dict(strategy=HYBRID, **row) for row in audit)
                selection_year = year
            for m in MULTIPLIERS:
                trader = STRATEGIES[HYBRID](multiplier=m)
                history[m].append(replay(HYBRID, trader, days, vols[index], threshold, daily_signals))
        if year < start:
            continue
        starting_drawdown = drawdowns[offsets[month] - 1]
        if starting_drawdown is None:
            excluded[label] = "insufficient pre-start drawdown history"
            continue
        for name in names:
            if name == HYBRID:
                outcome = history[selected][-1]
            else:
                outcome = replay(name, STRATEGIES[name](), days, vols[index], threshold, daily_signals)
            # Selection history keeps the trader's original two volatility groups.
            results.append(replace(outcome, starting_drawdown=starting_drawdown, horizon_months=horizon_months,
                                   condition="crisis" if is_crisis(starting_drawdown) else outcome.condition))
    return results, {month: reason for month, reason in excluded.items() if int(month[:4]) >= start}


def schedule_sensitivity(rows, horizon):
    counts = defaultdict(lambda: [0, 0])
    for row in rows:
        day = row.first_session
        offset = (day.year * 12 + day.month - 1) % horizon
        count = counts[offset]
        count[0] += row.floor_breach
        count[1] += 1
    if not counts:
        return "—", "—", f"0/{horizon}"
    offsets = sorted(counts)
    def rate(offset):
        hits, total = counts[offset]
        return hits / total
    def describe(offset):
        hits, total = counts[offset]
        return f"{hits}/{total} ({rate(offset):.2%}, offset {offset})"
    return describe(min(offsets, key=rate)), describe(max(offsets, key=rate)), f"{len(offsets)}/{horizon}"


def episode_report(rows, episodes):
    lines = ["", "## Leave-one-crisis-out sensitivity", "",
             "Each episode is a consecutive spell of crisis closes. Exclusions are evaluated separately. "
             "Change is the remaining breach rate minus the headline rate, in percentage points.", "",
             "| Crisis closes (inclusive) | Strategy | Launch state | Exposed breaches/starts | "
             "Remaining breaches/starts | Change (pp) |", "|---|---|---|---:|---:|---:|"]
    groups = defaultdict(list)
    for row in rows:
        groups[row.strategy, "all"].append(row)
        groups[row.strategy, row.condition].append(row)
    found = False
    for first, last in episodes:
        if not any(r.first_session <= last and r.last_session >= first for r in rows):
            continue
        found = True
        for name in dict.fromkeys(r.strategy for r in rows):
            for condition in ("all", *CONDITIONS):
                group = groups[name, condition]
                exposed = [r for r in group if r.first_session <= last and r.last_session >= first]
                hits = sum(r.floor_breach for r in exposed)
                total_hits = sum(r.floor_breach for r in group)
                remaining = len(group) - len(exposed)
                remaining_hits = total_hits - hits
                delta = "—"
                if remaining:
                    delta = f"{100 * (remaining_hits / remaining - total_hits / len(group)):+.2f}"
                lines.append(f"| {first} to {last} | {name} | {condition} | {hits}/{len(exposed)} | "
                             f"{remaining_hits}/{remaining} | {delta} |")
    if not found:
        lines.extend(["", "No crisis closes intersect eligible launches."])
    return lines


def summary_row(name, condition, rows, horizon=12):
    prefix = f"| {name} | {condition} | {len(rows)} |"
    sensitivity = " | ".join(schedule_sensitivity(rows, horizon)) + " |"
    if not rows:
        return prefix + " — | — | — | — | — | — | " + sensitivity
    breaches = sum(r.floor_breach for r in rows)
    rate = breaches / len(rows)
    ce = harmonic_mean(r.terminal_wealth for r in rows)
    returns = [r.total_return for r in rows]
    return (f"{prefix} ${ce:,.2f} | {breaches} ({rate:.2%}) | {'yes' if rate > BREACH_LIMIT else 'no'} |"
            f" {fmean(returns):.2%} | {min(returns):.2%} | {max(r.max_drawdown for r in rows):.2%} | " + sensitivity)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", action="append", choices=STRATEGIES)
    parser.add_argument("--horizon-months", type=int, default=12)
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
        selections = []
        episodes = []
        horizon = args.horizon_months
        results, excluded = run_years(sessions, names, start, end, horizon_months=horizon, episodes=episodes,
                                      selection_history=selections)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))

    from backtest_figures import write_figures

    included = sorted({r.start_month for r in results})
    lines = [f"# {horizon}-month backtest with monthly starts", "",
             f"Command: `{shlex.join(['python3', 'backtest.py', *(sys.argv[1:] if argv is None else argv)])}`",
             f"Data: `{args.data}` (SHA-256 `{hashlib.sha256(data).hexdigest()}`)",
             f"Requested start years: {start}–{end}",
             f"Eligible monthly starts: {len(included)}" + (f" ({included[0]} through {included[-1]})" if included else ""),
             "Excluded start months: " + (", ".join(f"{y} ({why})" for y, why in excluded.items()) or "none"), ""]
    table = [
        "| Strategy | Starting condition | Monthly starts | CE (γ = 2) | Finished below $95M | Observed above 2%? | Mean horizon return | Worst horizon return | Maximum drawdown | Lowest schedule rate | Highest schedule rate | Schedules represented |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in names:
        rows = [r for r in results if r.strategy == name]
        table.append(summary_row(name, "all", rows, horizon))
        for condition in CONDITIONS:
            table.append(summary_row(name, condition, [r for r in rows if r.condition == condition], horizon))
    lines.extend(table)
    lines.extend(["", f"Each monthly start begins with $100M and runs for {horizon} calendar months. CE is the harmonic mean of terminal wealth. "
                  "Each monthly launch counts once. Schedule sensitivity spans nonempty calendar offsets with disjoint "
                  "holding periods within each offset. This range measures timing sensitivity, not statistical confidence."])
    lines.extend(episode_report(results, episodes))
    Path("runs").mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    folder = Path(mkdtemp(prefix=stamp + "-", dir="runs"))
    with (folder / "annual_results.csv").open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=[f.name for f in fields(AnnualResult)])
        writer.writeheader()
        writer.writerows(asdict(row) for row in results)
    if selections:
        with (folder / "selection_history.csv").open("w", newline="", encoding="utf-8") as out:
            writer = csv.DictWriter(out, fieldnames=selections[0])
            writer.writeheader()
            writer.writerows(selections)
        from cushion_report import comparison_report
        lines.extend(comparison_report(folder, results, names, FLOOR))
    write_figures(folder, results, names, CONDITIONS, FLOOR, BREACH_LIMIT, horizon)
    lines.extend(["", "![Terminal wealth distribution](terminal_wealth_distribution.png)",
                  "", "![Starting volatility and terminal wealth](starting_volatility_vs_wealth.png)"])
    (folder / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(table), end="\n\n")
    print(folder)


if __name__ == "__main__":
    main()
