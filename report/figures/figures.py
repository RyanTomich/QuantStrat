"""Replay the report's monthly launches from daily market data and generate its tables and figures."""

import argparse
import sys
from pathlib import Path
from functools import cache
from statistics import harmonic_mean
import numpy as np
from matplotlib.figure import Figure
from matplotlib import rcParams
from matplotlib.ticker import PercentFormatter

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))
from plot_allocation_heatmaps import ReferencePath, VOLATILITY
import plot_backtest as bt
from stochastic_dp import ChancePolicy, WealthPolicy
from trader import StochasticDPTrader, FLOOR

progression_labels = ["DP", "+ equity cap", "+ downside shocks", "+ policy selection"]
comparison_labels = ["Full DP", "Volatility-capped smooth cushion"]

families = {
    "Fixed allocation": {"Fixed 60/40": "fixed-split"},
    "Glide path": {
        "65–35 glide": "glide-65-35",
        "Floor-aware glide": "floor-aware-glide",
    },
    "Volatility targeting": {
        "Volatility target": "vol-target",
        "Floor-aware vol target": "floor-aware-vol-target",
        "High-vol reduction": "high-vol-reduction",
    },
    "Cushion": {
        "Vol-scaled cushion": "vol-scaled-cushion",
        "Tiered cushion": "tiered-cushion-3x",
        "Smooth 4× cushion": "smooth-cushion-4x",
        "Fixed time-aware cushion": bt.FIXED_SMOOTH,
        "Time-aware smooth cushion": bt.TIME_AWARE,
        comparison_labels[1]: bt.VOL_CAPPED,
    },
}


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--table-only", action="store_true", help="Generate the report tables and skip plotting.")
args = parser.parse_args()

class GapUtility(WealthPolicy):
    """Use the three-shock transition law with the original terminal utility."""

    def _set_transitions(self, floor):
        ChancePolicy._set_transitions(self, floor)

    def _extend(self):
        values = np.empty_like(self.values)
        actions = np.empty_like(values, dtype=np.uint8)
        for z, switch in enumerate((0.01, 0.04)):
            mass, _, left, frac = self.segments[z]
            cont = (1 - switch) * self.values[z] + switch * self.values[1 - z]
            expected = (
                (cont[left] + frac * (cont[left + 1] - cont[left])) * mass[:, None]
            ).sum(axis=1)
            ranked = np.take_along_axis(expected, self.by_exposure, axis=0)
            best = np.take_along_axis(
                self.by_exposure, ranked.argmax(axis=0)[None, :], axis=0
            )
            actions[z] = best[0]
            values[z] = np.take_along_axis(expected, best, axis=0)[0]
        self.actions.append(actions)
        self.values = values


@cache
def policy(m):
    return GapUtility(FLOOR, multiplier=m)


class GapTrader(StochasticDPTrader):
    def policy(self, state, wealth, high):
        return policy(self.multiplier)


sessions = bt.read_sessions((ROOT / "F-F_Research_Data_Factors_daily.csv").read_bytes())
names = ["stochastic-dp", bt.HYBRID, "chance-constrained-dp"]
for candidates in families.values():
    names.extend(candidates.values())
selection_history = []
smooth_settings = []
results, _ = bt.run_years(
    sessions, names, 1950, 2025, horizon_months=12, selection_history=selection_history,
    vol_capped_scale_source="incumbent", smooth_launch_settings=smooth_settings,
)
mult = {r["selection_date"].year: r["selected_multiplier"] for r in selection_history}
offset = {s.date: i for i, s in enumerate(sessions)}
ret = [s.equity_return for s in sessions]
signals = bt.annual_volatility_signals(bt.rolling_daily_volatilities(ret, bt.DEFAULT_VOL_LOOKBACK))
by_month = {name: {} for name in names}
for row in results:
    group = by_month[row.strategy]
    if row.start_month in group:
        raise ValueError(f"Duplicate {row.strategy} launch for {row.start_month}")
    group[row.start_month] = row
baseline = by_month["stochastic-dp"]
base = [baseline[month] for month in sorted(baseline)]
if not base:
    raise ValueError("No eligible monthly launches for the report")


def launch_conditions(row):
    return (row.first_session, row.last_session, row.starting_volatility,
            row.median_volatility, row.condition)


def matched_wealth(name):
    group = by_month[name]
    if group.keys() != baseline.keys():
        raise ValueError(f"{name} covers different launch months than baseline DP")
    wealth = []
    for launch in base:
        row = group[launch.start_month]
        if launch_conditions(row) != launch_conditions(launch):
            raise ValueError(f"{name} has different launch inputs for {launch.start_month}")
        wealth.append(row.terminal_wealth)
    return np.array(wealth) / 1e6


def launch_inputs(r):
    a, b = offset[r.first_session], offset[r.last_session] + 1
    return (
        sessions[a:b],
        r.starting_volatility,
        r.median_volatility,
        signals[a - bt.DEFAULT_VOL_LOOKBACK : b - bt.DEFAULT_VOL_LOOKBACK],
    )


wealth_by_strategy = {name: matched_wealth(name) for name in names}
gap = [
    bt.replay("gap", GapTrader(mult[r.year]), *launch_inputs(r)).terminal_wealth
    for r in base
]
series = [
    wealth_by_strategy["stochastic-dp"],
    wealth_by_strategy[bt.HYBRID],
    np.array(gap) / 1e6,
    wealth_by_strategy["chance-constrained-dp"],
]

overview = dict(zip(progression_labels, series, strict=True))
overview[comparison_labels[1]] = wealth_by_strategy[bt.VOL_CAPPED]
conditions = np.array([r.condition for r in base])
floor_millions = FLOOR / 1e6

def meets_breach_limit(wealth):
    return all(np.mean(wealth[conditions == c] < floor_millions) <= bt.BREACH_LIMIT
               for c in bt.CONDITIONS)


eligible = {name for name, wealth in overview.items() if meets_breach_limit(wealth)}
lines = [
    r"\begin{tabular}{@{}lrrr@{}}", r"\hline",
    r"Strategy & CE (\$M) & Mean $W_T$ (\$M) & Breaches (\%) \\", r"\hline",
]
slide_lines = [
    r"\begin{tabular}{@{}lrr@{}}", r"\hline",
    r"Strategy & CE (\$M) & Breach rate \\", r"\hline",
]
for condition in ("all", *bt.CONDITIONS):
    mask = np.ones(len(base), dtype=bool) if condition == "all" else conditions == condition
    label = "All launches" if condition == "all" else condition.replace("-", " ").capitalize()
    lines.append(rf"\multicolumn{{4}}{{@{{}}l}}{{\itshape {label} ($n={mask.sum()}$)}} \\")
    metrics = {}
    for name, wealth in overview.items():
        sample = wealth[mask]
        metrics[name] = (harmonic_mean(sample), sample.mean(), np.count_nonzero(sample < floor_millions))
    slide_rows = {
        "Regular DP": metrics[progression_labels[0]],
        "Full DP": metrics[progression_labels[-1]],
        "Vol.-capped cushion": metrics[comparison_labels[1]],
    }
    slide_lines.append(rf"\multicolumn{{3}}{{@{{}}l}}{{\itshape {label} ($n={mask.sum()}$)}} \\")
    highest_ce = max(ce for ce, _, _ in slide_rows.values())
    fewest_breaches = min(count for _, _, count in slide_rows.values())
    for name, (ce, _, count) in slide_rows.items():
        ce_cell = rf"\textbf{{{ce:.2f}}}" if ce == highest_ce else f"{ce:.2f}"
        rate = rf"{100 * count / mask.sum():.2f}\%"
        rate_cell = rf"\textbf{{{rate}}}" if count == fewest_breaches else rate
        slide_lines.append(rf"{name} & {ce_cell} & {rate_cell} \\")
    slide_lines.append(r"\hline")
    best_ce = max((metrics[name][0] for name in eligible), default=None)
    best_mean = max(mean for _, mean, _ in metrics.values())
    fewest = min(count for _, _, count in metrics.values())
    for name, (ce, mean, count) in metrics.items():
        cells = [f"{ce:.2f}", f"{mean:.2f}", rf"{count} ({100 * count / mask.sum():.2f}\%)"]
        winners = [name in eligible and ce == best_ce, mean == best_mean, count == fewest]
        cells = [rf"\textbf{{{value}}}" if winner else value for value, winner in zip(cells, winners)]
        lines.append(" & ".join([name, *cells]) + r" \\")
    if condition != bt.CONDITIONS[-1]:
        lines.append(r"\hline")
lines.extend([r"\hline", r"\end{tabular}"])
(OUT / "strategy-overview.tex").write_text("\n".join(lines) + "\n")
slide_lines.append(r"\end{tabular}")
(OUT / "slide-performance.tex").write_text("\n".join(slide_lines) + "\n")

alternatives = {"Full DP": wealth_by_strategy["chance-constrained-dp"]}
for family, strategies in families.items():
    candidates = {label: wealth_by_strategy[name] for label, name in strategies.items()}
    qualifying = [name for name, wealth in candidates.items() if meets_breach_limit(wealth)]
    if qualifying:
        name = max(qualifying, key=lambda name: harmonic_mean(candidates[name]))
    elif family == "Fixed allocation" and len(candidates) == 1:
        # Retain the sole fixed-weight rule as an explicitly ineligible benchmark.
        name, = candidates
    else:
        raise ValueError(f"No eligible candidate in {family}")
    alternatives[name] = candidates[name]

lines = [
    r"\begin{tabular}{@{}lrrr@{}}", r"\hline",
    r"Strategy & CE (\$M) & Mean $W_T$ (\$M) & Breach rate (\%) \\", r"\hline",
]
alternative_metrics = {
    name: (harmonic_mean(wealth), wealth.mean(), 100 * np.mean(wealth < floor_millions))
    for name, wealth in alternatives.items()
}
ce_values, means, breach_rates = zip(*alternative_metrics.values())
best_values = (max(ce_values), max(means), min(breach_rates))
for name, values in alternative_metrics.items():
    cells = [
        rf"\textbf{{{value:.2f}}}" if value == best else f"{value:.2f}"
        for value, best in zip(values, best_values, strict=True)
    ]
    lines.append(" & ".join([name, *cells]) + r" \\")
lines.extend([r"\hline", r"\end{tabular}"])
(OUT / "alternatives-overview.tex").write_text("\n".join(lines) + "\n")
if args.table_only:
    sys.exit(0)

colors = ["#435465", "#007a72", "#b55c16", "#623f8d", "#b32658"]
styles = ["-", "--", ":", "-.", "-"]
rcParams.update({"font.size": 9, "pdf.fonttype": 42})

launch = base[-1]
days, daily_vol, median_vol, forecasts = launch_inputs(launch)
multiple = mult[launch.year]
cutoff = forecasts[0][1]
remaining = min(len(days), 252)
traders = [
    StochasticDPTrader(),
    StochasticDPTrader(multiple),
    GapTrader(multiple),
    bt.ChanceConstrainedDPTrader(multiple),
]

def plot_allocations(traders, labels, settings, filename, *, columns=2):
    rows = len(traders) // columns
    fig = Figure(figsize=(3.6 * columns, 2.0 * rows + 0.9), layout="constrained")
    axes = fig.subplots(rows, columns, sharex=True, sharey=True, squeeze=False)
    wealth_grid = np.arange(92_000_000, 130_000_001, 100_000)
    for index, (ax, trader, label) in enumerate(zip(axes.flat, traders, labels)):
        path = ReferencePath(trader, days[0].date, len(days), cutoff)
        bt.replay(label, path, days, daily_vol, median_vol, forecasts)
        grid = [[path.allocation(remaining, v, w) for w in wealth_grid] for v in VOLATILITY]
        im = ax.imshow(
            grid,
            extent=(91.95, 130.05, 4.5, 40.5),
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            cmap="viridis",
            vmin=0,
            vmax=100,
        )
        ax.axhline(100 * cutoff, color="white", ls="--", lw=0.85)
        ax.axvline(95, color="white", ls="--", lw=0.85)
        ax.set(xlim=(92, 130), xticks=[92, 100, 110, 120, 130], yticks=[5, 15, 25, 35, 40])
        ax.set_title(f"({chr(97 + index)})  {label}", fontsize=10, loc="left")
    fig.colorbar(
        im,
        ax=list(axes.flat),
        label="Equity allocation (%)",
        ticks=[0, 25, 50, 75, 100],
        shrink=0.9,
    )
    fig.supxlabel("Current portfolio wealth ($ millions)", fontsize=10)
    fig.supylabel("Annualized forecast volatility (%)", fontsize=10)
    fig.suptitle(
        f"{days[0].date} launch   ·   {remaining} sessions remaining   ·   {settings}",
        fontsize=10,
    )
    fig.savefig(OUT / filename)


def plot_cdf(curves, colors, styles, filename):
    # Preserve the original main panel's endpoints and default margins.
    left = min(map(min, series)) - 1
    right = max(map(max, series)) + 1
    fig = Figure(figsize=(7.1, 4.5), layout="constrained")
    ax = fig.subplots()
    inset = ax.inset_axes([0.58, 0.14, 0.385, 0.36])
    for (label, y), color, style in zip(curves.items(), colors, styles, strict=True):
        x = np.r_[min(left, min(y)), np.sort(y), max(right, max(y))]
        prob = np.r_[0, np.arange(1, len(y) + 1) / len(y), 1]
        for panel in (ax, inset):
            panel.step(x, prob, where="post", color=color, label=label, ls=style, lw=1.6)
        inset.scatter(95, np.mean(y <= 95), s=22, color=color, edgecolor="white", linewidth=0.5, zorder=4)
    for panel in (ax, inset):
        panel.axhline(0.02, color="#555555", ls="--", lw=0.9)
        panel.axvline(95, color="#666666", ls=":", lw=0.9)
        panel.grid(axis="y", alpha=0.16)
        panel.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    margin = (right - left) * rcParams["axes.xmargin"]
    ax.set(
        xlim=(left - margin, right + margin),
        xlabel="12-month terminal wealth ($ millions)",
        ylabel="Cumulative share of launches",
        ylim=(-0.025, 1.025),
    )
    inset.set(
        xlim=(94, 98), ylim=(0, 0.05), xticks=range(94, 99), yticks=[0, 0.02, 0.05]
    )
    inset.tick_params(labelsize=8)
    inset.set_title("Detail near the $95M floor", fontsize=8, pad=3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, frameon=False)
    fig.savefig(OUT / filename)


plot_allocations(traders, progression_labels, f"{multiple}× cushion multiplier", "wealth-volatility.pdf")
plot_cdf(overview, colors, styles, "terminal-cdf.pdf")

plot_cdf(alternatives, [colors[i] for i in (3, 0, 1, 2, 4)],
         ["-.", "--", "-", ":", "-"], "alternatives-cdf.pdf")

settings, = (
    r for r in smooth_settings
    if r["strategy"] == bt.VOL_CAPPED and r["start_month"] == launch.start_month
)
scale = settings["risk_scale"]
smooth_trader = bt.launch_trader(
    bt.VOL_CAPPED, multiple, None if settings["mode"] == "nominal warm-up" else scale,
    settings["scale_source"],
)
plot_allocations(
    [bt.ChanceConstrainedDPTrader(multiple), smooth_trader],
    comparison_labels, f"Full DP multiplier {multiple}×   ·   Cushion scale {scale:g}", "smooth-wealth-volatility.pdf",
)
plot_allocations(
    [StochasticDPTrader(), bt.ChanceConstrainedDPTrader(multiple), smooth_trader],
    ["Regular DP", *comparison_labels],
    f"Full DP multiplier {multiple}×   ·   Cushion scale {scale:g}",
    "slide-heatmaps.pdf", columns=3,
)
