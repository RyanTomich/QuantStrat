"""Regenerate the DP progression and smooth-cushion comparison from the archived monthly launch sample."""

import argparse
import csv
import sys
from pathlib import Path
from functools import cache
from statistics import harmonic_mean, stdev
import numpy as np
from matplotlib.figure import Figure
from matplotlib import rcParams
from matplotlib.ticker import PercentFormatter

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))
from allocation_heatmaps import ReferencePath, VOLATILITY
import backtest as bt
from stochastic_dp import ChancePolicy, WealthPolicy
from trader import StochasticDPTrader, FLOOR

run = ROOT / Path("runs/20261120T175115Z-84zev02c")
comparison_runs = {
    12: ROOT / "runs/20261120T184542Z-lhqkdp2f",
    24: ROOT / "runs/20261120T184548Z-uzs9td6_",
}


def read(name, folder=run):
    with (folder / name).open() as f:
        return list(csv.DictReader(f))


def write_comparison_table():
    lines = [
        r"\begin{tabular}{@{}llrrrrr@{}}",
        r"\hline",
        r"& & & \multicolumn{2}{c}{Chance-constrained DP} & \multicolumn{2}{c}{Smooth cushion} \\",
        r"\cline{4-5}\cline{6-7}",
        r"Horizon & Launch condition & Launches & CE (\$M) & Breaches (\%) & CE (\$M) & Breaches (\%) \\",
        r"\hline",
    ]
    for months, folder in comparison_runs.items():
        results = read("annual_results.csv", folder)
        samples = []
        for strategy in ("chance-constrained-dp", bt.VOL_CAPPED):
            sample = {}
            for row in results:
                if row["strategy"] != strategy:
                    continue
                key = (row["first_session"], row["last_session"], row["condition"])
                if key in sample or int(row["horizon_months"]) != months:
                    raise ValueError(f"Unexpected launch for {strategy} in {folder}: {key}")
                sample[key] = float(row["terminal_wealth"])
            samples.append(sample)
        if samples[0].keys() != samples[1].keys():
            raise ValueError(f"Strategies have different launches in {folder}")

        if months != next(iter(comparison_runs)):
            lines.append(r"\hline")
        for condition in ("all", *bt.CONDITIONS):
            keys = [key for key in samples[0] if condition == "all" or key[2] == condition]
            label = "All launches" if condition == "all" else condition.replace("-", " ").capitalize()
            cells = [f"{months} months" if condition == "all" else "", label, str(len(keys))]
            for sample in samples:
                wealth = [sample[key] for key in keys]
                breaches = sum(w < FLOOR for w in wealth)
                rate = f"{100 * breaches / len(wealth):.2f}" if breaches else "0"
                cells.extend([f"{harmonic_mean(wealth) / 1e6:.2f}", rf"{breaches} ({rate}\%)"])
            lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\hline", r"\end{tabular}"])
    (OUT / "strategy-comparison.tex").write_text("\n".join(lines) + "\n")


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--table-only", action="store_true", help="Generate both report tables and skip plotting.")
args = parser.parse_args()
write_comparison_table()

rows = read("annual_results.csv")
mult = {
    r["selection_date"][:4]: int(r["selected_multiplier"])
    for r in read("selection_history.csv")
}


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
offset = {s.date.isoformat(): i for i, s in enumerate(sessions)}
ret = [s.equity_return for s in sessions]
signals = bt.annual_volatility_signals(
    [stdev(ret[i : i + 63]) for i in range(len(ret) - 63)]
)
base = [r for r in rows if r["strategy"] == "stochastic-dp"]


def archived(name, source=rows):
    group = [r for r in source if r["strategy"] == name]
    assert [
        (r["first_session"], r["last_session"], r["starting_volatility"]) for r in group
    ] == [
        (r["first_session"], r["last_session"], r["starting_volatility"]) for r in base
    ]
    return np.array([float(r["terminal_wealth"]) for r in group]) / 1e6


def launch_inputs(r):
    a, b = offset[r["first_session"]], offset[r["last_session"]] + 1
    return (
        sessions[a:b],
        float(r["starting_volatility"]),
        float(r["median_volatility"]),
        signals[a - 63 : b - 63],
    )


gap = []
for r in base:
    inputs = launch_inputs(r)
    if len(gap) % 100 == 0:
        assert (
            abs(
                bt.replay("base", StochasticDPTrader(), *inputs).terminal_wealth
                - float(r["terminal_wealth"])
            )
            < 0.01
        )
    gap.append(
        bt.replay("gap", GapTrader(mult[r["year"]]), *inputs).terminal_wealth / 1e6
    )
series = [
    archived("stochastic-dp"),
    archived(bt.HYBRID),
    np.array(gap),
    archived("chance-constrained-dp"),
]

smooth_run = comparison_runs[12]
comparison = read("annual_results.csv", smooth_run)
dp = archived("chance-constrained-dp", comparison)
np.testing.assert_array_equal(dp, series[-1])
smooth = archived(bt.VOL_CAPPED, comparison)

overview = {
    "Standard DP": series[0],
    "DP with equity cap": series[1],
    "Capped DP with downside shocks": series[2],
    "Chance-constrained DP": dp,
    "Smooth cushion": smooth,
}
conditions = np.array([r["condition"] for r in base])
floor_millions = FLOOR / 1e6
eligible = {
    name for name, wealth in overview.items()
    if all(np.mean(wealth[conditions == c] < floor_millions) <= bt.BREACH_LIMIT
           for c in bt.CONDITIONS)
}
lines = [
    r"\begin{tabular}{@{}lrrr@{}}", r"\hline",
    r"Strategy & Objective: CE (\$M) & Mean $W_T$ (\$M) & Breaches (\%) \\", r"\hline",
]
for condition in ("all", *bt.CONDITIONS):
    mask = np.ones(len(base), dtype=bool) if condition == "all" else conditions == condition
    label = "All launches" if condition == "all" else condition.replace("-", " ").capitalize()
    lines.append(rf"\multicolumn{{4}}{{@{{}}l}}{{\itshape {label} ($n={mask.sum()}$)}} \\")
    metrics = {}
    for name, wealth in overview.items():
        sample = wealth[mask]
        metrics[name] = (harmonic_mean(sample), sample.mean(), np.count_nonzero(sample < floor_millions))
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
if args.table_only:
    sys.exit(0)

labels = [
    "Standard DP",
    "+ Equity cap",
    "+ Downside shock",
    "+ Terminal chance constraint",
]
colors = ["#435465", "#007a72", "#b55c16", "#623f8d"]
styles = ["-", "--", ":", "-."]
rcParams.update({"font.size": 9, "pdf.fonttype": 42})

launch = max(base, key=lambda row: row["first_session"])
days, daily_vol, median_vol, forecasts = launch_inputs(launch)
multiple = mult[launch["year"]]
cutoff = forecasts[0][1]
remaining = min(len(days), 252)
traders = [
    StochasticDPTrader(),
    StochasticDPTrader(multiple),
    GapTrader(multiple),
    bt.ChanceConstrainedDPTrader(multiple),
]

def plot_allocations(traders, labels, settings, filename):
    rows = len(traders) // 2
    fig = Figure(figsize=(7.2, 2.0 * rows + 0.9), layout="constrained")
    axes = fig.subplots(rows, 2, sharex=True, sharey=True)
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


# Use the four-stage plot's endpoints for both CDFs, including its default margins.
left = min(map(min, series)) - 1
right = max(map(max, series)) + 1


def plot_cdf(series, labels, colors, styles, filename, *, inset_max=96):
    fig = Figure(figsize=(7.1, 4.5), layout="constrained")
    ax = fig.subplots()
    inset = ax.inset_axes([0.58, 0.14, 0.385, 0.36])
    for y, label, color, style in zip(series, labels, colors, styles):
        x = np.r_[left, np.sort(y), right]
        prob = np.r_[0, np.arange(1, len(y) + 1) / len(y), 1]
        for panel in (ax, inset):
            panel.step(x, prob, where="post", color=color, label=label, ls=style, lw=1.6)
        inset.scatter(95, np.mean(y <= 95), s=22, color=color, edgecolor="white", linewidth=0.5, zorder=4)
    for panel in (ax, inset):
        panel.axhline(0.02, color="#555555", ls="--", lw=0.9)
        panel.axvline(95, color="#666666", ls=":", lw=0.9)
        panel.grid(axis="y", alpha=0.16)
        panel.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set(
        xlabel="12-month terminal wealth ($ millions)",
        ylabel="Cumulative share of launches",
        ylim=(-0.025, 1.025),
    )
    inset.set(
        xlim=(94, inset_max), ylim=(0, 0.05), xticks=range(94, inset_max + 1), yticks=[0, 0.02, 0.05]
    )
    inset.tick_params(labelsize=8)
    inset.set_title("Detail near the $95M floor", fontsize=8, pad=3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False)
    fig.savefig(OUT / filename)


plot_allocations(traders, labels, f"{multiple}× cushion multiplier", "wealth-volatility.pdf")
plot_cdf(series, labels, colors, styles, "terminal-cdf.pdf")

settings, = (
    r for r in read("smooth_launch_settings.csv", smooth_run)
    if r["strategy"] == bt.VOL_CAPPED and r["start_month"] == launch["start_month"]
)
assert settings["mode"] == "calibrated"
scale = float(settings["risk_scale"])
pair = ["Chance-constrained DP", "Smooth cushion"]
plot_allocations(
    [bt.ChanceConstrainedDPTrader(multiple), bt.VolCappedSmoothCushionTrader(scale)],
    pair, f"DP multiplier {multiple}×   ·   smooth scale {scale:g}", "smooth-wealth-volatility.pdf",
)
plot_cdf([dp, smooth], pair, [colors[-1], colors[1]], [styles[-1], "-"], "smooth-terminal-cdf.pdf", inset_max=98)
