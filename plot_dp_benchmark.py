"""Compare the four DP traders from completed backtest run directories."""

import argparse
import csv
from pathlib import Path
from statistics import fmean

from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

from backtest import BREACH_LIMIT, CONDITIONS, DP_VARIANTS, HYBRID


def plot(runs, output):
    names = [HYBRID, *DP_VARIANTS]
    labels = ["Combined baseline", "50/50 split", "Chance constrained", "Discounted floor"]
    fig = Figure(figsize=(10, 2.8 * len(runs) + .9), layout="constrained")
    axes = fig.subplots(len(runs), 2, squeeze=False, sharex="col")
    fig.suptitle("Terminal wealth and floor breaches", fontsize=16)
    for run, (wealth_ax, risk_ax) in zip(runs, axes):
        with (run / "annual_results.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        grouped = [[r for r in rows if r["strategy"] == name] for name in names]
        starts = {r["start_month"] for r in grouped[0]}
        if not starts or any({r["start_month"] for r in group} != starts for group in grouped):
            raise ValueError(f"Expected matching monthly starts for all four strategies in {run}")
        horizon = grouped[0][0]["horizon_months"]
        wealth_ax.set_title(f"{horizon} months   ·   {len(starts)} starts", loc="left")
        wealth_ax.set_yticks(range(len(names)), labels)
        risk_ax.set_yticks(range(len(names)), [])
        for ax in (wealth_ax, risk_ax):
            ax.set_ylim(len(names) - .5, -.5)
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.tick_params(axis="y", length=0)
            ax.grid(axis="x", alpha=.2)
            ax.set_axisbelow(True)
        means = [fmean(float(r["terminal_wealth"]) for r in group) / 1e6 for group in grouped]
        for i, mean in enumerate(means):
            gain = mean - means[0]
            wealth_ax.hlines(i, 0, gain, color="#7399ab", lw=2)
            wealth_ax.scatter(gain, i, color="#165873", zorder=3)
            wealth_ax.annotate(f"${mean:.2f}M", (gain, i), xytext=(7, 0),
                               textcoords="offset points", va="center", fontsize=9)
        for offset, marker, color, condition in zip((-.17, 0, .17), ("o", "s", "D"),
                                                  ("#146685", "#cf781b", "#87486c"), CONDITIONS):
            for i, group in enumerate(grouped):
                sample = [r for r in group if r["condition"] == condition]
                if sample:
                    rate = fmean(r["floor_breach"] == "True" for r in sample)
                    risk_ax.scatter(rate, i + offset, marker=marker, color=color, s=30,
                                    label=condition if i == 0 else None, zorder=3)
        risk_ax.axvline(BREACH_LIMIT, ls="--", lw=1.2, color="#333333", label="2% limit")
        risk_ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    axes[0, 1].set_title("Breaches by launch condition", loc="left")
    axes[-1, 0].margins(x=.27)
    axes[-1, 0].set_xlabel("Mean gain over combined baseline ($M)")
    axes[-1, 1].set_xlabel("Share finishing below $95M")
    axes[-1, 1].legend(loc="upper center", bbox_to_anchor=(.5, -.25), ncol=2,
                       frameon=False, fontsize=9)
    fig.savefig(output, dpi=165)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.runs, args.output)
