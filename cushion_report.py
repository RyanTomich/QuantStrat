"""Distribution and paired comparisons for the walk-forward cushion strategy."""

from math import ceil

import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

from cushion_selection import HYBRID


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
        return "insufficient contiguous starts"
    rng = np.random.default_rng(1950)
    estimates = []
    for _ in range(2000):
        blocks = rng.choice(starts, size=ceil(n / size))
        index = (blocks[:, None] + np.arange(size)).ravel()[:n]
        a, b = hybrid[index], other[index]
        estimates.append([ce(a) - ce(b), (a - b).mean()])
    lo, hi = np.percentile(estimates, [2.5, 97.5], axis=0) / 1e6
    return f"CE [{lo[0]:+.3f}, {hi[0]:+.3f}], mean [{lo[1]:+.3f}, {hi[1]:+.3f}]"


def comparison_report(folder, results, names, floor):
    rows = {name: sorted((r for r in results if r.strategy == name), key=lambda r: r.start_month)
            for name in names}
    starts = [r.start_month for r in rows[HYBRID]]
    months = np.array([12 * int(m[:4]) + int(m[5:]) for m in starts])
    wealth = {}
    lines = ["", "## Walk-forward historical comparison", "",
             "Terminal wealth and shortfall in millions of dollars. Worst 1% averages the lowest "
             "ceil(1% × starts) outcomes. Shortfall is the largest terminal deficit below the floor.", "",
             "| Strategy | Mean | Median | 5th percentile | Worst 1% mean | Largest shortfall |",
             "|---|---:|---:|---:|---:|---:|"]
    fig = Figure(figsize=(10, 4.5), layout="constrained")
    ax, zoom = fig.subplots(1, 2)
    floor_m = floor / 1e6
    for name, group in rows.items():
        if [r.start_month for r in group] != starts:
            raise ValueError("Paired comparison requires identical monthly starts")
        wealth[name] = np.array([r.terminal_wealth for r in group])
        values = np.sort(wealth[name]) / 1e6
        lines.append(f"| {name} | {values.mean():.3f} | {np.median(values):.3f} | "
                     f"{np.percentile(values, 5):.3f} | {values[:ceil(.01 * len(values))].mean():.3f} | "
                     f"{max(0, floor_m - values[0]):.3f} |")
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
    lines.extend(["", "![Terminal wealth and floor detail](cushion_distribution.png)", "",
                  "Paired gains in millions of dollars. Intervals are 95% percentiles from 2,000 "
                  "paired moving-block samples of 12 consecutive starts (seed 1950).", ""])
    for name in names:
        if name == HYBRID:
            continue
        a, b = wealth[HYBRID], wealth[name]
        lines.append(f"- {HYBRID} minus {name}: CE {(ce(a) - ce(b))/1e6:+.3f}, "
                     f"mean {(a - b).mean()/1e6:+.3f}. Intervals: {paired_intervals(months, a, b)}.")
    if HYBRID in rows and "smooth-cushion-4x" in wealth and "stochastic-dp" in wealth:
        smooth = ce(wealth["smooth-cushion-4x"])
        advantage = ce(wealth["stochastic-dp"]) - smooth
        if advantage > 0:
            lines.extend(["", f"Share of the uncapped DP's CE advantage over smooth cushion retained: "
                          f"{(ce(wealth[HYBRID]) - smooth) / advantage:.1%}."])
    lines.extend(["", "### Results by starting decade and launch state", "",
                  "| Decade | Launch state | Strategy | Starts | CE ($M) | Mean ($M) | Breaches |",
                  "|---|---|---|---:|---:|---:|---:|"])
    buckets = sorted({(r.year // 10 * 10, r.condition) for r in rows[HYBRID]})
    for decade, condition in buckets:
        for name, group in rows.items():
            subset = [r for r in group if r.year // 10 * 10 == decade and r.condition == condition]
            values = np.array([r.terminal_wealth for r in subset]) / 1e6
            lines.append(f"| {decade}s | {condition} | {name} | {len(values)} | {ce(values):.3f} | "
                         f"{values.mean():.3f} | {sum(r.floor_breach for r in subset)} |")
    return lines
