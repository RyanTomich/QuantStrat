"""Static figures built from the same annual outcomes as the backtest table."""

from itertools import groupby

from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter


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


def write_distribution(folder, results, names, conditions, floor_m, limits, breach_limit, horizon=12):
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


def write_figures(folder, results, names, conditions, floor, breach_limit, horizon=12, reference_paths=None,
                  *, expanded_grid=False):
    colors = ("#007088", "#a34917", "#952454")
    floor_m = floor / 1_000_000
    wealth = [floor_m, *(r.terminal_wealth / 1_000_000 for r in results)]
    pad = max(1, (max(wealth) - min(wealth)) * .05)
    limits = (min(wealth) - pad, max(wealth) + pad)

    def panels(title):
        fig = Figure(figsize=(8, 4.3 * len(names) + .4), layout="constrained")
        fig.suptitle(title, fontsize=16)
        axes = fig.subplots(len(names), 1, sharex=True, sharey=True, squeeze=False)[:, 0]
        for ax, name in zip(axes, names):
            ax.set_title(name, loc="left", fontsize=12)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=.17)
            ax.set_axisbelow(True)
        return fig, axes

    write_distribution(folder, results, names, conditions, floor_m, limits, breach_limit, horizon)
    diagnostic, scatter_axes = panels(f"Starting volatility and {horizon}-month terminal wealth")
    for name, scatter in zip(names, scatter_axes):
        rows = [r for r in results if r.strategy == name]
        scatter.axhspan(limits[0], floor_m, color="#fbe9e6", label="Below floor at T", zorder=0)
        scatter.axhline(floor_m, color="#505050", linestyle="--", linewidth=1, label=f"${floor_m:g}M floor")

        for condition, color in zip(conditions, colors):
            group = [r for r in rows if r.condition == condition]
            label = f"{condition.capitalize()} (n={len(group)})"
            scatter.scatter([r.starting_volatility for r in group],
                            [r.terminal_wealth / 1_000_000 for r in group],
                            color=color, alpha=.8, s=26, label=label)
        for index, row in enumerate(sorted(rows, key=lambda r: (r.terminal_wealth, r.start_month))[:3]):
            scatter.annotate(row.start_month, (row.starting_volatility, row.terminal_wealth / 1_000_000),
                             xytext=(5, 7 + index * 5), textcoords="offset points", fontsize=8)
        scatter.set(ylim=limits, ylabel="Terminal wealth ($ millions)")
        scatter.xaxis.set_major_formatter(PercentFormatter(1))
        scatter.legend(loc="upper left", bbox_to_anchor=(1, 1), frameon=False, fontsize=9)
        if not rows:
            scatter.text(.5, .5, "No eligible monthly starts", transform=scatter.transAxes, ha="center")
    scatter_axes[-1].set_xlabel("Starting daily equity volatility (prior 63 sessions)")
    diagnostic.savefig(folder / "starting_volatility_vs_wealth.png", dpi=160)

    if reference_paths is not None:
        from allocation_heatmaps import write_allocation_heatmaps
        write_allocation_heatmaps(folder, names, reference_paths, floor, expanded_grid=expanded_grid)
