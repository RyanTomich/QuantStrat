"""Draw the combined DP's policy surfaces and its archived walk-forward evidence."""

import csv
from pathlib import Path
from statistics import fmean

import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from cushion_selection import HYBRID, MULTIPLIERS
from stochastic_dp import WealthPolicy
from trader import FLOOR


RUNS = Path(__file__).resolve().parent / "runs"
OUT = RUNS / "dp_progression"
AUDITS = ("20261119T000641Z-0go0ai2d", "20261119T000649Z-rppw23q4")
BENCHMARKS = ("20261120T173059Z-e6wzujbz", "20261120T173109Z-emwslak2")
SESSIONS = [1, 5, 21, 63, 126, 252, 504]
STAGES = [
    ("simple", "Simple DP", {}, (0, 96, 1)),
    ("cap", "Flat cushion cap", dict(multiplier=8, tiered=False, fractional=False), (0, 96, 1)),
    ("fractional", "Flat cap with fractional positions", dict(multiplier=8, tiered=False), (1, 98, 252)),
    ("combined", "Tiered cap with fractional positions", dict(multiplier=8), (0, 105, 63)),
]


def read(run, filename):
    with (RUNS / run / filename).open() as stream:
        return list(csv.DictReader(stream))


def policy_map(key, title, options, highlight):
    policy = WealthPolicy(FLOOR, **options)
    wealth = np.arange(92, 112.01, .1).round(1)
    fig = Figure(figsize=(10, 3.5), layout="constrained")
    axes = fig.subplots(1, 2, sharey=True)
    fig.suptitle(title, fontsize=16, fontweight="semibold")
    for regime, ax in enumerate(axes):
        grid = [[100 * policy.weight(day, value * 1e6, bool(regime))
                 for value in wealth] for day in SESSIONS]
        im = ax.imshow(grid, vmin=0, vmax=100, cmap="viridis", origin="lower",
                       aspect="auto", interpolation="nearest", extent=(91.95, 112.05, -.5, 6.5))
        ax.set(title=["Ordinary volatility", "High volatility"][regime],
               xticks=[92, 95, 100, 105, 110], xlabel="Current wealth ($M)",
               yticks=range(len(SESSIONS)), yticklabels=SESSIONS)
        ax.axvline(95, color="white", ls="--", lw=1.3)
        ax.axvline(100, color="white", ls=":", lw=.85)
        if regime == highlight[0]:
            _, w, d = highlight
            ax.scatter(w, SESSIONS.index(d), marker="o", s=70, facecolors="none",
                       edgecolors="white", lw=2)
            ax.scatter(w, SESSIONS.index(d), marker="o", s=92, facecolors="none",
                       edgecolors="#303030", lw=.8)
    axes[0].set_ylabel("Trading sessions remaining")
    cb = fig.colorbar(im, ax=list(axes), ticks=[0, 25, 50, 75, 100], shrink=.85)
    cb.set_label("Equity allocation (%)")
    fig.savefig(OUT / f"{key}.png", dpi=180)


def selection_map():
    fig = Figure(figsize=(10, 4.2), layout="constrained")
    a, b = fig.subplots(2, 1, sharex=True)
    palette = ListedColormap(["#eef4f5", "#bed8e5", "#7eacc9", "#f5a77b", "#bb514e", "#793843"])
    bounds = [0, 1e-8, 1, 2, 4, 6, 100]
    for ax, run, horizon in zip((a, b), AUDITS, (12, 24)):
        data = read(run, "selection_history.csv")
        years = sorted({int(r["selection_date"][:4]) for r in data})
        mat = np.full((len(MULTIPLIERS), len(years)), np.nan)
        chosen = {}
        for r in data:
            y = int(r["selection_date"][:4])
            i = MULTIPLIERS.index(int(r["multiplier"]))
            rate = 100 * max(float(r[k + "_rate"]) for k in ("lower-volatility", "higher-volatility"))
            # Assign exact endpoints to the lower bin; selection accepts exactly 2%.
            mat[i, y - years[0]] = max(0, np.nextafter(rate, -np.inf))
            chosen[y] = MULTIPLIERS.index(int(r["selected_multiplier"]))
        ax.imshow(mat, origin="lower", aspect="auto", cmap=palette, norm=BoundaryNorm(bounds, palette.N),
                  extent=(years[0]-.5, years[-1]+.5, -.5, len(MULTIPLIERS)-.5), interpolation="nearest")
        ax.plot(chosen.keys(), chosen.values(), ".", ms=3, color="#172837")
        ax.set(title=f"{horizon}-month investments", yticks=range(len(MULTIPLIERS)),
               yticklabels=MULTIPLIERS, ylabel="Multiplier")
    b.set_xlabel("January selection year")
    fig.suptitle("Historical breach rates visible when selecting a multiplier", fontsize=15, fontweight="semibold")
    labels = ["0%", "0–1%", "1–2%", "2–4%", "4–6%", ">6%"]
    handles = [Line2D([], [], marker="s", color=c, ls="", label=l) for c, l in zip(palette.colors, labels)]
    handles.append(Line2D([], [], marker=".", color="#172837", ls="", label="Selected"))
    b.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, -.44), ncol=7, frameon=False, fontsize=8)
    fig.savefig(OUT / "selection.png", dpi=180)


def outcomes_map():
    names = [HYBRID, "split-cushion-dp", "chance-constrained-dp", "discounted-floor-dp"]
    labels = ["Combined DP", "50/50 split", "Chance-constrained DP", "Discounted-floor DP"]
    fig = Figure(figsize=(10, 3.15), layout="constrained")
    axes = fig.subplots(1, 2, sharey=True)
    means, worst = [], []
    for run in BENCHMARKS:
        rows = read(run, "annual_results.csv")
        groups = [[r for r in rows if r["strategy"] == n] for n in names]
        means.append([fmean(float(r["terminal_wealth"]) for r in group) / 1e6 for group in groups])
        worst.append([100 * max(fmean(r["floor_breach"] == "True" for r in group if r["condition"] == c)
                                for c in {r["condition"] for r in group}) for group in groups])
    gains = np.array(means).T - np.array(means)[:, 0]
    for i, (ax, values) in enumerate(zip(axes, (gains, np.array(worst).T))):
        if i:
            colors = ListedColormap(["#e4f1ed", "#f5c3b9"])
            ax.imshow(values >= 2, cmap=colors, vmin=0, vmax=1, aspect="auto")
        else:
            ax.imshow(values, cmap=ListedColormap(["#f0f5f8", "#dfecf2", "#b8d4e3", "#8fbad2"]),
                      vmin=0, vmax=values.max(), aspect="auto")
        for row, col in np.ndindex(values.shape):
            val = values[row, col]
            label = f"{val:.2f}%" if i else f"{val:+.3f}" if val else "0.000"
            ax.text(col, row, label, ha="center", va="center", fontsize=12, color="#182838")
        ax.set(xticks=[0, 1], xticklabels=["12 months", "24 months"], yticks=range(4), yticklabels=labels)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0].set_title("Mean terminal gain over combined ($M)")
    axes[1].set_title("Highest breach rate across launch groups")
    fig.savefig(OUT / "challengers.png", dpi=180)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for stage in STAGES:
        policy_map(*stage)
    selection_map()
    outcomes_map()
