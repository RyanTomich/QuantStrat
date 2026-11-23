"""Stock allocation on hypothetical states along one historical launch path."""

from copy import deepcopy
from dataclasses import replace
from functools import partial

import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

from trader import ChanceConstrainedDPTrader, dp_order


WEALTH = np.arange(92_000_000, 112_000_001, 100_000)
VOLATILITY = np.arange(5, 41) / 100
TIME_VOLATILITY = (.15, .30)


class ReferencePath:
    """Forward real trades during one replay and retain independent quotes at sampled dates."""

    def __init__(self, trader, launch_date, sessions, cutoff, settings=""):
        self.trader = trader
        self.launch_date = launch_date
        self.sessions = sessions
        self.cutoff = cutoff
        self.settings = settings
        self.rows = sorted({sessions, *(n for n in (1, 5, 21, 63, 126, 252, 504) if n <= sessions)})
        self.snapshots = {}

    def trade(self, state):
        remaining = self.sessions - state.day
        if remaining not in self.rows:
            return self.trader.trade(state)
        if isinstance(self.trader, ChanceConstrainedDPTrader):
            # The real opening trade selects once. Even launch-day hypothetical
            # quotes must keep that choice when wealth or volatility changes.
            order = self.trader.trade(state)
            policy = self.trader.selected_policy
            quote = partial(dp_order, policy=policy)
            if state.day == 0:
                self.settings += f", penalty ${100 * policy.penalty:g}M" if policy else ", cash fallback"
        else:
            opening = deepcopy(self.trader)
            def quote(hypothetical):
                return deepcopy(opening).trade(hypothetical)
            order = self.trader.trade(state)
        self.snapshots[remaining] = (state, quote)
        return order

    def allocation(self, remaining, volatility, wealth):
        state, quote = self.snapshots[remaining]
        total = state.equity_value + state.cash
        if total == 0:
            return np.nan  # Asset proportions are undefined after a total loss.
        equity = (state.equity_value / total) * wealth
        hypothetical = replace(state, equity_value=equity, cash=wealth - equity,
                               annual_volatility=volatility, annual_volatility_p80=self.cutoff)
        return 100 * (equity + quote(hypothetical)) / wealth


def write_allocation_heatmaps(folder, names, paths, floor, *, expanded_grid=False):
    reference = next(iter(paths.values()), None)
    wealth = np.arange(92_000_000, 130_000_001, 100_000) if expanded_grid else WEALTH
    volatility = np.arange(20, 161) / 400 if expanded_grid else VOLATILITY
    half_vol_step = (volatility[1] - volatility[0]) / 2
    wealth_ticks = [92, 95, 100, 105, 110] + ([115, 120, 125, 130] if expanded_grid else [])
    for by_time, filename in ((False, "stock_allocation_by_volatility.png"),
                              (True, "stock_allocation_by_time.png")):
        columns = 2 if by_time else 1
        fig = Figure(figsize=(9, 2.8 * len(names) + 1.0), layout="constrained")
        axes = fig.subplots(len(names), columns, sharex=True, sharey=True, squeeze=False)
        title = "Stock allocation by " + ("time remaining" if by_time else "volatility")
        if reference:
            slice_day = min(252, reference.sessions)
            caption = (f"Launch {reference.launch_date}   ·   {reference.sessions} sessions"
                       f"   ·   launch volatility cutoff {reference.cutoff:.2%}")
            if not by_time:
                caption += f"   ·   {slice_day} sessions remaining"
            title += "\n" + caption
        fig.suptitle(title, fontsize=12)
        for row, name in enumerate(names):
            for col, ax in enumerate(axes[row]):
                ax.set_title(name, loc="left", fontsize=10)
                if not reference:
                    ax.text(.5, .5, "No eligible monthly starts", ha="center", transform=ax.transAxes)
                    ax.set_axis_off()
                    continue
                path = paths[name]
                title = name + (f"   ·   {path.settings}" if path.settings else "")
                if by_time:
                    title += f"\n{TIME_VOLATILITY[col]:.0%} forecast volatility"
                    slices = [(n, TIME_VOLATILITY[col]) for n in path.rows]
                    height = (-.5, len(slices) - .5)
                    ax.set_yticks(range(len(slices)), path.rows)
                    if col == 0:
                        ax.set_ylabel("Sessions remaining")
                else:
                    slices = [(slice_day, vol) for vol in volatility]
                    height = (volatility[0] - half_vol_step, volatility[-1] + half_vol_step)
                    ax.set_ylabel("Annualized volatility")
                    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
                    if volatility[0] <= path.cutoff <= volatility[-1]:
                        ax.axhline(path.cutoff, color="white", ls="--", lw=1)
                grid = [[path.allocation(n, vol, w) for w in wealth] for n, vol in slices]
                im = ax.imshow(grid, cmap="viridis", vmin=0, vmax=100, aspect="auto", origin="lower",
                               interpolation="nearest", extent=(91.95, wealth[-1] / 1e6 + .05, *height))
                ax.set_title(title, loc="left", fontsize=9)
                ax.axvline(floor / 1e6, color="white", ls="--", lw=1)
                ax.axvline(100, color="white", ls=":", lw=1)
                ax.set_xticks(wealth_ticks)
                if row == len(names) - 1:
                    ax.set_xlabel("Current portfolio wealth ($M)")
        if reference:
            fig.colorbar(im, ax=list(axes.flat), label="Stock allocation (%)", ticks=[0, 25, 50, 75, 100])
        fig.savefig(folder / filename, dpi=150)
