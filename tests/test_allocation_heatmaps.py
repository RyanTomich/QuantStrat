import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np

from allocation_heatmaps import ReferencePath, TIME_VOLATILITY, VOLATILITY, WEALTH
from backtest import INITIAL, Session, main, replay
from trader import (ChanceConstrainedDPTrader, FixedSplitTrader, SplitDPTrader,
                    StochasticDPTrader, TradingState)


class AllocationHeatmapTests(unittest.TestCase):
    def test_fixed_split_grids_and_short_horizon(self):
        path = ReferencePath(FixedSplitTrader(), date(2020, 1, 1), 3, .2)
        for day in range(3):
            path.trade(TradingState(day, 3, 0, INITIAL, .15, .20))
        self.assertEqual(path.rows, [1, 3])
        for remaining, vol in [(3, v) for v in VOLATILITY] + [
                (n, v) for n in path.rows for v in TIME_VOLATILITY]:
            np.testing.assert_allclose([path.allocation(remaining, vol, w) for w in WEALTH], 60.)

    def test_launch_commitment_and_independent_stateful_quotes(self):
        for trader in (SplitDPTrader(8), ChanceConstrainedDPTrader(8)):
            with self.subTest(trader=type(trader).__name__):
                path = ReferencePath(trader, date(2000, 1, 1), 6, .20)
                returns = [.07, -.11, .04, -.05, .12, .03]
                days = [Session(path.launch_date + timedelta(days=i), r, .0001)
                        for i, r in enumerate(returns)]
                signals = [(.30 if i % 2 else .15, .20) for i in range(6)]
                actual = []
                replay("test", path, days, .15, .2, signals,
                       lambda wealth, equity: actual.append((wealth, 100 * equity / wealth)))
                # A plot can be drawn after the entire path has completed, in any cell order.
                with patch("stochastic_dp.chance_policy", side_effect=AssertionError("reselected")):
                    for n in path.rows:
                        i = path.sessions - n
                        w, target = actual[i]
                        self.assertAlmostEqual(path.allocation(n, signals[i][0], w), target)
                        grid = [(v, w) for v in (.15, .20, .30) for w in (96e6, 100e6, 105e6)]
                        forward = [path.allocation(n, v, w) for v, w in grid]
                        reverse = [path.allocation(n, v, w) for v, w in reversed(grid)]
                        self.assertEqual(forward, reverse[::-1])

    def test_launch_cutoff_and_shared_grid_cell(self):
        path = ReferencePath(StochasticDPTrader(), date(2020, 1, 1), 1, .15)
        path.trade(TradingState(0, 1, 0, INITIAL, .30, .15))
        n, w = path.rows[0], WEALTH[80]
        self.assertEqual(w, INITIAL)
        self.assertEqual(path.allocation(n, TIME_VOLATILITY[0], w),
                         path.allocation(min(path.sessions, 252), VOLATILITY[10], w))
        self.assertEqual(path.allocation(n, .15, w), 100.)
        self.assertEqual(path.allocation(n, np.nextafter(.15, 1.), w), 25.)

    def test_cli_standard_outputs_and_empty_run(self):
        with tempfile.TemporaryDirectory() as temp:
            previous = os.getcwd()
            self.addCleanup(os.chdir, previous)
            os.chdir(temp)
            data = Path("market.csv")
            start = date(2019, 1, 1)
            data.write_text(",Mkt-RF,RF\n" + "".join(
                f"{start + timedelta(days=i):%Y%m%d},0.01,0.00\n" for i in range(731)))
            for year, names in (("2020", ["fixed-split"]), ("2018", ["fixed-split", "vol-target"])):
                stdout = io.StringIO()
                options = [opt for name in names for opt in ("--strategy", name)]
                with redirect_stdout(stdout):
                    main(["--data", str(data), "--start-year", year, "--end-year", year,
                          "--horizon-months", "1", *options])
                folder = Path(stdout.getvalue().splitlines()[-1])
                summary = (folder / "summary.md").read_text()
                for name in ("terminal_wealth_distribution", "stock_allocation_by_time",
                             "stock_allocation_by_volatility", "starting_volatility_vs_wealth"):
                    self.assertIn(f"({name}.png)", summary)
                    self.assertGreater((folder / f"{name}.png").stat().st_size, 1000)
                self.assertEqual("no reference launch" in summary, year == "2018")
