import unittest
from dataclasses import replace
from datetime import date
from statistics import stdev
from unittest.mock import patch

from plot_allocation_heatmaps import ReferencePath
from plot_backtest import FIXED_SMOOTH, VOL_CAPPED, Session, launch_trader, replay, run_years
from cushion_selection import TIME_AWARE
from trader import FLOOR, TradingState, VolCappedSmoothCushionTrader, funding_floor


class VolCappedSmoothTests(unittest.TestCase):
    def setUp(self):
        self.trader = VolCappedSmoothCushionTrader(1.5)
        self.state = TradingState(0, 252, 0, 100e6, .20)

    def weight(self, millions, vol):
        state = replace(self.state, cash=millions * 1e6, annual_volatility=vol)
        return self.trader.trade(state) / state.cash

    def test_proposed_weights_and_heatmap_quotes(self):
        path = ReferencePath(self.trader, date(1999, 1, 4), 252, .1655)
        path.trade(replace(self.state, equity_value=30e6, cash=70e6))
        for millions, vol, percent in ((100, .20, 29.7), (112, .30, 48.2), (120, .30, 49.6), (120, .40, 24.0)):
            actual = self.weight(millions, vol) * 100
            self.assertAlmostEqual(actual, percent, delta=.1)
            self.assertAlmostEqual(path.allocation(252, vol, millions * 1e6), actual)

    def test_monotone_allocation_and_ceiling(self):
        wealths = (90, 95, 100, 102.5, 105, 112, 120, 130, 1e6)
        vols = (0., .05, .099, .1, .101, .2, .3, .4, 1.)
        for vol in vols:
            weights = [self.weight(w, vol) for w in wealths]
            self.assertEqual(weights, sorted(weights))
            self.assertGreaterEqual(weights[0], 0)
            self.assertLessEqual(weights[-1], 1 / (1 + (vol / .3)**4) + 1e-15)
        for w in wealths:
            weights = [self.weight(w, vol) for vol in vols]
            self.assertEqual(weights, sorted(weights, reverse=True))

    def test_slopes_at_old_multiplier_and_full_investment_boundaries(self):
        def slopes(function, x, step):
            return (function(x) - function(x-step)) / step, (function(x+step) - function(x)) / step

        # At 10% volatility the old rule first reached full investment in its
        # 7M-15M cushion band, where unscaled exposure is 3*cushion - 10M.
        floor = funding_floor(FLOOR, 252) / 1e6
        factor = 1.25 * 1.5 * 3
        corner = factor * (3 * floor + 10) / (3 * factor - 1)
        for function, x, step in [(lambda v: self.weight(corner, v), .10, 1e-7),
                                  (lambda w: self.weight(w, .10), corner, 1e-5)]:
            left, right = slopes(function, x, step)
            self.assertAlmostEqual(left, right, delta=1e-4)

    def test_floor_cash_terminal_and_extremes(self):
        floor = funding_floor(FLOOR, 252)
        for equity in (0, floor - 1, floor):
            self.assertEqual(self.trader.trade(replace(self.state, equity_value=equity, cash=0)), -equity)
        mixed = replace(self.state, equity_value=40e6, cash=60e6)
        self.assertEqual(VolCappedSmoothCushionTrader(0).trade(mixed), -40e6)
        self.assertEqual(self.trader.trade(replace(mixed, day=252, annual_volatility=None)), 0)
        with self.assertRaises(ValueError):
            self.trader.trade(replace(mixed, annual_volatility=None))
        self.assertAlmostEqual(self.trader.trade(mixed) + 40e6, self.trader.trade(self.state))
        for vol in (0., 1e308):
            order = self.trader.trade(replace(mixed, equity_value=1e308, cash=1e308, annual_volatility=vol))
            self.assertTrue(-1e308 <= order <= 1e308)

    def test_scale_commitment_warmup_and_standalone_replay(self):
        sessions = [Session(date(y, m, d), (-1)**d * .008, .0001)
                    for y in range(1991, 2005) for m in range(1, 13) for d in range(1, 9)]

        def choices(history, cutoff, conditions, limit):
            self.assertTrue(all(r.strategy == TIME_AWARE for rows in history.values() for r in rows))
            return {2000: (None, [{"mode": "nominal warm-up"}]),
                    2001: (1.25, [{"mode": "calibrated"}]),
                    2002: (0., [{"mode": "cash fallback"}])}[cutoff.year]
        with patch("plot_backtest.select_smooth_scale", side_effect=choices):
            settings = []
            rows, _ = run_years(sessions, [TIME_AWARE, VOL_CAPPED], 2000, 2002, smooth_launch_settings=settings)
            alone, _ = run_years(sessions, [VOL_CAPPED], 2000, 2002)
        self.assertEqual(alone, [r for r in rows if r.strategy == VOL_CAPPED])
        for incumbent, capped in zip(settings[::2], settings[1::2]):
            self.assertEqual(dict(incumbent, strategy=VOL_CAPPED), capped)
        warmup = launch_trader(VOL_CAPPED, None, None)
        self.assertEqual((warmup.annual_funding_rate, warmup.risk_scale), (0., 1.))
        # The synthetic 30-session sample alternates signs at constant magnitude.
        vol = stdev([s.equity_return for s in sessions[:30]]) * 252**.5
        for row, scale in zip(alone[::12], (None, 1.25, 0.)):
            days = [s for s in sessions if row.first_session <= s.date <= row.last_session]
            check = replay(VOL_CAPPED, launch_trader(VOL_CAPPED, None, scale), days,
                           vol, vol, [(vol, vol)] * len(days))
            self.assertAlmostEqual(row.terminal_wealth, check.terminal_wealth, places=6)
        with patch("plot_backtest.select_smooth_scale", side_effect=AssertionError("unexpected selection")):
            fixed_settings = []
            fixed, _ = run_years(sessions, [VOL_CAPPED, FIXED_SMOOTH], 2000, 2000,
                                 vol_capped_scale_source="fixed-1", smooth_launch_settings=fixed_settings)
            self.assertTrue(fixed)
            self.assertTrue(all(s["risk_scale"] == 1 and s["mode"] == "fixed" for s in fixed_settings))
