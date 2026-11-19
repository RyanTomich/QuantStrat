import unittest
from sys import float_info

from trader import (
    AggressiveTieredCushionTrader,
    FixedSplitTrader, FloorAwareGlideTrader, FloorAwareVolatilityTargetTrader,
    GentleGlideTrader, HighVolatilityReductionTrader, SmoothCushionTrader,
    Trader, TradingState, VolatilityScaledCushionTrader, VolatilityTargetTrader,
)


class VolatilityTargetTests(unittest.TestCase):
    def test_smooth_cushion_targets(self):
        trader = SmoothCushionTrader()
        for wealth, forecast, expected in [
            (94, .15, 0), (98, .15, 7.5), (100, .15, 15), (102, .15, 27.5),
            (110, .15, 87.5), (115, .30, 64.0625), (120, .30, 87.5), (125, .30, 112.5),
            (120, .15, 120), (100, 0, 22.5),
        ]:
            state = TradingState(0, 2, 50e6, (wealth - 50) * 1e6, forecast)
            self.assertAlmostEqual((state.equity_value + trader.trade(state)) / 1e6, expected)
        self.assertEqual(trader.trade(TradingState(2, 2, 0, 100e6)), 0)

    def test_smooth_cushion_marginal_exposure_at_boundaries(self):
        trader = SmoothCushionTrader()
        for wealth, slope in [(98e6, 1), (102e6, 3), (110e6, 3), (120e6, 4)]:
            targets = [trader.trade(TradingState(0, 2, 0, wealth + shift, .30))
                       for shift in (-1, 0, 1)]
            for left, right in zip(targets, targets[1:]):
                self.assertAlmostEqual(right - left, 1.25 * slope, places=5)

    def test_aggressive_upper_tier(self):
        trader = AggressiveTieredCushionTrader()
        for wealth, expected in [(95, 0), (99, 8), (100, 10), (101, 16), (110, 70), (120, 120)]:
            state = TradingState(1, 2, 50e6, (wealth - 50) * 1e6, .15)
            self.assertAlmostEqual((state.equity_value + trader.trade(state)) / 1e6, expected)

    def test_volatility_scaled_cushion_positions(self):
        trader = VolatilityScaledCushionTrader()
        for equity, cash, forecast, expected in [
            (0, 100, .10, 15), (0, 100, .15, 10), (15, 85, .30, 5),
            (0, 100, 0, 15), (10, 85, .12, 0), (5, 89, .15, 0),
            (10, 190, .08, 200), (0, 100, .60, 2.5),
        ]:
            with self.subTest(equity=equity, cash=cash, forecast=forecast):
                state = TradingState(0, 2, equity * 1e6, cash * 1e6, forecast)
                self.assertAlmostEqual(state.equity_value + trader.trade(state), expected * 1e6)
        state = TradingState(0, 2, float_info.max, float_info.max, .60)
        self.assertAlmostEqual(trader.trade(state), -47_500_000)

    def test_weights(self):
        continuous = VolatilityTargetTrader()
        conditional = HighVolatilityReductionTrader()
        for trader, forecast, p80, expected in [
            (continuous, 0, None, 100), (continuous, .15, None, 60),
            (continuous, .30, None, 30), (VolatilityTargetTrader(.12), .30, None, 40),
            (conditional, .21, .22, 60), (conditional, .22, .22, 60),
            (conditional, .30, .22, 30), (conditional, .10, .08, 60),
        ]:
            for equity in (0, 85):
                with self.subTest(trader=trader, forecast=forecast, equity=equity):
                    state = TradingState(0, 2, equity, 100 - equity, forecast, p80)
                    order = trader.trade(state)
                    self.assertAlmostEqual(equity + order, expected)
                    self.assertTrue(-equity <= order <= state.cash)

    def test_floor_aware_positions(self):
        trader = FloorAwareVolatilityTargetTrader()
        for equity, cash, expected in [(0, 100, 10), (10, 87, 4), (10, 85, 0),
                                       (10, 84, 0), (100, 100, 120)]:
            state = TradingState(0, 1, equity * 1e6, cash * 1e6, .15)
            self.assertAlmostEqual(state.equity_value + trader.trade(state), expected * 1e6)

    def test_required_inputs_and_terminal_day(self):
        for trader in (VolatilityTargetTrader(), FloorAwareVolatilityTargetTrader(),
                       HighVolatilityReductionTrader(), VolatilityScaledCushionTrader()):
            with self.assertRaisesRegex(ValueError, "annual_volatility"):
                trader.trade(TradingState(0, 1, 90, 10))
            self.assertEqual(trader.trade(TradingState(1, 1, 90, 10)), 0)
        with self.assertRaisesRegex(ValueError, "annual_volatility_p80"):
            HighVolatilityReductionTrader().trade(TradingState(0, 1, 90, 10, .15))
        for value in (-1, float("inf"), float("nan")):
            for signals in [(value, None), (.15, value)]:
                with self.assertRaises(ValueError):
                    TradingState(0, 1, 0, 100, *signals)
            with self.assertRaises(ValueError):
                VolatilityTargetTrader(value)
        with self.assertRaises(ValueError):
            VolatilityTargetTrader(0)


class GlidePathTests(unittest.TestCase):
    def test_gentle_glide(self):
        for day, target in enumerate([65, 50, 35]):
            self.assertAlmostEqual(GentleGlideTrader().trade(TradingState(day, 3, 0, 100)), target)

    def test_floor_aware_equity_targets(self):
        # Day, equity, cash, and expected post-trade equity in millions.
        cases = [(0, 0, 100, 10), (0, 15, 82, 4), (1, 20, 75, 0),
                 (1, 20, 74, 0), (2, 40, 85, 25), (3, 40, 54, 40)]
        for day, equity, cash, expected in cases:
            with self.subTest(day=day, cash=cash):
                state = TradingState(day, 3, equity * 1e6, cash * 1e6)
                order = FloorAwareGlideTrader().trade(state)
                self.assertAlmostEqual(state.equity_value + order, expected * 1e6)

    def test_schedule_across_balances(self):
        trader = GentleGlideTrader()
        for day, weight in enumerate([.65, .575, .50, .425, .35]):
            for equity, cash in [(0, 100), (180, 20)]:
                with self.subTest(day=day, equity=equity):
                    order = trader.trade(TradingState(day, 5, equity, cash))
                    self.assertAlmostEqual((equity + order) / (equity + cash), weight)

    def test_single_session_and_terminal_valuation(self):
        trader = GentleGlideTrader()
        self.assertEqual(trader.trade(TradingState(0, 1, 0, 100)), 65)
        for horizon in (1, 5):
            self.assertEqual(trader.trade(TradingState(horizon, horizon, 90, 10)), 0)


class FixedSplitTests(unittest.TestCase):
    def test_daily_orders(self):
        cases = [
            # day, equity, cash, target weight, expected dollar order
            (0, 0, 100_000_000, 0.6, 60_000_000),
            (1, 66_000_000, 40_000_000, 0.6, -2_400_000),
            (2, 54_000_000, 40_000_000, 0.6, 2_400_000),
            (3, 60, 40, 0.6, 0),
            (4, 60, 40, 0, -60),
            (5, 60, 40, 1, 40),
            (6, 0, 0, 0.6, 0),
            (251, 60, 40, 0.7, 10),
        ]
        for day, equity, cash, weight, expected in cases:
            with self.subTest(day=day):
                trader: Trader = FixedSplitTrader(weight)
                order = trader.trade(TradingState(day, 252, equity, cash))
                self.assertAlmostEqual(order, expected)
                self.assertGreaterEqual(cash - order, 0)
                self.assertGreaterEqual(equity + order, 0)
                self.assertAlmostEqual((cash - order) + (equity + order), cash + equity)
                self.assertAlmostEqual(equity + order, weight * (equity + cash))

    def test_default_and_terminal_valuation(self):
        trader = FixedSplitTrader()
        self.assertEqual(trader.trade(TradingState(0, 504, 0, 100)), 60)
        self.assertEqual(trader.trade(TradingState(504, 504, 80, 20)), 0)

    def test_large_balances(self):
        state = TradingState(0, 252, float_info.max, float_info.max)
        self.assertEqual(FixedSplitTrader(0.5).trade(state), 0)
        self.assertEqual(FixedSplitTrader(1).trade(TradingState(0, 1, 1e20, 1)), 1)

    def test_invalid_weights_and_balances(self):
        for value in (-1, float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    FixedSplitTrader(value)
                with self.assertRaises(ValueError):
                    TradingState(0, 1, value, 0)
                with self.assertRaises(ValueError):
                    TradingState(0, 1, 0, value)
        with self.assertRaises(ValueError):
            FixedSplitTrader(1.01)

    def test_invalid_trading_days(self):
        for day, horizon in [(-1, 1), (2, 1), (0, 0), (0.5, 1), (0, 1.5), (True, 1)]:
            with self.subTest(day=day, horizon=horizon), self.assertRaises(ValueError):
                TradingState(day, horizon, 0, 100)
