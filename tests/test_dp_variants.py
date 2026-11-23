import unittest
from datetime import date, timedelta

from plot_backtest import Session, replay
from stochastic_dp import ChancePolicy, WealthPolicy, cap_weight
from trader import (FLOOR, ChanceConstrainedDPTrader, DiscountedFloorDPTrader,
                    SplitDPTrader, StochasticDPTrader, TradingState)


class VariantTests(unittest.TestCase):
    def test_split_reproduces_independent_sleeves_daily(self):
        returns = [.08, -.15, .03, -.02, .10, -.08]
        days = [Session(date(2000, 1, 1) + timedelta(days=i), r, .0001)
                for i, r in enumerate(returns)]
        signals = [(.15 if i % 2 == 0 else .30, .20) for i in range(len(days))]

        def trace(trader):
            path = []
            result = replay("test", trader, days, .15, .2, signals,
                            lambda wealth, equity: path.append((wealth, equity)))
            path.append((result.terminal_wealth, 0))
            return path

        split = trace(SplitDPTrader(8))
        for combined, capped, uncapped in zip(split, trace(StochasticDPTrader(8)),
                                               trace(StochasticDPTrader())):
            for actual, a, b in zip(combined, capped, uncapped):
                self.assertAlmostEqual(actual, (a + b) / 2, delta=1e-6)

    def test_split_repeated_quote_uses_same_opening_balances(self):
        trader = SplitDPTrader(8)
        trader.trade(TradingState(0, 3, 0, 100e6, .15, .20))
        state = TradingState(1, 3, 50e6, 47e6, .30, .20)
        self.assertEqual(trader.trade(state), trader.trade(state))

    def test_discount_used_in_planning_and_execution(self):
        policy = WealthPolicy(FLOOR, multiplier=8, discount_floor=True)
        self.assertEqual(policy._floor_at(0), FLOOR)
        # At the nominal floor the last day's positive interest funds a small position.
        state = TradingState(0, 1, 0, FLOOR, .15, .2)
        expected = 8 * (FLOOR - FLOOR / (1 + .02 / 252))
        self.assertAlmostEqual(DiscountedFloorDPTrader(8).trade(state), expected, delta=1e-6)
        self.assertEqual(StochasticDPTrader(8).trade(state), 0.)
        self.assertGreater(cap_weight(100e6, policy._floor_at(504), 8),
                           cap_weight(100e6, policy._floor_at(252), 8))

    def test_chance_estimates_match_one_day_under_gap_model(self):
        wealth = 100e6
        policy = ChancePolicy(FLOOR, 12, 0., wealth)
        for high, gap, mass in ((False, -.10, .001), (True, -.20, .004)):
            weight = policy.weight(1, wealth, high)
            self.assertAlmostEqual(weight, .6)
            self.assertLess(wealth * (1 + .02 / 252 + weight * gap), FLOOR)
            mean, risk = policy.launch_score(1, high)
            self.assertAlmostEqual(mean, 1 + (.02 + .05 * weight) / 252)
            self.assertAlmostEqual(risk, mass)

    def test_chance_limit_and_launch_commitment(self):
        state = TradingState(0, 2, 0, 100e6, .3, .2)
        loose = ChanceConstrainedDPTrader(12)
        strict = ChanceConstrainedDPTrader(12, breach_limit=.001)
        self.assertGreater(loose.trade(state), strict.trade(state))
        selected = strict.selected_policy
        self.assertLess(selected.launch_score(2, True)[1], .001)
        strict.trade(TradingState(1, 2, 10e6, 85e6, .1, .2))
        self.assertIs(strict.selected_policy, selected)
