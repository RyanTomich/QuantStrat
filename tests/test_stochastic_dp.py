import math
import unittest

from stochastic_dp import WealthPolicy, cap_weight
from trader import FLOOR, StochasticDPTrader, TradingState, _wealth_policy


WEIGHTS = (0, .25, .5, .75, 1)


def branches(wealth, weight, high):
    sigma = .30 if high else .15
    for sign in (-1, 1):
        equity_return = .07 / 252 + sign * sigma / math.sqrt(252)
        yield wealth * (1 + weight * equity_return + (1 - weight) * .02 / 252)


def one_day_value(wealth, weight, high):
    return sum(-100e6 / end - 5 * max(0, (95e6 - end) / 100e6)
               for end in branches(wealth, weight, high)) / 2


class StochasticDPTests(unittest.TestCase):
    def test_one_session(self):
        policy = WealthPolicy(FLOOR)
        for wealth in (94.5e6, 95e6, 96e6, 100e6):
            for high in (False, True):
                expected = max(WEIGHTS, key=lambda a: one_day_value(wealth, a, high))
                self.assertEqual(policy.weight(1, wealth, high), expected)

    def test_two_session_scenario_tree(self):
        wealth = 96e6

        def value(first):
            total = 0
            for balance in branches(wealth, first, False):
                for high, probability in ((False, .99), (True, .01)):
                    best = max(one_day_value(balance, a, high) for a in WEIGHTS)
                    total += .5 * probability * best
            return total

        first = max(WEIGHTS, key=value)
        self.assertEqual(WealthPolicy(FLOOR).weight(2, wealth, False), first)
        next_actions = {max(WEIGHTS, key=lambda a: one_day_value(balance, a, False))
                        for balance in branches(wealth, first, False)}
        self.assertGreater(len(next_actions), 1)

    def test_grid_lookup_cache_and_ties(self):
        policy = WealthPolicy(FLOOR)
        self.assertEqual(policy.weight(1, 95.25e6, False), 0)
        self.assertEqual(policy.weight(1, 95.25e6 + 1, False), .25)
        for outside, boundary in ((1, 50e6), (1e12, 200e6)):
            self.assertEqual(policy.weight(1, outside, True), policy.weight(1, boundary, True))
        original = policy.actions[0].copy()
        policy.weight(10, 100e6, False)
        self.assertTrue((policy.actions[0] == original).all())
        policy.values.fill(1)  # Every action has exactly equal continuation value.
        self.assertEqual(policy.weight(11, 100e6, False), 0)

    def test_trade_contract(self):
        trader = StochasticDPTrader()
        for forecast, target in ((.2, 1), (.20001, .25)):
            state = TradingState(9, 10, 60e6, 40e6, forecast, .2)
            order = trader.trade(state)
            self.assertEqual(order, target * 100e6 - state.equity_value)
            self.assertTrue(-state.equity_value <= order <= state.cash)
        self.assertEqual(trader.trade(TradingState(0, 1, 0, 0)), 0)
        self.assertEqual(trader.trade(TradingState(1, 1, 0, 100e6)), 0)
        for signals in ((None, .2), (.2, None)):
            with self.assertRaises(ValueError):
                trader.trade(TradingState(0, 1, 0, 100e6, *signals))

    def test_cap_uses_exact_balance(self):
        trader = StochasticDPTrader(multiplier=4)
        for wealth in (94.9e6, FLOOR, 96.037e6):
            equity = 30e6
            order = trader.trade(TradingState(0, 1, equity, wealth - equity, .12, .18))
            self.assertAlmostEqual(equity + order, 4 * max(0, wealth - FLOOR))
            self.assertTrue(-equity <= order <= wealth - equity)

    def test_combined_caps_and_available_actions(self):
        combined = _wealth_policy(8)
        self.assertIsNot(combined, _wealth_policy())
        self.assertIs(combined, _wealth_policy(8))
        for wealth, dollars in [
            (94e6, 0), (FLOOR, 0), (96e6, 8e6), (100e6 - 1, 40e6 - 8),
            (100e6, 40e6), (100e6 + 1, 40e6 + 12), (102e6, 64e6), (120e6, 120e6),
        ]:
            cap = cap_weight(wealth, FLOOR, 8)
            self.assertAlmostEqual(cap * wealth, dollars)
            actions = {combined._action_weight(i, cap) for i in range(len(combined.targets))}
            self.assertTrue({min(w, cap) for w in WEIGHTS} <= actions)
            if wealth == 96e6:
                self.assertEqual({round(w * wealth) for w in actions}, {0, 2e6, 4e6, 6e6, 8e6})

    def test_combined_two_session_tree_and_off_grid_trade(self):
        wealth = 100.9e6

        def actions(balance):
            cushion = max(0, balance - FLOOR)
            cap = min(balance, 8 * min(cushion, 5e6) + 12 * max(0, cushion - 5e6))
            return sorted({min(w, cap / balance) for w in WEIGHTS}
                          | {w * cap / balance for w in WEIGHTS})

        def value(first):
            return sum(.5 * prob * max(one_day_value(b, w, high) for w in actions(b))
                       for b in branches(wealth, first, True)
                       for high, prob in ((True, .96), (False, .04)))

        best = max(actions(wealth), key=value)
        self.assertAlmostEqual(best * wealth, (40e6 + 12 * .9e6) / 2)
        trader = StochasticDPTrader(8)
        for offset in (0, 23_000):
            order = trader.trade(TradingState(0, 2, 0, wealth + offset, .3, .2))
            self.assertAlmostEqual(order, best * wealth + 6 * offset)
