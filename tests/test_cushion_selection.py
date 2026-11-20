import unittest
from dataclasses import replace
from datetime import date
from unittest.mock import patch

from backtest import AnnualResult, STRATEGIES, BREACH_LIMIT, VOLATILITY_GROUPS, Session, replay, run_years
from cushion_selection import HYBRID, MULTIPLIERS, select_multiplier


class CushionSelectionTests(unittest.TestCase):
    def setUp(self):
        self.cutoff = date(2020, 1, 2)
        self.history = {}
        for m in MULTIPLIERS:
            self.history[m] = [
                AnnualResult(HYBRID, 2018, "2018-01", date(2018, 1, 2), date(2018, 12, 31),
                             252, .01, .01, condition, 100e6 + m * 1e6, 0, 0, False)
                for condition in VOLATILITY_GROUPS for _ in range(50)
            ]

    def choice(self):
        return select_multiplier(self.history, self.cutoff, VOLATILITY_GROUPS, BREACH_LIMIT)

    def test_exact_limit_one_failing_group_and_unfinished_starts(self):
        group = self.history[12]
        group[0] = replace(group[0], terminal_wealth=94e6, floor_breach=True)
        self.assertEqual(self.choice()[0], 12)  # Exactly 1/50 in the lower group.
        group[1] = replace(group[0])
        selected, audit = self.choice()
        self.assertEqual(selected, 8)
        group.extend([replace(group[-1], last_session=self.cutoff)] * 100)
        self.assertEqual(self.choice(), (selected, audit))

    def test_fallbacks_and_ties(self):
        for m, rows in self.history.items():
            rows[:2] = [replace(r, terminal_wealth=90e6, floor_breach=True) for r in rows[:2]]
        selected, audit = self.choice()
        self.assertEqual((selected, audit[0]["reason"]), (2, "no qualifying multiplier"))
        for rows in self.history.values():
            rows.pop()
        self.assertEqual(self.choice()[1][0]["reason"], "insufficient history")
        self.history = {m: [self.history[2][-1]] * 50 + [self.history[2][20]] * 50 for m in MULTIPLIERS}
        selected, audit = self.choice()
        self.assertEqual((selected, audit[0]["reason"]), (2, "qualified"))

    def test_requires_mean_gain_as_well_as_ce(self):
        # The tighter policy has a higher mean but a lower harmonic mean.
        base = self.history[2][0]
        self.history[2][0] = replace(base, terminal_wealth=1200e6)
        self.assertEqual(self.choice()[0], 2)

    def test_future_returns_cannot_change_january_choice_or_earlier_orders(self):
        sessions = [Session(date(y, m, d), (-1)**d * (.004 if m <= 6 else .014), .0001)
                    for y in range(1997, 2022) for m in range(1, 13) for d in range(1, 9)]
        shock = date(2020, 2, 3)

        def run(data):
            orders = {}
            def trace_january(name, trader, days, *args):
                if days[0].date != date(2020, 1, 1):
                    return replay(name, trader, days, *args)
                prefix = []
                orders[name, trader.multiplier] = prefix
                class Recorder:
                    def trade(self, state):
                        order = trader.trade(state)
                        if days[state.day].date <= shock:
                            prefix.append(order)
                        return order
                return replay(name, Recorder(), days, *args)

            audit = []
            with patch("backtest.replay", trace_january):
                rows, _ = run_years(data, [HYBRID], 2020, 2020, selection_history=audit)
            return orders, audit, rows

        original, audit, rows = run(sessions)
        revised, changed, other = run([replace(s, equity_return=-.3) if s.date == shock else s
                                       for s in sessions])
        self.assertEqual(audit, changed)
        self.assertEqual(audit[0]["reason"], "qualified")
        self.assertEqual(set(original), {(HYBRID, m) for m in MULTIPLIERS})
        self.assertEqual({r["strategy"] for r in audit}, {HYBRID})
        self.assertEqual(original, revised)
        self.assertNotEqual(rows[0].terminal_wealth, other[0].terminal_wealth)

    def test_dp_strategy_choices(self):
        self.assertEqual({name for name in STRATEGIES if "dp" in name},
                         {"stochastic-dp", HYBRID, "split-cushion-dp", "chance-constrained-dp",
                          "discounted-floor-dp"})
