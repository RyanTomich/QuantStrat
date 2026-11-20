import unittest
from dataclasses import replace
from datetime import date
from unittest.mock import patch

from backtest import AnnualResult, BREACH_LIMIT, CONDITIONS, Session, run_years
from cushion_selection import SMOOTH_SCALES, TIME_AWARE, select_smooth_scale
from trader import FLOOR, SmoothCushionTrader, TimeAwareSmoothCushionTrader, TradingState


class SmoothPolicyTests(unittest.TestCase):
    def test_orders_and_funding_horizon(self):
        trader = TimeAwareSmoothCushionTrader()
        for n, expected in [(252, 4_700_000), (1, 18_850)]:
            self.assertAlmostEqual(trader.trade(TradingState(0, n, 0, FLOOR, .15)),
                                   expected, delta=expected * .001)
        floor = FLOOR * (1 + .02 / 252) ** -252
        for wealth in (floor, floor - 1):
            self.assertEqual(trader.trade(TradingState(0, 252, wealth, 0, .2)), -wealth)
        self.assertEqual(trader.trade(TradingState(0, 252, 0, 200e6, .05)), 200e6)
        self.assertEqual(trader.trade(TradingState(252, 252, FLOOR, 0)), 0)
        self.assertEqual(TimeAwareSmoothCushionTrader(0).trade(TradingState(0, 1, 105e6, 0, .05)), -105e6)

    def test_nominal_equivalence_and_continuity(self):
        smooth = SmoothCushionTrader()
        zero_rate = TimeAwareSmoothCushionTrader(annual_funding_rate=0)
        for cushion in (0, 3e6, 5e6, 7e6, 15e6, 25e6, 32e6):
            for vol in (.05, .1, .15, .5):
                state = TradingState(3, 252, 5e6, FLOOR + cushion - 5e6, vol)
                self.assertEqual(zero_rate.trade(state), smooth.trade(state))
                # Probe both sides of each wealth breakpoint without saturating at 100% equity.
                self.assertLess(abs(zero_rate.trade(replace(state, cash=state.cash + .001))
                                    - zero_rate.trade(replace(state, cash=state.cash - .001))), .04)
            state = TradingState(3, 252, 0, FLOOR + cushion, .1)
            self.assertLess(abs(zero_rate.trade(replace(state, annual_volatility=.1 - 1e-12))
                                - zero_rate.trade(replace(state, annual_volatility=.1 + 1e-12))), .01)


class SmoothSelectionTests(unittest.TestCase):
    def setUp(self):
        self.cutoff = date(2020, 1, 2)
        base = AnnualResult(TIME_AWARE, 2018, "2018-01", date(2018, 1, 2), date(2018, 12, 31),
                            252, .01, .02, CONDITIONS[0], FLOOR, 0, 0, False)
        self.history = {s: [replace(base, condition=c) for c in CONDITIONS for _ in range(50)]
                        for s in SMOOTH_SCALES}

    def choice(self):
        return select_smooth_scale(self.history, self.cutoff, CONDITIONS, BREACH_LIMIT)

    def test_mean_tie_and_crisis_constraint(self):
        self.assertEqual(self.choice()[0], 0)
        top = self.history[1.5]
        top[0] = replace(top[0], terminal_wealth=1e9)
        self.assertEqual(self.choice()[0], 1.5)
        top[-1] = replace(top[-1], floor_breach=True, terminal_wealth=94e6)
        self.assertEqual(self.choice()[0], 0)  # 1/150 overall, 1/50 in crisis.
        for rows in self.history.values():
            rows[-1] = top[-1]
        selected, audit = self.choice()
        self.assertEqual((selected, audit[0]["mode"]), (0, "cash fallback"))
        for rows in self.history.values():
            rows[-1] = replace(rows[-1], last_session=self.cutoff)
        selected, audit = self.choice()
        self.assertEqual((selected, audit[0]["mode"]), (None, "nominal warm-up"))

    def test_year_start_audit_and_launch_commitment(self):
        sessions = [Session(date(y, m, d), (-1)**d * (.005 if m <= 6 else .012), .0001)
                    for y in range(1991, 2005) for m in range(1, 13) for d in range(1, 9)]
        # Exercise two successive January choices and compare overlapping launches
        # to independent replays at the two corresponding fixed scales.
        def choices(history, cutoff, conditions, limit):
            _, audit = select_smooth_scale(history, cutoff, conditions, limit)
            return (1. if cutoff.year == 2001 else 0.), audit

        audit = []
        names = [TIME_AWARE, "fixed-time-aware-smooth-cushion"]
        with patch("backtest.select_smooth_scale", side_effect=choices):
            rows, _ = run_years(sessions, names, 2001, 2002, smooth_selection_history=audit)
            changed = [replace(s, equity_return=-.40) if s.date == date(2001, 8, 1) else s
                       for s in sessions]
            after = []
            run_years(changed, names, 2001, 2002, smooth_selection_history=after)
        self.assertEqual(audit[:len(SMOOTH_SCALES)], after[:len(SMOOTH_SCALES)])
        for adaptive, fixed in zip(rows[::2], rows[1::2]):
            expected = fixed.terminal_wealth if adaptive.year == 2001 else 100e6 * 1.0001**adaptive.sessions
            self.assertAlmostEqual(adaptive.terminal_wealth, expected, delta=.01)
