import unittest
from dataclasses import replace
from datetime import date, timedelta
from math import sqrt
from statistics import stdev
from unittest.mock import patch

from backtest import (
    CONDITIONS, INITIAL, STRATEGIES, Session, annual_volatility_signals, read_sessions,
    closing_drawdowns, crisis_episodes, episode_report, is_crisis, replay, run_years,
    schedule_sensitivity, summary_row,
)
from trader import FixedSplitTrader, VolatilityTargetTrader


class BacktestTests(unittest.TestCase):
    def test_prior_percentile(self):
        daily = [.04, .01, .03, .02, .05, .20]
        signals = annual_volatility_signals(daily)
        self.assertEqual([vol for vol, _ in signals], [v * sqrt(252) for v in daily])
        self.assertEqual([p80 for _, p80 in signals],
                         [None] + [v * sqrt(252) for v in [.04, .04, .04, .04, .04]])

    def test_signal_timing_and_overlapping_replays(self):
        start = date(2018, 1, 1)
        days = [Session(start + timedelta(days=i), (-1)**i * .01, 0)
                for i in range((date(2022, 1, 1) - start).days)]
        shock = (date(2020, 2, 1) - start).days
        revised = days.copy()
        revised[shock] = replace(days[shock], equity_return=-.30)
        names = ["vol-target", "floor-aware-vol-target", "high-vol-reduction"]

        def capture(sessions):
            traces = []

            class RecordingTrader(VolatilityTargetTrader):
                def trade(self, state):
                    if state.day == 0:
                        traces.append([])
                    order = super().trade(state)
                    traces[-1].append((state.annual_volatility, state.annual_volatility_p80, order))
                    return order

            with patch.dict(STRATEGIES, {"record": RecordingTrader}):
                rows, _ = run_years(sessions, [*names, "record"], 2020, 2020)
            self.assertEqual(len(rows), 12 * 4)
            return traces

        original = capture(days)
        changed = capture(revised)
        self.assertEqual(original[1][0], changed[1][0])  # February opening trade
        self.assertEqual(original[0][31][:2], original[1][0][:2])
        self.assertEqual(changed[0][32][:2], changed[1][1][:2])
        for offset in (0, 1):
            past = [s.equity_return for s in revised[shock + offset - 63:shock + offset]]
            self.assertAlmostEqual(changed[1][offset][0], stdev(past) * sqrt(252))
        self.assertGreater(changed[1][1][0], original[1][1][0])

    def test_rolling_peak_expires_and_exact_threshold_counts(self):
        days = [Session(date(2019, 2, 28), 0, 0), Session(date(2020, 2, 28), .25, 0),
                Session(date(2020, 2, 29), -.2, 0), Session(date(2021, 2, 28), 0, 0)]
        values = closing_drawdowns(days)
        self.assertIsNone(values[0])
        self.assertTrue(is_crisis(values[2]))
        self.assertEqual(values[-1], 0)
        self.assertEqual(list(crisis_episodes(days, values)), [(days[2].date, days[2].date)])

    def test_launch_state_uses_prior_close_and_crisis_overrides_low_volatility(self):
        start = date(2018, 1, 1)
        days = [Session(start + timedelta(days=i), 0, 0)
                for i in range((date(2021, 1, 1) - start).days)]
        shock = (date(2020, 2, 1) - start).days
        days[shock] = replace(days[shock], equity_return=-.25)
        episodes = []
        rows, _ = run_years(days, ["fixed-split"], 2020, 2020, horizon_months=1, episodes=episodes)
        self.assertEqual(rows[1].condition, "lower-volatility")
        self.assertEqual(episodes, [(rows[1].first_session, days[-1].date)])
        self.assertEqual(rows[-1].condition, "crisis")
        self.assertEqual(rows[-1].starting_volatility, 0)
        days[shock] = replace(days[shock], equity_return=.25)
        rows, _ = run_years(days, ["fixed-split"], 2020, 2020, horizon_months=1)
        self.assertEqual(rows[2].condition, "higher-volatility")
        self.assertEqual(rows[2].starting_drawdown, 0)

    def test_disjoint_offsets_with_missing_month_and_sparse_states(self):
        for horizon in (12, 24):
            template = replay("test", FixedSplitTrader(), [Session(date(2020, 1, 1), 0, 0)], 0, 0)
            rows = []
            shock = horizon
            for month in range(horizon * 2):
                if month == 2:
                    continue
                opening = date(2020 + month // 12, month % 12 + 1, 1)
                rows.append(replace(template, first_session=opening,
                                    floor_breach=month <= shock < month + horizon))
            low, high, represented = schedule_sensitivity(rows, horizon)
            self.assertEqual(low, "0/1 (0.00%, offset 2)")
            self.assertEqual(high, "1/2 (50.00%, offset 0)")
            self.assertEqual(represented, f"{horizon}/{horizon}")
            self.assertEqual(sum(r.floor_breach for r in rows), horizon - 1)
            self.assertEqual(schedule_sensitivity([], horizon), ("—", "—", f"0/{horizon}"))

    def test_episode_exclusion_includes_survivors_and_both_endpoints(self):
        day = date(2020, 1, 2)
        sample = replay("test", FixedSplitTrader(), [Session(day, 0, 0)], 0, 0)
        rows = [replace(sample, last_session=day + timedelta(days=4), floor_breach=True),
                replace(sample, first_session=day - timedelta(days=3)),
                replace(sample, first_session=day + timedelta(days=1), last_session=day + timedelta(days=4))]
        lines = episode_report(rows, [(day, day), (day + timedelta(days=4), day + timedelta(days=4))])
        for end in [day, day + timedelta(days=4)]:
            self.assertIn(f"| {end} to {end} | test | all | 1/2 | 0/1 | -33.33 |", lines)
            self.assertIn(f"| {end} to {end} | test | crisis | 0/0 | 0/0 | — |", lines)

    def test_input(self):
        header = b"source note\n\n,Mkt-RF,RF\n"
        days = read_sessions(header + b"20200102,1,0.2\n\nCopyright 2026\n")
        self.assertEqual(days, [Session(date(2020, 1, 2), .012, .002)])
        for bad in [b"20200102,1,\n", b"20200102,nan,1", b"20200230,1,0",
                    b"20200102,1,0\n20200102,1,0", b"20200102,-99.99,0"]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                read_sessions(header + bad)

    def test_trade_timing_compounding_and_drawdown(self):
        days = [Session(date(2020, 1, 2), -.25, .02), Session(date(2020, 1, 3), .5, .01)]
        for weight, terminal, drawdown in [(0, INITIAL * 1.02 * 1.01, 0),
                                           (1, INITIAL * .75 * 1.5, .25)]:
            result = replay("test", FixedSplitTrader(weight), days, 0, 0)
            self.assertEqual(result.terminal_wealth, terminal)
            self.assertEqual(result.max_drawdown, drawdown)

    def test_gentle_glide_rebalances_before_returns(self):
        days = [Session(date(2020, 1, 2), .10, 0), Session(date(2020, 1, 3), -.10, .01)]
        result = replay("glide-65-35", STRATEGIES["glide-65-35"](), days, 0, 0)
        # First day gains 6.5%. Second day loses 3.5% on equity and earns 0.65% on cash.
        self.assertAlmostEqual(result.terminal_wealth, INITIAL * 1.065 * .9715)
        self.assertAlmostEqual(result.max_drawdown, .0285)

    def test_breach_threshold_and_ce(self):
        def outcome(ret):
            return replay("test", FixedSplitTrader(1), [Session(date(2020, 1, 2), ret, 0)], 0, 0)

        floor = outcome(-.05)
        loss = outcome(-.5)
        self.assertFalse(floor.floor_breach)
        self.assertTrue(loss.floor_breach)
        rows = [loss, floor, floor]
        expected_ce = 3 / (1 / 50_000_000 + 2 / 95_000_000)
        line = summary_row("test", "all", rows)
        self.assertIn(f"${expected_ce:,.2f}", line)
        self.assertIn("1 (33.33%) | yes", line)
        self.assertIn("| 0 | —", summary_row("test", CONDITIONS[1], []))

    def test_invalid_order(self):
        class BadTrader:
            def trade(self, state):
                return state.cash + 1

        with self.assertRaisesRegex(ValueError, "bad on 2020-01-02"):
            replay("bad", BadTrader(), [Session(date(2020, 1, 2), 0, 0)], 0, 0)

    def test_monthly_resets_shared_windows_and_prestart_information(self):
        class BuyOnce:
            def __init__(self):
                self.calls = 0

            def trade(self, state):
                if self.calls != state.day:
                    raise AssertionError("unexpected state carried across years")
                self.calls += 1
                return state.cash

        start = date(2018, 1, 1)
        days = [Session(start + timedelta(days=i), (i % 7) * .0001, 0)
                for i in range((date(2022, 1, 1) - start).days)]
        revised = [replace(s, equity_return=.01) if s.date.year >= 2020 else s for s in days]
        with patch.dict(STRATEGIES, {"once": BuyOnce}):
            rows, excluded = run_years(days, ["fixed-split", "once"], 2020, 2021)
            changed, _ = run_years(revised, ["once"], 2020, 2020)
        self.assertEqual(list(excluded), [f"2021-{m:02d}" for m in range(2, 13)])
        self.assertEqual([(r.start_month, r.strategy) for r in rows],
                         [(f"2020-{m:02d}", name) for m in range(1, 13)
                          for name in ("fixed-split", "once")]
                         + [("2021-01", name) for name in ("fixed-split", "once")])
        for r in rows[:2]:
            self.assertEqual((r.starting_volatility, r.median_volatility, r.condition),
                             (changed[0].starting_volatility, changed[0].median_volatility, changed[0].condition))

        self.assertEqual((rows[2].first_session, rows[2].last_session, rows[2].sessions),
                         (date(2020, 2, 1), date(2021, 1, 31), 366))
        shortened, excluded = run_years(days[:-1], ["fixed-split"], 2021, 2021)
        self.assertFalse(shortened)
        self.assertEqual(excluded["2021-01"], "incomplete one-year window")
