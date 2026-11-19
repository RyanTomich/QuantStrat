# QuantStrat

## Download the daily Fama/French factors

1. Open the [Kenneth R. French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html).
2. Under **U.S. Research Returns Data**, select **CSV** beside **Fama/French 3 Factors [Daily]**.
3. Unzip the download and place `F-F_Research_Data_Factors_daily.csv` in the repository root, alongside this README.

## Daily trading policy

Call `Trader.trade(state)` once per trading session with the current equity market value and cash balance, including accrued interest. The returned dollar amount buys equity when positive and sells equity when negative. The caller executes the order and carries the resulting balances forward.

```python
from trader import FixedSplitTrader, TradingState

trader = FixedSplitTrader(equity_weight=0.60)
state = TradingState(day=0, horizon_days=252, equity_value=0, cash=100_000_000)
order = trader.trade(state)  # Buy $60M of equity.
equity_after_trade = state.equity_value + order
cash_after_trade = state.cash - order
```

Day 0 is the initial investment. The trader rebalances daily through day `horizon_days - 1` and returns zero on the terminal valuation day. Orders assume fractional-dollar trading and negligible costs.

Run the contract tests with `python3 -m unittest discover -s tests`.

## Historical backtest

Run from the repository root with Python 3.10 or newer. Each month starts a fresh trader with $100M in cash on its first trading session. The investment runs through the last session of the final month of the selected horizon (12 months by default). Adjacent starts share all but one month of returns.

```sh
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 backtest.py --strategy fixed-split --start-year 1950 --end-year 2025 --horizon-months 12
python3 backtest.py --strategy fixed-split --start-year 1950 --end-year 2025 --horizon-months 24
```

The year bounds select start months, so a holding period may extend into the following year. Omit the bounds to use the full eligible history. Use `--data PATH` to select a daily Fama/French CSV. The default strategy, `fixed-split`, targets 60% equity every session.

Each order executes before the session earns its returns. Equity earns the market excess return plus RF, and cash earns RF. A launch is in crisis if the previous close of the compounded equity total-return index is at least 20% below its highest close in the trailing 12 calendar months. The window includes the previous close and excludes its year-ago anniversary (February 28 for a February 29 close). A full year of history is required. Otherwise, the launch state compares the prior 63 sessions' daily equity volatility with the median of all earlier 63-session windows. At least 252 earlier windows are required. A value equal to the median goes in the lower-volatility group.

`glide-65-35` reduces equity linearly from 65% on the opening session to 35% on the final session. A single-session horizon uses 65%. `floor-aware-glide` uses an 80% to 20% schedule and caps equity dollars at twice the positive difference between current wealth and $95M. It starts with $10M in equity at $100M of wealth, targets at most $4M at $97M, and holds cash at or below the floor. Compare the three policies with this command.

```sh
python3 backtest.py --strategy fixed-split --strategy glide-65-35 --strategy floor-aware-glide
```

The volatility traders forecast annual equity volatility daily using the sample standard deviation of the preceding 63 sessions' equity returns multiplied by √252. They share a 9% annual portfolio volatility target, configurable with the trader constructor's `target_vol` argument.

- `vol-target` divides the target by the forecast and caps the equity weight at 100%. A zero forecast also gives 100% equity.
- `floor-aware-vol-target` applies the same rule and limits equity dollars to twice the positive cushion above $95M.
- `high-vol-reduction` holds 60% equity through the 80th percentile of all prior forecasts. Strictly above that threshold, it uses the smaller of 60% and the continuous target weight. The percentile uses nearest rank and includes pre-start history.

```sh
python3 backtest.py --strategy vol-target --strategy floor-aware-vol-target --strategy high-vol-reduction
```

`vol-scaled-cushion` uses the same lagged forecast to size exposure against the positive cushion above $95M. It invests the cushion times `min(3, 0.30 / annual_volatility)` in equity, up to available wealth. At zero volatility it uses the maximum multiplier of three.

```sh
python3 backtest.py --strategy tiered-cushion-3x --strategy vol-scaled-cushion --strategy floor-aware-vol-target --strategy floor-aware-glide
```

`tiered-cushion-3x` uses the volatility multiplier on the first $5M above the floor and triples it on additional cushion. It recalculates daily and caps equity at total wealth. At $110M wealth and 15% forecast volatility, it holds $70M in equity.

`smooth-cushion-4x` uses 1.25 times the volatility-scaled base multiplier. For each additional dollar of cushion, its factor is 1 through $3M, rises linearly to 3 at $7M, holds at 3 through $15M, rises linearly to 4 at $25M, and holds at 4 thereafter. Total equity is the sum across the occupied bands, capped at available wealth.

Each invocation prints the full metrics table for every strategy, overall and by launch state, then the path to its run folder. The table includes certainty-equivalent wealth, terminal floor breaches (count and rate), mean and worst cumulative horizon return, and the largest within-horizon drawdown. Each folder contains four files.

- `annual_results.csv` records the strategy, start year and month, first and last session, session count, starting daily volatility, historical median daily volatility, launch drawdown, horizon in months, condition, terminal dollar wealth, total return, maximum drawdown, and terminal floor-breach flag. Returns and volatilities are decimals. Drawdown is a positive loss fraction measured from the running peak, including starting wealth.
- `summary.md` records the command, data checksum, eligible start count, date range, excluded months, and results overall and by starting condition. Each monthly start has equal weight. Compare strategies on certainty-equivalent terminal wealth (the harmonic mean for γ = 2) alongside the observed frequency of terminal wealth below $95M in each condition.
- `terminal_wealth_distribution.png` shows one cumulative distribution per strategy across all eligible monthly starts. Each dot on the $95M floor marker shows the share of starts that finished strictly below the floor. The lower panel shows each strategy’s observed breach rate for each starting condition against the dashed 2% line. The 2% comparison uses observed frequencies.
- `starting_volatility_vs_wealth.png` plots each monthly start against its pre-start daily volatility, with the three lowest terminal wealth outcomes labeled by start month. The scatter plot uses a separate panel for each strategy and common axes for comparison. The shaded area below $95M marks terminal breaches. A finish on the line meets the floor.

The main table counts each eligible launch once. For schedule sensitivity, it assigns calendar months to offsets using `(12 × year + month − 1) modulo horizon_months`. Holding periods within each offset are disjoint. Within each state the table shows the lowest and highest nonempty offset rates, their breach counts and denominators, and how many offsets contributed. Ties use the lowest offset. The range describes sensitivity to launch timing.

The crisis table treats each consecutive spell of crisis closes as a separate episode. It removes all launches whose holding period includes any close in that episode, including survivors, and shows the change in the remaining breach rate in percentage points. Each exclusion starts from the full sample and preserves the original launch labels. A dash indicates an undefined rate or change.

## Stochastic dynamic programming

`stochastic-dp` chooses among 0%, 25%, 50%, 75%, and 100% equity each day. It uses current wealth, trading sessions remaining, and the pre-trade volatility forecast. A forecast strictly above its historical 80th percentile selects the high-volatility regime. The benchmark evaluates the policy in all three launch states.

The policy solves a fixed model by backward induction. Cash earns 2%/252 daily. Equity earns 7%/252 plus or minus the regime's annual volatility divided by √252, with equal probabilities. Annual volatility is 15% in the ordinary regime and 30% in the high regime. After each session, the regime switches with probability 1% from ordinary to high or 4% from high to ordinary, independently of the return shock. These assumptions determine the policy. Historical replays use the observed equity and cash returns.

With terminal wealth expressed as a multiple of $100M, the objective is expected terminal utility `-1/x - 5 × max(0, 0.95 - x)`. Finishing below $95M incurs a penalty proportional to the shortfall. The backtest measures how often this happens against the project's 2% limit.

The solver uses a $50M–$200M grid in $100K increments. It linearly interpolates future values between grid points and extends the boundary slopes outside the grid. For trading, it uses the nearest grid point, rounding midpoint ties down and using the endpoint beyond either boundary. Equal action values favor the smaller equity allocation.

```sh
python3 backtest.py --strategy stochastic-dp --strategy fixed-split --strategy floor-aware-glide --start-year 1950 --end-year 2025
```

### Primary strategy: combined cushion-capped DP

`cushion-capped-dp-combined` uses the same utility and market model as the `stochastic-dp` baseline. Its equity cap applies the selected multiplier to the first $5M above the $95M floor and 1.5 times that multiplier to additional cushion, up to total wealth. The solver considers the baseline targets clipped to this cap plus positions at 25%, 50%, and 75% of the cap. With an 8× multiplier and $96M in wealth, the available equity positions are $0M, $2M, $4M, $6M, and $8M. At or below $95M the portfolio holds cash.

Each January the backtest compares multipliers 2, 4, 6, 8, and 12 on the same historical starts. Only investments completed strictly before the year's first session are available for selection, including eligible pre-start-year history. Each of the two original volatility groups used for multiplier selection needs at least 50 completions and at most 2% terminal breaches. Start with the tightest qualifying multiplier. Move to a looser qualifying multiplier only when both certainty-equivalent wealth and mean terminal dollars improve over the current choice. Each investment keeps its opening year's multiplier through maturity. If history is insufficient or every multiplier fails, use two and record the reason.

```sh
python3 backtest.py --strategy stochastic-dp --strategy cushion-capped-dp-combined --start-year 1950 --end-year 2025
```

The run folder includes `selection_history.csv` with one row per candidate and selection year. The report compares terminal wealth, losses near the floor, and results by starting decade and launch state. Paired 12-month block bootstrap intervals summarize historical uncertainty for differences in CE and mean wealth. The selection audit reports the two volatility groups used by the policy.
