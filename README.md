# QuantStrat

How much of a $100M portfolio should a trader hold in equity if at most 2% of launches in each market condition may finish below $95M? QuantStrat replays daily U.S. equity and cash returns and ranks strategies that meet that limit by certainty-equivalent terminal wealth (the harmonic mean of their final balances).

## Data

The daily Fama/French factors supply both returns on the same trading dates. Equity earns the market excess return plus the risk-free rate (`Mkt-RF + RF`), and cash earns `RF`. The loader converts both from percentages to decimals.

1. Open the [Kenneth R. French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html).
2. Under **U.S. Research Returns Data**, select **CSV** beside **Fama/French 3 Factors [Daily]**.
3. Unzip the download and place `F-F_Research_Data_Factors_daily.csv` in the repository root, alongside this README.

Run all commands from the repository root. Install the dependencies once with `python3 -m pip install -r requirements.txt`.

## Trader

Fixed-split traders rebalance to a constant equity weight. Glide paths change that weight with time, volatility targets adjust it with market risk, and cushion rules size equity holdings from wealth above a floor or reserve. Dynamic programming (DP) traders look up an equity allocation using wealth, volatility regime, and sessions remaining.

Call `trader.trade(state)` before each session’s returns. A positive dollar order buys equity and a negative one sells it. The caller transfers that amount between equity and cash, applies the session’s returns to the resulting balances, and carries them into the next session.

```python
from trader import FixedSplitTrader, TradingState

trader = FixedSplitTrader(equity_weight=0.60)
state = TradingState(day=0, horizon_days=252, equity_value=0, cash=100_000_000)
order = trader.trade(state)  # Buy $60M of equity.
equity_after_trade = state.equity_value + order
cash_after_trade = state.cash - order
```

## Historical backtest

Trading signals estimate volatility from the previous 30 completed trading sessions. Set `--vol-lookback` on the CLI or `vol_lookback` in `run_years` to use a different window.

Each strategy starts with $100M in cash on the first trading day of each eligible month and trades the same overlapping windows (12 months by default). A launch is in crisis if the prior close was at least 20% below the market’s trailing 12-month high. Of the remaining launches, those whose prior 63-session volatility exceeds its historical median are higher-volatility launches. The rest are lower-volatility launches.

This experiment compares a fixed 60/40 allocation with a glide path that reduces equity from 65% to 35%.

```sh
python3 plot_backtest.py --strategy fixed-split --strategy glide-65-35 --start-year 2010 --end-year 2019
```

Repeat `--strategy` to compare traders. The years select launch dates, and `--horizon-months` sets the holding period. Run `python3 plot_backtest.py --help` for strategy names and other options. The equivalent Python call returns per-launch results and excluded starts.

```python
from pathlib import Path
from plot_backtest import read_sessions, run_years

sessions = read_sessions(Path("F-F_Research_Data_Factors_daily.csv").read_bytes())
results, excluded = run_years(sessions, ["fixed-split", "glide-65-35"], 2010, 2019)
```

The CLI prints its new directory under `runs/`. Open `summary.csv` to compare strategies by launch condition, `annual_results.csv` for individual launches, and `run_metadata.csv` for the command and data checksum.

| Experiment figure | What it shows |
| --- | --- |
| `terminal_wealth_distribution.png` | The share of launches finishing at or below each wealth level, with detail near $95M and breach rates by launch condition. |
| `stock_allocation_by_volatility.png` | Equity weight across portfolio wealth and forecast volatility for the latest eligible launch. |
| `stock_allocation_by_time.png` | Equity weight across wealth and sessions remaining at 15% and 30% forecast volatility. |
| `cushion_distribution.png` | Terminal wealth across all launches and close to the floor, for comparisons that include `cushion-capped-dp-combined`. |

## Generating figures

The report uses the default 30-session trading lookback. To recreate its figures, run the following command with the **September 2026 CRSP** version of the factor CSV (SHA-256 `3f7910502dd6f188748210e5145d95642de1d8cff56c2cf0fa85985299610ff0`).

```sh
python3 report/figures/figures.py
```

The script calibrates from earlier market history and replays 910 overlapping 12-month launches from January 1950 through October 2025 directly from the CSV. It writes four PDFs and two LaTeX tables to `report/figures/`. Use `--table-only` to rerun the experiments and write just the tables.

- `wealth-volatility.pdf` shows how each DP refinement changes equity allocation. `smooth-wealth-volatility.pdf` compares Full DP with the volatility-capped smooth cushion.
- `terminal-cdf.pdf` compares terminal wealth across the DP refinements and the smooth cushion. `alternatives-cdf.pdf` compares Full DP with one representative from each alternative strategy family.
