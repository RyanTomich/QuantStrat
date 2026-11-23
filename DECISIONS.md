# QuantStrat decision log

## Reproduce the report

From the repository root, place the September 2026 Fama/French daily factors file, `F-F_Research_Data_Factors_daily.csv`, beside `plot_backtest.py`. With Python 3.10 or later, run the following commands.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python report/figures/figures.py
```

The last command writes the figures and tables to `report/figures/`. Add `--table-only` to regenerate just the tables. 

| Report item | Output and calculation |
| --- | --- |
| Figures 1–2 · Equity allocations | `wealth-volatility.pdf` shows the four DP stages. `smooth-wealth-volatility.pdf` compares Full DP with the capped smooth cushion. Both use the October 2025 launch with 251 sessions remaining. |
| Figure 3 and Table 1 · DP refinements | `terminal-cdf.pdf` and `strategy-overview.tex` use the same 910 overlapping 12-month launches from January 1950 through October 2025. Adding the equity cap, downside shocks and launch policy selection changed the baseline’s 27 breaches to 2, then 1, then 1. |
| Figure 4 and Table 2 · Alternative traders | `alternatives-cdf.pdf` and `alternatives-overview.tex` use the highest-CE trader in each family that stayed at or below 2% observed breaches in every launch group. Fixed 60/40 is the ineligible benchmark. |
| Slide plots and numbers | The same command produces `slide-heatmaps.pdf` and `slide-performance.tex`. |

```sh
python plot_backtest.py --start-year 1950 --end-year 2025 --horizon-months 12 \
  --vol-lookback 30 --vol-capped-scale-source incumbent \
  --strategy chance-constrained-dp --strategy vol-capped-smooth-cushion \
  --strategy time-aware-smooth-cushion
```

## Current traders

Pass the first-column name to `python plot_backtest.py --strategy`. The registry and launch settings are in [plot_backtest.py](plot_backtest.py), and the implementations are in [trader.py](trader.py).

| CLI strategy | Implementation | Rule and role in the experiments |
| --- | --- | --- |
| `fixed-split` | `FixedSplitTrader` | Rebalances daily to 60% equity. Fixed-allocation benchmark in Table 2. |
| `glide-65-35` | `GentleGlideTrader` | Reduces equity linearly from 65% to 35%. Tested as the time-only glide path. |
| `floor-aware-glide` | `FloorAwareGlideTrader` | Runs an 80%–20% glide with equity capped at twice the cushion above $95M. Selected glide path in Table 2. |
| `vol-target` | `VolatilityTargetTrader` | Targets 9% annual portfolio volatility, with equity limited to 100%. Tested as the basic volatility rule. |
| `floor-aware-vol-target` | `FloorAwareVolatilityTargetTrader` | Adds the twice-cushion cap to the 9% target. Selected volatility rule in Table 2. |
| `high-vol-reduction` | `HighVolatilityReductionTrader` | Holds 60% equity in ordinary conditions. Above the volatility cutoff, applies the 9% target with a 60% ceiling. |
| `vol-scaled-cushion` | `VolatilityScaledCushionTrader` | Invests up to three times the cushion, reducing the multiplier as volatility rises above 10%. First cushion benchmark. |
| `tiered-cushion-3x` | `AggressiveTieredCushionTrader` | Triples the base multiplier on cushion dollars above $5M. Compared with a smooth increase in exposure. |
| `smooth-cushion-4x` | `SmoothCushionTrader` | Gradually raises marginal exposure from one to four times the scaled base multiplier. Uses the nominal $95M floor. |
| `fixed-time-aware-smooth-cushion` | `TimeAwareSmoothCushionTrader` | Discounts the floor to its present value and fixes the risk scale at one. Control for annual scale selection. |
| `time-aware-smooth-cushion` | `TimeAwareSmoothCushionTrader` | Selects the risk scale from completed launches. Selected cushion in Table 2 ($108.63M CE). |
| `stochastic-dp` | `StochasticDPTrader` | Chooses equity by expected terminal utility under two volatility regimes. Baseline DP in Table 1. |
| `cushion-capped-dp-combined` | `StochasticDPTrader` | Uses a selected multiplier, a tiered equity cap and fractional positions. The “+ equity cap” stage in Table 1. |
| `split-cushion-dp` | `SplitDPTrader` | Starts half the wealth in capped DP and half in uncapped DP, then tracks each separately. Trailed Full DP at both horizons. |
| `discounted-floor-dp` | `DiscountedFloorDPTrader` | Uses the floor’s present value in capped DP. Exceeded the breach limit at both horizons. |
| **`chance-constrained-dp`** | **`ChanceConstrainedDPTrader`** | **Full DP. At launch, selects the highest modeled mean wealth among policies below 2% modeled breach risk, with cash as the fallback. Primary strategy in the report.** |
| **`vol-capped-smooth-cushion`** | **`VolCappedSmoothCushionTrader`** | **Adds a continuous volatility-dependent equity ceiling. Inherits the time-aware scale in the report and had zero breaches.** |


### 2026-11-18 (UTC)

#### Annual benchmarks and monthly starts

**03:41:30 · [Establish a fixed-allocation baseline](runs/20261118T034130Z-9ubn11an/summary.md).** Daily rebalancing to 60% equity produced $106.88M CE over 98 calendar years (1928–2025), with 13 finishes below $95M. Kept 60/40 as a reference throughout the project.

**03:58:31 · [Clarify what a breach measures](runs/20261118T035831Z-7fy9kvxd/summary.md).** Labeled the 13 terminal breaches separately from maximum drawdown. The 98 annual outcomes matched the baseline.

**04:01:28 · [Plot the fixed-split outcomes](runs/20261118T040128Z-go3yzgvv/summary.md).** Plotted terminal wealth and starting volatility for the same 98 annual launches. Kept the distribution plot to show how close finishes came to the floor.

**~04:05 · [Vary the fixed equity weight](runs/split-comparison-Lb1lJUJl/summary.md).** Across the same 98 years, raising equity from 50% to 60%, 70% and 80% raised the breach count from 9 to 13, 15 and 19. Even 50/50 exceeded 2%, so the paper uses 60/40 only as a benchmark. The folder saves the four-way comparison without its generating command.

**04:26:15 · [Make the benchmark explicit](runs/20261118T042615Z-7pu4f80f/summary.md).** The explicit `--strategy fixed-split` run reproduced the default’s $106.88M CE and all 98 annual outcomes.

**04:27:02 · [Try a calendar glide path](runs/20261118T042702Z-buurur3g/summary.md).** Reducing equity over time cut the annual breach count from 13 to 12, while CE fell from $106.88M to $106.48M. The generic glide path gave little floor protection and was later replaced in the comparison by explicit 65–35 and floor-aware rules.

**04:32:48 · [Use monthly launch dates](runs/20261118T043248Z-yeun0kcz/summary.md).** Starting the same two rules every month expanded the sample from 98 annual launches to 1,179 one-year investments. Fixed split breached 12.21% and the glide path 11.28%. Kept monthly starts to sample more entry dates.

**04:47:22 · [Tie the glide path to the floor](runs/20261118T044722Z-u2kyvg6t/summary.md).** The 65–35 glide breached in 120 of 1,179 starts. The floor-aware 80–20 glide finished below the floor zero times, with $104.21M CE versus $106.43M for the 65–35 rule. Kept the floor-aware glide as the family benchmark.

#### Volatility targets and cushion rules

**18:39:23 · [Test whether volatility targeting protects the floor](runs/20261118T183923Z-sjy4u7ot/summary.md).** Plain volatility targeting breached 175 times and a high-volatility reduction breached 139 times in 1,179 starts. The floor-aware volatility target and floor-aware glide both had zero breaches. Retained the floor-aware target for the comparison.

**18:41:34 · [Replot the volatility-target comparison](runs/20261118T184134Z-intr6tvq/summary.md).** Updated the terminal distribution plot. The launch CSV matched the preceding run, including the 14.84% breach rate for plain volatility targeting.

**18:48:08 · [Compare the revised volatility plots](runs/20261118T184808Z-ggwr0kmi/summary.md).** Revised both the terminal distribution and starting-volatility plots. The four strategies reproduced the same launch outcomes.

**18:49:53 · [Scale the cushion by volatility](runs/20261118T184953Z-rrlnnm0p/summary.md).** The volatility-scaled cushion achieved $104.38M CE versus $104.21M for each floor-aware comparator, with zero breaches for all three. Kept it as the first cushion benchmark.

**18:52:34 · [Compare all eight early rules together](runs/20261118T185234Z-brckhewu/summary.md).** With common launch dates, the volatility target had the highest CE ($107.44M) and 175 breaches. The volatility-scaled cushion had the highest CE among the three zero-breach rules ($104.38M). Used conditional eligibility to narrow the alternatives.

**19:00:57 · [Add a tiered cushion](runs/20261118T190057Z-467ypx6d/summary.md).** Allowing different equity exposure across cushion bands raised CE from $104.38M for the volatility-scaled cushion to $104.92M with zero breaches. Carried the tiered rule into a multiplier comparison.

**19:02:43 · [Increase the tiered exposure to 3×](runs/20261118T190243Z-43nbilla/summary.md).** The 3× variant raised CE to $105.49M from $104.92M, again with zero breaches across 1,179 launches. Kept the 3× version as the tiered comparator.

**~19:10 · [Match the 12- and 24-month start dates](runs/matched-12-24m-lyi4mkq4/matched_results.csv).** Compared five rules on the same 1,167 monthly starts at both maturities. The 3× tiered cushion had zero breaches and CE of $105.49M and $112.55M. Fixed 60/40 breached 144 and 123 times. Kept the two-horizon check.

**~19:22 · [Smooth the tier transitions](runs/smooth-cushion-matched-9kklzay6/summary.md).** On the matched starts, the smooth 4× rule edged out the 3× tiered rule at both horizons ($105.56M versus $105.49M at one year and $112.64M versus $112.55M at two). Both had zero breaches. Kept the smooth rule for scaling trials.

**~19:27 · [Raise the smooth multiplier by 25%](runs/smooth-125-matched-2sstjakk/matched_results.csv).** On the same 1,167 starts, the updated version raised CE from $105.56M to $106.22M at 12 months and from $112.64M to $113.76M at 24 months, with zero breaches throughout. Kept the 1.25 factor in the smooth allocation rule.

#### DP, equity caps and horizon checks

**23:05:26 · [Introduce baseline dynamic programming](runs/20261118T230526Z-m76uiobd/summary.md).** On the later evaluation sample of 910 launches, baseline DP had $108.87M CE and 34 breaches, versus 72 breaches for fixed split and zero for floor-aware glide. Kept DP as the starting point for improving floor protection.

**23:12:37 · [Rank DP against nine alternatives](runs/20261118T231237Z-xqqtxf9r/summary.md).** DP still breached in 3.74% of launches. The smooth 4× cushion had the best CE among eligible rules ($106.83M) and zero breaches. Kept it as the comparator for a capped DP.

**23:38:51 · [Cap DP equity using a walk-forward multiplier](runs/20261118T233851Z-g80n329z/summary.md).** Selecting the cushion multiplier from completed launches reduced breaches from 34 to 3, with CE falling from $108.87M to $108.38M. Kept the cap and saved each annual choice in `selection_history.csv`.

**23:40:05 · [Repeat the capped-DP comparison](runs/20261118T234005Z-oa11qmuz/summary.md).** Repeated the three-strategy comparison. Both the launch outcomes and annual multiplier selections matched the 23:38 run.

**23:49:07 · [Combine tiered caps with fractional positions](runs/20261118T234907Z-ajsvc919/summary.md).** Flat cap, fractional-only, tiered-only and combined DP had 3, 1, 2 and 1 breaches, respectively. The combined rule had the highest CE ($108.60M). Kept both features as the capped-DP baseline.

**23:51:49 · [Separate crisis starts in the one-year benchmark](runs/20261118T235149Z-js8akw2u/summary.md).** Compared fixed split with smooth 4× over 910 launches, reporting crisis starts separately and checking sensitivity to individual crisis episodes. Fixed split breached 72 times and smooth 4× never breached. Kept crisis as a reporting group.

**23:51:49 · [Extend the crisis comparison to two years](runs/20261118T235149Z-tnwd8buf/summary.md).** At 24 months, fixed split breached in 60 of 898 launches, while smooth 4× again had zero breaches.

**23:54:49 · [Add DP and a volatility-scaled cushion to the one-year check](runs/20261118T235449Z-5saov80o/summary.md).** Baseline DP breached 34 times versus zero for both cushion rules. Smooth 4× had $106.83M CE versus $105.19M for the volatility-scaled rule. Kept smooth 4× as the stronger cushion reference.

**23:54:51 · [Check those four rules at two years](runs/20261118T235451Z-r5ghkvf8/summary.md).** At two years, baseline DP breached in 39 of 898 launches. Both cushions had zero breaches, and smooth 4× led the volatility-scaled rule in CE ($115.29M versus $111.17M).

**23:56:47 · [Isolate the combined cap’s effect](runs/20261118T235647Z-kj21dnqh/summary.md).** Across 910 one-year launches, the combined cap cut DP breaches from 34 to one at a CE cost of about $0.27M. Kept uncapped DP for the ablation and the combined cap for further development.

**23:57:48 · [Benchmark combined DP against both cushion rules](runs/20261118T235748Z-1b5rn0zv/summary.md).** On the five-way one-year comparison, combined DP had $108.60M CE and one breach. Smooth 4× had $106.83M CE and zero breaches. Kept combined DP as the higher-CE eligible candidate.

**23:57:56 · [Repeat the combined-DP benchmark over two years](runs/20261118T235756Z-iln15ukd/summary.md).** Combined DP had $117.22M CE and zero breaches, versus $117.52M and 39 breaches for uncapped DP. Kept the combined rule for the two-year policy comparison.

### 2026-11-19 (UTC)

**00:06:41 · [Check one-year results across launch schedules](runs/20261119T000641Z-0go0ai2d/summary.md).** The four-rule rerun retained one combined-DP breach. Its pooled lower-volatility rate was 1/440 (0.23%), but one nonoverlapping calendar offset had 1/33 (3.03%). Kept the pooled rate for evaluation and the offset range as a timing check.

**00:06:49 · [Check two-year schedules](runs/20261119T000649Z-rppw23q4/summary.md).** Combined DP and smooth 4× both had zero breaches in 898 starts, so each nonempty offset also had zero. Combined DP led smooth 4× in CE ($117.22M versus $115.29M). Kept the annual multiplier audit for later policy plots.

### 2026-11-20 (UTC)

#### DP variants and smooth-cushion calibration

**17:30:59 · [Test three alternatives to the combined DP](runs/20261120T173059Z-e6wzujbz/summary.md).** At 12 months, combined, 50/50 split, chance-constrained and discounted-floor DP had 1, 5, 1 and 27 breaches. Chance-constrained DP improved CE over combined ($108.80M versus $108.60M). Kept it as Full DP and dropped discounted-floor DP from contention.

**17:31:09 · [Compare the four DP variants over two years](runs/20261120T173109Z-emwslak2/summary.md).** Chance-constrained DP had the highest CE ($117.90M) and zero breaches. The split policy breached four times and discounted-floor DP breached 36 times. Retained chance-constrained DP at both horizons. The two runs also underlie the standalone `runs/dp_benchmark.png` (`plot_dp_benchmark.py` in commit `7725d51`).

**17:51:15 · [Discount the smooth reserve and calibrate its risk scale](runs/20261120T175115Z-84zev02c/summary.md).** At one year the calibrated time-aware cushion achieved $108.53M CE with two breaches, versus $107.50M and zero for the same rule at fixed scale. The original smooth 4× had $106.83M CE. Kept the calibrated rule as the main smooth alternative.

**17:51:29 · [Test smooth-scale calibration over two years](runs/20261120T175129Z-mmbjx1p8/summary.md).** The ranking reversed at 24 months. Fixed scale achieved $117.06M CE versus $115.16M for the calibrated rule, both without breaches. Kept fixed scale as the two-year control for the subsequent volatility-cap test.

#### Allocation plots and volatility ceilings

**~17:51 · [Show what each cap changes](runs/dp_progression/combined.png).** The historical `plot_dp_progression.py` draws four policy maps plus annual multiplier selection and challenger outcomes. The maps show how fractional positions add choices inside the flat cap and how the upper tier expands exposure as wealth rises. Kept the staged comparison and later redrew it for the report. The folder also has a separate 252-session volatility map.

**~18:04 · [Explain the time-aware cushion](runs/smooth_cushion_argument/funding_and_selection.png).** At 15% forecast volatility, the saved chart shows how discounting the reserve permits more equity with 504 sessions remaining than with 252. Its second panel shows the one- and two-year risk scales selected by launch year. Kept the funded-reserve explanation in the paper.

**~18:14 · [Compare smooth and DP allocations side by side](runs/smooth_dp_comparison/outcomes.png).** The time and volatility maps compare the time-aware cushion with combined and chance-constrained DP. The outcome plots show the 12- and 24-month distributions, including near-floor finishes and breaches by group. Kept the floor detail and allocation maps in the final comparison.

**18:24:45 · [Add allocation plots to the one-year comparison](runs/20261120T182445Z-2uflzp0a/summary.md).** Added wealth-by-time and wealth-by-volatility plots to the eight-strategy comparison. The launch CSV matched the 17:51:15 run. Kept the volatility view for the paper.

**18:24:51 · [Plot the two-year DP allocations](runs/20261120T182451Z-_zqr_cov/summary.md).** Added the same allocation plots for the two-year DP comparison. The rerun reproduced zero breaches for Full DP and 36 for discounted-floor DP.

**18:45:42 · [Cap the smooth rule continuously as volatility rises](runs/20261120T184542Z-lhqkdp2f/summary.md).** At 12 months the capped rule inherited the time-aware rule’s launch scale. It removed both breaches and lowered CE from $108.53M to $108.24M, versus $108.80M and one breach for Full DP. Kept capped smooth as the zero-breach comparator and Full DP as the higher-CE strategy.

**18:45:48 · [Test the volatility ceiling with fixed scale at two years](runs/20261120T184548Z-uzs9td6_/summary.md).** With scale fixed at one, capped smooth, uncapped time-aware smooth and Full DP all had zero breaches across 898 starts. Their CEs were $116.57M, $117.06M and $117.90M, respectively. Kept this as a two-year sensitivity check. The paper’s one-year comparison uses the inherited scale.

### 2026-11-22 · Final report replay

Adopted a 30-session trading lookback after the decade-by-decade comparison described in the report. On the final replay, Full DP had the highest overall CE among eligible strategies ($109.07M) and one breach in 910 launches. Kept Full DP as the paper’s primary strategy and the volatility-capped smooth cushion as the zero-breach comparison ($108.37M CE). [Report driver](report/figures/figures.py) · [Final results](report/figures/strategy-overview.tex)

## Commit history

| Commit time (UTC) | Commit | Message |
| --- | --- | --- |
| 2026-11-18 02:44:40 | `fb0c45c` | Initial commit |
| 2026-11-19 00:22:19 | `91c87ad` | Implement portfolio backtesting and cushion-capped trading strategies |
| 2026-11-20 19:03:01 | `7725d51` | Add DP and smooth cushion strategies, allocation heatmaps, and backtest reports |
| 2026-11-20 23:01:33 | `4b61a5f` | report draft |
| 2026-11-23 01:30:53 | `20ecbbb` | Consolidate backtest tooling and update the strategy report and slides |
