"""Daily trading policies for one equity portfolio and interest-bearing cash."""

from dataclasses import dataclass
from functools import cache
from math import isfinite
from typing import Protocol


FLOOR = 95_000_000.0


@dataclass(frozen=True)
class TradingState:
    """Pre-trade dollar balances and trading-session count since activation."""

    day: int
    horizon_days: int
    equity_value: float
    cash: float
    annual_volatility: float | None = None
    annual_volatility_p80: float | None = None

    def __post_init__(self) -> None:
        if type(self.day) is not int or type(self.horizon_days) is not int:
            raise ValueError("day and horizon_days must be integers")
        if self.horizon_days <= 0 or not 0 <= self.day <= self.horizon_days:
            raise ValueError("Expected a positive horizon and 0 <= day <= horizon_days")
        for balance in (self.equity_value, self.cash):
            if not isfinite(balance) or balance < 0:
                raise ValueError("Balances must be finite and nonnegative")
        for signal in (self.annual_volatility, self.annual_volatility_p80):
            if signal is not None and (not isfinite(signal) or signal < 0):
                raise ValueError("Volatility signals must be finite and nonnegative")


class Trader(Protocol):
    def trade(self, state: TradingState) -> float:
        """Return the equity order in dollars. Positive buys and negative sells."""
        ...


@dataclass(frozen=True)
class FixedSplitTrader:
    """Restore the target equity weight every trading day before the horizon."""

    equity_weight: float = 0.60

    def __post_init__(self) -> None:
        if not 0 <= self.equity_weight <= 1:
            raise ValueError("equity_weight must be finite and between 0 and 1")

    def trade(self, state: TradingState) -> float:
        if state.day == state.horizon_days:
            return 0.0
        return _rebalance_order(state, self.equity_weight)


def _glide_order(state: TradingState, opening_weight: float, decline: float) -> float:
    if state.day == state.horizon_days:
        return 0.0
    progress = state.day / (state.horizon_days - 1) if state.horizon_days > 1 else 0.0
    return _rebalance_order(state, opening_weight - decline * progress)


class GentleGlideTrader:
    """Reduce equity exposure linearly from 65% to 35%."""

    def trade(self, state: TradingState) -> float:
        return _glide_order(state, 0.65, 0.30)


class FloorAwareGlideTrader:
    """Limit the 80%-to-20% glide to twice the dollars above the floor."""

    def trade(self, state: TradingState) -> float:
        return _floor_capped_order(state, _glide_order(state, 0.80, 0.60))


@dataclass(frozen=True)
class VolatilityTargetTrader:
    """Size the equity holding to target annual portfolio volatility."""

    target_vol: float = 0.09

    def __post_init__(self) -> None:
        if not isfinite(self.target_vol) or self.target_vol <= 0:
            raise ValueError("target_vol must be finite and positive")

    def _target_weight(self, state: TradingState) -> float:
        forecast = state.annual_volatility
        if forecast is None:
            raise ValueError("annual_volatility is required")
        if forecast <= self.target_vol:
            return 1.0
        return self.target_vol / forecast

    def trade(self, state: TradingState) -> float:
        if state.day == state.horizon_days:
            return 0.0
        return _rebalance_order(state, self._target_weight(state))


class FloorAwareVolatilityTargetTrader(VolatilityTargetTrader):
    """Cap the volatility-targeted position at twice the cushion above the floor."""

    def trade(self, state: TradingState) -> float:
        return _floor_capped_order(state, super().trade(state))


class HighVolatilityReductionTrader(VolatilityTargetTrader):
    """Reduce the usual 60% equity weight only above the historical 80th percentile."""

    def _target_weight(self, state: TradingState) -> float:
        weight = super()._target_weight(state)
        if state.annual_volatility_p80 is None:
            raise ValueError("annual_volatility_p80 is required")
        if state.annual_volatility > state.annual_volatility_p80:
            return min(0.60, weight)
        return 0.60


class VolatilityScaledCushionTrader:
    """Invest up to three times the cushion, scaling the multiple by forecast volatility."""

    def trade(self, state: TradingState) -> float:
        if state.day == state.horizon_days:
            return 0.0
        forecast = state.annual_volatility
        if forecast is None:
            raise ValueError("annual_volatility is required")
        multiplier = 3.0 if forecast <= 0.10 else 0.30 / forecast
        return self._equity_order(state, multiplier)

    def _equity_order(self, state: TradingState, multiplier: float) -> float:
        return _floor_capped_order(state, state.cash, multiplier)


class AggressiveTieredCushionTrader(VolatilityScaledCushionTrader):
    """Triple the equity multiplier on the portion of the cushion above $5M."""

    def _equity_order(self, state: TradingState, multiplier: float) -> float:
        first_tier = 5_000_000.0
        if state.equity_value + state.cash <= FLOOR + first_tier:
            return super()._equity_order(state, multiplier)
        # The effective floor makes the two allocations meet at the tier boundary.
        effective_floor = FLOOR + first_tier - first_tier / 3
        return _floor_capped_order(state, state.cash, 3 * multiplier, effective_floor)


class SmoothCushionTrader(VolatilityScaledCushionTrader):
    """Ramp marginal equity exposure from one to three to four times the base multiplier."""

    def _equity_order(self, state: TradingState, multiplier: float) -> float:
        multiplier *= 1.25
        cushion = max(0.0, state.equity_value + state.cash - FLOOR)
        if cushion >= 25e6:
            # The first $25M contributes $70M before scaling; each further dollar adds four.
            return _floor_capped_order(state, state.cash, 4 * multiplier, FLOOR + 25e6 - 70e6 / 4)

        first_ramp = min(max(cushion - 3e6, 0.0), 4e6)
        middle = min(max(cushion - 7e6, 0.0), 8e6)
        second_ramp = max(cushion - 15e6, 0.0)
        # Each ramp contributes its occupied width times its average marginal factor.
        exposure = (min(cushion, 3e6) + first_ramp * (1 + first_ramp / 4e6)
                    + 3 * middle + second_ramp * (3 + second_ramp / 20e6))
        return min(state.cash, multiplier * exposure - state.equity_value)


def _floor_capped_order(state: TradingState, order: float, multiplier: float = 2.0,
                        floor: float = FLOOR) -> float:
    if state.day == state.horizon_days:
        return 0.0
    cushion = max(0.0, (state.equity_value + state.cash) - floor)
    if not isfinite(cushion) and multiplier < 1:
        # Total wealth may overflow even when the net order is representable.
        capped_order = (multiplier * state.cash - (1 - multiplier) * state.equity_value
                        - multiplier * floor)
    else:
        capped_order = multiplier * cushion - state.equity_value
    return min(order, capped_order)


@cache
def _wealth_policy(multiplier=None):
    from stochastic_dp import WealthPolicy

    return WealthPolicy(FLOOR, multiplier=multiplier)


@dataclass(frozen=True)
class StochasticDPTrader:
    """Maximize expected floor-aware terminal utility under a fixed two-regime model."""

    multiplier: float | None = None

    def trade(self, state: TradingState) -> float:
        wealth = state.equity_value + state.cash
        if state.day == state.horizon_days or wealth == 0:
            return 0.0
        if state.annual_volatility is None or state.annual_volatility_p80 is None:
            raise ValueError("annual_volatility and annual_volatility_p80 are required")
        high = state.annual_volatility > state.annual_volatility_p80
        policy = _wealth_policy(self.multiplier)
        weight = policy.weight(state.horizon_days - state.day, wealth, high)
        return _rebalance_order(state, weight)


def _rebalance_order(state: TradingState, equity_weight: float) -> float:
    if equity_weight == 1:
        return state.cash

    wealth = state.equity_value + state.cash
    if isfinite(wealth):
        order = equity_weight * wealth - state.equity_value
    else:
        # The net order is representable even when total wealth overflows.
        order = equity_weight * state.cash - (1 - equity_weight) * state.equity_value
    return max(-state.equity_value, min(order, state.cash))
