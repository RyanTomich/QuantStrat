"""Finite-horizon equity allocation with two volatility regimes."""

from threading import Lock

import numpy as np


def cap_weight(wealth, floor, multiplier):
    """Maximum equity fraction. The upper tier adds half the base multiple above $5M."""
    if multiplier is None:
        return 1.0
    cushion = np.maximum(0, 1 - floor / wealth)
    cushion += 0.5 * np.maximum(0, 1 - (floor + 5_000_000) / wealth)
    return np.minimum(1, multiplier * cushion)


class WealthPolicy:
    """Share action tables across horizons, indexed by trading sessions remaining."""

    weights = (0.0, 0.25, 0.5, 0.75, 1.0)
    minimum = 50_000_000
    maximum = 200_000_000

    def __init__(self, floor: float, spacing: int = 100_000, multiplier: float | None = None):
        self.floor = floor
        self.multiplier = multiplier
        if multiplier is not None and (not np.isfinite(multiplier) or multiplier <= 0):
            raise ValueError("Expected a finite positive cushion multiplier")
        self.spacing = spacing
        wealth = np.arange(self.minimum, self.maximum + 1, spacing)
        normalized = wealth / 100_000_000
        terminal = -1 / normalized - 5 * np.maximum(0, (floor - wealth) / 100_000_000)
        self.values = np.stack((terminal, terminal))
        self.actions = []
        self.lock = Lock()

        # Precompute the two successor wealth locations for every action and grid point.
        # Clamp the segment index only. Fractions outside [0, 1] extend its slope.
        self.segments = []
        # Each action specifies a target fraction of wealth or of the effective cap.
        self.targets = [(w, False) for w in self.weights]
        if multiplier is not None:
            self.targets.extend((fraction, True) for fraction in (.25, .5, .75))
        cap = cap_weight(wealth, floor, multiplier)
        weights = np.stack([np.broadcast_to(self._action_weight(i, cap), wealth.shape)
                            for i in range(len(self.targets))])
        self.by_exposure = np.argsort(weights, axis=0, kind="stable")
        weights = weights[:, None, :]
        signs = np.array([-1, 1])[:, None]
        for volatility in (0.15, 0.30):
            premium = 0.05 / 252 + signs * volatility / np.sqrt(252)
            next_wealth = wealth * (1 + 0.02 / 252 + weights * premium)
            location = (next_wealth - self.minimum) / spacing
            left = np.clip(np.floor(location).astype(int), 0, len(wealth) - 2)
            self.segments.append((left, location - left))

    def _action_weight(self, action, cap):
        fraction, of_cap = self.targets[action]
        return fraction * cap if of_cap else np.minimum(fraction, cap)

    def _extend(self):
        values = np.empty_like(self.values)
        actions = np.empty(self.values.shape, dtype=np.uint8)
        for regime, switch in enumerate((0.01, 0.04)):
            continuation = (1 - switch) * self.values[regime] + switch * self.values[1 - regime]
            left, fraction = self.segments[regime]
            outcomes = continuation[left] + fraction * (continuation[left + 1] - continuation[left])
            expected = outcomes.mean(axis=1)
            # Sort by exposure at each balance so equal values favor the smaller position.
            ranked = np.take_along_axis(expected, self.by_exposure, axis=0)
            best = ranked.argmax(axis=0)[None, :]
            actions[regime] = np.take_along_axis(self.by_exposure, best, axis=0)[0]
            values[regime] = np.take_along_axis(expected, actions[regime][None, :], axis=0)[0]
        self.actions.append(actions)
        self.values = values

    def weight(self, remaining: int, wealth: float, high_volatility: bool) -> float:
        bounded = max(self.minimum, min(self.maximum, wealth))
        offset, remainder = divmod(bounded - self.minimum, self.spacing)
        index = int(offset) + (remainder > self.spacing / 2)
        with self.lock:
            while len(self.actions) < remaining:
                self._extend()
            action = self.actions[remaining - 1][int(high_volatility), index]
        cap = cap_weight(wealth, self.floor, self.multiplier)
        return self._action_weight(action, cap)
