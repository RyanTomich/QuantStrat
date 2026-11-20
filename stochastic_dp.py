"""Finite-horizon equity allocation with two volatility regimes."""

from functools import cache
from threading import Lock

import numpy as np

from trader import funding_floor


def cap_weight(wealth, floor, multiplier, *, tiered=True):
    """Maximum equity fraction. The upper tier adds half the base multiple above $5M."""
    if multiplier is None:
        return 1.0
    cushion = np.maximum(0, 1 - floor / wealth)
    if tiered:
        cushion += 0.5 * np.maximum(0, 1 - (floor + 5_000_000) / wealth)
    return np.minimum(1, multiplier * cushion)


class WealthPolicy:
    """Share action tables across horizons, indexed by trading sessions remaining."""

    weights = (0.0, 0.25, 0.5, 0.75, 1.0)
    minimum = 50_000_000
    maximum = 200_000_000

    def __init__(self, floor: float, spacing: int = 100_000, multiplier: float | None = None,
                 discount_floor: bool = False, *, tiered=True, fractional=True):
        self.floor = floor
        self.discount_floor = discount_floor
        self.tiered = tiered
        self.multiplier = multiplier
        if multiplier is not None and (not np.isfinite(multiplier) or multiplier <= 0):
            raise ValueError("Expected a finite positive cushion multiplier")
        self.spacing = spacing
        wealth = np.arange(self.minimum, self.maximum + 1, spacing)
        self.wealth = wealth
        normalized = wealth / 100_000_000
        terminal = -1 / normalized - 5 * np.maximum(0, (floor - wealth) / 100_000_000)
        self.values = np.stack((terminal, terminal))
        self.actions = []
        self.lock = Lock()

        # Each action specifies a target fraction of wealth or of the effective cap.
        self.targets = [(w, False) for w in self.weights]
        if multiplier is not None and fractional:
            self.targets.extend((fraction, True) for fraction in (.25, .5, .75))
        self._set_transitions(floor)

    def _floor_at(self, remaining):
        if self.discount_floor:
            return funding_floor(self.floor, remaining)
        return self.floor

    def _set_transitions(self, floor):
        wealth = self.wealth
        self.segments = []
        cap = cap_weight(wealth, floor, self.multiplier, tiered=self.tiered)
        weights = np.stack([np.broadcast_to(self._action_weight(i, cap), wealth.shape)
                            for i in range(len(self.targets))])
        self.by_exposure = np.argsort(weights, axis=0, kind="stable")
        weights = weights[:, None, :]
        signs = np.array([-1, 1])[:, None]
        for volatility in (0.15, 0.30):
            premium = 0.05 / 252 + signs * volatility / np.sqrt(252)
            next_wealth = wealth * (1 + 0.02 / 252 + weights * premium)
            location = (next_wealth - self.minimum) / self.spacing
            left = np.clip(np.floor(location).astype(int), 0, len(wealth) - 2)
            self.segments.append((left, location - left))

    def _action_weight(self, action, cap):
        fraction, of_cap = self.targets[action]
        return fraction * cap if of_cap else np.minimum(fraction, cap)

    def _extend(self):
        if self.discount_floor:
            self._set_transitions(self._floor_at(len(self.actions) + 1))
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

    def _index(self, wealth):
        bounded = max(self.minimum, min(self.maximum, wealth))
        offset, remainder = divmod(bounded - self.minimum, self.spacing)
        return int(offset) + (remainder > self.spacing / 2)

    def weight(self, remaining: int, wealth: float, high_volatility: bool) -> float:
        index = self._index(wealth)
        with self.lock:
            while len(self.actions) < remaining:
                self._extend()
            action = self.actions[remaining - 1][int(high_volatility), index]
        cap = cap_weight(wealth, self._floor_at(remaining), self.multiplier, tiered=self.tiered)
        return self._action_weight(action, cap)


class ChancePolicy(WealthPolicy):
    """Price a terminal breach in units of initial wealth and track its probability."""

    def __init__(self, floor, multiplier, penalty, launch_wealth):
        super().__init__(floor, multiplier=multiplier)
        self.penalty = penalty
        self.launch_index = self._index(launch_wealth)
        self.launch_scores = []
        self.means = np.stack([self.wealth, self.wealth]) / 100_000_000
        self.risks = np.asarray([self.wealth < floor] * 2, dtype=float)

    def _set_transitions(self, floor):
        cap = cap_weight(self.wealth, floor, self.multiplier)
        weights = np.array([self._action_weight(i, cap) for i in range(len(self.targets))])
        self.by_exposure = np.argsort(weights, axis=0, kind="stable")
        self.segments = []
        # Rare gaps supplement the ordinary shocks. Center the mixture so its
        # unconditional equity premium remains the incumbent's 5% / 252.
        for sigma, gap, probability in ((.15, -.10, .001), (.30, -.20, .004)):
            center = (.05 / 252 - probability * gap) / (1 - probability)
            ordinary = sigma / np.sqrt(252)
            excess = np.array([center - ordinary, center + ordinary, gap])
            mass = np.array([(1 - probability) / 2] * 2 + [probability])
            successor = self.wealth * (1 + .02 / 252 + weights[:, None, :] * excess[:, None])
            location = (successor - self.minimum) / self.spacing
            left = np.clip(np.floor(location).astype(int), 0, len(self.wealth) - 2)
            self.segments.append((mass, successor, left, location - left))

    def _extend(self):
        means = np.empty_like(self.means)
        risks = np.empty_like(self.risks)
        actions = np.empty_like(self.risks, dtype=np.uint8)
        for regime, switch in enumerate((.01, .04)):
            mass, successor, left, fraction = self.segments[regime]
            if self.actions:
                mean = (1 - switch) * self.means[regime] + switch * self.means[1 - regime]
                risk = (1 - switch) * self.risks[regime] + switch * self.risks[1 - regime]
                branch_mean = mean[left] + fraction * (mean[left + 1] - mean[left])
                branch_risk = risk[left] + fraction.clip(0, 1) * (risk[left + 1] - risk[left])
            else:
                # Preserve the terminal discontinuity, including equality at the floor.
                branch_mean = successor / 100_000_000
                branch_risk = successor < self.floor
            expected = (branch_mean * mass[:, None]).sum(axis=1)
            breach = (branch_risk * mass[:, None]).sum(axis=1)
            score = expected - self.penalty * breach
            ranked = np.take_along_axis(score, self.by_exposure, axis=0)
            best = np.take_along_axis(self.by_exposure, ranked.argmax(axis=0)[None, :], axis=0)
            actions[regime] = best[0]
            means[regime] = np.take_along_axis(expected, best, axis=0)[0]
            risks[regime] = np.take_along_axis(breach, best, axis=0)[0]
        self.actions.append(actions)
        self.means, self.risks = means, risks
        self.launch_scores.append((means[:, self.launch_index].copy(), risks[:, self.launch_index].copy()))

    def launch_score(self, remaining, high):
        with self.lock:
            while len(self.actions) < remaining:
                self._extend()
            means, risks = self.launch_scores[remaining - 1]
            return means[int(high)], risks[int(high)]


@cache
def chance_policy(floor, multiplier, penalty, launch_wealth):
    return ChancePolicy(floor, multiplier, penalty, launch_wealth)
