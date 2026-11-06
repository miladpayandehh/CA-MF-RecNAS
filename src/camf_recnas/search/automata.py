from __future__ import annotations

import numpy as np

from camf_recnas.search.operators import OPERATOR_NAMES


class CostAwareLearningAutomata:
    """Multi-action Linear Reward-Inaction update with cost-aware EMA rewards."""

    def __init__(self, learning_rate: float = 0.1, beta: float = 0.3):
        if not 0 < learning_rate <= 1:
            raise ValueError("learning_rate must be in (0, 1]")
        if not 0 < beta <= 1:
            raise ValueError("beta must be in (0, 1]")
        self.learning_rate = learning_rate
        self.beta = beta
        self.probabilities = np.full(len(OPERATOR_NAMES), 1.0 / len(OPERATOR_NAMES))
        self.ema_rewards = np.zeros(len(OPERATOR_NAMES), dtype=np.float64)

    @staticmethod
    def allocate_slots(total: int, probabilities: np.ndarray, minimum: int = 2) -> np.ndarray:
        if total < len(probabilities) * minimum:
            raise ValueError("Population size is too small for minimum slots per operator")
        p = np.asarray(probabilities, dtype=np.float64)
        p = p / p.sum()
        remaining = total - len(p) * minimum
        raw = p * remaining
        base = np.floor(raw).astype(int) + minimum
        leftover = total - int(base.sum())
        order = np.argsort(raw - np.floor(raw))[::-1]
        for index in order[:leftover]:
            base[index] += 1
        return base

    def update(self, raw_rewards: np.ndarray) -> dict[str, object]:
        rewards = np.asarray(raw_rewards, dtype=np.float64)
        if rewards.shape != self.ema_rewards.shape:
            raise ValueError("One reward per crossover operator is required")
        self.ema_rewards = (1.0 - self.beta) * self.ema_rewards + self.beta * rewards
        threshold = float(self.ema_rewards.mean())
        rewarded = np.flatnonzero((self.ema_rewards > threshold) & (self.ema_rewards > 0))
        if len(rewarded):
            next_p = (1.0 - self.learning_rate) * self.probabilities
            next_p[rewarded] += self.learning_rate / len(rewarded)
            self.probabilities = next_p / next_p.sum()
        # Numerically enforce the simplex invariant.
        self.probabilities = np.maximum(self.probabilities, 0)
        self.probabilities /= self.probabilities.sum()
        return {"ema_rewards": self.ema_rewards.copy(), "threshold": threshold,
                "rewarded_operators": [OPERATOR_NAMES[i] for i in rewarded],
                "probabilities": self.probabilities.copy()}
