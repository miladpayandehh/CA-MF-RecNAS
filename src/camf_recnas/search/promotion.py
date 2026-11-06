from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import RidgeCV

from camf_recnas.models.encoding import Architecture, BLOCK_TYPES, ACTIVATIONS
from camf_recnas.objectives.hypervolume import hypervolume_3d


@dataclass
class Prediction:
    objectives: np.ndarray
    standard_deviation: np.ndarray
    full_cost_hours: float


def architecture_features(architecture: Architecture, low_record) -> np.ndarray:
    """Features listed in the manuscript: low NDCG, size, depth, histograms, latency."""
    block_hist = [sum(g.kind == kind for g in architecture.blocks) for kind in BLOCK_TYPES]
    activation_hist = [sum(g.activation == activation for g in architecture.blocks)
                       for activation in ACTIVATIONS]
    latency = max(float(low_record.latency_seconds_p95), 1e-9)
    values = [float(low_record.ndcg10), float(low_record.parameter_count),
              float(len(architecture.blocks)), *map(float, block_hist),
              *map(float, activation_hist), float(latency)]
    return np.asarray(values, dtype=np.float64)


class MultiFidelityCalibrator:
    """Per-dataset ridge calibration with 5-fold alpha selection and residual uncertainty."""

    def __init__(self, alphas: list[float] | None = None):
        self.alphas = alphas or [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
        self.models: list[RidgeCV] = []
        self.residual_std: np.ndarray | None = None
        self.x_mean: np.ndarray | None = None
        self.xtx_inverse: np.ndarray | None = None
        self.cost_model: RidgeCV | None = None
        self.cost_residual_std: float = 0.0
        self.n_features: int | None = None
        self.is_fitted = False

    def fit(self, features: np.ndarray, full_objectives: np.ndarray,
            full_cost_hours: np.ndarray) -> "MultiFidelityCalibrator":
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(full_objectives, dtype=np.float64)
        costs = np.asarray(full_cost_hours, dtype=np.float64)
        if x.ndim != 2 or y.shape != (len(x), 3) or len(x) < 5:
            raise ValueError("Calibration requires X[n, features], Y[n, 3], and at least 5 rows")
        if len(costs) != len(x):
            raise ValueError("One full-training cost is required per calibration architecture")
        self.n_features = x.shape[1]
        self.models = []
        residuals = []
        alphas = self.alphas
        for j in range(3):
            model = RidgeCV(alphas=alphas, cv=min(5, len(x)), scoring="neg_mean_squared_error")
            model.fit(x, y[:, j])
            self.models.append(model)
            residuals.append(y[:, j] - model.predict(x))
        self.residual_std = np.asarray([max(float(np.std(r, ddof=1)), 1e-8) for r in residuals])
        self.x_mean = x.mean(axis=0)
        centered = x - self.x_mean
        selected_alpha = float(np.median([m.alpha_ for m in self.models]))
        gram = centered.T @ centered + selected_alpha * np.eye(x.shape[1])
        self.xtx_inverse = np.linalg.pinv(gram)
        safe_cost = np.maximum(costs, 1e-9)
        self.cost_model = RidgeCV(alphas=alphas, cv=min(5, len(x)),
                                  scoring="neg_mean_squared_error")
        self.cost_model.fit(x, np.log(safe_cost))
        cost_residual = np.log(safe_cost) - self.cost_model.predict(x)
        self.cost_residual_std = max(float(np.std(cost_residual, ddof=1)), 1e-8)
        self.is_fitted = True
        return self

    def predict(self, features: np.ndarray) -> Prediction:
        if not self.is_fitted:
            raise RuntimeError("Calibrator has not been fitted")
        x = np.asarray(features, dtype=np.float64).reshape(1, -1)
        if x.shape[1] != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {x.shape[1]}")
        predicted = np.asarray([model.predict(x)[0] for model in self.models])
        # Objective values are normalized and clipped to [0,1] by definition.
        predicted = np.clip(predicted, 0.0, 1.0)
        centered = x[0] - self.x_mean
        leverage = float(np.sqrt(max(1.0 + centered @ self.xtx_inverse @ centered, 1.0)))
        std = self.residual_std * leverage
        log_cost = float(self.cost_model.predict(x)[0])
        cost = float(np.exp(np.clip(log_cost, -20, 20)))
        return Prediction(predicted, std, max(cost, 1e-8))

    def promotion_score(self, features: np.ndarray, full_front_objectives: np.ndarray,
                        reference: tuple[float, float, float] = (1.05, 1.05, 1.05),
                        z: float = 1.6448536269514722, epsilon: float = 1e-8) -> float:
        """LCB of predicted 3-D HV improvement divided by estimated full cost.

        The uncertainty of the hypervolume gain is estimated with a first-order
        finite-difference delta approximation, matching the manuscript's description.
        """
        pred = self.predict(features)
        ref = np.asarray(reference, dtype=np.float64)
        front = np.asarray(full_front_objectives, dtype=np.float64).reshape(-1, 3)
        baseline = hypervolume_3d(front, ref)

        def improvement(point: np.ndarray) -> float:
            return max(0.0, hypervolume_3d(np.vstack([front, point]), ref) - baseline)

        utility = improvement(pred.objectives)
        gradient = np.zeros(3, dtype=np.float64)
        for j in range(3):
            step = 1e-4 * max(1.0, abs(float(pred.objectives[j])))
            plus, minus = pred.objectives.copy(), pred.objectives.copy()
            plus[j] += step
            minus[j] -= step
            gradient[j] = (improvement(plus) - improvement(minus)) / (2 * step)
        sigma_utility = float(np.sqrt(np.sum((gradient * pred.standard_deviation) ** 2)))
        lower_confidence_bound = utility - z * sigma_utility
        return float(lower_confidence_bound / (epsilon + pred.full_cost_hours))
