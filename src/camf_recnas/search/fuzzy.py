from __future__ import annotations

import numpy as np
from scipy.integrate import trapezoid


def _triangle(x: np.ndarray, peak: float, left: float, right: float) -> np.ndarray:
    y = np.zeros_like(x, dtype=np.float64)
    if peak > left:
        mask = (x >= left) & (x <= peak)
        y[mask] = (x[mask] - left) / (peak - left)
    else:
        y[x == peak] = 1.0
    if right > peak:
        mask = (x > peak) & (x <= right)
        y[mask] = (right - x[mask]) / (right - peak)
    y[x == peak] = 1.0
    return np.clip(y, 0.0, 1.0)


class FuzzyRateController:
    """Mamdani FIS using HVP and genotypic-diversity inputs from the paper."""

    PC = {
        "L": (0.55, 0.40, 0.70),
        "M": (0.70, 0.55, 0.90),
        "H": (0.90, 0.70, 0.95),
    }
    PM = {
        "L": (0.02, 0.005, 0.05),
        "M": (0.10, 0.05, 0.18),
        "H": (0.25, 0.15, 0.30),
    }
    # Rows: HVP Low/Medium/High; columns: GD Low/Medium/High; outputs: (pc, pm).
    RULES = (
        (("H", "H"), ("H", "M"), ("M", "M")),
        (("H", "M"), ("M", "M"), ("M", "L")),
        (("M", "M"), ("M", "L"), ("L", "L")),
    )

    def __init__(self, resolution: int = 1001):
        if resolution < 101:
            raise ValueError("Fuzzy output resolution must be at least 101")
        self.resolution = resolution

    @staticmethod
    def _hvp_membership(value: float) -> dict[str, float]:
        x = float(np.clip(value, -1.0, 1.0))
        low = 1.0 if x <= 0 else (1.0 - x / 0.05 if x < 0.05 else 0.0)
        med = max(0.0, 1.0 - abs(x - 0.05) / 0.10)
        high = 0.0 if x <= 0.05 else ((x - 0.05) / 0.10 if x < 0.15 else 1.0)
        return {"L": float(np.clip(low, 0, 1)), "M": float(np.clip(med, 0, 1)),
                "H": float(np.clip(high, 0, 1))}

    @staticmethod
    def _gd_membership(value: float) -> dict[str, float]:
        x = float(np.clip(value, 0.0, 1.0))
        low = 1.0 if x <= 0.20 else (1.0 - (x - 0.20) / 0.15 if x < 0.35 else 0.0)
        med = max(0.0, 1.0 - abs(x - 0.45) / 0.25)
        high = 0.0 if x <= 0.55 else ((x - 0.55) / 0.15 if x < 0.70 else 1.0)
        return {"L": float(np.clip(low, 0, 1)), "M": float(np.clip(med, 0, 1)),
                "H": float(np.clip(high, 0, 1))}

    def _infer_output(self, activations: list[tuple[float, str]], kind: str) -> float:
        supports = self.PC if kind == "pc" else self.PM
        low, high = ((0.4, 0.95) if kind == "pc" else (0.005, 0.30))
        grid = np.linspace(low, high, self.resolution)
        aggregate = np.zeros_like(grid)
        for strength, label in activations:
            peak, left, right = supports[label]
            membership = _triangle(grid, peak, left, right)
            aggregate = np.maximum(aggregate, np.minimum(strength, membership))
        denominator = float(trapezoid(aggregate, grid))
        if denominator <= 1e-12:
            return float((low + high) / 2)
        return float(trapezoid(grid * aggregate, grid) / denominator)

    def update(self, hv_progress: float, genotypic_diversity: float) -> tuple[float, float]:
        hv = self._hvp_membership(hv_progress)
        gd = self._gd_membership(genotypic_diversity)
        pc_rules: list[tuple[float, str]] = []
        pm_rules: list[tuple[float, str]] = []
        labels = ("L", "M", "H")
        for row_idx, h_label in enumerate(labels):
            for col_idx, d_label in enumerate(labels):
                strength = hv[h_label] * gd[d_label]  # product t-norm
                pc_label, pm_label = self.RULES[row_idx][col_idx]
                if strength > 0:
                    pc_rules.append((strength, pc_label))
                    pm_rules.append((strength, pm_label))
        return (self._infer_output(pc_rules, "pc"), self._infer_output(pm_rules, "pm"))
