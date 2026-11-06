from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon


def mean_confidence_interval(values: list[float] | np.ndarray, confidence_z: float = 1.96) -> tuple[float, float, float]:
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        raise ValueError("At least one observation is required")
    mean = float(x.mean())
    if x.size == 1:
        return mean, mean, mean
    half_width = float(confidence_z * x.std(ddof=1) / np.sqrt(x.size))
    return mean, mean - half_width, mean + half_width


def wilcoxon_paired(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> float:
    x, y = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("Paired samples must be one-dimensional arrays of equal length")
    if np.allclose(x, y):
        return 1.0
    return float(wilcoxon(x, y, alternative="two-sided", zero_method="wilcox").pvalue)


def benjamini_hochberg(p_values: list[float] | np.ndarray) -> np.ndarray:
    """Return BH-adjusted p-values in the original order."""
    p = np.asarray(p_values, dtype=np.float64)
    n = len(p)
    if np.any((p < 0) | (p > 1)):
        raise ValueError("p-values must be in [0, 1]")
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    adjusted_sorted = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty(n, dtype=np.float64)
    adjusted[order] = np.clip(adjusted_sorted, 0.0, 1.0)
    return adjusted
