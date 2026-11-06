from __future__ import annotations

import numpy as np


def _hypervolume_2d(points: np.ndarray, reference: np.ndarray) -> float:
    if points.size == 0:
        return 0.0
    valid = points[(points[:, 0] < reference[0]) & (points[:, 1] < reference[1])]
    if len(valid) == 0:
        return 0.0
    valid = np.unique(valid, axis=0)
    ys = np.unique(valid[:, 0])
    area = 0.0
    best_y2 = float(reference[1])
    for idx, y in enumerate(ys):
        best_y2 = min(best_y2, float(valid[valid[:, 0] <= y, 1].min()))
        next_y = float(ys[idx + 1]) if idx + 1 < len(ys) else float(reference[0])
        if next_y > y and best_y2 < reference[1]:
            area += (next_y - float(y)) * (float(reference[1]) - best_y2)
    return float(max(area, 0.0))


def hypervolume_3d(points: np.ndarray, reference: tuple[float, float, float] | np.ndarray) -> float:
    """Exact 3-D dominated hypervolume for minimization (axis-aligned reference box).

    The implementation integrates exact 2-D slices and is intended for small Pareto
    populations used in architecture search, not very high-dimensional objectives.
    """
    pts = np.asarray(points, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or ref.shape != (3,):
        raise ValueError("Expected points [n, 3] and reference [3]")
    if len(pts) == 0:
        return 0.0
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    pts = pts[np.all(pts < ref, axis=1)]
    if len(pts) == 0:
        return 0.0
    xs = np.unique(pts[:, 0])
    volume = 0.0
    for idx, x in enumerate(xs):
        next_x = float(xs[idx + 1]) if idx + 1 < len(xs) else float(ref[0])
        if next_x <= x:
            continue
        active = pts[pts[:, 0] <= x][:, 1:]
        volume += (next_x - float(x)) * _hypervolume_2d(active, ref[1:])
    return float(max(volume, 0.0))
