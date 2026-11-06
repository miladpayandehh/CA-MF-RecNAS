from __future__ import annotations

import numpy as np


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(a <= b) and np.any(a < b))


def nondominated_sort(objectives: np.ndarray) -> list[list[int]]:
    values = np.asarray(objectives, dtype=np.float64)
    n = len(values)
    dominates_set: list[list[int]] = [[] for _ in range(n)]
    dominated_count = np.zeros(n, dtype=np.int64)
    fronts: list[list[int]] = [[]]
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if dominates(values[p], values[q]):
                dominates_set[p].append(q)
            elif dominates(values[q], values[p]):
                dominated_count[p] += 1
        if dominated_count[p] == 0:
            fronts[0].append(p)
    rank = 0
    while fronts[rank]:
        next_front: list[int] = []
        for p in fronts[rank]:
            for q in dominates_set[p]:
                dominated_count[q] -= 1
                if dominated_count[q] == 0:
                    next_front.append(q)
        rank += 1
        fronts.append(next_front)
    return fronts[:-1]


def crowding_distance(objectives: np.ndarray, front: list[int]) -> dict[int, float]:
    values = np.asarray(objectives, dtype=np.float64)
    distances = {idx: 0.0 for idx in front}
    if len(front) <= 2:
        return {idx: float("inf") for idx in front}
    local = values[front]
    for objective_idx in range(values.shape[1]):
        order = np.argsort(local[:, objective_idx], kind="mergesort")
        distances[front[int(order[0])]] = float("inf")
        distances[front[int(order[-1])]] = float("inf")
        low = local[order[0], objective_idx]
        high = local[order[-1], objective_idx]
        if high <= low:
            continue
        for j in range(1, len(order) - 1):
            idx = front[int(order[j])]
            if np.isfinite(distances[idx]):
                distances[idx] += float((local[order[j + 1], objective_idx] -
                                         local[order[j - 1], objective_idx]) / (high - low))
    return distances


def nsga2_select(objectives: np.ndarray, n_select: int) -> list[int]:
    fronts = nondominated_sort(objectives)
    chosen: list[int] = []
    for front in fronts:
        remaining = n_select - len(chosen)
        if remaining <= 0:
            break
        if len(front) <= remaining:
            chosen.extend(front)
        else:
            distance = crowding_distance(objectives, front)
            chosen.extend(sorted(front, key=lambda i: distance[i], reverse=True)[:remaining])
            break
    return chosen
