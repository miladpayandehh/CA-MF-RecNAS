from __future__ import annotations

import numpy as np

from .dataset import InteractionDataset


def few_shot_user_protocol(dataset: InteractionDataset, k: int = 3,
                           user_fraction: float = 0.20, seed: int = 42) -> InteractionDataset:
    """Retain k train events for a random 20% user subset; hold the rest out.

    This is the manuscript's few-shot proxy, not a true unseen-user cold-start split.
    The returned validation set is unchanged; held-out train events are appended to test.
    """
    if k < 1 or not 0 < user_fraction <= 1:
        raise ValueError("k must be >=1 and user_fraction must be in (0, 1]")
    rng = np.random.default_rng(seed)
    selected_count = max(1, int(round(dataset.n_users * user_fraction)))
    selected_users = set(rng.choice(dataset.n_users, size=selected_count, replace=False).tolist())
    keep_rows, held_out = [], []
    for user in range(dataset.n_users):
        rows = dataset.train[dataset.train[:, 0].astype(int) == user]
        rows = rows[np.argsort(rows[:, 3], kind="mergesort")]
        if user in selected_users and len(rows) > k:
            keep_rows.extend(rows[:k])
            held_out.extend(rows[k:])
        else:
            keep_rows.extend(rows)
    new_test = np.vstack([dataset.test, np.asarray(held_out).reshape(-1, 4)]) if held_out else dataset.test.copy()
    return InteractionDataset(dataset.n_users, dataset.n_items,
                              np.asarray(keep_rows).reshape(-1, 4),
                              dataset.validation.copy(), new_test,
                              dataset.user_ids.copy(), dataset.item_ids.copy(),
                              f"{dataset.name}-fewshot-k{k}")


def controlled_sparsity(dataset: InteractionDataset, remove_fraction: float,
                        seed: int = 42) -> InteractionDataset:
    """Randomly remove a fraction of train interactions and drop degree-zero IDs.

    This follows the paper's evaluation caveat: users/items with no remaining training
    events are excluded from train, validation and test for this stress condition.
    """
    if not 0 <= remove_fraction < 1:
        raise ValueError("remove_fraction must be in [0, 1)")
    rng = np.random.default_rng(seed)
    keep_mask = rng.random(len(dataset.train)) >= remove_fraction
    train = dataset.train[keep_mask].copy()
    if len(train) == 0:
        raise ValueError("Sparsification removed every training interaction")
    active_users = np.unique(train[:, 0].astype(int))
    active_items = np.unique(train[:, 1].astype(int))
    user_map = {old: new for new, old in enumerate(active_users.tolist())}
    item_map = {old: new for new, old in enumerate(active_items.tolist())}

    def remap_rows(rows: np.ndarray) -> np.ndarray:
        if len(rows) == 0:
            return np.empty((0, 4), dtype=np.float64)
        rows = rows[np.isin(rows[:, 0].astype(int), active_users) &
                    np.isin(rows[:, 1].astype(int), active_items)].copy()
        if len(rows):
            rows[:, 0] = [user_map[int(x)] for x in rows[:, 0]]
            rows[:, 1] = [item_map[int(x)] for x in rows[:, 1]]
        return rows

    return InteractionDataset(len(active_users), len(active_items), remap_rows(train),
                              remap_rows(dataset.validation), remap_rows(dataset.test),
                              [dataset.user_ids[i] for i in active_users],
                              [dataset.item_ids[i] for i in active_items],
                              f"{dataset.name}-sparsity-{remove_fraction:g}")
