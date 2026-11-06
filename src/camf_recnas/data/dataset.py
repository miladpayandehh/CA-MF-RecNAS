from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class InteractionDataset:
    n_users: int
    n_items: int
    train: np.ndarray  # [user, item, rating, timestamp]
    validation: np.ndarray
    test: np.ndarray
    user_ids: list[str]
    item_ids: list[str]
    name: str = "dataset"

    def __post_init__(self) -> None:
        for field_name in ("train", "validation", "test"):
            array = np.asarray(getattr(self, field_name))
            if array.ndim != 2 or array.shape[1] != 4:
                raise ValueError(f"{field_name} must have shape [n, 4], got {array.shape}")
            setattr(self, field_name, array.astype(np.float64, copy=False))

    @property
    def train_users(self) -> np.ndarray:
        return self.train[:, 0].astype(np.int64)

    @property
    def train_items(self) -> np.ndarray:
        return self.train[:, 1].astype(np.int64)

    def histories(self, max_length: int | None = None) -> dict[int, list[int]]:
        frame = pd.DataFrame(self.train, columns=["user", "item", "rating", "timestamp"])
        result: dict[int, list[int]] = {}
        for user, group in frame.sort_values("timestamp").groupby("user"):
            values = group.item.astype(int).tolist()
            result[int(user)] = values[-max_length:] if max_length else values
        return result

    def history_matrix(self, max_length: int = 50) -> np.ndarray:
        matrix = np.full((self.n_users, max_length), -1, dtype=np.int64)
        for user, items in self.histories(max_length).items():
            matrix[user, -len(items):] = items
        return matrix

    def low_fidelity_train(self, fraction: float = 0.05, seed: int = 42) -> np.ndarray:
        """Uniform per-user subsampling, retaining at least one train interaction/user."""
        rng = np.random.default_rng(seed)
        frame = pd.DataFrame(self.train, columns=["user", "item", "rating", "timestamp"])
        selected = []
        for _, group in frame.groupby("user", sort=True):
            n = max(1, int(np.ceil(len(group) * fraction)))
            positions = rng.choice(len(group), size=min(n, len(group)), replace=False)
            selected.append(group.iloc[np.sort(positions)].to_numpy())
        if not selected:
            return np.empty((0, 4), dtype=np.float64)
        return np.concatenate(selected, axis=0)


def leave_one_out_split(ratings: pd.DataFrame, name: str = "dataset") -> InteractionDataset:
    """Timestamp-ordered leave-one-out: last=test, second-last=validation."""
    data = ratings.copy()
    data["timestamp"] = pd.to_numeric(data["timestamp"], errors="coerce").fillna(0)
    data = data.sort_values(["user_id", "timestamp", "item_id"], kind="mergesort")
    user_ids = sorted(data.user_id.astype(str).unique().tolist())
    item_ids = sorted(data.item_id.astype(str).unique().tolist())
    u_map = {value: i for i, value in enumerate(user_ids)}
    i_map = {value: i for i, value in enumerate(item_ids)}
    data["u"] = data.user_id.astype(str).map(u_map)
    data["i"] = data.item_id.astype(str).map(i_map)
    train_rows, val_rows, test_rows = [], [], []
    for _, group in data.groupby("u", sort=True):
        rows = group[["u", "i", "rating", "timestamp"]].to_numpy(dtype=np.float64)
        if len(rows) < 3:
            continue
        train_rows.extend(rows[:-2])
        val_rows.append(rows[-2])
        test_rows.append(rows[-1])
    train = np.asarray(train_rows, dtype=np.float64).reshape(-1, 4)
    validation = np.asarray(val_rows, dtype=np.float64).reshape(-1, 4)
    test = np.asarray(test_rows, dtype=np.float64).reshape(-1, 4)
    used_users = sorted(set(train[:, 0].astype(int).tolist())) if len(train) else []
    # Users with no train history are excluded; the 5-core paper protocol should prevent this.
    if len(used_users) != len(user_ids):
        keep = set(used_users)
        user_remap = {old: new for new, old in enumerate(used_users)}
        train = train[np.isin(train[:, 0].astype(int), list(keep))]
        validation = validation[np.isin(validation[:, 0].astype(int), list(keep))]
        test = test[np.isin(test[:, 0].astype(int), list(keep))]
        for arr in (train, validation, test):
            if len(arr):
                arr[:, 0] = np.asarray([user_remap[int(x)] for x in arr[:, 0]], dtype=np.float64)
        user_ids = [user_ids[i] for i in used_users]
    used_items = sorted(set(train[:, 1].astype(int).tolist()) |
                        set(validation[:, 1].astype(int).tolist()) |
                        set(test[:, 1].astype(int).tolist()))
    item_remap = {old: new for new, old in enumerate(used_items)}
    for arr in (train, validation, test):
        if len(arr):
            arr[:, 1] = np.asarray([item_remap[int(x)] for x in arr[:, 1]], dtype=np.float64)
    item_ids = [item_ids[i] for i in used_items]
    return InteractionDataset(len(user_ids), len(item_ids), train, validation, test,
                               user_ids, item_ids, name)


def save_processed_dataset(dataset: InteractionDataset, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, n_users=np.asarray(dataset.n_users), n_items=np.asarray(dataset.n_items),
                        train=dataset.train, validation=dataset.validation, test=dataset.test,
                        user_ids=np.asarray(dataset.user_ids, dtype=str),
                        item_ids=np.asarray(dataset.item_ids, dtype=str), name=np.asarray(dataset.name))
    metadata_path = path.with_suffix(".json")
    metadata_path.write_text(json.dumps({"name": dataset.name, "n_users": dataset.n_users,
                                        "n_items": dataset.n_items,
                                        "train_interactions": len(dataset.train),
                                        "validation_interactions": len(dataset.validation),
                                        "test_interactions": len(dataset.test)}, indent=2), encoding="utf-8")


def load_processed_dataset(path: str | Path) -> InteractionDataset:
    with np.load(path, allow_pickle=False) as data:
        return InteractionDataset(int(data["n_users"]), int(data["n_items"]), data["train"],
                                  data["validation"], data["test"],
                                  data["user_ids"].astype(str).tolist(),
                                  data["item_ids"].astype(str).tolist(), str(data["name"]))
