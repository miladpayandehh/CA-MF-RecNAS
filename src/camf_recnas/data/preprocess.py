from __future__ import annotations

import gzip
import json
import re
from pathlib import Path
from typing import Iterator

import pandas as pd

CANONICAL_COLUMNS = ["user_id", "item_id", "rating", "timestamp"]


def _read_amazon_jsonl(path: Path) -> pd.DataFrame:
    opener = gzip.open if path.suffix == ".gz" else open
    rows = []
    with opener(path, "rt", encoding="utf-8", errors="replace") as stream:
        for line_no, line in enumerate(stream, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_no}: {exc}") from exc
            rows.append((row.get("reviewerID"), row.get("asin"), row.get("overall"),
                         row.get("unixReviewTime", row.get("reviewTime", 0))))
    return pd.DataFrame(rows, columns=CANONICAL_COLUMNS)


def _read_netflix_combined(path: Path) -> pd.DataFrame:
    """Parse Netflix Prize combined_data_N.txt (movie headers end with ':')."""
    rows: list[tuple[str, str, float, int]] = []
    current_movie: str | None = None
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line_no, raw in enumerate(stream, 1):
            line = raw.strip()
            if not line:
                continue
            if line.endswith(":"):
                current_movie = line[:-1]
                continue
            if current_movie is None:
                raise ValueError(f"Netflix rating found before movie header at line {line_no}")
            parts = line.split(",")
            if len(parts) != 3:
                raise ValueError(f"Malformed Netflix row at {path}:{line_no}: {line!r}")
            customer, rating, date = parts
            # The original Netflix file uses YYYY-MM-DD. Keep date sortable as an integer.
            timestamp = int(re.sub(r"-", "", date))
            rows.append((customer, current_movie, float(rating), timestamp))
    return pd.DataFrame(rows, columns=CANONICAL_COLUMNS)


def load_raw_ratings(dataset: str, path: str | Path) -> pd.DataFrame:
    """Read a source dataset and return user_id/item_id/rating/timestamp columns."""
    path = Path(path)
    if dataset == "ml-100k":
        file_path = path / "u.data" if path.is_dir() else path
        df = pd.read_csv(file_path, sep="\t", names=CANONICAL_COLUMNS, engine="python")
    elif dataset == "ml-1m":
        file_path = path / "ratings.dat" if path.is_dir() else path
        df = pd.read_csv(file_path, sep="::", names=CANONICAL_COLUMNS, engine="python")
    elif dataset == "amazon-movies-tv":
        file_path = path
        if path.is_dir():
            candidates = list(path.glob("*Movies*TV*5*.json*"))
            if not candidates:
                candidates = list(path.glob("*.json*"))
            if not candidates:
                raise FileNotFoundError(f"No Amazon Movies and TV JSON file found in {path}")
            file_path = candidates[0]
        df = _read_amazon_jsonl(file_path)
    elif dataset == "netflix":
        if path.is_dir():
            files = sorted(path.glob("combined_data_*.txt"))
            if not files:
                raise FileNotFoundError(f"No combined_data_*.txt files found in {path}")
            frames = [_read_netflix_combined(p) for p in files]
            df = pd.concat(frames, ignore_index=True)
        else:
            df = _read_netflix_combined(path)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    df = df[CANONICAL_COLUMNS].copy()
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").fillna(0)
    df = df.dropna(subset=["user_id", "item_id", "rating"])
    return df


def iterative_k_core(
    ratings: pd.DataFrame, min_user_interactions: int = 5, min_item_interactions: int = 5
) -> pd.DataFrame:
    """Iteratively remove users/items below their minimum degree until stable."""
    result = ratings.copy()
    while not result.empty:
        old_len = len(result)
        user_counts = result.groupby("user_id")["item_id"].transform("size")
        result = result.loc[user_counts >= min_user_interactions]
        if result.empty:
            break
        item_counts = result.groupby("item_id")["user_id"].transform("size")
        result = result.loc[item_counts >= min_item_interactions]
        if len(result) == old_len:
            break
    return result.reset_index(drop=True)


def preprocess_ratings(
    ratings: pd.DataFrame,
    positive_rating_threshold: float = 4.0,
    min_user_interactions: int = 5,
    min_item_interactions: int = 5,
) -> pd.DataFrame:
    """Apply paper's positive-feedback threshold, duplicate removal and min-5 filtering."""
    required = set(CANONICAL_COLUMNS)
    missing = required.difference(ratings.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    df = ratings[CANONICAL_COLUMNS].copy()
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").fillna(0)
    df = df.dropna(subset=["user_id", "item_id", "rating"])
    df = df.loc[df["rating"] >= positive_rating_threshold]
    # The paper says to remove duplicates but does not define a tie-break rule;
    # retain the latest timestamp, with stable sorting for deterministic ties.
    df = df.sort_values(["timestamp", "user_id", "item_id"], kind="mergesort")
    df = df.drop_duplicates(subset=["user_id", "item_id"], keep="last")
    return iterative_k_core(df, min_user_interactions, min_item_interactions)


def _netflix_target_sample(
    ratings: pd.DataFrame, max_users: int, max_items: int, max_interactions: int, seed: int
) -> pd.DataFrame:
    """Deterministic sampling utility; not claimed to reproduce the paper's undisclosed sample."""
    import numpy as np

    rng = np.random.default_rng(seed)
    users = ratings["user_id"].drop_duplicates().to_numpy()
    if len(users) > max_users:
        users = rng.choice(users, size=max_users, replace=False)
    subset = ratings[ratings.user_id.isin(users)]
    item_counts = subset.item_id.value_counts()
    keep_items = item_counts.head(max_items).index
    subset = subset[subset.item_id.isin(keep_items)]
    if len(subset) > max_interactions:
        subset = subset.sample(n=max_interactions, random_state=seed)
    return subset.sort_values(["user_id", "timestamp", "item_id"]).reset_index(drop=True)
