from __future__ import annotations

import time

import numpy as np
import torch


def ranking_metrics_from_ranks(ranks: np.ndarray, k: int = 10) -> dict[str, float]:
    ranks = np.asarray(ranks, dtype=np.int64)
    hits = ranks <= k
    ndcg = np.where(hits, 1.0 / np.log2(ranks + 1), 0.0)
    return {
        f"HR@{k}": float(hits.mean()) if len(hits) else 0.0,
        f"Recall@{k}": float(hits.mean()) if len(hits) else 0.0,
        f"NDCG@{k}": float(ndcg.mean()) if len(ndcg) else 0.0,
    }


@torch.no_grad()
def evaluate_ranking(model, dataset, split: str = "validation", device: str = "cpu",
                     max_sequence_length: int = 50, batch_users: int = 128,
                     item_chunk_size: int = 2048, k: int = 10) -> dict[str, float]:
    """All-item ranking evaluation with one or multiple held-out positives per user.

    On the ordinary leave-one-out split each user has one positive, so HR@k and
    Recall@k are equal as stated in the manuscript. Few-shot stress evaluation can
    hold out several interactions per user; in that case HR and Recall are distinct.
    """
    if split not in {"validation", "test"}:
        raise ValueError("split must be 'validation' or 'test'")
    model.eval()
    target = dataset.validation if split == "validation" else dataset.test
    if len(target) == 0:
        return {f"HR@{k}": 0.0, f"Recall@{k}": 0.0, f"NDCG@{k}": 0.0}

    # Group held-out positives by user before batching; this avoids splitting a
    # multi-positive user's targets across separate batches.
    positives_by_user: dict[int, set[int]] = {}
    for row in target:
        positives_by_user.setdefault(int(row[0]), set()).add(int(row[1]))
    evaluated_users = np.asarray(sorted(positives_by_user), dtype=np.int64)
    histories_np = dataset.history_matrix(max_sequence_length)
    train_seen: list[set[int]] = [set() for _ in range(dataset.n_users)]
    for row in dataset.train:
        train_seen[int(row[0])].add(int(row[1]))
    all_items = torch.arange(dataset.n_items, device=device)
    hits_per_user: list[float] = []
    recalls_per_user: list[float] = []
    ndcgs_per_user: list[float] = []
    effective_k = min(k, dataset.n_items)

    for start in range(0, len(evaluated_users), batch_users):
        user_batch = evaluated_users[start:start + batch_users]
        users = torch.as_tensor(user_batch, dtype=torch.long, device=device)
        histories = torch.as_tensor(histories_np[user_batch], dtype=torch.long, device=device)
        hidden = model.encode_users(users, histories, model._evaluation_adjacency)
        score_parts = []
        for item_start in range(0, dataset.n_items, item_chunk_size):
            item_ids = all_items[item_start:item_start + item_chunk_size]
            expanded_h = hidden[:, None, :].expand(-1, len(item_ids), -1)
            expanded_i = item_ids[None, :].expand(len(users), -1)
            score_parts.append(model.score_from_hidden(expanded_h, expanded_i))
        scores = torch.cat(score_parts, dim=1)

        for row_idx, user in enumerate(user_batch.tolist()):
            seen = train_seen[user]
            if seen:
                scores[row_idx, torch.as_tensor(sorted(seen), dtype=torch.long,
                                                 device=scores.device)] = -torch.inf
            top_items = scores[row_idx].topk(effective_k).indices.cpu().numpy().tolist()
            relevant = positives_by_user[user]
            hit_positions = [rank for rank, item in enumerate(top_items) if item in relevant]
            hit_count = len(hit_positions)
            ideal_count = min(effective_k, len(relevant))
            dcg = sum(1.0 / np.log2(rank + 2) for rank in hit_positions)
            idcg = sum(1.0 / np.log2(rank + 2) for rank in range(ideal_count))
            hits_per_user.append(1.0 if hit_count else 0.0)
            recalls_per_user.append(hit_count / max(len(relevant), 1))
            ndcgs_per_user.append(dcg / idcg if idcg > 0 else 0.0)

    return {f"HR@{k}": float(np.mean(hits_per_user)) if hits_per_user else 0.0,
            f"Recall@{k}": float(np.mean(recalls_per_user)) if recalls_per_user else 0.0,
            f"NDCG@{k}": float(np.mean(ndcgs_per_user)) if ndcgs_per_user else 0.0}


@torch.no_grad()
def measure_latency(model, dataset, device: str = "cpu", warmup: int = 100,
                    repeats: int = 1000, max_sequence_length: int = 50) -> float:
    """Return p95 single-user inference latency in seconds."""
    model.eval()
    user = torch.tensor([0], dtype=torch.long, device=device)
    history = torch.as_tensor(dataset.history_matrix(max_sequence_length)[:1],
                              dtype=torch.long, device=device)
    item = torch.tensor([0], dtype=torch.long, device=device)
    adjacency = model._evaluation_adjacency
    use_cuda = str(device).startswith("cuda")
    for _ in range(warmup):
        _ = model(user, item, history, adjacency)
    if use_cuda:
        torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        _ = model(user, item, history, adjacency)
        if use_cuda:
            torch.cuda.synchronize()
        samples.append(time.perf_counter() - start)
    return float(np.percentile(samples, 95))
