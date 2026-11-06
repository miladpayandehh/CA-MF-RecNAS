from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _training_seen_by_user(dataset) -> list[set[int]]:
    seen_by_user: list[set[int]] = [set() for _ in range(dataset.n_users)]
    for row in dataset.train:
        seen_by_user[int(row[0])].add(int(row[1]))
    return seen_by_user


def _mask_seen_training_items(scores: torch.Tensor, users: torch.Tensor,
                              seen_by_user: list[set[int]]) -> None:
    for row_index, user in enumerate(users.tolist()):
        seen = seen_by_user[user]
        if seen:
            scores[row_index, torch.as_tensor(sorted(seen), dtype=torch.long,
                                               device=scores.device)] = -torch.inf


@torch.no_grad()
def catalog_coverage(model, dataset, k: int = 10, device: str = "cpu",
                     max_sequence_length: int = 50, user_batch_size: int = 128) -> float:
    model.eval()
    histories = torch.as_tensor(dataset.history_matrix(max_sequence_length), dtype=torch.long,
                                device=device)
    adjacency = model._evaluation_adjacency
    item_ids = torch.arange(dataset.n_items, device=device)
    recommended: set[int] = set()
    seen_by_user = _training_seen_by_user(dataset)
    for start in range(0, dataset.n_users, user_batch_size):
        users = torch.arange(start, min(start + user_batch_size, dataset.n_users), device=device)
        hidden = model.encode_users(users, histories[users], adjacency)
        scores = model.score_from_hidden(hidden[:, None, :].expand(-1, dataset.n_items, -1),
                                         item_ids[None, :].expand(len(users), -1))
        _mask_seen_training_items(scores, users, seen_by_user)
        for row in scores.topk(min(k, dataset.n_items), dim=1).indices.cpu().numpy():
            recommended.update(map(int, row))
    return len(recommended) / max(dataset.n_items, 1)


@torch.no_grad()
def intra_list_similarity(model, dataset, k: int = 10, device: str = "cpu",
                          max_sequence_length: int = 50, user_batch_size: int = 128) -> float:
    model.eval()
    histories = torch.as_tensor(dataset.history_matrix(max_sequence_length), dtype=torch.long,
                                device=device)
    adjacency = model._evaluation_adjacency
    item_ids = torch.arange(dataset.n_items, device=device)
    embeddings = F.normalize(model.item_embedding.weight, dim=-1)
    scores_all = []
    seen_by_user = _training_seen_by_user(dataset)
    for start in range(0, dataset.n_users, user_batch_size):
        users = torch.arange(start, min(start + user_batch_size, dataset.n_users), device=device)
        hidden = model.encode_users(users, histories[users], adjacency)
        scores = model.score_from_hidden(hidden[:, None, :].expand(-1, dataset.n_items, -1),
                                         item_ids[None, :].expand(len(users), -1))
        _mask_seen_training_items(scores, users, seen_by_user)
        scores_all.extend(scores.topk(min(k, dataset.n_items), dim=1).indices)
    values = []
    for selected in scores_all:
        vectors = embeddings[selected]
        similarity = vectors @ vectors.T
        upper = torch.triu(similarity, diagonal=1)
        selected_count = int(selected.numel())
        denom = selected_count * (selected_count - 1) / 2
        values.append(float(upper.sum().cpu()) / denom if denom > 0 else 0.0)
    return float(np.mean(values)) if values else 0.0
