from __future__ import annotations

import copy
import random
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from camf_recnas.config import TrainingConfig
from camf_recnas.data.dataset import InteractionDataset
from camf_recnas.models.encoding import Architecture
from camf_recnas.models.recommender import Recommender
from camf_recnas.objectives.metrics import evaluate_ranking, measure_latency


@dataclass
class EvaluationRecord:
    architecture_id: str
    ndcg10: float
    hr10: float
    parameter_count: int
    latency_seconds_p95: float
    wall_time_seconds: float
    training_time_seconds: float
    objective: list[float]
    fidelity: str
    epochs: int
    metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this PyTorch installation")
    return device


def make_training_view(dataset: InteractionDataset, train_rows: np.ndarray) -> InteractionDataset:
    """Same validation/test targets with an alternate training graph (e.g. 5% fidelity)."""
    return InteractionDataset(dataset.n_users, dataset.n_items, train_rows,
                              dataset.validation, dataset.test, dataset.user_ids,
                              dataset.item_ids, dataset.name)


def _build_positive_sets(train_rows: np.ndarray, n_users: int) -> list[set[int]]:
    positives = [set() for _ in range(n_users)]
    for row in train_rows:
        positives[int(row[0])].add(int(row[1]))
    return positives


def _sample_negatives(users: np.ndarray, positives: list[set[int]], n_items: int,
                      rng: np.random.Generator) -> np.ndarray:
    negatives = rng.integers(0, n_items, size=len(users), dtype=np.int64)
    for idx, user in enumerate(users):
        attempts = 0
        while int(negatives[idx]) in positives[int(user)]:
            negatives[idx] = int(rng.integers(0, n_items))
            attempts += 1
            if attempts > 1000:
                available = list(set(range(n_items)).difference(positives[int(user)]))
                if not available:
                    raise RuntimeError(f"User {user} has interacted with every item")
                negatives[idx] = int(rng.choice(available))
                break
    return negatives


def _make_optimizer(model: nn.Module, config: TrainingConfig):
    name = config.optimizer.lower()
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=config.learning_rate,
                               momentum=config.momentum, weight_decay=config.weight_decay)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=config.learning_rate,
                                weight_decay=config.weight_decay)
    raise ValueError(f"Unsupported optimizer: {config.optimizer}")


def _objective_vector(ndcg10: float, params: int, latency_seconds: float,
                      parameter_reference: float = 1_000_000,
                      latency_reference_seconds: float = 1e-3) -> list[float]:
    return [float(1.0 - ndcg10),
            float(np.log10(max(params, 1) / parameter_reference)),
            float(np.log10(max(latency_seconds, 1e-9) / latency_reference_seconds))]


def fit_and_evaluate(
    architecture: Architecture,
    dataset: InteractionDataset,
    config: TrainingConfig,
    epochs: int,
    device: torch.device,
    seed: int = 42,
    split: str = "validation",
    measure_inference_latency: bool = True,
    parameter_reference: float = 1_000_000,
    latency_reference_seconds: float = 1e-3,
    fidelity_label: str | None = None,
) -> tuple[Recommender, EvaluationRecord]:
    """Train with BPR and evaluate full-catalog ranking on validation or test.

    Each positive training interaction is paired with a uniformly resampled negative
    item at every epoch. The evaluation split is never sampled for negatives.
    """
    seed_everything(seed)
    start = time.perf_counter()
    model = Recommender(dataset.n_users, dataset.n_items, architecture,
                        embedding_dropout=config.embedding_dropout).to(device)
    train_tensor = torch.as_tensor(dataset.train, dtype=torch.float32, device=device)
    adjacency = model.build_normalized_adjacency(train_tensor, dataset.n_users,
                                                 dataset.n_items, device)
    model._evaluation_adjacency = adjacency
    histories = torch.as_tensor(dataset.history_matrix(config.max_sequence_length),
                                dtype=torch.long, device=device)
    rows = dataset.train[:, :2].astype(np.int64)
    if len(rows) == 0:
        raise ValueError("Training set is empty")
    tensor_dataset = TensorDataset(torch.as_tensor(rows[:, 0], dtype=torch.long),
                                   torch.as_tensor(rows[:, 1], dtype=torch.long))
    loader = DataLoader(tensor_dataset, batch_size=min(config.batch_size, len(rows)),
                        shuffle=True, drop_last=False)
    positives = _build_positive_sets(dataset.train, dataset.n_users)
    rng = np.random.default_rng(seed)
    optimizer = _make_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    model.train()
    epoch_losses: list[float] = []
    training_start = time.perf_counter()
    for _epoch in range(epochs):
        running_loss = 0.0
        seen = 0
        for batch_users, batch_items in loader:
            batch_users = batch_users.to(device)
            batch_items = batch_items.to(device)
            negative_np = _sample_negatives(batch_users.detach().cpu().numpy(), positives,
                                            dataset.n_items, rng)
            negatives = torch.as_tensor(negative_np, dtype=torch.long, device=device)
            user_histories = histories[batch_users]
            optimizer.zero_grad(set_to_none=True)
            positive_scores = model(batch_users, batch_items, user_histories, adjacency)
            negative_scores = model(batch_users, negatives, user_histories, adjacency)
            # Mean Bayesian Personalized Ranking objective plus L2 weight decay via optimizer.
            loss = -torch.nn.functional.logsigmoid(positive_scores - negative_scores).mean()
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach().cpu()) * len(batch_users)
            seen += len(batch_users)
        scheduler.step()
        epoch_losses.append(running_loss / max(seen, 1))
    training_elapsed = time.perf_counter() - training_start

    model.eval()
    metrics = evaluate_ranking(model, dataset, split=split, device=str(device),
                               max_sequence_length=config.max_sequence_length,
                               batch_users=config.eval_batch_users,
                               item_chunk_size=config.item_chunk_size, k=10)
    if measure_inference_latency:
        latency = measure_latency(model, dataset, str(device), config.latency_warmup,
                                  config.latency_repeats, config.max_sequence_length)
    else:
        latency = 0.0
    elapsed = time.perf_counter() - start
    params = model.parameter_count
    ndcg = metrics["NDCG@10"]
    objective = _objective_vector(ndcg, params, latency if latency > 0 else 1e-3,
                                  parameter_reference, latency_reference_seconds)
    record = EvaluationRecord(
        architecture_id=architecture.fingerprint,
        ndcg10=ndcg,
        hr10=metrics["HR@10"],
        parameter_count=params,
        latency_seconds_p95=latency,
        wall_time_seconds=elapsed,
        training_time_seconds=training_elapsed,
        objective=objective,
        fidelity=fidelity_label or ("full" if epochs >= config.full_epochs else "low"),
        epochs=epochs,
        metrics={**metrics, "bpr_loss_by_epoch": epoch_losses,
                 "device": str(device), "optimizer": config.optimizer,
                 "training_config": asdict(config)},
    )
    return model, record
