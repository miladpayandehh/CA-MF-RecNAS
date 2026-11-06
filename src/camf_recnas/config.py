from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    min_user_interactions: int = 5
    min_item_interactions: int = 5
    positive_rating_threshold: float = 4.0
    low_fidelity_fraction: float = 0.05


@dataclass
class TrainingConfig:
    optimizer: str = "sgd"
    learning_rate: float = 0.025
    momentum: float = 0.9
    weight_decay: float = 0.0003
    batch_size: int = 256
    embedding_dropout: float = 0.2
    low_fidelity_epochs: int = 5
    full_epochs: int = 300
    max_sequence_length: int = 50
    eval_batch_users: int = 128
    item_chunk_size: int = 2048
    latency_warmup: int = 100
    latency_repeats: int = 1000


@dataclass
class SearchConfig:
    population_size: int = 40
    generations: int = 80
    max_blocks: int = 8
    calibration_size: int = 100
    promotion_budget: int = 10
    learning_automata_rate: float = 0.1
    reward_ema_beta: float = 0.3
    crossover_probability_initial: float = 0.7
    mutation_probability_initial: float = 0.1
    min_crossover_probability: float = 0.4
    max_crossover_probability: float = 0.95
    min_mutation_probability: float = 0.005
    max_mutation_probability: float = 0.30
    ridge_alphas: list[float] = field(default_factory=lambda: [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100])
    promotion_z: float = 1.6448536269514722
    epsilon: float = 1e-8
    reference_point: list[float] = field(default_factory=lambda: [1.05, 1.05, 1.05])
    parameter_reference: float = 1_000_000
    latency_reference_seconds: float = 1e-3
    normalization_lower: list[float] = field(default_factory=lambda: [0.20, 0.0, 0.0])
    normalization_upper: list[float] = field(default_factory=lambda: [0.60, 1.0, 1.0])
    runs: int = 10


@dataclass
class ExperimentConfig:
    seed: int = 42
    device: str = "auto"
    max_sequence_length: int = 50
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    search: SearchConfig = field(default_factory=SearchConfig)


def _build_dataclass(cls: type, values: dict[str, Any]):
    allowed = cls.__dataclass_fields__.keys()
    return cls(**{k: v for k, v in values.items() if k in allowed})


def load_config(path: str | Path) -> ExperimentConfig:
    """Load YAML while ignoring unknown keys only at the top-level safely."""
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return ExperimentConfig(
        seed=int(raw.get("seed", 42)),
        device=str(raw.get("device", "auto")),
        max_sequence_length=int(raw.get("max_sequence_length", 50)),
        data=_build_dataclass(DataConfig, raw.get("data", {})),
        training=_build_dataclass(TrainingConfig, raw.get("training", {})),
        search=_build_dataclass(SearchConfig, raw.get("search", {})),
    )
