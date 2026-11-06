from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from typing import Any

ACTIVATIONS = ("ReLU", "LeakyReLU", "PReLU", "ELU", "SELU")
EMBEDDING_DIMS = (16, 32, 64, 128, 256)
ATTENTION_HEADS = (1, 2, 4, 8)
DWC_KERNELS = (1, 3, 5)
HEAD_WIDTHS = (64, 128, 256, 512)
HEAD_DROPOUTS = (0.0, 0.1, 0.2, 0.3, 0.5)
ENCODERS = ("product", "concat", "attn", "gcn")
BLOCK_TYPES = ("gcn", "attention", "dwc", "mlp")


@dataclass(frozen=True)
class BlockGene:
    kind: str
    activation: str = "ReLU"
    heads: int = 1
    kernel_size: int = 3
    residual: bool = True


@dataclass(frozen=True)
class Architecture:
    embedding_dim: int
    interaction_encoder: str
    blocks: tuple[BlockGene, ...]
    head_width: int
    head_dropout: float
    side_info: str = "none"
    repair_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict, compare=False, hash=False)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["blocks"] = [asdict(block) for block in self.blocks]
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Architecture":
        return cls(
            embedding_dim=int(value["embedding_dim"]),
            interaction_encoder=str(value["interaction_encoder"]),
            blocks=tuple(BlockGene(**block) for block in value["blocks"]),
            head_width=int(value["head_width"]),
            head_dropout=float(value["head_dropout"]),
            side_info=str(value.get("side_info", "none")),
            repair_count=int(value.get("repair_count", 0)),
            metadata=dict(value.get("metadata", {})),
        )

    @property
    def canonical_json(self) -> str:
        """Canonical phenotype serialization; excludes audit-only metadata and unused genes."""
        blocks = []
        for block in self.blocks:
            gene = {"kind": block.kind, "activation": block.activation}
            if block.kind == "attention":
                gene["heads"] = block.heads
            if block.kind == "dwc":
                gene["kernel_size"] = block.kernel_size
            if block.kind == "mlp":
                gene["residual"] = block.residual
            blocks.append(gene)
        value = {
            "embedding_dim": self.embedding_dim,
            "interaction_encoder": self.interaction_encoder,
            "blocks": blocks,
            "head_width": self.head_width,
            "head_dropout": self.head_dropout,
            "side_info": self.side_info,
        }
        return json.dumps(value, sort_keys=True, separators=(",", ":"))

    @property
    def fingerprint(self) -> str:
        return hashlib.sha1(self.canonical_json.encode("utf-8")).hexdigest()

    def validate(self, max_blocks: int = 8, allow_side_info: bool = False) -> None:
        if self.embedding_dim not in EMBEDDING_DIMS:
            raise ValueError(f"Invalid embedding dimension: {self.embedding_dim}")
        if self.interaction_encoder not in ENCODERS:
            raise ValueError(f"Invalid interaction encoder: {self.interaction_encoder}")
        if not 1 <= len(self.blocks) <= max_blocks:
            raise ValueError(f"Number of blocks must be in [1, {max_blocks}]")
        if self.head_width not in HEAD_WIDTHS:
            raise ValueError(f"Invalid head width: {self.head_width}")
        if self.head_dropout not in HEAD_DROPOUTS:
            raise ValueError(f"Invalid head dropout: {self.head_dropout}")
        if self.side_info not in ("none", "demographics", "categories"):
            raise ValueError(f"Invalid side-information type: {self.side_info}")
        if self.side_info != "none" and not allow_side_info:
            raise ValueError("Side-information genes require a dataset-specific feature schema.")
        for index, block in enumerate(self.blocks):
            if block.kind not in BLOCK_TYPES:
                raise ValueError(f"Invalid block type at {index}: {block.kind}")
            if block.activation not in ACTIVATIONS:
                raise ValueError(f"Invalid activation at {index}: {block.activation}")
            if block.kind == "attention":
                if block.heads not in ATTENTION_HEADS or self.embedding_dim % block.heads:
                    raise ValueError(f"Invalid attention head count at block {index}")
            if block.kind == "dwc" and block.kernel_size not in DWC_KERNELS:
                raise ValueError(f"Invalid DWC kernel at block {index}")


class SearchSpace:
    """Typed architecture grammar and deterministic validity repair."""

    def __init__(self, max_blocks: int = 8, allow_side_info: bool = False):
        self.max_blocks = max_blocks
        self.allow_side_info = allow_side_info

    def sample(self, rng: random.Random) -> Architecture:
        dim = rng.choice(EMBEDDING_DIMS)
        depth = rng.randint(1, self.max_blocks)
        blocks = tuple(self._sample_block(rng, dim) for _ in range(depth))
        arch = Architecture(
            embedding_dim=dim,
            interaction_encoder=rng.choice(ENCODERS),
            blocks=blocks,
            head_width=rng.choice(HEAD_WIDTHS),
            head_dropout=rng.choice(HEAD_DROPOUTS),
            side_info="none",
        )
        return self.repair(arch)

    @staticmethod
    def _sample_block(rng: random.Random, dim: int) -> BlockGene:
        kind = rng.choice(BLOCK_TYPES)
        heads = rng.choice([h for h in ATTENTION_HEADS if dim % h == 0]) if kind == "attention" else 1
        kernel = rng.choice(DWC_KERNELS) if kind == "dwc" else 3
        residual = bool(rng.getrandbits(1)) if kind == "mlp" else True
        return BlockGene(kind=kind, activation=rng.choice(ACTIVATIONS), heads=heads,
                         kernel_size=kernel, residual=residual)

    def repair(self, architecture: Architecture) -> Architecture:
        """Repair only typed/domain violations; all implemented blocks preserve d."""
        repairs = 0
        dim = architecture.embedding_dim if architecture.embedding_dim in EMBEDDING_DIMS else 32
        encoder = architecture.interaction_encoder if architecture.interaction_encoder in ENCODERS else "product"
        blocks: list[BlockGene] = []
        for block in architecture.blocks[: self.max_blocks]:
            kind = block.kind if block.kind in BLOCK_TYPES else "mlp"
            activation = block.activation if block.activation in ACTIVATIONS else "ReLU"
            heads = block.heads if kind == "attention" else 1
            if kind == "attention" and (heads not in ATTENTION_HEADS or dim % heads):
                heads = max(h for h in ATTENTION_HEADS if dim % h == 0)
                repairs += 1
            kernel = block.kernel_size if kind == "dwc" else 3
            if kind == "dwc" and kernel not in DWC_KERNELS:
                kernel = 3
                repairs += 1
            residual = bool(block.residual) if kind == "mlp" else True
            blocks.append(BlockGene(kind, activation, heads, kernel, residual))
        if not blocks:
            blocks = [BlockGene("mlp", "ReLU")]
            repairs += 1
        width = architecture.head_width if architecture.head_width in HEAD_WIDTHS else 128
        dropout = architecture.head_dropout if architecture.head_dropout in HEAD_DROPOUTS else 0.2
        side_info = architecture.side_info if architecture.side_info in ("none", "demographics", "categories") else "none"
        if side_info != "none" and not self.allow_side_info:
            side_info = "none"
            repairs += 1
        repaired = Architecture(dim, encoder, tuple(blocks), width, dropout, side_info,
                                architecture.repair_count + repairs, architecture.metadata)
        repaired.validate(self.max_blocks, self.allow_side_info)
        return repaired
