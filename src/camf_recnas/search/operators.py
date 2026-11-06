from __future__ import annotations

import random
from dataclasses import replace

from camf_recnas.models.encoding import (
    ACTIVATIONS, ATTENTION_HEADS, DWC_KERNELS, EMBEDDING_DIMS, ENCODERS,
    HEAD_DROPOUTS, HEAD_WIDTHS, Architecture, BlockGene, SearchSpace,
)

OPERATOR_NAMES = ("sequence", "substructure", "gene_wise")


def _swap_block_pair(a: BlockGene, b: BlockGene) -> tuple[BlockGene, BlockGene]:
    # Typed genes are exchanged only within the same block family.
    if a.kind != b.kind:
        return a, b
    return replace(a, activation=b.activation, heads=b.heads, kernel_size=b.kernel_size,
                   residual=b.residual), replace(b, activation=a.activation, heads=a.heads,
                   kernel_size=a.kernel_size, residual=a.residual)


def crossover(a: Architecture, b: Architecture, operator: str,
              rng: random.Random, space: SearchSpace) -> tuple[Architecture, Architecture]:
    if operator not in OPERATOR_NAMES:
        raise ValueError(f"Unknown crossover operator: {operator}")
    if operator == "sequence":
        left_a = list(a.blocks)
        left_b = list(b.blocks)
        cut_a = rng.randrange(len(left_a))
        cut_b = rng.randrange(len(left_b))
        child_a_blocks = left_a[:cut_a] + left_b[cut_b:]
        child_b_blocks = left_b[:cut_b] + left_a[cut_a:]
        child_a = replace(a, blocks=tuple(child_a_blocks[:space.max_blocks]))
        child_b = replace(b, blocks=tuple(child_b_blocks[:space.max_blocks]))
    elif operator == "substructure":
        left_a, left_b = list(a.blocks), list(b.blocks)
        i, j = rng.randrange(len(left_a)), rng.randrange(len(left_b))
        ga, gb = left_a[i], left_b[j]
        if ga.kind == gb.kind:
            # Exchange one typed block and its residual-connection flag.
            left_a[i], left_b[j] = _swap_block_pair(ga, gb)
        child_a, child_b = replace(a, blocks=tuple(left_a)), replace(b, blocks=tuple(left_b))
    else:
        # Gene-wise exchange: top-level genes and block fields are type constrained.
        dim_a, dim_b = a.embedding_dim, b.embedding_dim
        if rng.random() < 0.5:
            dim_a, dim_b = dim_b, dim_a
        enc_a, enc_b = a.interaction_encoder, b.interaction_encoder
        if rng.random() < 0.5:
            enc_a, enc_b = enc_b, enc_a
        width_a, width_b = a.head_width, b.head_width
        if rng.random() < 0.5:
            width_a, width_b = width_b, width_a
        drop_a, drop_b = a.head_dropout, b.head_dropout
        if rng.random() < 0.5:
            drop_a, drop_b = drop_b, drop_a
        ba, bb = list(a.blocks), list(b.blocks)
        for idx in range(min(len(ba), len(bb))):
            if ba[idx].kind == bb[idx].kind and rng.random() < 0.5:
                ba[idx], bb[idx] = _swap_block_pair(ba[idx], bb[idx])
        child_a = replace(a, embedding_dim=dim_a, interaction_encoder=enc_a,
                          head_width=width_a, head_dropout=drop_a, blocks=tuple(ba))
        child_b = replace(b, embedding_dim=dim_b, interaction_encoder=enc_b,
                          head_width=width_b, head_dropout=drop_b, blocks=tuple(bb))
    return space.repair(child_a), space.repair(child_b)


def mutate(architecture: Architecture, rng: random.Random, space: SearchSpace,
           mutation_probability: float) -> Architecture:
    """Structural mutation plus independent gene-level mutation at p_m."""
    arch = architecture
    blocks = list(arch.blocks)
    # Structural add/remove/swap; it is attempted at p_m.
    if rng.random() < mutation_probability:
        operation = rng.choice(("add", "remove", "swap"))
        if operation == "add" and len(blocks) < space.max_blocks:
            blocks.insert(rng.randrange(len(blocks) + 1), space._sample_block(rng, arch.embedding_dim))
        elif operation == "remove" and len(blocks) > 1:
            blocks.pop(rng.randrange(len(blocks)))
        elif operation == "swap" and len(blocks) > 1:
            i, j = rng.sample(range(len(blocks)), 2)
            blocks[i], blocks[j] = blocks[j], blocks[i]
    # Independent gene-level mutation, following the manuscript's p_m rule.
    dim = arch.embedding_dim
    encoder = arch.interaction_encoder
    width = arch.head_width
    dropout = arch.head_dropout
    if rng.random() < mutation_probability:
        dim = rng.choice(EMBEDDING_DIMS)
    if rng.random() < mutation_probability:
        encoder = rng.choice(ENCODERS)
    if rng.random() < mutation_probability:
        width = rng.choice(HEAD_WIDTHS)
    if rng.random() < mutation_probability:
        dropout = rng.choice(HEAD_DROPOUTS)
    for idx, gene in enumerate(blocks):
        activation = gene.activation
        heads = gene.heads
        kernel = gene.kernel_size
        residual = gene.residual
        if rng.random() < mutation_probability:
            activation = rng.choice(ACTIVATIONS)
        if gene.kind == "attention" and rng.random() < mutation_probability:
            heads = rng.choice([h for h in ATTENTION_HEADS if dim % h == 0])
        if gene.kind == "dwc" and rng.random() < mutation_probability:
            kernel = rng.choice(DWC_KERNELS)
        if gene.kind == "mlp" and rng.random() < mutation_probability:
            residual = not residual
        blocks[idx] = BlockGene(gene.kind, activation, heads, kernel, residual)
    return space.repair(replace(arch, embedding_dim=dim, interaction_encoder=encoder,
                                head_width=width, head_dropout=dropout, blocks=tuple(blocks)))
