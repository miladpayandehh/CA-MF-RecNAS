from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .encoding import Architecture, BlockGene


def make_activation(name: str) -> nn.Module:
    table = {
        "ReLU": nn.ReLU,
        "LeakyReLU": nn.LeakyReLU,
        "PReLU": nn.PReLU,
        "ELU": nn.ELU,
        "SELU": nn.SELU,
    }
    if name not in table:
        raise ValueError(f"Unknown activation: {name}")
    return table[name]()


class GraphConvBlock(nn.Module):
    def __init__(self, dim: int, activation: str):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.activation = make_activation(activation)

    def forward(self, entities: torch.Tensor, normalized_adjacency: torch.Tensor) -> torch.Tensor:
        aggregated = torch.sparse.mm(normalized_adjacency, entities)
        return self.activation(self.linear(aggregated))


class AttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float, activation: str):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.activation = make_activation(activation)

    def forward(self, sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # True in key_padding_mask means padded position.
        safe_mask = mask.clone()
        empty = safe_mask.all(dim=1)
        if empty.any():
            safe_mask[empty, -1] = False
            sequence = sequence.clone()
            sequence[empty, -1] = 0.0
        output, _ = self.attention(sequence, sequence, sequence,
                                   key_padding_mask=safe_mask, need_weights=False)
        output = self.norm(sequence + output)
        return self.activation(output) * (~mask).unsqueeze(-1)


class DepthwiseConvBlock(nn.Module):
    def __init__(self, dim: int, kernel_size: int, activation: str):
        super().__init__()
        self.depthwise = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim)
        self.pointwise = nn.Conv1d(dim, dim, kernel_size=1)
        self.batch_norm = nn.BatchNorm1d(dim)
        self.activation = make_activation(activation)

    def forward(self, sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = sequence.transpose(1, 2)
        x = self.batch_norm(self.pointwise(self.depthwise(x))).transpose(1, 2)
        return self.activation(x) * (~mask).unsqueeze(-1)


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, activation: str, residual: bool):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.activation = make_activation(activation)
        self.residual = residual

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        transformed = self.fc2(self.activation(self.fc1(hidden)))
        return hidden + transformed if self.residual else transformed


class Recommender(nn.Module):
    """Typed sequential/graph recommender implementing the manuscript's block families.

    Note: the user-item GCN is implemented on the full training bipartite graph. The
    manuscript does not specify side-feature tensors/encoders, so side_info must be 'none'.
    """

    def __init__(self, n_users: int, n_items: int, architecture: Architecture,
                 embedding_dropout: float = 0.2):
        super().__init__()
        architecture.validate(max_blocks=8, allow_side_info=False)
        self.n_users, self.n_items = n_users, n_items
        self.architecture = architecture
        dim = architecture.embedding_dim
        self.user_embedding = nn.Embedding(n_users, dim)
        self.item_embedding = nn.Embedding(n_items, dim)
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        self.concat_projection = (nn.Linear(2 * dim, dim)
                                  if architecture.interaction_encoder == "concat" else None)
        self.attention_encoder_w = (nn.Linear(2 * dim, dim // 2)
                                    if architecture.interaction_encoder == "attn" else None)
        self.attention_encoder_v = (nn.Linear(dim // 2, 1, bias=False)
                                    if architecture.interaction_encoder == "attn" else None)
        self.gcn_encoder = (GraphConvBlock(dim, "ReLU")
                            if architecture.interaction_encoder == "gcn" else None)
        self.blocks = nn.ModuleList()
        for gene in architecture.blocks:
            if gene.kind == "gcn":
                self.blocks.append(GraphConvBlock(dim, gene.activation))
            elif gene.kind == "attention":
                self.blocks.append(AttentionBlock(dim, gene.heads, architecture.head_dropout, gene.activation))
            elif gene.kind == "dwc":
                self.blocks.append(DepthwiseConvBlock(dim, gene.kernel_size, gene.activation))
            elif gene.kind == "mlp":
                self.blocks.append(ResidualMLPBlock(dim, gene.activation, gene.residual))
            else:
                raise ValueError(f"Unsupported block: {gene.kind}")
        self.head = nn.Linear(2 * dim, architecture.head_width)
        self.head_activation = nn.ELU()
        self.head_dropout = nn.Dropout(architecture.head_dropout)
        self.output = nn.Linear(architecture.head_width, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

    @staticmethod
    def build_normalized_adjacency(train: torch.Tensor, n_users: int, n_items: int,
                                   device: torch.device) -> torch.Tensor:
        """Build D^-1/2(A+I)D^-1/2 for the symmetric user-item graph."""
        users = train[:, 0].long()
        items = train[:, 1].long() + n_users
        src = torch.cat([users, items])
        dst = torch.cat([items, users])
        nodes = n_users + n_items
        self_ids = torch.arange(nodes, device=device)
        src = torch.cat([src, self_ids])
        dst = torch.cat([dst, self_ids])
        indices = torch.stack([src, dst])
        values = torch.ones(indices.shape[1], device=device, dtype=torch.float32)
        adjacency = torch.sparse_coo_tensor(indices, values, (nodes, nodes), device=device).coalesce()
        degree = torch.zeros(nodes, device=device, dtype=torch.float32)
        degree.index_add_(0, adjacency.indices()[0], adjacency.values())
        inv_sqrt = degree.clamp_min(1).pow(-0.5)
        norm_values = adjacency.values() * inv_sqrt[adjacency.indices()[0]] * inv_sqrt[adjacency.indices()[1]]
        return torch.sparse_coo_tensor(adjacency.indices(), norm_values, adjacency.shape,
                                       device=device).coalesce()

    def encode_users(self, users: torch.Tensor, histories: torch.Tensor,
                     normalized_adjacency: torch.Tensor) -> torch.Tensor:
        mask = histories < 0
        safe_history = histories.clamp_min(0)
        user_emb = self.user_embedding(users)
        seq = self.item_embedding(safe_history) * (~mask).unsqueeze(-1)
        # Histories are left-padded, so the most recent interaction is at the final slot.
        last_item = seq[:, -1]
        entities = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        if self.architecture.interaction_encoder == "product":
            hidden = user_emb * last_item
        elif self.architecture.interaction_encoder == "concat":
            hidden = self.concat_projection(torch.cat([user_emb, last_item], dim=-1))
        elif self.architecture.interaction_encoder == "attn":
            query = user_emb.unsqueeze(1).expand_as(seq)
            weights = self.attention_encoder_v(torch.tanh(self.attention_encoder_w(
                torch.cat([query, seq], dim=-1)))).squeeze(-1)
            weights = weights.masked_fill(mask, torch.finfo(weights.dtype).min)
            all_empty = mask.all(dim=1)
            if all_empty.any():
                weights[all_empty] = 0.0
            weights = torch.softmax(weights, dim=1) * (~mask)
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
            hidden = torch.sum(weights.unsqueeze(-1) * seq, dim=1)
        elif self.architecture.interaction_encoder == "gcn":
            entities = self.gcn_encoder(entities, normalized_adjacency)
            hidden = entities[users]
            seq = F.embedding(safe_history, entities[self.n_users:]) * (~mask).unsqueeze(-1)
        else:
            raise ValueError(self.architecture.interaction_encoder)

        for gene, block in zip(self.architecture.blocks, self.blocks):
            if gene.kind == "gcn":
                entities = block(entities, normalized_adjacency)
                hidden = entities[users]
                seq = F.embedding(safe_history, entities[self.n_users:]) * (~mask).unsqueeze(-1)
            elif gene.kind == "attention":
                seq = block(seq, mask)
                hidden = (seq.sum(dim=1) / (~mask).sum(dim=1, keepdim=True).clamp_min(1))
            elif gene.kind == "dwc":
                seq = block(seq, mask)
                hidden = seq.sum(dim=1) / (~mask).sum(dim=1, keepdim=True).clamp_min(1)
            elif gene.kind == "mlp":
                hidden = block(hidden)
        return self.embedding_dropout(hidden)

    def score_from_hidden(self, hidden: torch.Tensor, target_items: torch.Tensor) -> torch.Tensor:
        item_emb = self.item_embedding(target_items)
        joined = torch.cat([hidden, item_emb], dim=-1)
        x = self.head_activation(self.head(joined))
        x = self.head_dropout(x)
        return self.output(x).squeeze(-1)

    def forward(self, users: torch.Tensor, target_items: torch.Tensor,
                histories: torch.Tensor, normalized_adjacency: torch.Tensor) -> torch.Tensor:
        hidden = self.encode_users(users, histories, normalized_adjacency)
        return self.score_from_hidden(hidden, target_items)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
