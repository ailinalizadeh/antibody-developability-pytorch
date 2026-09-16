from __future__ import annotations

import torch
from torch import nn

from .data import VOCAB


class SequenceEncoder(nn.Module):
    """Encode one amino-acid chain with embedding + Conv1D + max pooling."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 32,
        channels: int = 64,
        kernel_size: int = 5,
        padding_idx: int = 0,
        dropout: float = 0.20,
    ):
        super().__init__()
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=padding_idx,
        )
        self.conv = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [batch, length]
        mask = tokens.ne(self.padding_idx).unsqueeze(1)  # [batch, 1, length]

        x = self.embedding(tokens)                       # [batch, length, embed]
        x = x.transpose(1, 2)                           # [batch, embed, length]
        x = self.conv(x)                                # [batch, channels, length]
        x = self.activation(x)
        x = self.dropout(x)

        # Ignore padded positions during global max pooling.
        x = x.masked_fill(~mask, torch.finfo(x.dtype).min)
        pooled = x.max(dim=2).values                    # [batch, channels]

        # Protect against an all-padding input.
        all_padding = ~mask.any(dim=2).squeeze(1)
        if all_padding.any():
            pooled[all_padding] = 0.0

        return pooled


class AntibodyDevelopabilityCNN(nn.Module):
    """CNN classifier for paired heavy/light antibody sequences."""

    def __init__(
        self,
        embedding_dim: int = 32,
        channels: int = 64,
        hidden_dim: int = 64,
        dropout: float = 0.30,
    ):
        super().__init__()
        vocab_size = len(VOCAB)

        # Separate encoders allow heavy and light chains to learn different filters.
        self.heavy_encoder = SequenceEncoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            channels=channels,
            dropout=dropout,
        )
        self.light_encoder = SequenceEncoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            channels=channels,
            dropout=dropout,
        )

        self.classifier = nn.Sequential(
            nn.Linear(channels * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        heavy: torch.Tensor,
        light: torch.Tensor,
    ) -> torch.Tensor:
        heavy_repr = self.heavy_encoder(heavy)
        light_repr = self.light_encoder(light)
        combined = torch.cat([heavy_repr, light_repr], dim=1)
        logits = self.classifier(combined).squeeze(1)
        return logits
