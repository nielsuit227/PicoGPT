import torch
from models.attention import MultiHead
from models.feedforward import FeedForward
from torch import nn


class Block(nn.Module):
    def __init__(
        self, n_embeddings: int, block_size: int, n_heads: int, dropout: float
    ) -> None:
        super().__init__()
        self.heads = MultiHead(
            n_embeddings=n_embeddings,
            block_size=block_size,
            n_heads=n_heads,
            head_size=n_embeddings // n_heads,
            dropout=dropout,
        )
        self.ffwd = FeedForward(n_embeddings, dropout)
        self.layer_norm_head = nn.LayerNorm(n_embeddings)
        self.layer_norm_ffwd = nn.LayerNorm(n_embeddings)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        data = data + self.heads(self.layer_norm_head(data))
        return data + self.ffwd(self.layer_norm_ffwd(data))
