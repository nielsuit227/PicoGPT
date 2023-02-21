import torch
import torch.nn.functional as F
from torch import nn


__all__ = ["Head", "MultiHead"]

class Head(nn.Module):
    tril: torch.Tensor

    def __init__(
        self, n_embeddings: int, block_size: int, head_size: int, dropout: float
    ) -> None:
        super().__init__()
        self.n_embeddings = n_embeddings
        self.block_size = block_size
        self.head_size = head_size
        self.keys = nn.Linear(n_embeddings, head_size, bias=False)
        self.queries = nn.Linear(n_embeddings, head_size, bias=False)
        self.values = nn.Linear(n_embeddings, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        # This is not a parameter
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        For cross-attention, this function needs to be extended, and the keys/queries are
        calculated with the cross attention component.

        For encoders, the masking with lower triangular matrix should be omitted.
        """
        B, T, C = data.shape  # batch x block x embeddings
        k = self.keys(data)  # B T head_size
        q = self.queries(data)  # B T head_size

        weights = q @ k.transpose(1, 2) * self.head_size**-0.5  # B T T
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # B T T
        weights = F.softmax(weights, dim=-1)  # B T T
        weights = self.dropout(weights)

        # B T T @ B T head_size => B T C
        return weights @ self.values(data)


class MultiHead(nn.Module):
    def __init__(
        self,
        n_embeddings: int,
        block_size: int,
        n_heads: int,
        head_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(n_embeddings, block_size, head_size, dropout) for _ in range(n_heads)]
        )
        self.projection = nn.Linear(n_embeddings, n_embeddings)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        data = torch.cat([h(data) for h in self.heads], dim=-1)
        data = self.projection(data)
        return self.dropout(data)
