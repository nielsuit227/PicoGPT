import torch
from torch import nn


class FeedForward(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        """
        Factor 4 of hidden layers taken from Attention is All You Need
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.net(data)
