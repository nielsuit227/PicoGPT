import torch
from torch import nn


class LayerNorm(nn.Module):
    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        mean = data.mean(dim=1, keepdim=True)
        std = data.std(dim=1, keepdim=True)
        return (data - mean) / (std + self.eps)
