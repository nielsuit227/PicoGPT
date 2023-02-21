import torch
import torch.nn.functional as F
from models.block import Block
from torch import nn


class GPT(nn.Module):
    """Inspiration from the ingeneous Andrej Karpathy [1]

    This is brilliantly build up, starting with simple n-gram.
    Adding self attention
    Adding multi heads (very beneficial!)
    Adding feedforward
    Introducing multiple blocks of multihead + ffwd
    Introducing residual connections (very beneficial!)
    Introducing a linear projection after heads & ffwd (again very beneficial)
    Adding layer norm
    Adding dropout
    Adding more blocks

    This is only the pre-training step, where the output is generative, but 
    preconditioned. To make it answer a prompt
    """

    def __init__(
        self,
        vocab_size: int,
        block_size: int = 8,
        n_embeddings: int = 32,
        n_heads: int = 4,
        n_layers: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        print(f"Initializing enmbeddings with vocabulary size: {vocab_size}")
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embeddings = n_embeddings
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        self.token_embedding = nn.Embedding(vocab_size, n_embeddings)
        self.position_embedding = nn.Embedding(block_size, n_embeddings)
        self.blocks = nn.Sequential(
            *[
                Block(
                    n_embeddings=n_embeddings,
                    block_size=block_size,
                    n_heads=4,
                    dropout=dropout,
                ),
            ]
            * n_layers
        )
        self.layer_norm = nn.LayerNorm(n_embeddings)
        self.linear_head = nn.Linear(n_embeddings, vocab_size)
        print(
            f"GPT initialized, total parameters: {sum(p.numel() for p in self.parameters()) / 1e6}M"
        )

    def forward(
        self, data: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = data.shape  # batch x block

        # First we get token embeddings (optimized)
        token_emb = self.token_embedding(data)

        # We get position embeddings (optimized)
        pos_emb = self.position_embedding(
            torch.arange(T, device="cpu")
        )  # block x embed

        # We add the two
        x = token_emb + pos_emb  # Ba T C

        # Parse through Multi-Attention heads & layer norm
        x = self.blocks(x)
        x = self.layer_norm(x)

        # Make sure output is of dim B, T, vocab_size
        logits = self.linear_head(x)  # B T vocab

        loss = None
        if targets is not None:
            # We need to reshape to satisfy F.cross_entropy (n x C)
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, data: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        for _ in range(max_new_tokens):
            crop = data[:, -self.block_size :]
            logits, loss = self(crop)  # Again B T C
            logits = logits[:, -1, :]  # Take last T (block), B C
            probabilities = F.softmax(logits, dim=-1)  # B C
            idx_next = torch.multinomial(probabilities, num_samples=1)  # B
            data = torch.cat((data, idx_next), dim=1)  # B i
        return data  # B Tokens
