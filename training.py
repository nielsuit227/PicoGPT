import numpy as np
import torch
from models.encoder import Enc
from models.gpt import GPT

# Set seed
torch.manual_seed(1337)

# Parameters
n_heads = 4
n_layers = 4
n_embeddings = 64
block_size = 64
batch_size = 64
learning_rate = 3e-4
training_iterations = 10_000
eval_iterations = 300
dropout = 0.2

# Load data
file_ = "data.txt"
try:
    data = open(file_).read()
except FileNotFoundError:
    data = open("pytorch/shakespear_transformer/" + file_).read()

# Encoding
n = int(len(data) * 0.9)
enc = Enc(data)
data_tensor = torch.tensor(enc.encode(data), dtype=torch.long)
train_data = data_tensor[:n]
val_data = data_tensor[n:]


def get_batch(data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y


@torch.no_grad()  # no backprop needed, so pytorch can be memory efficient and not store intermediate vars
def estimate_loss(step: int) -> None:
    model.eval()  # Change mode
    train_loss = np.mean(
        [model(*get_batch(train_data))[1].item() for i in range(eval_iterations)]
    )
    test_loss = np.mean(
        [model(*get_batch(train_data))[1].item() for i in range(eval_iterations)]
    )
    model.train()  # Change mode
    print(
        f"Step: {str(step).ljust(5)}         Train: {train_loss:6f}, Test: {test_loss:.6f}"
    )


# Instantiate the model
model = GPT(
    enc.n_vocab,
    block_size=block_size,
    n_embeddings=n_embeddings,
    n_heads=n_heads,
    n_layers=n_layers,
    dropout=dropout,
).to("cpu")

# Train the model
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
for step in range(training_iterations):
    # Eval
    if step % eval_iterations == 0:
        estimate_loss(step)

    # Forward pass
    xb, yb = get_batch(train_data)
    logits, loss = model(xb, yb)

    # Backprop
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(
    enc.decode(
        model.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[
            0
        ].tolist()
    )
)
