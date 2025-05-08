import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from gensim.models import KeyedVectors
import os

# -------------------------
# 1. Load pre-trained Word2Vec
# -------------------------
# Download GoogleNews vectors from Google Drive:
# https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit
# After downloading, place the file in the same directory as this script
w2v_path = 'GoogleNews-vectors-negative300.bin'  # Note: remove .gz extension if you unzipped the file

# Add device selection for multi-GPU support
if torch.cuda.is_available():
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        device = torch.device('cuda')
    else:
        print("Using single GPU")
        device = torch.device('cuda')
else:
    print("Using CPU")
    device = torch.device('cpu')
print(f"Primary device: {device}")

# Check if the file exists
if not os.path.exists(w2v_path):
    print(f"Error: {w2v_path} not found!")
    print("Please download the Google News vectors from:")
    print("https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit")
    print("After downloading, place the file in the same directory as this script.")
    exit(1)

w2v = KeyedVectors.load_word2vec_format(w2v_path, binary=True)

# -------------------------
# 2. Prepare training edges
# -------------------------
# Positive examples (u depends on v)
pos_edges = [
    ('dog', 'animal'),
    ('cat', 'animal'),
    ('car', 'vehicle'),
    ('bike', 'vehicle'),
]

# Build a small vocab from edges
words = list({w for edge in pos_edges for w in edge})
word2idx = {w: i for i, w in enumerate(words)}

# Build embedding matrix X (N x d)
X = np.stack([w2v[w] for w in words])  # shape (N, 300)
X_tensor = torch.tensor(X, dtype=torch.float).to(device)

# Negative examples: simple reversal
neg_edges = [(v, u) for (u, v) in pos_edges]

# Convert edges to index pairs
pos_idx = [(word2idx[u], word2idx[v]) for u, v in pos_edges]
neg_idx = [(word2idx[u], word2idx[v]) for u, v in neg_edges]

# -------------------------
# 3. Define the projection model
# -------------------------
class OrderProjector(nn.Module):
    def __init__(self, input_dim, proj_dim):
        super().__init__()
        self.lin = nn.Linear(input_dim, proj_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.lin(x))

# Instantiate model
d, D = X.shape[1], 16  # project from 300 to 16 dims
model = OrderProjector(d, D).to(device)

# -------------------------
# 4. Define order-violation loss
# -------------------------
def order_violation(f_u, f_v):
    viol = torch.clamp(f_v - f_u, min=0)
    return (viol**2).sum(dim=1)

def loss_pos(f_u, f_v):
    return order_violation(f_u, f_v).mean()

def loss_neg(f_u, f_v, margin=1.0):
    score = order_violation(f_u, f_v)
    return torch.clamp(margin - score, min=0).mean()

# -------------------------
# 5. Training loop
# -------------------------
optimizer = optim.Adam(model.parameters(), lr=1e-2)
epochs = 100

for epoch in range(epochs):
    # positive batch
    u_pos = torch.tensor([u for u, v in pos_idx], dtype=torch.long, device=device)
    v_pos = torch.tensor([v for u, v in pos_idx], dtype=torch.long, device=device)
    f_u_pos = model(X_tensor[u_pos])
    f_v_pos = model(X_tensor[v_pos])
    lp = loss_pos(f_u_pos, f_v_pos)

    # negative batch
    u_neg = torch.tensor([u for u, v in neg_idx], dtype=torch.long, device=device)
    v_neg = torch.tensor([v for u, v in neg_idx], dtype=torch.long, device=device)
    f_u_neg = model(X_tensor[u_neg])
    f_v_neg = model(X_tensor[v_neg])
    ln = loss_neg(f_u_neg, f_v_neg)

    loss = lp + ln
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Save the model weights
save_path = 'order_projector.pt'
torch.save(model.state_dict(), save_path)
print(f"Model weights saved to {save_path}")

# -------------------------
# 6. Demo: compute penalty for test pairs
# -------------------------
test_pairs = [
    ('dog', 'animal'),
    ('dog', 'cat'),
    ('car', 'vehicle'),
    ('bike', 'car'),
]

print("\nOrder-violation penalties (lower ≈ parent relationship):")
for u, v in test_pairs:
    iu, iv = word2idx[u], word2idx[v]
    # Ensure test tensors are on the correct device
    f_u = model(X_tensor[iu].unsqueeze(0).to(device))
    f_v = model(X_tensor[iv].unsqueeze(0).to(device))
    pen = order_violation(f_u, f_v).item()
    print(f"  {u:4s} → {v:7s} | penalty = {pen:.4f}")
