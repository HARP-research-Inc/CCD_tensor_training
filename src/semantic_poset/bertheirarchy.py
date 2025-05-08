import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sentence_transformers import SentenceTransformer

# -------------------------
# 1. Load SBERT instead of Word2Vec
# -------------------------
# You can choose any SBERT model; here we use a compact one:
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Device setup (multi-GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    print(f"Found {torch.cuda.device_count()} GPUs, using DataParallel")

# -------------------------
# 2. Prepare training edges
# -------------------------
pos_edges = [
    ('dog', 'animal'),
    ('cat', 'animal'),
    ('car', 'vehicle'),
    ('bike', 'vehicle'),
]
words = list({w for edge in pos_edges for w in edge})
word2idx = {w: i for i, w in enumerate(words)}

# Encode with SBERT (returns a torch.Tensor of shape [N, d])
X_tensor = sbert_model.encode(words, convert_to_tensor=True).to(device)

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

d = X_tensor.size(1)    # SBERT dimension (e.g. 384 or 768)
D = 16                  # choose your low-dim projection size
model = OrderProjector(d, D).to(device)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

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
    # Positive batch
    u_pos = torch.tensor([u for u, v in pos_idx], dtype=torch.long, device=device)
    v_pos = torch.tensor([v for u, v in pos_idx], dtype=torch.long, device=device)
    f_u_pos = model(X_tensor[u_pos])
    f_v_pos = model(X_tensor[v_pos])
    lp = loss_pos(f_u_pos, f_v_pos)

    # Negative batch
    u_neg = torch.tensor([u for u, v in neg_idx], dtype=torch.long, device=device)
    v_neg = torch.tensor([v for u, v in neg_idx], dtype=torch.long, device=device)
    f_u_neg = model(X_tensor[u_neg])
    f_v_neg = model(X_tensor[v_neg])
    ln = loss_neg(f_u_neg, f_v_neg)

    loss = lp + ln
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# -------------------------
# 6. Save the model weights
# -------------------------
save_path = 'order_projector_sbert.pt'
torch.save(model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
           save_path)
print(f"Model weights saved to {save_path}")

# -------------------------
# 7. Demo: compute penalty for test pairs
# -------------------------
print("\nOrder-violation penalties (lower ≈ parent relationship):")
test_pairs = [
    ('dog', 'animal'),
    ('dog', 'cat'),
    ('car', 'vehicle'),
    ('bike', 'car'),
]
for u, v in test_pairs:
    iu, iv = word2idx[u], word2idx[v]
    f_u = model(X_tensor[iu].unsqueeze(0))
    f_v = model(X_tensor[iv].unsqueeze(0))
    pen = order_violation(f_u, f_v).item()
    print(f"  {u:4s} → {v:7s} | penalty = {pen:.4f}")
