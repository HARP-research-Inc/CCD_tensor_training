import torch
import torch.nn as nn
import numpy as np
from sentence_transformers import SentenceTransformer

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Define your vocab and load SBERT + projection model
words = ['dog','cat','car','bike','animal','vehicle']
word2idx = {w:i for i,w in enumerate(words)}

# Load SBERT embeddings
sbert = SentenceTransformer('all-MiniLM-L6-v2')
X = sbert.encode(words, convert_to_tensor=True).to(device)  # shape (N, d)

# Define the same projector class
class OrderProjector(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.act = nn.ReLU()
    def forward(self, x):
        return self.act(self.lin(x))

# Instantiate and load your trained weights
d = X.size(1)    # e.g. 384
D = 16           # projection dimension you used
model = OrderProjector(d, D).to(device)
model.load_state_dict(torch.load('order_projector_sbert.pt', map_location=device))
model.eval()

# 2. Project all vectors into the order space
F = model(X)    # tensor of shape (N, D)

# 3. Build the domination matrix: dom[i,j] = True iff F[i] <= F[j] coordinatewise
dom = (F.unsqueeze(1) <= F.unsqueeze(0)).all(dim=2)  # shape (N, N)

# 4. For each node i, find its immediate parents:
#    candidates = { j | j!=i and dom[i,j] }
#    immediate = those j with no k s.t. dom[i,k] and dom[k,j]
N = F.size(0)
parents = [[] for _ in range(N)]
for i in range(N):
    cands = [j for j in range(N) if j!=i and dom[i,j]]
    imm = []
    for j in cands:
        if not any(k!=i and k!=j and dom[i,k] and dom[k,j] for k in cands):
            imm.append(j)
    parents[i] = imm

# 5. If you want a single parent per node, pick the nearest by L1 distance
parent = [-1]*N
for i in range(N):
    imm = parents[i]
    if not imm: continue
    dists = [(F[j]-F[i]).abs().sum().item() for j in imm]
    parent[i] = imm[int(np.argmin(dists))]

# 6. Build children lists
children = {i: [] for i in range(N)}
for i,p in enumerate(parent):
    if p >= 0:
        children[p].append(i)

# 7. Print the resulting forest
def print_tree(i, level=0):
    print('  '*level + words[i])
    for c in children[i]:
        print_tree(c, level+1)

roots = [i for i,p in enumerate(parent) if p < 0]
for r in roots:
    print_tree(r)
