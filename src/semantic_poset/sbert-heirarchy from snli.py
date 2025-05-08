import torch
import torch.nn as nn
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1) Define the projector architecture
class OrderProjector(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.act = nn.ReLU()
    def forward(self, x):
        return self.act(self.lin(x))

# 2) Load pretrained SBERT and trained projector
sbert = SentenceTransformer('all-MiniLM-L6-v2')
proj_dim = 16
model = OrderProjector(sbert.get_sentence_embedding_dimension(), proj_dim).to(device)
state = torch.load('order_projector_sbert.pt', map_location=device)
model.load_state_dict(state)
model.eval()

# 3) Load SNLI validation split
snli = load_dataset('snli', split='validation')

# 4) Filter to entailment vs non-entailment
snli = snli.filter(lambda ex: ex['label'] != -1)  # remove '-1' examples
labels = np.array([1 if l==0 else 0 for l in snli['label']])  # 0: entailment, else non

# 5) Encode premises and hypotheses in batches
batch_size = 64
sentences = snli['premise'] + snli['hypothesis']
# encode separately to avoid mixing
prem_embeds = sbert.encode(snli['premise'], convert_to_tensor=True).to(device)
hyp_embeds = sbert.encode(snli['hypothesis'], convert_to_tensor=True).to(device)

# 6) Project all embeddings
with torch.no_grad():
    F_p = model(prem_embeds)
    F_h = model(hyp_embeds)

# 7) Compute penalties
viol = torch.clamp(F_h - F_p, min=0)  # shape (N, D)
penalties = viol.pow(2).sum(dim=1).cpu().numpy()

# 8) Classify by thresholds and plot results
import matplotlib.pyplot as plt

thresholds = np.logspace(-6, 0, 20)  # 20 thresholds from 10^-6 to 1
accuracies = []

for threshold in thresholds:
    preds = (penalties < threshold).astype(int)  # 1=entailment, 0=non
    acc = (preds == labels).mean()
    accuracies.append(acc)
    print(f"Threshold {threshold:.6f}: accuracy {acc:.4f}")

# Plot accuracy vs threshold
plt.figure(figsize=(10, 6))
plt.semilogx(thresholds, accuracies, 'b-', marker='o')
plt.grid(True)
plt.xlabel('Threshold')
plt.ylabel('Accuracy')
plt.title('SNLI Entailment Accuracy vs Classification Threshold')
plt.savefig('threshold_accuracy.png')
plt.close()

# Use best threshold for final prediction
best_threshold = thresholds[np.argmax(accuracies)]
preds = (penalties < best_threshold).astype(int)  # 1=entailment, 0=non
print(f"\nBest threshold: {best_threshold:.6f}")

# 9) Accuracy
accuracy = (preds == labels).mean()
print(f"SNLI entailment accuracy: {accuracy:.4f}")
