import math
import torch
import torch.nn as nn
import torch.optim as optim
import geoopt
from torch.utils.data import DataLoader
from datasets import load_dataset
from torch_struct import MaxSpanningArborescence

# -----------------------------------------------------------------------------
# 1) Model components
# -----------------------------------------------------------------------------
class TinyPosRouter(nn.Module):
    def __init__(self, emb_dim=64, n_tags=17):
        super().__init__()
        # shared conv encoder
        self.conv = nn.Conv1d(emb_dim, emb_dim, kernel_size=3, padding=1, groups=emb_dim)
        self.lin = nn.Linear(emb_dim, n_tags)
    def forward(self, h):  # h: [B, T, emb_dim]
        # conv over time
        x = h.transpose(1,2)         # [B, emb_dim, T]
        x = self.conv(x).transpose(1,2)  # [B, T, emb_dim]
        scores = self.lin(x)         # [B, T, n_tags]
        return torch.log_softmax(scores, dim=-1)

class HyperbolicExpert(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        # Mobius linear + bias in tangent
        self.mobius = geoopt.layers.MobiusLinear(dim, dim, bias=True, manifold=geoopt.PoincareBall())
        # gated nonlinearity
        self.glu = nn.GLU(dim)
        # layer norm in tangent
        self.norm = geoopt.layers.ManifoldLayerNorm(dim, manifold=geoopt.PoincareBall())
    def forward(self, x):
        # x is on the Poincare ball
        y = self.mobius(x)
        # transport to tangent, apply GLU, retract
        y_tan = geoopt.manifolds.poincare.math.logmap0(y, c=1.0)
        y_glu = self.glu(y_tan)
        y_norm = self.norm(y_glu)
        return geoopt.manifolds.poincare.math.expmap0(y_norm, c=1.0)

class MoEParser(nn.Module):
    def __init__(self, vocab_size, char_emb_dim=128, token_emb_dim=64, n_tags=17,
                 expert_dim=32, n_rels=40):
        super().__init__()
        # embedding
        self.token_embed = nn.Embedding(vocab_size, token_emb_dim)
        # router
        self.router = TinyPosRouter(emb_dim=token_emb_dim, n_tags=n_tags)
        # experts
        self.experts = nn.ModuleList([HyperbolicExpert(expert_dim) for _ in range(n_tags)])
        # relation subspaces (product manifold dims)
        # here we share expert_dim across all rels for simplicity
        self.rel_T = nn.Parameter(torch.randn(n_rels, expert_dim, expert_dim))
        self.rel_S = nn.Parameter(torch.randn(n_rels, expert_dim, expert_dim))
        # curvature (unused in this simplified script)
        # decoder
        self.mst = MaxSpanningArborescence()

    def forward(self, tokens):
        # tokens: [B, T]
        B, T = tokens.size()
        # 1) embed
        h0 = self.token_embed(tokens)           # [B, T, D]
        # 2) router log-probs
        logp = self.router(h0)                  # [B, T, n_tags]
        p = logp.exp()                          # [B, T, n_tags]
        # 3) expert outputs
        # project first to Poincare ball tangent0
        x0 = geoopt.manifolds.poincare.math.logmap0(h0, c=1.0)  # [B, T, D]
        x = geoopt.manifolds.poincare.math.expmap0(x0, c=1.0)
        # collect expert outputs
        y = []
        for k, expert in enumerate(self.experts):
            yk = expert(x)                      # [B, T, d]
            y.append(yk)
        y_stack = torch.stack(y, dim=2)        # [B, T, n_tags, d]
        # 4) fuse by weighted average
        p_ = p.unsqueeze(-1)                   # [B, T, n_tags, 1]
        y_avg = (y_stack * p_).sum(2)          # [B, T, d]
        # 5) score arcs per relation
        # build score tensor [B, n_rels, T, T]
        scores = []
        for r in range(self.rel_T.size(0)):
            Th = torch.einsum("btd,df->btf", y_avg, self.rel_T[r])
            Sd = torch.einsum("btd,df->btf", y_avg, self.rel_S[r])
            # dot-product head × dep
            sr = torch.einsum("bhd,bdd->bhd", Th, Sd).mean(-1)  # simplify
            scores.append(sr)
        score = torch.stack(scores, dim=1)     # [B, n_rels, T, T]
        # marginalise over relations
        score_rel = score.max(1)[0]            # [B, T, T]
        # 6) decode MST
        # torch_struct expects [B, T, T] with leaf prohibition on diagonal
        mask = torch.eye(T, device=tokens.device).bool()[None]
        score_rel = score_rel.masked_fill(mask, -1e9)
        tree = self.mst(score_rel)             # log-potentials
        return tree, score_rel, logp

# -----------------------------------------------------------------------------
# 2) Data Loading (UD)
# -----------------------------------------------------------------------------
def collate_batch(batch, tokenizer, max_len=64):
    # tokenize sentences; pad to max_len
    tokens = [tokenizer.encode(x["forms"]) for x in batch]
    seqs = [torch.tensor(t[:max_len]) for t in tokens]
    seqs = nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0)
    return seqs

# -----------------------------------------------------------------------------
# 3) Training loop
# -----------------------------------------------------------------------------
def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        tokens = batch.to(device)
        tree, score_rel, logp = model(tokens)
        # gold-heads tensor from dataset (not shown: you must extract UD head indices)
        # heads: [B, T] (with head index for each token)
        # relation labels: [B, T]
        # here we fake dummy targets:
        gold_heads = torch.zeros_like(tokens)
        gold_rels  = torch.zeros_like(tokens)

        # arc loss: margin-softmax (simplified)
        # log_Z = torch.logsumexp(score_rel.view(tokens.size(0), -1), dim=1)
        # arc_ll = torch.gather(score_rel, dim=-1, index=gold_heads.unsqueeze(-1)).squeeze(-1)
        # loss_arc = (log_Z - arc_ll).mean()
        loss_arc = torch.nn.functional.cross_entropy(
            score_rel.view(-1, score_rel.size(-1)), gold_heads.view(-1), ignore_index=0
        )

        # router loss
        loss_pos = torch.nn.functional.nll_loss(
            logp.view(-1, logp.size(-1)), gold_rels.view(-1), ignore_index=0
        )

        loss = loss_arc + 0.1 * loss_pos
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load UD dataset (e.g. English-GUM)
    ds = load_dataset("universal_dependencies", "en_gum", split="train")
    vocab = {tok: i+1 for i, tok in enumerate(ds.features["forms"].feature.tokenizer.get_vocab())}
    model = MoEParser(vocab_size=len(vocab)+1).to(device)
    optimizer = geoopt.optim.RiemannianAdam(model.parameters(), lr=1e-3)

    # dummy tokenizer stub (you should plug in a real BPE/char tokenizer)
    class StubTok:
        def encode(self, forms): return [vocab.get(x, 0) for x in forms]
    tokenizer = StubTok()

    dataloader = DataLoader(ds, batch_size=32,
                            collate_fn=lambda b: collate_batch(b, tokenizer),
                            shuffle=True)

    for epoch in range(1, 11):
        loss = train_epoch(model, dataloader, optimizer, device)
        print(f"Epoch {epoch} – loss: {loss:.4f}")

if __name__ == "__main__":
    main()
