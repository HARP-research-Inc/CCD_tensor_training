#!/usr/bin/env python3
# pos_router_train.py
#
# Train the tiny depth-wise-CNN POS tagger used as the router
# in the MoE dependency-parser architecture.

# ---------------------------------------------------------------------#
# 0.  Dependencies
# ---------------------------------------------------------------------#
# pip install torch datasets tqdm sentencepiece
import math, argparse
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm

###############################################################################
# 1.  Hyper-parameters
###############################################################################
EMB_DIM      = 64          # token embedding size
DW_KERNEL    = 3           # depth-wise conv width   (±1 token context)
N_TAGS       = 17          # Universal-POS
BATCH_SIZE   = 256
LR           = 2e-3
EPOCHS       = 10
MAX_LEN      = 64          # truncate very long sentences

###############################################################################
# 2.  Tiny router model
###############################################################################
class DepthWiseCNNRouter(nn.Module):
    """Token EMB → depth-wise Conv → POS logits (length preserved)."""
    def __init__(self, vocab_size: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, EMB_DIM, padding_idx=0)

        # depth-wise separable Conv:
        self.dw = nn.Conv1d(
            EMB_DIM, EMB_DIM, kernel_size=DW_KERNEL,
            padding=DW_KERNEL // 2,
            groups=EMB_DIM, bias=True
        )                       # depth-wise (channel-wise) conv
        self.pw = nn.Conv1d(EMB_DIM, EMB_DIM, kernel_size=1)  # point-wise mix
        self.act = nn.ReLU()
        self.lin = nn.Linear(EMB_DIM, N_TAGS)

    def forward(self, token_ids, mask):
        """
        token_ids : [B, T] int64
        mask      : [B, T] bool  (True on real tokens, False on padding)
        returns   : log-probs  [B, T, N_TAGS]
        """
        x = self.emb(token_ids)               # [B, T, D]
        x = x.transpose(1, 2)                 # -> [B, D, T]  for Conv1d
        x = self.pw(self.act(self.dw(x)))
        x = x.transpose(1, 2)                 # back to [B, T, D]
        logits = self.lin(x)                  # [B, T, 17]
        # Use −inf on padding positions so CE ignores them
        logits = logits.masked_fill(~mask.unsqueeze(-1), -1e4)
        return torch.log_softmax(logits, dim=-1)


###############################################################################
# 3.  UD data → tensors
###############################################################################
def build_vocab(train):
    """Map every token form to a unique integer id (0 = PAD)."""
    vocab = {"<PAD>": 0}
    for ex in train:
        for tok in ex["tokens"]:
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab

def encode_sent(ex, vocab):
    ids = [vocab.get(tok, 0) for tok in ex["tokens"]][:MAX_LEN]
    pos = ex["upos"][:MAX_LEN]
    return {"ids": ids, "upos": pos}

def collate(batch):
    max_len = max(len(x["ids"]) for x in batch)
    ids   = torch.zeros(len(batch), max_len, dtype=torch.long)
    upos  = torch.full((len(batch), max_len), -100, dtype=torch.long)
    mask  = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for i, ex in enumerate(batch):
        n = len(ex["ids"])
        ids[i, :n]  = torch.tensor(ex["ids"])
        upos[i, :n] = torch.tensor(ex["upos"], dtype=torch.long)
        mask[i, :n] = True
    return ids, upos, mask

###############################################################################
# 4.  Training / validation loops
###############################################################################
def run_epoch(model, loader, optimiser=None, device="cpu"):
    train = optimiser is not None
    model.train() if train else model.eval()
    total_loss, total_tok, correct = 0.0, 0, 0

    for ids, upos, mask in tqdm(loader, leave=False):
        ids, upos, mask = ids.to(device), upos.to(device), mask.to(device)
        logp = model(ids, mask)                # [B, T, 17]
        loss = nn.functional.nll_loss(
            logp.transpose(1,2), upos, reduction="sum"
        )


        if train:
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        total_loss  += loss.item()
        total_tok   += mask.sum().item()
        pred        = logp.argmax(-1)
        correct     += ((pred == upos) & mask).sum().item()

    ppl = math.exp(total_loss / total_tok)
    acc = correct / total_tok
    return ppl, acc

###############################################################################
# 5.  Main
###############################################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--treebank", default="en_gum",
                        help="Any UD code accepted by datasets (e.g. en_ewt, fr_sequoia)")
    args = parser.parse_args()

    print("Loading UD dataset …")
    ds_train = load_dataset("universal_dependencies", args.treebank, split="train", trust_remote_code=True)
    ds_val   = load_dataset("universal_dependencies", args.treebank, split="validation")

    vocab = build_vocab(ds_train)
    train_enc = ds_train.map(lambda ex: encode_sent(ex, vocab))
    val_enc   = ds_val  .map(lambda ex: encode_sent(ex, vocab))

    train_loader = DataLoader(train_enc, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate)
    val_loader   = DataLoader(val_enc, batch_size=BATCH_SIZE,
                              shuffle=False, collate_fn=collate)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = DepthWiseCNNRouter(len(vocab)).to(device)
    opt    = optim.AdamW(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        train_ppl, train_acc = run_epoch(model, train_loader, opt, device)
        val_ppl,   val_acc   = run_epoch(model, val_loader, None, device)
        print(f"epoch {epoch:02d} | "
              f"train acc {train_acc*100:5.2f}% | "
              f"val acc {val_acc*100:5.2f}% | "
              f"val ppl {val_ppl:4.2f}")

    torch.save(model.state_dict(), f"router_{args.treebank}.pt")
    print("✓ finished; weights saved.")

if __name__ == "__main__":
    main()
