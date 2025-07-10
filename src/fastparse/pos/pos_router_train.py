#!/usr/bin/env python3
# pos_router_train.py
#
# Train the tiny depth-wise-CNN POS tagger used as the router
# in the MoE dependency-parser architecture.
#
# 🚀 GPU OPTIMIZATIONS:
# - cuDNN benchmark mode for faster convolutions
# - Mixed precision training (AMP) for 2x throughput
# - Multi-worker DataLoader with prefetching
# - Non-blocking GPU transfers
# - Pre-tensorified datasets to avoid Python overhead

# ---------------------------------------------------------------------#
# 0.  Dependencies
# ---------------------------------------------------------------------#
# pip install torch datasets tqdm sentencepiece
import math, argparse
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from datasets import load_dataset
from tqdm import tqdm

###############################################################################
# 1.  Hyper-parameters
###############################################################################
EMB_DIM      = 64          # token embedding size
DW_KERNEL    = 3           # depth-wise conv width   (±1 token context)
N_TAGS       = 18          # Universal-POS (dataset has 18 tags: 0-17)
BATCH_SIZE   = 8192  # Try 4096, 8192 for better GPU utilization (adjust based on VRAM)
LR_HIGH      = 4e-2  # Initial learning rate
LR_LOW       = 2e-2  # Reduced learning rate after $schedule_max train acc
EPOCHS       = 30
MAX_LEN      = 64          # truncate very long sentences
schedule_max = 0.98

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
        logits = self.lin(x)                  # [B, T, 18]
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
    # No clipping needed - dataset has exactly 18 tags (0-17) and model expects 18
    return {"ids": ids, "upos": pos}

def collate(batch):
    max_len = max(len(x["ids"]) for x in batch)
    ids   = torch.zeros(len(batch), max_len, dtype=torch.long)
    upos  = torch.full((len(batch), max_len), -100, dtype=torch.long)
    mask  = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for i, ex in enumerate(batch):
        n = len(ex["ids"])
        # Data is already tensorified, no need for torch.tensor()
        ids[i, :n]  = ex["ids"]
        upos[i, :n] = ex["upos"]
        mask[i, :n] = True
    return ids, upos, mask

###############################################################################
# 4.  Training / validation loops
###############################################################################
def run_epoch(model, loader, optimiser=None, device="cpu", scaler=None):
    train = optimiser is not None
    model.train() if train else model.eval()
    total_loss, total_tok, correct = 0.0, 0, 0

    for ids, upos, mask in tqdm(loader, leave=False):
        # Non-blocking transfers to GPU
        ids   = ids.to(device,   non_blocking=True)
        upos  = upos.to(device,  non_blocking=True)
        mask  = mask.to(device,  non_blocking=True)
        
        if train and scaler is not None:
            # Mixed precision training
            with torch.cuda.amp.autocast():
                logp = model(ids, mask)                # [B, T, 18]
                loss = nn.functional.nll_loss(
                    logp.transpose(1,2), upos, reduction="sum", ignore_index=-100
                )
            
            optimiser.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
        else:
            # Standard training/validation
            logp = model(ids, mask)                # [B, T, 18]
            loss = nn.functional.nll_loss(
                logp.transpose(1,2), upos, reduction="sum", ignore_index=-100
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
    parser.add_argument("--treebank", default="en_ewt",
                        help="Any UD code accepted by datasets (e.g. en_ewt, en_gum, fr_sequoia)")
    parser.add_argument("--combine", action="store_true",
                        help="Combine multiple English treebanks for more training data")
    args = parser.parse_args()
    
    # Enable cuDNN autotuning for faster convolutions
    torch.backends.cudnn.benchmark = True

    print("Loading UD dataset …")
    if args.combine:
        print("🔥 Loading combined English treebanks for maximum training data!")
        # Load multiple English treebanks
        treebanks = ["en_ewt", "en_gum"]  # Add more if available
        train_datasets = []
        val_datasets = []
        
        for tb in treebanks:
            try:
                ds_train_tb = load_dataset("universal_dependencies", tb, split="train", trust_remote_code=True)
                ds_val_tb = load_dataset("universal_dependencies", tb, split="validation", trust_remote_code=True)
                train_datasets.append(ds_train_tb)
                val_datasets.append(ds_val_tb)
                print(f"  ✓ Loaded {tb}: {len(ds_train_tb)} train, {len(ds_val_tb)} val")
            except Exception as e:
                print(f"  ❌ Failed to load {tb}: {e}")
        
        # Concatenate datasets
        from datasets import concatenate_datasets
        ds_train = concatenate_datasets(train_datasets)
        ds_val = concatenate_datasets(val_datasets)
        print(f"🎯 Combined total: {len(ds_train)} train, {len(ds_val)} val sentences")
    else:
        ds_train = load_dataset("universal_dependencies", args.treebank, split="train", trust_remote_code=True)
        ds_val   = load_dataset("universal_dependencies", args.treebank, split="validation", trust_remote_code=True)
        print(f"📊 {args.treebank}: {len(ds_train)} train, {len(ds_val)} val sentences")

    vocab = build_vocab(ds_train)
    
    # Debug: Check upos value ranges
    all_upos = []
    for ex in ds_train:
        all_upos.extend(ex["upos"])
    print(f"POS tag range in dataset: {min(all_upos)} to {max(all_upos)}")
    print(f"Model expects: 0 to {N_TAGS-1}")
    
    train_enc = ds_train.map(lambda ex: encode_sent(ex, vocab))
    val_enc   = ds_val  .map(lambda ex: encode_sent(ex, vocab))
    
    # Pre-tensorify datasets to avoid Python tensor creation overhead
    train_enc = train_enc.with_format("torch", columns=["ids", "upos"], output_all_columns=True)
    val_enc   = val_enc.with_format("torch", columns=["ids", "upos"], output_all_columns=True)

    train_loader = DataLoader(
        train_enc,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate,
        num_workers=4,       # parallelize Python preprocessing
        pin_memory=True,     # speed up host→GPU transfer
        prefetch_factor=2,   # each worker preloads 2 batches
        persistent_workers=True  # keep workers alive between epochs
    )
    val_loader = DataLoader(
        val_enc,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate,
        num_workers=2,       # fewer workers for validation
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = DepthWiseCNNRouter(len(vocab)).to(device)
    opt    = optim.AdamW(model.parameters(), lr=LR_HIGH)
    
    # Initialize AMP scaler for mixed precision training
    scaler = GradScaler() if device == "cuda" else None
    
    lr_switched = False  # Track if we've already switched LR
    
    print(f"🚀 Starting training on {device}")
    print(f"🔥 GPU optimization enabled: cuDNN benchmark, AMP, optimized DataLoader")

    for epoch in range(1, EPOCHS + 1):
        train_ppl, train_acc = run_epoch(model, train_loader, opt, device, scaler)
        val_ppl,   val_acc   = run_epoch(model, val_loader, None, device, None)
        
        # Switch learning rate when train accuracy hits 95%
        if train_acc >= schedule_max and not lr_switched:
            print(f"🎯 Train accuracy reached {train_acc*100:.2f}% - switching LR from {LR_HIGH} to {LR_LOW}")
            for param_group in opt.param_groups:
                param_group['lr'] = LR_LOW
            lr_switched = True
        
        current_lr = opt.param_groups[0]['lr']
        print(f"epoch {epoch:02d} | "
              f"train acc {train_acc*100:5.2f}% | "
              f"val acc {val_acc*100:5.2f}% | "
              f"val ppl {val_ppl:4.2f} | "
              f"lr {current_lr:.1e}")

    # Save model with dataset info
    if args.combine:
        model_name = "router_combined.pt"
    else:
        model_name = f"router_{args.treebank}.pt"
    
    torch.save(model.state_dict(), model_name)
    print(f"✓ finished; weights saved to {model_name}")

if __name__ == "__main__":
    main()
