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
# - Label smoothing for better calibration
# - Temperature scaling for probability calibration

# Run: python .\pos_router_train.py --combine

# ---------------------------------------------------------------------#
# 0.  Dependencies
# ---------------------------------------------------------------------#
# pip install torch datasets tqdm sentencepiece scikit-learn
import math, argparse
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report
from collections import defaultdict

###############################################################################
# 1.  Hyper-parameters
###############################################################################
EMB_DIM      = 48          # token embedding size
DW_KERNEL    = 3           # depth-wise conv width   (±1 token context)
N_TAGS       = 18          # Universal-POS (dataset has 18 tags: 0-17)
BATCH_SIZE   = 25911  # Optimal for GPU utilization - NOT full dataset size!
LR_MAX       = 7e-2  # Peak learning rate after warm-up
LR_MIN       = 1e-4  # Minimum learning rate at end of cosine decay
EPOCHS       = 80    # Total training epochs
WARMUP_EPOCHS = 3    # Warm-up epochs (gradual LR increase)
MAX_LEN      = 64    # truncate very long sentences
LABEL_SMOOTHING = 0.1      # Label smoothing factor for better calibration
TEMP_SCALING = True        # Enable temperature scaling

# Universal POS tag names for better reporting
UPOS_TAGS = [
    "NOUN", "PUNCT", "ADP", "NUM", "SYM", "SCONJ", "ADJ", "PART", "DET", "CCONJ", 
    "PROPN", "PRON", "X", "ADV", "INTJ", "VERB", "AUX", "SPACE"
]

###############################################################################
# 2.  Enhanced router model with more capacity and temperature scaling
###############################################################################
class DepthWiseCNNRouter(nn.Module):
    """Enhanced Token EMB → depth-wise Conv → POS logits with temperature scaling."""
    def __init__(self, vocab_size: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, EMB_DIM, padding_idx=0)
        self.emb_dropout = nn.Dropout(0.1)  # Regularization
        
        # First depth-wise separable Conv layer
        self.dw1 = nn.Conv1d(
            EMB_DIM, EMB_DIM, kernel_size=DW_KERNEL,
            padding=DW_KERNEL // 2,
            groups=EMB_DIM, bias=True
        )
        self.pw1 = nn.Conv1d(EMB_DIM, EMB_DIM, kernel_size=1)
        self.norm1 = nn.LayerNorm(EMB_DIM)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.1)
        
        # Second depth-wise separable Conv layer for more capacity
        self.dw2 = nn.Conv1d(
            EMB_DIM, EMB_DIM, kernel_size=DW_KERNEL,
            padding=DW_KERNEL // 2,
            groups=EMB_DIM, bias=True
        )
        self.pw2 = nn.Conv1d(EMB_DIM, EMB_DIM, kernel_size=1)
        self.norm2 = nn.LayerNorm(EMB_DIM)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.1)
        
        # Final classification layer
        self.lin = nn.Linear(EMB_DIM, N_TAGS)
        
        # Temperature parameter for calibration
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, token_ids, mask, use_temperature=False):
        """
        token_ids : [B, T] int64
        mask      : [B, T] bool  (True on real tokens, False on padding)
        use_temperature : bool - whether to apply temperature scaling
        returns   : log-probs  [B, T, N_TAGS]
        """
        x = self.emb(token_ids)               # [B, T, D]
        x = self.emb_dropout(x)
        
        # First conv layer
        x = x.transpose(1, 2)                 # -> [B, D, T]  for Conv1d
        x = self.pw1(self.act1(self.dw1(x)))  # depth-wise + point-wise
        x = x.transpose(1, 2)                 # back to [B, T, D]
        x = self.norm1(x)
        x = self.dropout1(x)
        
        # Second conv layer
        #x = x.transpose(1, 2)                 # -> [B, D, T]  for Conv1d
        #x = self.pw2(self.act2(self.dw2(x)))  # depth-wise + point-wise
        #x = x.transpose(1, 2)                 # back to [B, T, D]
        #x = self.norm2(x)
        #x = self.dropout2(x)
        
        # Final classification
        logits = self.lin(x)                  # [B, T, 18]
        
        # Apply temperature scaling if requested
        if use_temperature:
            logits = logits / self.temperature
        
        # Use −inf on padding positions so CE ignores them
        logits = logits.masked_fill(~mask.unsqueeze(-1), -1e4)
        return F.log_softmax(logits, dim=-1)

###############################################################################
# 3.  Label smoothing loss function
###############################################################################
class LabelSmoothingLoss(nn.Module):
    """Label smoothing loss for better calibration."""
    def __init__(self, smoothing=0.1, ignore_index=-100):
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        
    def forward(self, pred, target):
        """
        pred: [B, C, T] log probabilities
        target: [B, T] target labels
        """
        B, C, T = pred.shape
        pred = pred.transpose(1, 2).contiguous().view(-1, C)  # [B*T, C]
        target = target.view(-1)  # [B*T]
        
        # Create one-hot with label smoothing
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (C - 1))
        
        # Only apply to non-ignored indices
        mask = target != self.ignore_index
        if mask.any():
            true_dist[mask] = true_dist[mask].scatter_(1, target[mask].unsqueeze(1), 1.0 - self.smoothing)
        
        # Compute loss only on non-ignored tokens
        loss = -true_dist * pred
        loss = loss.sum(dim=1)
        loss = loss[mask].mean() if mask.any() else torch.tensor(0.0, device=pred.device)
        
        return loss

###############################################################################
# 4.  UD data → tensors
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

def augment_dataset(ds, augment_factor=1.5):
    """Apply data augmentation to increase dataset size."""
    if augment_factor <= 1.0:
        return ds
    
    print(f"🔄 Applying data augmentation (factor: {augment_factor}x)")
    
    # Simple augmentation: randomly duplicate sentences with slight variations
    augmented_data = []
    original_data = list(ds)
    
    import random
    random.seed(42)  # For reproducibility
    
    # Add original data
    augmented_data.extend(original_data)
    
    # Add augmented versions
    target_size = int(len(original_data) * augment_factor)
    while len(augmented_data) < target_size:
        # Randomly select a sentence to augment
        orig_ex = random.choice(original_data)
        
        # Simple augmentation: randomly drop/duplicate tokens (keeping POS alignment)
        if len(orig_ex["tokens"]) > 3:  # Only for sentences with enough tokens
            # Randomly select augmentation type
            aug_type = random.choice(["original", "truncate", "repeat_short"])
            
            if aug_type == "truncate" and len(orig_ex["tokens"]) > 5:
                # Truncate sentence (keep first 70-90% of tokens)
                keep_ratio = random.uniform(0.7, 0.9)
                keep_len = max(3, int(len(orig_ex["tokens"]) * keep_ratio))
                aug_ex = {
                    "tokens": orig_ex["tokens"][:keep_len],
                    "upos": orig_ex["upos"][:keep_len]
                }
            elif aug_type == "repeat_short" and len(orig_ex["tokens"]) <= 8:
                # For short sentences, create slight variations
                aug_ex = orig_ex.copy()
            else:
                # Keep original
                aug_ex = orig_ex.copy()
            
            augmented_data.append(aug_ex)
        else:
            # For short sentences, just duplicate
            augmented_data.append(orig_ex.copy())
    
    print(f"📈 Dataset augmented: {len(original_data)} → {len(augmented_data)} sentences")
    return augmented_data

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
# 5.  Training / validation loops with per-class analysis
###############################################################################
def run_epoch(model, loader, optimiser=None, device="cpu", scaler=None, criterion=None, 
              epoch=None, detailed_analysis=False):
    train = optimiser is not None
    model.train() if train else model.eval()
    total_loss, total_tok, correct = 0.0, 0, 0
    
    # For detailed analysis
    all_preds = []
    all_targets = []
    per_class_stats = defaultdict(lambda: {'correct': 0, 'total': 0, 'loss': 0.0})

    for ids, upos, mask in tqdm(loader, leave=False):
        # Non-blocking transfers to GPU
        ids   = ids.to(device,   non_blocking=True)
        upos  = upos.to(device,  non_blocking=True)
        mask  = mask.to(device,  non_blocking=True)
        
        if train and scaler is not None:
            # Mixed precision training
            with torch.cuda.amp.autocast():
                logp = model(ids, mask)                # [B, T, 18]
                if criterion is not None:
                    loss = criterion(logp.transpose(1,2), upos)
                else:
                    loss = F.nll_loss(
                        logp.transpose(1,2), upos, reduction="sum", ignore_index=-100
                    )
            
            optimiser.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
        else:
            # Standard training/validation
            # Apply temperature scaling during validation
            use_temp = not train and TEMP_SCALING
            logp = model(ids, mask, use_temperature=use_temp)    # [B, T, 18]
            if criterion is not None:
                loss = criterion(logp.transpose(1,2), upos)
            else:
                loss = F.nll_loss(
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
        
        # Collect predictions for detailed analysis
        if detailed_analysis:
            valid_mask = mask & (upos != -100)
            if valid_mask.any():
                all_preds.extend(pred[valid_mask].cpu().numpy())
                all_targets.extend(upos[valid_mask].cpu().numpy())
                
                # Per-class statistics
                for i in range(N_TAGS):
                    class_mask = valid_mask & (upos == i)
                    if class_mask.any():
                        class_correct = ((pred == upos) & class_mask).sum().item()
                        class_total = class_mask.sum().item()
                        per_class_stats[i]['correct'] += class_correct
                        per_class_stats[i]['total'] += class_total

    ppl = math.exp(total_loss / total_tok) if total_tok > 0 else float('inf')
    acc = correct / total_tok if total_tok > 0 else 0.0
    
    # Generate detailed analysis
    analysis = {}
    if detailed_analysis and all_preds:
        try:
            # Classification report
            report = classification_report(
                all_targets, all_preds, 
                target_names=UPOS_TAGS[:N_TAGS], 
                output_dict=True, 
                zero_division=0
            )
            analysis['classification_report'] = report
            
            # Per-class accuracy
            per_class_acc = {}
            for i, stats in per_class_stats.items():
                if stats['total'] > 0:
                    per_class_acc[UPOS_TAGS[i]] = stats['correct'] / stats['total']
            analysis['per_class_accuracy'] = per_class_acc
            
        except Exception as e:
            print(f"Warning: Could not generate detailed analysis: {e}")
    
    return ppl, acc, analysis

###############################################################################
# 6.  Temperature scaling calibration
###############################################################################
def calibrate_temperature(model, val_loader, device):
    """Calibrate temperature parameter using validation set."""
    print("🌡️  Calibrating temperature for better probability calibration...")
    
    model.eval()
    logits_list = []
    targets_list = []
    
    # Collect logits and targets
    with torch.no_grad():
        for ids, upos, mask in tqdm(val_loader, desc="Collecting logits", leave=False):
            ids = ids.to(device, non_blocking=True)
            upos = upos.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            
            # Get logits before softmax (without temperature)
            x = model.emb(ids)
            x = model.emb_dropout(x)
            x = x.transpose(1, 2)
            x = model.pw1(model.act1(model.dw1(x)))
            x = x.transpose(1, 2)
            x = model.norm1(x)
            x = model.dropout1(x)
            logits = model.lin(x)
            
            # Collect valid positions
            valid_mask = mask & (upos != -100)
            if valid_mask.any():
                logits_list.append(logits[valid_mask])
                targets_list.append(upos[valid_mask])
    
    if not logits_list:
        print("No valid data for temperature calibration")
        return
    
    all_logits = torch.cat(logits_list)
    all_targets = torch.cat(targets_list)
    
    # Optimize temperature
    temp_optimizer = optim.LBFGS([model.temperature], lr=0.01, max_iter=100)
    
    def temperature_loss():
        temp_optimizer.zero_grad()
        scaled_logits = all_logits / model.temperature
        loss = F.cross_entropy(scaled_logits, all_targets)
        loss.backward()
        return loss
    
    temp_optimizer.step(temperature_loss)
    
    print(f"📊 Optimal temperature: {model.temperature.item():.4f}")

###############################################################################
# 7.  Main
###############################################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--treebank", default="en_ewt",
                        help="Any UD code accepted by datasets (e.g. en_ewt, en_gum, fr_sequoia)")
    parser.add_argument("--combine", action="store_true",
                        help="Combine multiple English treebanks for more training data")
    parser.add_argument("--augment", action="store_true",
                        help="Apply data augmentation techniques")
    parser.add_argument("--no-label-smoothing", action="store_true",
                        help="Disable label smoothing")
    parser.add_argument("--no-temp-scaling", action="store_true",
                        help="Disable temperature scaling")
    args = parser.parse_args()
    
    # Enable cuDNN autotuning for faster convolutions
    torch.backends.cudnn.benchmark = True
    
    # Configure label smoothing and temperature scaling
    global LABEL_SMOOTHING, TEMP_SCALING
    if args.no_label_smoothing:
        LABEL_SMOOTHING = 0.0
    if args.no_temp_scaling:
        TEMP_SCALING = False

    print("Loading UD dataset …")
    
    if args.combine:
        print("🔥 Loading combined English treebanks for maximum training data!")
        # Load multiple English treebanks for maximum training data
        english_treebanks = [
            "en_ewt",      # English Web Treebank (12,543 sentences)
            "en_gum",      # Georgetown University Multilayer (8,551 sentences) 
            "en_lines",    # English LinES (4,564 sentences)
            "en_partut",   # English ParTUT (1,781 sentences)
            "en_pronouns", # English Pronouns (305 sentences)
            "en_esl",      # English ESL (5,124 sentences)
        ]
        
        train_datasets = []
        val_datasets = []
        
        for tb in english_treebanks:
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
        print(f"🎯 Combined English total: {len(ds_train)} train, {len(ds_val)} val sentences")
    else:
        # Single treebank mode
        ds_train = load_dataset("universal_dependencies", args.treebank, split="train", trust_remote_code=True)
        ds_val = load_dataset("universal_dependencies", args.treebank, split="validation", trust_remote_code=True)
        print(f"📊 Single treebank {args.treebank}: {len(ds_train)} train, {len(ds_val)} val sentences")

    vocab = build_vocab(ds_train)
    
    # Apply data augmentation if requested
    if args.augment:
        ds_train = augment_dataset(ds_train, augment_factor=1.5)
        # Convert back to dataset format if needed
        if not hasattr(ds_train, 'map'):
            # Convert list back to dataset-like object
            from datasets import Dataset
            ds_train = Dataset.from_list(ds_train)
    
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
    opt    = optim.AdamW(model.parameters(), lr=LR_MIN, weight_decay=1e-4)  # Start with min LR for warm-up
    
    # Initialize loss function with label smoothing
    criterion = None
    if LABEL_SMOOTHING > 0:
        criterion = LabelSmoothingLoss(smoothing=LABEL_SMOOTHING)
        print(f"📊 Using label smoothing with α={LABEL_SMOOTHING}")
    
    # Initialize AMP scaler for mixed precision training
    scaler = GradScaler() if device == "cuda" else None
    
    # Cosine annealing scheduler with warm-up
    def get_lr(epoch):
        if epoch < WARMUP_EPOCHS:
            # Linear warm-up from LR_MIN to LR_MAX
            return LR_MIN + (LR_MAX - LR_MIN) * epoch / WARMUP_EPOCHS
        else:
            # Cosine decay from LR_MAX to LR_MIN
            progress = (epoch - WARMUP_EPOCHS) / (EPOCHS - WARMUP_EPOCHS)
            return LR_MIN + (LR_MAX - LR_MIN) * 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(opt, lambda epoch: get_lr(epoch) / LR_MIN)
    
    print(f"🚀 Starting training on {device}")
    print(f"🔥 GPU optimization enabled: cuDNN benchmark, AMP, optimized DataLoader")
    print(f"📈 Cosine annealing LR: {LR_MIN:.1e} → {LR_MAX:.1e} → {LR_MIN:.1e} (warmup: {WARMUP_EPOCHS} epochs)")
    if TEMP_SCALING:
        print(f"🌡️  Temperature scaling enabled")

    for epoch in range(1, EPOCHS + 1):
        # Training with label smoothing
        train_ppl, train_acc, _ = run_epoch(
            model, train_loader, opt, device, scaler, criterion, epoch
        )
        
        # Validation with detailed analysis every 10 epochs
        detailed = (epoch % 10 == 0) or (epoch == EPOCHS)
        val_ppl, val_acc, analysis = run_epoch(
            model, val_loader, None, device, None, criterion, epoch, detailed_analysis=detailed
        )
        
        # Step the learning rate scheduler
        scheduler.step()
        
        current_lr = opt.param_groups[0]['lr']
        temp_str = f" | temp {model.temperature.item():.3f}" if TEMP_SCALING else ""
        phase = "warmup" if epoch <= WARMUP_EPOCHS else "cosine"
        print(f"epoch {epoch:02d} | "
              f"train acc {train_acc*100:5.2f}% | "
              f"val acc {val_acc*100:5.2f}% | "
              f"val ppl {val_ppl:4.2f} | "
              f"lr {current_lr:.1e} ({phase}){temp_str}")
        
        # Print detailed analysis
        if detailed and analysis:
            print("\n📊 Detailed Per-Class Analysis:")
            if 'per_class_accuracy' in analysis:
                sorted_classes = sorted(analysis['per_class_accuracy'].items(), 
                                      key=lambda x: x[1], reverse=True)
                for class_name, acc in sorted_classes:
                    print(f"  {class_name:8s}: {acc*100:5.2f}%")
            
            if 'classification_report' in analysis:
                print(f"\n📋 Macro avg F1: {analysis['classification_report']['macro avg']['f1-score']*100:.2f}%")
                print(f"   Weighted F1: {analysis['classification_report']['weighted avg']['f1-score']*100:.2f}%")
            print()

    # Apply temperature scaling calibration
    if TEMP_SCALING:
        calibrate_temperature(model, val_loader, device)
        
        # Re-evaluate with calibrated temperature
        print("🔍 Re-evaluating with calibrated temperature...")
        cal_ppl, cal_acc, cal_analysis = run_epoch(
            model, val_loader, None, device, None, criterion, 
            epoch=EPOCHS, detailed_analysis=True
        )
        print(f"📊 Calibrated results: acc {cal_acc*100:.2f}%, ppl {cal_ppl:.2f}")

    # Save model with dataset info
    if args.combine:
        model_name = "router_combined.pt"
    else:
        model_name = f"router_{args.treebank}.pt"
    
    torch.save(model.state_dict(), model_name)
    print(f"✓ finished; weights saved to {model_name}")

if __name__ == "__main__":
    main()
