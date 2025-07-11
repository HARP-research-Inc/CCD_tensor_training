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
# BATCH_SIZE will be auto-calculated based on dataset size
LR_MAX       = 7e-2  # Lower LR for smaller Penn dataset
LR_MIN       = 1e-4  # Minimum learning rate at end of cosine decay
EPOCHS       = 80    # Total training epochs
WARMUP_EPOCHS = 3    # Warm-up epochs (gradual LR increase)
MAX_LEN      = 64    # truncate very long sentences
LABEL_SMOOTHING = 0.1      # Label smoothing factor for better calibration
TEMP_SCALING = True        # Enable temperature scaling

# Compute node optimizations
NUM_WORKERS_TRAIN = 48     # 64 cores: use 48 for training (leave some for system)
NUM_WORKERS_VAL = 16       # 16 for validation
PREFETCH_FACTOR = 4        # Higher prefetch for compute nodes
PIN_MEMORY = True          # Always pin memory on compute nodes

# Universal POS tag names for better reporting (MUST match UD dataset order!)
UPOS_TAGS = [
    "NOUN", "PUNCT", "ADP", "NUM", "SYM", "SCONJ", "ADJ", "PART", "DET", "CCONJ", 
    "PROPN", "PRON", "X", "_", "ADV", "INTJ", "VERB", "AUX"
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
        x = x.transpose(1, 2)                 # -> [B, D, T]  for Conv1d
        x = self.pw2(self.act2(self.dw2(x)))  # depth-wise + point-wise
        x = x.transpose(1, 2)                 # back to [B, T, D]
        x = self.norm2(x)
        x = self.dropout2(x)
        
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
        # Return sum (not mean) to be consistent with F.nll_loss(reduction="sum") for perplexity calculation
        loss = loss[mask].sum() if mask.any() else torch.tensor(0.0, device=pred.device)
        
        return loss

###############################################################################
# 4.  Penn Treebank support
###############################################################################

def penn_to_universal_tag_mapping():
    """Convert Penn Treebank POS tags to Universal POS tags."""
    return {
        # Nouns
        'NN': 'NOUN', 'NNS': 'NOUN', 'NNP': 'PROPN', 'NNPS': 'PROPN',
        
        # Verbs  
        'VB': 'VERB', 'VBD': 'VERB', 'VBG': 'VERB', 'VBN': 'VERB', 'VBP': 'VERB', 'VBZ': 'VERB',
        
        # Auxiliaries (modal verbs)
        'MD': 'AUX',
        
        # Adjectives
        'JJ': 'ADJ', 'JJR': 'ADJ', 'JJS': 'ADJ',
        
        # Adverbs
        'RB': 'ADV', 'RBR': 'ADV', 'RBS': 'ADV', 'WRB': 'ADV',
        
        # Pronouns
        'PRP': 'PRON', 'PRP$': 'PRON', 'WP': 'PRON', 'WP$': 'PRON',
        
        # Determiners
        'DT': 'DET', 'PDT': 'DET', 'WDT': 'DET',
        
        # Prepositions
        'IN': 'ADP',
        
        # Conjunctions
        'CC': 'CCONJ',
        
        # Numbers
        'CD': 'NUM',
        
        # Particles
        'RP': 'PART', 'TO': 'PART',
        
        # Interjections
        'UH': 'INTJ',
        
        # Symbols and punctuation
        'SYM': 'SYM',
        '.': 'PUNCT', ',': 'PUNCT', ':': 'PUNCT', ';': 'PUNCT', 
        '!': 'PUNCT', '?': 'PUNCT', '``': 'PUNCT', "''": 'PUNCT',
        '(': 'PUNCT', ')': 'PUNCT', '"': 'PUNCT', '#': 'PUNCT', '$': 'PUNCT',
        
        # Other/Unknown
        'FW': 'X',     # Foreign words
        'LS': 'X',     # List markers
        'POS': 'PART', # Possessive endings
        'EX': 'PRON',  # Existential there
        
        # Default for any unmapped tags
        'X': 'X'
    }

def load_penn_treebank_data(penn_path=None):
    """
    Load Penn Treebank data with standard WSJ splits.
    
    Returns:
        train_data, val_data, test_data: Lists of {'tokens': [...], 'upos': [...]}
    """
    import os
    from nltk.corpus import treebank
    
    if penn_path and os.path.exists(penn_path):
        print(f"🏛️  Loading full Penn Treebank from {penn_path}")
        # TODO: Implement full Penn Treebank loading
        # This would require parsing .mrg files from the full LDC distribution
        print("⚠️  Full Penn Treebank parsing not yet implemented")
        print("   Falling back to NLTK sample with manual splits")
    
    print("📚 Using NLTK Penn Treebank sample with train/val/test splits")
    
    # Get all sentences from NLTK
    sents = list(treebank.tagged_sents())
    
    # Create train/val/test splits (80/10/10)
    total = len(sents)
    train_end = int(0.8 * total)
    val_end = int(0.9 * total)
    
    train_sents = sents[:train_end]
    val_sents = sents[train_end:val_end]
    test_sents = sents[val_end:]
    
    print(f"📊 Penn Treebank splits:")
    print(f"   Train: {len(train_sents):,} sentences")
    print(f"   Val:   {len(val_sents):,} sentences") 
    print(f"   Test:  {len(test_sents):,} sentences")
    
    # Convert to Universal POS format
    tag_mapping = penn_to_universal_tag_mapping()
    
    def convert_sentences(sentences):
        converted = []
        for sent in sentences:
            tokens = []
            upos = []
            
            for word, penn_tag in sent:
                # Skip empty/trace elements
                if penn_tag == '-NONE-' or not word or not word.strip():
                    continue
                    
                # Clean up word
                word = word.strip()
                
                # Convert tag
                base_tag = penn_tag.split('-')[0]  # Remove suffixes like -TMP
                universal_tag = tag_mapping.get(base_tag, 'X')
                
                # Special handling for auxiliaries (context-dependent)
                if base_tag in ['VBZ', 'VBP', 'VBD', 'VB'] and word.lower() in {
                    'be', 'am', 'is', 'are', 'was', 'were', 'being', 'been',
                    'have', 'has', 'had', 'having', 'do', 'does', 'did'
                }:
                    universal_tag = 'AUX'
                
                tokens.append(word)
                upos.append(UPOS_TAGS.index(universal_tag))
            
            if tokens:  # Only add non-empty sentences
                converted.append({
                    'tokens': tokens,
                    'upos': upos
                })
        
        return converted
    
    train_data = convert_sentences(train_sents)
    val_data = convert_sentences(val_sents)
    test_data = convert_sentences(test_sents)
    
    print(f"✅ Converted to Universal POS format")
    
    return train_data, val_data, test_data

###############################################################################
# 5.  UD data → tensors
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

def calculate_batch_size(dataset_size, target_batches_per_epoch=25):
    """
    Auto-calculate batch size based on dataset size to ensure reasonable number of batches per epoch.
    
    Args:
        dataset_size: Number of training samples
        target_batches_per_epoch: Target number of batches per epoch (default: 25)
    
    Returns:
        Optimal batch size (power of 2 between 64 and 8192)
    """
    # Calculate ideal batch size
    ideal_batch_size = max(1, dataset_size // target_batches_per_epoch)
    
    # Round to nearest power of 2 and clamp to reasonable range
    powers_of_2 = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
    
    # Find closest power of 2
    best_batch_size = min(powers_of_2, key=lambda x: abs(x - ideal_batch_size))
    
    # Special cases for very small datasets
    if dataset_size < 1000:
        best_batch_size = min(64, dataset_size // 10)  # At least 10 batches for small datasets
    elif dataset_size < 5000:
        best_batch_size = min(256, best_batch_size)  # Cap at 256 for small-medium datasets
    
    # Ensure at least some batches per epoch
    if best_batch_size >= dataset_size:
        best_batch_size = max(32, dataset_size // 8)  # At least 8 batches
    
    return best_batch_size

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
    confusion_matrix = defaultdict(lambda: defaultdict(int))  # confusion_matrix[true_class][pred_class] = count

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
                valid_preds = pred[valid_mask].cpu().numpy()
                valid_targets = upos[valid_mask].cpu().numpy()
                
                all_preds.extend(valid_preds)
                all_targets.extend(valid_targets)
                
                # Build confusion matrix
                for true_label, pred_label in zip(valid_targets, valid_preds):
                    confusion_matrix[true_label][pred_label] += 1
                
                # Per-class statistics
                for i in range(N_TAGS):
                    class_mask = valid_mask & (upos == i)
                    if class_mask.any():
                        class_correct = ((pred == upos) & class_mask).sum().item()
                        class_total = class_mask.sum().item()
                        per_class_stats[i]['correct'] += class_correct
                        per_class_stats[i]['total'] += class_total

    # Calculate perplexity from total sum loss (both criterion and nll_loss return sums now)
    avg_loss_per_token = total_loss / total_tok if total_tok > 0 else float('inf')
    ppl = math.exp(avg_loss_per_token) if avg_loss_per_token < 100 else float('inf')  # Avoid overflow
    acc = correct / total_tok if total_tok > 0 else 0.0
    
    # Generate detailed analysis
    analysis = {}
    if detailed_analysis and all_preds:
        try:
            # Determine which classes are actually present in the data
            present_classes = sorted(set(all_targets + all_preds))
            present_class_names = [UPOS_TAGS[i] for i in present_classes if i < len(UPOS_TAGS)]
            
            # Classification report only for present classes
            report = classification_report(
                all_targets, all_preds, 
                labels=present_classes,
                target_names=present_class_names, 
                output_dict=True, 
                zero_division=0
            )
            analysis['classification_report'] = report
            analysis['present_classes'] = present_classes  # Store for reference
            
            # Per-class accuracy
            per_class_acc = {}
            for i, stats in per_class_stats.items():
                if stats['total'] > 0:
                    per_class_acc[UPOS_TAGS[i]] = stats['correct'] / stats['total']
            analysis['per_class_accuracy'] = per_class_acc
            
            # Confusion matrix analysis
            analysis['confusion_matrix'] = confusion_matrix
            
        except Exception as e:
            print(f"Warning: Could not generate detailed analysis: {e}")
    
    return ppl, acc, analysis

def run_epoch_with_sgdr(model, loader, optimiser, device, scaler, criterion, scheduler, step_count, epoch):
    """Training epoch with SGDR step-based learning rate scheduling."""
    model.train()
    total_loss, total_tok, correct = 0.0, 0, 0

    for ids, upos, mask in tqdm(loader, leave=False):
        # Non-blocking transfers to GPU
        ids   = ids.to(device,   non_blocking=True)
        upos  = upos.to(device,  non_blocking=True)
        mask  = mask.to(device,  non_blocking=True)
        
        if scaler is not None:
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
            # Standard training
            logp = model(ids, mask)    # [B, T, 18]
            if criterion is not None:
                loss = criterion(logp.transpose(1,2), upos)
            else:
                loss = F.nll_loss(
                    logp.transpose(1,2), upos, reduction="sum", ignore_index=-100
                )
            
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        # SGDR: Step the scheduler after each batch
        scheduler.step()
        step_count += 1

        total_loss  += loss.item()
        total_tok   += mask.sum().item()
        pred        = logp.argmax(-1)
        correct     += ((pred == upos) & mask).sum().item()

    # Calculate perplexity from total sum loss (both criterion and nll_loss return sums now)
    avg_loss_per_token = total_loss / total_tok if total_tok > 0 else float('inf')
    ppl = math.exp(avg_loss_per_token) if avg_loss_per_token < 100 else float('inf')  # Avoid overflow
    acc = correct / total_tok if total_tok > 0 else 0.0
    
    return ppl, acc, step_count

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
                        help="Combine multiple UD English treebanks ONLY (no Penn Treebank)")
    parser.add_argument("--augment", action="store_true",
                        help="Apply data augmentation techniques")
    parser.add_argument("--no-label-smoothing", action="store_true",
                        help="Disable label smoothing")
    parser.add_argument("--no-temp-scaling", action="store_true",
                        help="Disable temperature scaling")
    parser.add_argument("--share", action="store_true",
                        help="If multiple GPUs are available, force use of cuda:1 (compute node sharing)")
    parser.add_argument("--gpu", type=int, default=None,
                        help="Force specific GPU ID (overrides auto-selection)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override default batch size for compute node")
    parser.add_argument("--workers", type=int, default=None,
                        help="Override number of dataloader workers")
    parser.add_argument("--compute-node", action="store_true",
                        help="Enable all compute node optimizations")
    parser.add_argument("--penn-treebank", action="store_true",
                        help="Train on Penn Treebank WSJ ONLY (pure Penn Treebank training)")
    parser.add_argument("--penn-path", type=str, default=None,
                        help="Path to full Penn Treebank directory")
    parser.add_argument("--combined-penn", action="store_true",
                        help="Combine UD treebanks WITH Penn Treebank (experimental)")
    parser.add_argument("--cosine", action="store_true",
                        help="Use standard Cosine Annealing scheduler (default: SGDR)")
    parser.add_argument("--sgdr-t0", type=int, default=None,
                        help="SGDR: Steps in first cycle (auto-detected if not provided)")
    parser.add_argument("--sgdr-mult", type=float, default=2.0,
                        help="SGDR: Cycle length multiplier (default: 2.0)")
    args = parser.parse_args()

    # Validate argument combinations
    if args.penn_treebank and args.combine:
        print("❌ ERROR: Cannot use --penn-treebank and --combine together!")
        print("   💡 Use --penn-treebank for PURE Penn Treebank training")
        print("   💡 Use --combine for PURE UD treebank training")
        print("   💡 Use --combined-penn to combine UD + Penn Treebank (experimental)")
        exit(1)
        
    if args.penn_treebank and args.combined_penn:
        print("❌ ERROR: Cannot use --penn-treebank and --combined-penn together!")
        print("   💡 Use --penn-treebank for PURE Penn Treebank training")
        print("   💡 Use --combined-penn to combine UD + Penn Treebank")
        exit(1)
        
    if args.combine and args.combined_penn:
        print("❌ ERROR: Cannot use --combine and --combined-penn together!")
        print("   💡 Use --combine for PURE UD treebank training")
        print("   💡 Use --combined-penn to combine UD + Penn Treebank")
        exit(1)

    # Enable cuDNN autotuning for faster convolutions
    torch.backends.cudnn.benchmark = True
    
    # Compute node optimizations
    torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
    torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for compute nodes
    torch.backends.cudnn.allow_tf32 = True
    
    # Set optimal thread count for 64-core machine
    torch.set_num_threads(min(32, torch.get_num_threads()))  # Cap at 32 to avoid oversubscription
    print(f"🧵 PyTorch threads: {torch.get_num_threads()}")
    
    # Memory optimization for compute nodes
    import os
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'  # Reduce memory fragmentation

    # Configure label smoothing and temperature scaling
    global LABEL_SMOOTHING, TEMP_SCALING, NUM_WORKERS_TRAIN, NUM_WORKERS_VAL
    if args.no_label_smoothing:
        LABEL_SMOOTHING = 0.0
    if args.no_temp_scaling:
        TEMP_SCALING = False
    
    # Apply compute node overrides
    global PREFETCH_FACTOR
    if args.compute_node:
        print("🖥️  Compute node mode enabled: applying all optimizations")
        # More aggressive settings for compute nodes
        NUM_WORKERS_TRAIN = min(56, NUM_WORKERS_TRAIN)  # Leave 8 cores for system
        PREFETCH_FACTOR = 6
        
    if args.workers:
        NUM_WORKERS_TRAIN = args.workers
        NUM_WORKERS_VAL = max(1, args.workers // 3)
        print(f"🧵 Workers override: {NUM_WORKERS_TRAIN} train, {NUM_WORKERS_VAL} val")

    print("Loading dataset …")
    
    if args.penn_treebank:
        # Load Penn Treebank WSJ data ONLY
        print("🏛️  Penn Treebank WSJ Training Mode (PURE)")
        train_data, val_data, test_data = load_penn_treebank_data(args.penn_path)
        
        # Convert to datasets format
        from datasets import Dataset
        ds_train = Dataset.from_list(train_data)
        ds_val = Dataset.from_list(val_data)
        
        print(f"📊 Penn Treebank: {len(ds_train)} train, {len(ds_val)} val sentences")
        
    elif args.combined_penn:
        # Load UD treebanks AND Penn Treebank (experimental)
        print("🔬 EXPERIMENTAL: Combined UD + Penn Treebank Training")
        print("⚠️  Warning: Mixing different data sources - monitor for domain mismatch!")
        
        # First load UD treebanks
        english_treebanks = [
            "en_ewt",      # English Web Treebank (12,543 sentences)
            "en_gum",      # Georgetown University Multilayer (8,551 sentences) 
            "en_lines",    # English LinES (4,564 sentences)
            "en_partut",   # English ParTUT (1,781 sentences)
        ]
        
        print(f"🔥 Loading UD treebanks: {english_treebanks}")
        
        # Combine UD training sets
        combined_train = []
        combined_val = []
        
        for tb in english_treebanks:
            try:
                ds_train_tb = load_dataset("universal_dependencies", tb, split="train", trust_remote_code=True)
                ds_val_tb = load_dataset("universal_dependencies", tb, split="validation", trust_remote_code=True)
                
                # Convert to list and extend
                combined_train.extend(list(ds_train_tb))
                combined_val.extend(list(ds_val_tb))
                
                print(f"  ✓ Loaded UD {tb}: {len(ds_train_tb)} train, {len(ds_val_tb)} val sentences")
                
            except Exception as e:
                print(f"  ❌ Failed to load UD {tb}: {e}")
                continue
        
        # Then add Penn Treebank data
        print("🏛️  Adding Penn Treebank data...")
        penn_train, penn_val, penn_test = load_penn_treebank_data(args.penn_path)
        
        combined_train.extend(penn_train)
        combined_val.extend(penn_val)
        
        print(f"  ✓ Added Penn Treebank: {len(penn_train)} train, {len(penn_val)} val sentences")
        
        # Convert back to datasets
        from datasets import Dataset
        ds_train = Dataset.from_list(combined_train)
        ds_val = Dataset.from_list(combined_val)
        
        print(f"🎯 Combined UD+Penn dataset: {len(ds_train)} train, {len(ds_val)} val sentences")
        print(f"📊 Data sources: UD={len(combined_train)-len(penn_train)}, Penn={len(penn_train)}")
        
    elif args.combine:
        # Load multiple UD English treebanks ONLY (no Penn Treebank)
        print("🔥 Combined UD Training Mode (NO Penn Treebank)")
        english_treebanks = [
            "en_ewt",      # English Web Treebank (12,543 sentences)
            "en_gum",      # Georgetown University Multilayer (8,551 sentences) 
            "en_lines",    # English LinES (4,564 sentences)
            "en_partut",   # English ParTUT (1,781 sentences)
        ]
        
        print(f"📚 Loading UD-only treebanks: {english_treebanks}")
        
        # Combine training sets
        combined_train = []
        combined_val = []
        
        for tb in english_treebanks:
            try:
                ds_train_tb = load_dataset("universal_dependencies", tb, split="train", trust_remote_code=True)
                ds_val_tb = load_dataset("universal_dependencies", tb, split="validation", trust_remote_code=True)
                
                # Convert to list and extend
                combined_train.extend(list(ds_train_tb))
                combined_val.extend(list(ds_val_tb))
                
                print(f"  ✓ Loaded {tb}: {len(ds_train_tb)} train, {len(ds_val_tb)} val sentences")
                
            except Exception as e:
                print(f"  ❌ Failed to load {tb}: {e}")
                continue
        
        # Convert back to datasets
        from datasets import Dataset
        ds_train = Dataset.from_list(combined_train)
        ds_val = Dataset.from_list(combined_val)
        
        print(f"🎯 Combined dataset: {len(ds_train)} train, {len(ds_val)} val sentences")
        
        # Fallback to single treebank if combine failed
        if len(ds_train) == 0:
            print("⚠️  Combined loading failed, falling back to single treebank")
            ds_train = load_dataset("universal_dependencies", "en_ewt", split="train", trust_remote_code=True)
            ds_val = load_dataset("universal_dependencies", "en_ewt", split="validation", trust_remote_code=True)
            print(f"📊 Fallback to en_ewt: {len(ds_train)} train, {len(ds_val)} val sentences")
        
    else:
        ds_train = load_dataset("universal_dependencies", args.treebank, split="train", trust_remote_code=True)
        ds_val   = load_dataset("universal_dependencies", args.treebank, split="validation", trust_remote_code=True)
        print(f"📊 Single treebank {args.treebank}: {len(ds_train)} train, {len(ds_val)} val sentences")

    vocab = build_vocab(ds_train)
    if args.augment:
        ds_train = augment_dataset(ds_train, augment_factor=1.5)
        from datasets import Dataset
        if not hasattr(ds_train, 'map'):
            ds_train = Dataset.from_list(ds_train)

    # Pre-tensorify and DataLoader setup with auto-scaling batch size
    train_enc = ds_train.map(lambda ex: encode_sent(ex, vocab))
    val_enc   = ds_val  .map(lambda ex: encode_sent(ex, vocab))
    train_enc = train_enc.with_format("torch", columns=["ids", "upos"], output_all_columns=True)
    val_enc   = val_enc  .with_format("torch", columns=["ids", "upos"], output_all_columns=True)

    # Auto-calculate optimal batch size unless manually overridden
    if args.batch_size:
        BATCH_SIZE = args.batch_size
        print(f"📦 Manual batch size: {BATCH_SIZE}")
    else:
        BATCH_SIZE = calculate_batch_size(len(ds_train))
        expected_batches = len(ds_train) // BATCH_SIZE
        print(f"📦 Auto-calculated batch size: {BATCH_SIZE} ({expected_batches} batches/epoch)")

    train_loader = DataLoader(
        train_enc, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate, num_workers=NUM_WORKERS_TRAIN, pin_memory=PIN_MEMORY,
        prefetch_factor=PREFETCH_FACTOR, persistent_workers=True,
        drop_last=False  # Don't drop last batch for small datasets
    )
    val_loader = DataLoader(
        val_enc, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate, num_workers=NUM_WORKERS_VAL, pin_memory=PIN_MEMORY,
        prefetch_factor=PREFETCH_FACTOR, persistent_workers=True
    )

    # —— Compute node GPU selection logic ——
    if torch.cuda.is_available():
        ngpu = torch.cuda.device_count()
        print(f"🖥️  Compute node detected: {ngpu} GPU(s) available")
        
        # Smart GPU selection for compute nodes
        if args.gpu is not None:
            # Manual GPU override
            if args.gpu < ngpu:
                selected = args.gpu
                print(f"🎯 Manual GPU selection: GPU {selected}")
            else:
                print(f"⚠️  GPU {args.gpu} not available, using GPU 0")
                selected = 0
        elif args.share and ngpu >= 2:
            # Use GPU 1 if sharing (common on compute nodes)
            selected = 1
            print(f"🤝 Sharing mode: using GPU {selected}")
        elif ngpu >= 4:
            # On 4-GPU systems, prefer GPU 0 unless busy
            try:
                # Check GPU memory usage to pick least busy GPU
                gpu_memory = []
                for i in range(ngpu):
                    torch.cuda.set_device(i)
                    allocated = torch.cuda.memory_allocated(i)
                    total = torch.cuda.get_device_properties(i).total_memory
                    usage = allocated / total
                    gpu_memory.append((i, usage))
                    print(f"  GPU {i}: {usage*100:.1f}% memory used")
                
                # Select GPU with lowest memory usage
                selected = min(gpu_memory, key=lambda x: x[1])[0]
                print(f"🎯 Auto-selected GPU {selected} (lowest memory usage)")
            except Exception as e:
                print(f"⚠️  GPU detection failed: {e}, using GPU 0")
                selected = 0
        else:
            selected = 0
            
        torch.cuda.set_device(selected)
        device = torch.device(f"cuda:{selected}")
        
        # Display GPU info
        gpu_props = torch.cuda.get_device_properties(selected)
        memory_gb = gpu_props.total_memory / 1024**3
        print(f"🚀 Using {gpu_props.name} (GPU {selected})")
        print(f"💾 GPU Memory: {memory_gb:.1f}GB total")
        print(f"🔋 Compute Capability: {gpu_props.major}.{gpu_props.minor}")
        
        # Set memory allocation strategy for compute nodes
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
            torch.cuda.set_per_process_memory_fraction(0.95, selected)  # Use 95% of GPU memory
            
    else:
        device = torch.device("cpu")
        print("⚠️  CUDA not available, falling back to CPU")

    # Model, optimizer, criterion, scaler, scheduler
    model     = DepthWiseCNNRouter(len(vocab)).to(device)
    optimiser = optim.AdamW(model.parameters(), lr=LR_MIN, weight_decay=1e-4)

    criterion = None
    if LABEL_SMOOTHING > 0:
        criterion = LabelSmoothingLoss(smoothing=LABEL_SMOOTHING)
        print(f"📊 Using label smoothing with α={LABEL_SMOOTHING}")

    # Updated GradScaler for compute nodes (PyTorch 2.7+)
    if device.type.startswith("cuda"):
        try:
            from torch.amp import GradScaler
            scaler = GradScaler('cuda')
        except ImportError:
            # Fallback for older PyTorch versions
            from torch.cuda.amp import GradScaler
            scaler = GradScaler()
    else:
        scaler = None

    # Choose scheduler type (SGDR is default, standard cosine only when requested)
    use_cosine = args.cosine  # Only use standard cosine when explicitly requested
    
    if use_cosine:
        # Standard cosine annealing with warmup
        def get_lr(epoch):
            if epoch < WARMUP_EPOCHS:
                return LR_MIN + (LR_MAX - LR_MIN) * epoch / WARMUP_EPOCHS
            progress = (epoch - WARMUP_EPOCHS) / (EPOCHS - WARMUP_EPOCHS)
            return LR_MIN + (LR_MAX - LR_MIN) * 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = optim.lr_scheduler.LambdaLR(optimiser, lambda epoch: get_lr(epoch) / LR_MIN)
        print(f"📈 Standard Cosine Annealing: {LR_MIN:.1e} → {LR_MAX:.1e} → {LR_MIN:.1e}")
    else:
        # Cosine Annealing with Warm Restarts (SGDR) - DEFAULT
        # Use shorter cycles for better exploration and regularization
        T_0 = args.sgdr_t0 if args.sgdr_t0 else max(10, len(train_loader) // 4)
        
        # For Penn Treebank (small dataset), use more frequent restarts
        if args.penn_treebank and len(train_loader) < 200:
            T_0 = args.sgdr_t0 if args.sgdr_t0 else max(5, len(train_loader) // 6)
            T_mult = args.sgdr_mult if hasattr(args, 'sgdr_mult') else 1.5
            eta_min_ratio = 0.01  # Lower minimum
        else:
            T_mult = args.sgdr_mult
            eta_min_ratio = 0.1   # Higher minimum for stability
        
        eta_min = LR_MAX * eta_min_ratio
        
        print(f"🔄 SGDR Scheduler (default):")
        print(f"   • First cycle: {T_0} steps")
        print(f"   • Cycle multiplier: {T_mult}x") 
        print(f"   • LR range: {eta_min:.1e} → {LR_MAX:.1e}")
        
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimiser,
            T_0=T_0,           # Steps in first cycle
            T_mult=int(T_mult) if T_mult == int(T_mult) else 2,  # Integer multiplier
            eta_min=eta_min,   # Minimum LR (not zero for stability)
            last_epoch=-1
        )

    print(f"🚀 Starting training on {device}")
    print(f"🔥 Compute node optimizations enabled:")
    print(f"   • cuDNN benchmark, AMP, TF32")
    print(f"   • DataLoader: {NUM_WORKERS_TRAIN} train workers, {NUM_WORKERS_VAL} val workers")
    print(f"   • Prefetch factor: {PREFETCH_FACTOR}")
    if TEMP_SCALING:
        print(f"🌡️  Temperature scaling enabled")
    
    # Performance monitoring for compute nodes
    print(f"\n📊 COMPUTE NODE PERFORMANCE BASELINE:")
    print(f"   • Dataset size: {len(ds_train):,} train, {len(ds_val):,} val sentences")
    print(f"   • Vocabulary size: {len(vocab):,} tokens")
    print(f"   • Steps per epoch: {len(train_loader):,}")
    print(f"   • Total training steps: {len(train_loader) * EPOCHS:,}")
    
    # Check which POS classes are present in the dataset
    print(f"   • POS tag analysis:")
    sample_upos = []
    for i, batch in enumerate(train_loader):
        if i >= 5:  # Check first 5 batches
            break
        _, upos, mask = batch
        valid_upos = upos[mask & (upos != -100)]
        sample_upos.extend(valid_upos.cpu().numpy().tolist())
    
    present_classes = sorted(set(sample_upos))
    missing_classes = [i for i in range(N_TAGS) if i not in present_classes]
    
    print(f"     - Classes present: {len(present_classes)}/{N_TAGS}")
    if missing_classes:
        missing_names = [UPOS_TAGS[i] for i in missing_classes if i < len(UPOS_TAGS)]
        print(f"     - Missing classes: {missing_names} (indices: {missing_classes})")
    else:
        print(f"     - All {N_TAGS} Universal POS classes present ✓")
    
    # Add SGDR cycle info now that we have train_loader
    if not use_cosine:
        sgdr_T_0 = args.sgdr_t0 if args.sgdr_t0 else max(10, len(train_loader) // 4)
        if args.penn_treebank and len(train_loader) < 200:
            sgdr_T_0 = args.sgdr_t0 if args.sgdr_t0 else max(5, len(train_loader) // 6)
        print(f"   • SGDR first cycle: {sgdr_T_0} steps ({sgdr_T_0 / len(train_loader):.1f} epochs)")
    
    if torch.cuda.is_available():
        print(f"   • GPU memory allocated: {torch.cuda.memory_allocated(device) / 1024**3:.2f}GB")
    print()

    # Training loop variables
    step_count = 0
    
    # SGDR cycle tracking variables
    if not use_cosine:
        sgdr_T_0 = args.sgdr_t0 if args.sgdr_t0 else max(10, len(train_loader) // 4)
        if args.penn_treebank and len(train_loader) < 200:
            sgdr_T_0 = args.sgdr_t0 if args.sgdr_t0 else max(5, len(train_loader) // 6)
            sgdr_T_mult = args.sgdr_mult if hasattr(args, 'sgdr_mult') else 1.5
        else:
            sgdr_T_mult = args.sgdr_mult
        
        sgdr_current_cycle_length = sgdr_T_0
        sgdr_cycle_step = 0
        sgdr_total_steps_in_cycles = 0
        sgdr_cycle_num = 1
        sgdr_eta_min = LR_MAX * (0.01 if args.penn_treebank and len(train_loader) < 200 else 0.1)
        previous_lr = LR_MIN  # Initialize for SGDR tracking
    
    for epoch in range(1, EPOCHS + 1):
        if use_cosine:
            # Standard training with epoch-based scheduling
            train_ppl, train_acc, _ = run_epoch(
                model, train_loader, optimiser, device, scaler, criterion, epoch
            )
            # Step the scheduler after each epoch
            scheduler.step()
        else:
            # Training with SGDR step-based scheduling (default)
            train_ppl, train_acc, step_count = run_epoch_with_sgdr(
                model, train_loader, optimiser, device, scaler, criterion, scheduler, step_count, epoch
            )
        
        # Validation with detailed analysis every 10 epochs
        detailed = (epoch % 10 == 0) or (epoch == EPOCHS)
        val_ppl, val_acc, analysis = run_epoch(
            model, val_loader, None, device, None, criterion, epoch, detailed_analysis=detailed
        )
        
        # Get current learning rate once
        current_lr = optimiser.param_groups[0]['lr']
        
        # Update SGDR cycle tracking and determine phase
        if use_cosine:
            # Standard cosine annealing phases
            phase = "warmup" if epoch <= WARMUP_EPOCHS else "cosine"
        else:
            # Update SGDR cycle tracking - detect restarts by LR increases
            if epoch > 1:
                if current_lr > previous_lr * 1.5:  # LR jumped up significantly = restart
                    sgdr_cycle_num += 1
                    sgdr_cycle_step = 0
                    sgdr_current_cycle_length = int(sgdr_current_cycle_length * sgdr_T_mult)
                else:
                    sgdr_cycle_step += len(train_loader)
            else:
                sgdr_cycle_step += len(train_loader)
                
            # SGDR phase detection based on cycle position
            cycle_progress = sgdr_cycle_step / sgdr_current_cycle_length if sgdr_current_cycle_length > 0 else 0
            
            if cycle_progress < 0.1:
                phase = "restart"  # Just restarted, LR climbing from minimum
            elif cycle_progress < 0.3:
                phase = "early"    # Early in cycle, LR still climbing
            elif cycle_progress < 0.7:
                phase = "mid"      # Middle of cycle, near maximum LR
            else:
                phase = "late"     # Late in cycle, LR falling toward minimum
                
            # Add cycle info to phase
            phase = f"{phase}-C{sgdr_cycle_num}"
            
            # Update previous LR for next iteration
            previous_lr = current_lr
        
        temp_str = f" | temp {model.temperature.item():.3f}" if TEMP_SCALING else ""
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
            
            # Show confusion for problematic classes (accuracy < 20%)
            if 'confusion_matrix' in analysis and 'per_class_accuracy' in analysis:
                print("\n🔍 Confusion Analysis for Low-Accuracy Classes:")
                
                problematic_classes = [
                    (class_name, acc) for class_name, acc in analysis['per_class_accuracy'].items() 
                    if acc < 0.20  # Less than 20% accuracy
                ]
                
                for class_name, acc in sorted(problematic_classes, key=lambda x: x[1]):
                    class_idx = UPOS_TAGS.index(class_name)
                    confusion_data = analysis['confusion_matrix'][class_idx]
                    
                    if confusion_data:  # Only show if there were instances of this class
                        total_instances = sum(confusion_data.values())
                        print(f"\n  📋 True={class_name} ({total_instances} instances, {acc*100:.1f}% correct):")
                        
                        # Show top 5 predictions for this true class
                        top_predictions = sorted(confusion_data.items(), key=lambda x: x[1], reverse=True)[:5]
                        for pred_idx, count in top_predictions:
                            if pred_idx < len(UPOS_TAGS):
                                pred_name = UPOS_TAGS[pred_idx]
                                percentage = (count / total_instances) * 100
                                marker = "✓" if pred_idx == class_idx else "✗"
                                print(f"    {marker} Predicted as {pred_name:8s}: {count:4d} ({percentage:5.1f}%)")
                
                # Also show precision analysis for the most confused predicted classes
                print("\n🎯 Precision Analysis for Most Confused Predictions:")
                
                # Calculate predictions totals (how many times each class was predicted)
                predicted_totals = defaultdict(int)
                for true_idx in analysis['confusion_matrix']:
                    for pred_idx, count in analysis['confusion_matrix'][true_idx].items():
                        predicted_totals[pred_idx] += count
                
                # For problematic true classes, show what makes up their main predicted classes
                for class_name, acc in sorted(problematic_classes, key=lambda x: x[1])[:3]:  # Top 3 worst
                    class_idx = UPOS_TAGS.index(class_name)
                    confusion_data = analysis['confusion_matrix'][class_idx]
                    
                    if confusion_data:
                        # Get the top 2 most common wrong predictions
                        wrong_predictions = [(pred_idx, count) for pred_idx, count in confusion_data.items() 
                                           if pred_idx != class_idx]
                        top_wrong = sorted(wrong_predictions, key=lambda x: x[1], reverse=True)[:2]
                        
                        for pred_idx, count in top_wrong:
                            if pred_idx < len(UPOS_TAGS) and predicted_totals[pred_idx] > 0:
                                pred_name = UPOS_TAGS[pred_idx]
                                
                                # Calculate what makes up this predicted class
                                pred_breakdown = defaultdict(int)
                                for true_idx in analysis['confusion_matrix']:
                                    if pred_idx in analysis['confusion_matrix'][true_idx]:
                                        pred_breakdown[true_idx] += analysis['confusion_matrix'][true_idx][pred_idx]
                                
                                total_pred = predicted_totals[pred_idx]
                                correct_pred = pred_breakdown.get(pred_idx, 0)
                                false_positive = total_pred - correct_pred
                                
                                precision = (correct_pred / total_pred) * 100 if total_pred > 0 else 0
                                false_pos_rate = (false_positive / total_pred) * 100 if total_pred > 0 else 0
                                
                                print(f"\n    🎯 Predicted as {pred_name} ({total_pred} total predictions):")
                                print(f"      ✓ Correct {pred_name}: {correct_pred:4d} ({precision:5.1f}% - True Positives)")
                                print(f"      ✗ Wrong {pred_name}:   {false_positive:4d} ({false_pos_rate:5.1f}% - False Positives)")
                                
                                # Show top contributors to false positives
                                false_contributors = [(true_idx, cnt) for true_idx, cnt in pred_breakdown.items() 
                                                    if true_idx != pred_idx]
                                top_contributors = sorted(false_contributors, key=lambda x: x[1], reverse=True)[:3]
                                
                                if top_contributors:
                                    print(f"      📊 Main false positive sources:")
                                    for true_idx, contrib_count in top_contributors:
                                        if true_idx < len(UPOS_TAGS):
                                            true_name = UPOS_TAGS[true_idx]
                                            contrib_pct = (contrib_count / total_pred) * 100
                                            print(f"         • True {true_name} → Pred {pred_name}: {contrib_count:3d} ({contrib_pct:4.1f}%)")
                
                # Special diagnostic for VERB class failure
                if 'VERB' in [i for i, _ in problematic_classes]:
                    print("\n🚨 VERB Diagnostic Analysis:")
                    verb_idx = UPOS_TAGS.index('VERB')
                    if verb_idx in analysis['confusion_matrix']:
                        verb_confusion = analysis['confusion_matrix'][verb_idx]
                        total_verbs = sum(verb_confusion.values())
                        
                        if total_verbs > 0:
                            print(f"   📊 VERB instances: {total_verbs}")
                            print(f"   🎯 Top VERB → ? confusions:")
                            
                            sorted_confusions = sorted(verb_confusion.items(), key=lambda x: x[1], reverse=True)
                            for pred_idx, count in sorted_confusions[:8]:  # Top 8
                                if pred_idx < len(UPOS_TAGS):
                                    pred_name = UPOS_TAGS[pred_idx]
                                    pct = (count / total_verbs) * 100
                                    status = "CORRECT" if pred_idx == verb_idx else "WRONG"
                                    print(f"      VERB → {pred_name:8s}: {count:3d} ({pct:5.1f}%) [{status}]")
                            
                            # Check if this is data imbalance
                            verb_acc = analysis['per_class_accuracy'].get('VERB', 0)
                            aux_acc = analysis['per_class_accuracy'].get('AUX', 0)
                            print(f"   ⚖️  VERB accuracy: {verb_acc*100:.1f}% vs AUX accuracy: {aux_acc*100:.1f}%")
                            
                            # Check if VERBs are being overwhelmed by other classes
                            all_class_totals = {}
                            for class_idx in range(len(UPOS_TAGS)):
                                if class_idx in analysis['confusion_matrix']:
                                    all_class_totals[UPOS_TAGS[class_idx]] = sum(analysis['confusion_matrix'][class_idx].values())
                            
                            if all_class_totals:
                                sorted_totals = sorted(all_class_totals.items(), key=lambda x: x[1], reverse=True)
                                print(f"   📈 Class frequency ranking:")
                                for rank, (class_name, total) in enumerate(sorted_totals[:10], 1):
                                    pct_of_dataset = (total / sum(all_class_totals.values())) * 100
                                    marker = "🎯" if class_name == 'VERB' else "  "
                                    print(f"      {marker} #{rank:2d}: {class_name:8s} - {total:4d} instances ({pct_of_dataset:5.1f}%)")
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
    if args.penn_treebank:
        model_name = "router_penn_wsj.pt"
    elif args.combined_penn:
        model_name = "router_combined_penn.pt"
    elif args.combine:
        model_name = "router_combined.pt"
    else:
        model_name = f"router_{args.treebank}.pt"
    
    torch.save(model.state_dict(), model_name)
    print(f"✓ finished; weights saved to {model_name}")

if __name__ == "__main__":
    main()
