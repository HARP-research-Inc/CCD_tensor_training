#!/usr/bin/env python3
"""
Modular POS Router Training Script

A clean, modular version of the POS tagger training with all functionality preserved.
Uses the new modular architecture with focused, reusable components.
"""

import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from datasets import load_dataset, Dataset
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, f1_score
from collections import defaultdict

# Import our modular components
from models.router import DepthWiseCNNRouter
from losses.label_smoothing import LabelSmoothingLoss
from training.early_stopping import EarlyStopping
from training.adaptive_batch import AdaptiveBatchSizer, create_adaptive_dataloader
from training.temperature import calibrate_temperature
from data.penn_treebank import load_penn_treebank_data
from data.preprocessing import (
    build_vocab, encode_sent, augment_dataset, 
    calculate_batch_size, collate
)

# Constants from original script
EMB_DIM = 48
DW_KERNEL = 3
N_TAGS = 18
LR_MAX = 7e-2
LR_MIN = 1e-4
EPOCHS = 100
WARMUP_EPOCHS = 3
MAX_LEN = 64
LABEL_SMOOTHING = 0.1
TEMP_SCALING = True

# Compute node optimizations
NUM_WORKERS_TRAIN = 48
NUM_WORKERS_VAL = 16
PREFETCH_FACTOR = 4
PIN_MEMORY = True

# Universal POS tag names
UPOS_TAGS = [
    "NOUN", "PUNCT", "ADP", "NUM", "SYM", "SCONJ", "ADJ", "PART", "DET", "CCONJ", 
    "PROPN", "PRON", "X", "_", "ADV", "INTJ", "VERB", "AUX"
]

def run_epoch(model, loader, optimiser=None, device="cpu", scaler=None, criterion=None, 
              epoch=None, detailed_analysis=False, calculate_f1=True, f1_average='macro'):
    """Run a single training or validation epoch with full detailed analysis."""
    train = optimiser is not None
    model.train() if train else model.eval()
    total_loss, total_tok, correct = 0.0, 0, 0
    
    # For detailed analysis
    all_preds = []
    all_targets = []
    per_class_stats = defaultdict(lambda: {'correct': 0, 'total': 0, 'loss': 0.0})
    confusion_matrix = defaultdict(lambda: defaultdict(int))

    # Create progress bar with epoch info
    mode = "Train" if train else "Val"
    desc = f"{mode} Epoch {epoch}" if epoch is not None else mode
    for ids, upos, mask in tqdm(loader, desc=desc, leave=True):
        # Non-blocking transfers to GPU
        ids = ids.to(device, non_blocking=True)
        upos = upos.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        
        if train and scaler is not None:
            # Mixed precision training
            with torch.cuda.amp.autocast():
                logp = model(ids, mask)
                if criterion is not None:
                    loss = criterion(logp.transpose(1,2), upos)
                else:
                    loss = F.nll_loss(logp.transpose(1,2), upos, reduction="sum", ignore_index=-100)
            
            optimiser.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
        else:
            # Standard training/validation
            # Apply temperature scaling during validation
            use_temp = not train and TEMP_SCALING
            logp = model(ids, mask, use_temperature=use_temp)
            if criterion is not None:
                loss = criterion(logp.transpose(1,2), upos)
            else:
                loss = F.nll_loss(logp.transpose(1,2), upos, reduction="sum", ignore_index=-100)
            
            if train:
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

        total_loss += loss.item()
        total_tok += mask.sum().item()
        pred = logp.argmax(-1)
        correct += ((pred == upos) & mask).sum().item()
        
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

    # Calculate metrics
    avg_loss_per_token = total_loss / total_tok if total_tok > 0 else float('inf')
    ppl = math.exp(avg_loss_per_token) if avg_loss_per_token < 100 else float('inf')
    acc = correct / total_tok if total_tok > 0 else 0.0
    
    # Calculate F1 score if requested
    f1_score_val = 0.0
    if calculate_f1 and all_preds and all_targets:
        try:
            f1_score_val = f1_score(all_targets, all_preds, average=f1_average, zero_division=0)
        except Exception as e:
            print(f"Warning: Could not calculate {f1_average} F1: {e}")
            f1_score_val = 0.0

    # Generate detailed analysis (restored from original)
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
            analysis['present_classes'] = present_classes
            
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
    
    return ppl, acc, f1_score_val, analysis

def run_epoch_with_sgdr(model, loader, optimiser, device, scaler, criterion, scheduler, step_count, epoch, 
                        adaptive_batch_sizer=None, train_dataset=None, collate_fn=None, 
                        num_workers=None, pin_memory=None, prefetch_factor=None):
    """Training epoch with SGDR step-based learning rate scheduling (restored from original)."""
    model.train()
    total_loss, total_tok, correct = 0.0, 0, 0
    current_loader = loader
    
    # Track batch size changes for reporting
    batch_size_changes = []
    
    # Create progress bar with epoch and batch size info
    initial_batch_size = adaptive_batch_sizer.get_current_batch_size() if adaptive_batch_sizer else 512
    desc = f"Train Epoch {epoch} [BS={initial_batch_size}]" if adaptive_batch_sizer else f"Train Epoch {epoch}"
    
    # Create progress bar object that we can update
    pbar = tqdm(current_loader, desc=desc, leave=True)
    for batch_idx, (ids, upos, mask) in enumerate(pbar):
        # Update batch size with adaptive batch sizer
        if adaptive_batch_sizer is not None:
            old_batch_size = adaptive_batch_sizer.get_current_batch_size()
            new_batch_size = adaptive_batch_sizer.update_batch_size(
                model, current_loader, device, criterion, epoch
            )
            
            # Recreate DataLoader if batch size changed significantly
            if abs(new_batch_size - old_batch_size) > 0.1 * old_batch_size and batch_idx < len(current_loader) - 1:
                batch_size_changes.append({
                    'batch_idx': batch_idx,
                    'old_size': old_batch_size,
                    'new_size': new_batch_size,
                    'stats': adaptive_batch_sizer.get_statistics()
                })
                
                # Update progress bar description with new batch size
                pbar.set_description(f"Train Epoch {epoch} [BS={new_batch_size}]")
        
        # Non-blocking transfers to GPU
        ids = ids.to(device, non_blocking=True)
        upos = upos.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        
        if scaler is not None:
            # Mixed precision training
            with torch.cuda.amp.autocast():
                logp = model(ids, mask)
                if criterion is not None:
                    loss = criterion(logp.transpose(1,2), upos)
                else:
                    loss = F.nll_loss(logp.transpose(1,2), upos, reduction="sum", ignore_index=-100)
            
            optimiser.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
        else:
            # Standard training
            logp = model(ids, mask)
            if criterion is not None:
                loss = criterion(logp.transpose(1,2), upos)
            else:
                loss = F.nll_loss(logp.transpose(1,2), upos, reduction="sum", ignore_index=-100)
            
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        # SGDR: Step the scheduler after each batch
        scheduler.step()
        step_count += 1

        total_loss += loss.item()
        total_tok += mask.sum().item()
        pred = logp.argmax(-1)
        correct += ((pred == upos) & mask).sum().item()

    # Calculate metrics
    avg_loss_per_token = total_loss / total_tok if total_tok > 0 else float('inf')
    ppl = math.exp(avg_loss_per_token) if avg_loss_per_token < 100 else float('inf')
    acc = correct / total_tok if total_tok > 0 else 0.0
    
    return ppl, acc, 0.0, step_count, batch_size_changes

def main():
    parser = argparse.ArgumentParser(
        description="Train POS tagger with intelligent batch sizing, early stopping, and advanced optimization techniques."
    )
    parser.add_argument("--treebank", default="en_ewt",
                        help="Any UD code accepted by datasets (e.g. en_ewt, en_gum, fr_sequoia)")
    parser.add_argument("--combine", action="store_true",
                        help="Combine multiple UD English treebanks ONLY (no Penn Treebank)")
    parser.add_argument("--penn-treebank", action="store_true",
                        help="Train on Penn Treebank WSJ ONLY (pure Penn Treebank training)")
    parser.add_argument("--combined-penn", action="store_true",
                        help="Combine UD treebanks WITH Penn Treebank (experimental)")
    parser.add_argument("--augment", action="store_true",
                        help="Apply data augmentation techniques")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override default batch size")
    
    # Early stopping arguments (restored from original)
    parser.add_argument("--fixed-epochs", action="store_true",
                        help="Use fixed epoch training instead of early stopping")
    parser.add_argument("--max-epochs", type=int, default=500,
                        help="Maximum epochs for early stopping (default: 500)")
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience (epochs without improvement, default: 15)")
    parser.add_argument("--min-delta", type=float, default=1e-4,
                        help="Minimum improvement to count as progress (default: 1e-4)")
    parser.add_argument("--monitor", type=str, default="macro_f1", 
                        choices=["val_loss", "val_acc", "val_ppl", "macro_f1", "weighted_f1"],
                        help="Metric to monitor for early stopping (default: macro_f1)")
    parser.add_argument("--use-weighted-f1", action="store_true",
                        help="Use weighted F1 instead of macro F1")
    parser.add_argument("--no-f1", action="store_true",
                        help="Disable F1 calculation entirely")
    
    # Restored missing arguments from original
    parser.add_argument("--no-label-smoothing", action="store_true",
                        help="Disable label smoothing")
    parser.add_argument("--no-temp-scaling", action="store_true",
                        help="Disable temperature scaling")
    parser.add_argument("--share", action="store_true",
                        help="If multiple GPUs are available, force use of cuda:1 (compute node sharing)")
    parser.add_argument("--gpu", type=int, default=None,
                        help="Force specific GPU ID (overrides auto-selection)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Override number of dataloader workers")
    parser.add_argument("--compute-node", action="store_true",
                        help="Enable all compute node optimizations")
    parser.add_argument("--penn-path", type=str, default=None,
                        help="Path to full Penn Treebank directory")
    parser.add_argument("--cosine", action="store_true",
                        help="Use standard Cosine Annealing scheduler (default: SGDR)")
    parser.add_argument("--sgdr-t0", type=int, default=None,
                        help="SGDR: Steps in first cycle (auto-detected if not provided)")
    parser.add_argument("--sgdr-mult", type=float, default=2.0,
                        help="SGDR: Cycle length multiplier (default: 2.0)")
    
    # Adaptive batch sizing arguments (restored from original)
    parser.add_argument("--adaptive-batch", action="store_true",
                        help="Enable CABS (Coupled Adaptive Batch Size) for better generalization")
    parser.add_argument("--noise-threshold", type=float, default=0.1,
                        help="Noise threshold θ for adaptive batch sizing (default: 0.1)")
    parser.add_argument("--min-batch-size", type=int, default=512,
                        help="Minimum batch size for adaptive sizing (default: 512)")
    parser.add_argument("--max-batch-adaptive", type=int, default=2048,
                        help="Maximum batch size for adaptive sizing (default: 2048)")
    parser.add_argument("--pilot-batch-size", type=int, default=1024,
                        help="Pilot batch size for gradient variance estimation (default: 1024)")
    parser.add_argument("--small-batch-early", action="store_true",
                        help="Start with small batches for better exploration")
    parser.add_argument("--variance-estimation-freq", type=int, default=5,
                        help="How often to re-estimate gradient variance (every N steps, default: 5)")
    
    args = parser.parse_args()
    
    # Validation (restored from original)
    if args.penn_treebank and args.combine:
        print("❌ ERROR: Cannot use --penn-treebank and --combine together!")
        exit(1)
    if args.penn_treebank and args.combined_penn:
        print("❌ ERROR: Cannot use --penn-treebank and --combined-penn together!")
        exit(1)
    if args.combine and args.combined_penn:
        print("❌ ERROR: Cannot use --combine and --combined-penn together!")
        exit(1)

    # GPU optimizations (restored from original)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_num_threads(min(32, torch.get_num_threads()))
    
    # Configure global variables
    global LABEL_SMOOTHING, TEMP_SCALING, NUM_WORKERS_TRAIN, NUM_WORKERS_VAL, PREFETCH_FACTOR
    if args.no_label_smoothing:
        LABEL_SMOOTHING = 0.0
    if args.no_temp_scaling:
        TEMP_SCALING = False
    if args.compute_node:
        NUM_WORKERS_TRAIN = min(56, NUM_WORKERS_TRAIN)
        PREFETCH_FACTOR = 6
    if args.workers:
        NUM_WORKERS_TRAIN = args.workers
        NUM_WORKERS_VAL = max(1, args.workers // 3)

    # Configure training strategy
    if args.fixed_epochs:
        MAX_EPOCHS = EPOCHS
        USE_EARLY_STOPPING = False
        CALCULATE_F1 = False
        F1_AVERAGE = 'macro'
        print(f"📅 Fixed epochs mode: training for exactly {MAX_EPOCHS} epochs")
    else:
        MAX_EPOCHS = args.max_epochs
        USE_EARLY_STOPPING = True
        
        if (args.monitor in ['macro_f1', 'weighted_f1']) and args.no_f1:
            print("⚠️  Warning: F1 monitoring conflicts with --no-f1")
            args.monitor = 'val_acc'
        
        CALCULATE_F1 = (args.monitor in ['macro_f1', 'weighted_f1']) and not args.no_f1
        F1_AVERAGE = 'weighted' if args.use_weighted_f1 or args.monitor == 'weighted_f1' else 'macro'
        
        print(f"🛑 Early stopping mode: max {MAX_EPOCHS} epochs, patience {args.patience}")
        print(f"   📊 Monitoring: {args.monitor}")
        if CALCULATE_F1:
            print(f"   🎯 {F1_AVERAGE.title()} F1 calculation enabled")

    print("Loading dataset …")
    
    # Dataset loading (restored from original)
    if args.penn_treebank:
        print("🏛️  Penn Treebank WSJ Training Mode (PURE)")
        train_data, val_data, test_data = load_penn_treebank_data(args.penn_path)
        ds_train = Dataset.from_list(train_data)
        ds_val = Dataset.from_list(val_data)
        print(f"📊 Penn Treebank: {len(ds_train)} train, {len(ds_val)} val sentences")
        
    elif args.combined_penn:
        print("🔬 EXPERIMENTAL: Combined UD + Penn Treebank Training")
        # Load UD first
        english_treebanks = ["en_ewt", "en_gum", "en_lines", "en_partut"]
        combined_train = []
        combined_val = []
        
        for tb in english_treebanks:
            try:
                ds_train_tb = load_dataset("universal_dependencies", tb, split="train", trust_remote_code=True)
                ds_val_tb = load_dataset("universal_dependencies", tb, split="validation", trust_remote_code=True)
                combined_train.extend(list(ds_train_tb))
                combined_val.extend(list(ds_val_tb))
                print(f"  ✓ Loaded UD {tb}: {len(ds_train_tb)} train, {len(ds_val_tb)} val sentences")
            except Exception as e:
                print(f"  ❌ Failed to load UD {tb}: {e}")
        
        # Add Penn Treebank
        penn_train, penn_val, penn_test = load_penn_treebank_data(args.penn_path)
        combined_train.extend(penn_train)
        combined_val.extend(penn_val)
        print(f"  ✓ Added Penn Treebank: {len(penn_train)} train, {len(penn_val)} val sentences")
        
        ds_train = Dataset.from_list(combined_train)
        ds_val = Dataset.from_list(combined_val)
        print(f"🎯 Combined UD+Penn dataset: {len(ds_train)} train, {len(ds_val)} val sentences")
        
    elif args.combine:
        print("🔥 Combined UD Training Mode (NO Penn Treebank)")
        english_treebanks = ["en_ewt", "en_gum", "en_lines", "en_partut"]
        combined_train = []
        combined_val = []
        
        for tb in english_treebanks:
            try:
                ds_train_tb = load_dataset("universal_dependencies", tb, split="train", trust_remote_code=True)
                ds_val_tb = load_dataset("universal_dependencies", tb, split="validation", trust_remote_code=True)
                combined_train.extend(list(ds_train_tb))
                combined_val.extend(list(ds_val_tb))
                print(f"  ✓ Loaded {tb}: {len(ds_train_tb)} train, {len(ds_val_tb)} val sentences")
            except Exception as e:
                print(f"  ❌ Failed to load {tb}: {e}")
        
        ds_train = Dataset.from_list(combined_train)
        ds_val = Dataset.from_list(combined_val)
        print(f"🎯 Combined dataset: {len(ds_train)} train, {len(ds_val)} val sentences")
        
    else:
        ds_train = load_dataset("universal_dependencies", args.treebank, split="train", trust_remote_code=True)
        ds_val = load_dataset("universal_dependencies", args.treebank, split="validation", trust_remote_code=True)
        print(f"📊 Single treebank {args.treebank}: {len(ds_train)} train, {len(ds_val)} val sentences")

    # Data processing
    vocab = build_vocab(ds_train)
    if args.augment:
        ds_train = augment_dataset(ds_train, augment_factor=1.5)
        if not hasattr(ds_train, 'map'):
            ds_train = Dataset.from_list(ds_train)
    
    train_enc = ds_train.map(lambda ex: encode_sent(ex, vocab))
    val_enc = ds_val.map(lambda ex: encode_sent(ex, vocab))
    train_enc = train_enc.with_format("torch", columns=["ids", "upos"], output_all_columns=True)
    val_enc = val_enc.with_format("torch", columns=["ids", "upos"], output_all_columns=True)

    # Batch sizing strategy (restored from original)
    adaptive_batch_sizer = None
    
    if args.adaptive_batch:
        adaptive_batch_sizer = AdaptiveBatchSizer(
            min_batch_size=args.min_batch_size,
            max_batch_size=args.max_batch_adaptive,
            noise_threshold=args.noise_threshold,
            pilot_batch_size=args.pilot_batch_size,
            small_batch_early=args.small_batch_early,
            variance_estimation_freq=args.variance_estimation_freq
        )
        BATCH_SIZE = adaptive_batch_sizer.get_current_batch_size()
        print(f"📦 Adaptive batch sizing: starts at {BATCH_SIZE}")
        
        train_loader = create_adaptive_dataloader(
            train_enc, adaptive_batch_sizer, collate, NUM_WORKERS_TRAIN, PIN_MEMORY, PREFETCH_FACTOR
        )
        val_batch_size = min(args.pilot_batch_size, len(ds_val) // 10) if len(ds_val) > 100 else len(ds_val)
        val_loader = DataLoader(
            val_enc, batch_size=val_batch_size, shuffle=False,
            collate_fn=collate, num_workers=NUM_WORKERS_VAL, pin_memory=PIN_MEMORY,
            prefetch_factor=PREFETCH_FACTOR, persistent_workers=True
        )
    else:
        if args.batch_size:
            BATCH_SIZE = args.batch_size
        else:
            BATCH_SIZE = calculate_batch_size(len(ds_train))
        
        train_loader = DataLoader(
            train_enc, batch_size=BATCH_SIZE, shuffle=True,
            collate_fn=collate, num_workers=NUM_WORKERS_TRAIN, pin_memory=PIN_MEMORY,
            prefetch_factor=PREFETCH_FACTOR, persistent_workers=True, drop_last=False
        )
        val_loader = DataLoader(
            val_enc, batch_size=BATCH_SIZE, shuffle=False,
            collate_fn=collate, num_workers=NUM_WORKERS_VAL, pin_memory=PIN_MEMORY,
            prefetch_factor=PREFETCH_FACTOR, persistent_workers=True
        )

    # GPU selection (restored from original)
    if torch.cuda.is_available():
        ngpu = torch.cuda.device_count()
        print(f"🖥️  Compute node detected: {ngpu} GPU(s) available")
        
        if args.gpu is not None:
            selected = args.gpu if args.gpu < ngpu else 0
        elif args.share and ngpu >= 2:
            selected = 1
        elif ngpu >= 4:
            # Select GPU with lowest memory usage
            try:
                gpu_memory = []
                for i in range(ngpu):
                    torch.cuda.set_device(i)
                    allocated = torch.cuda.memory_allocated(i)
                    total = torch.cuda.get_device_properties(i).total_memory
                    usage = allocated / total
                    gpu_memory.append((i, usage))
                selected = min(gpu_memory, key=lambda x: x[1])[0]
                print(f"🎯 Auto-selected GPU {selected} (lowest memory usage)")
            except:
                selected = 0
        else:
            selected = 0
            
        torch.cuda.set_device(selected)
        device = torch.device(f"cuda:{selected}")
        
        gpu_props = torch.cuda.get_device_properties(selected)
        memory_gb = gpu_props.total_memory / 1024**3
        print(f"🚀 Using {gpu_props.name} (GPU {selected})")
        print(f"💾 GPU Memory: {memory_gb:.1f}GB total")
    else:
        device = torch.device("cpu")
        print("⚠️  CUDA not available, falling back to CPU")

    # Model setup
    model = DepthWiseCNNRouter(len(vocab)).to(device)
    optimiser = optim.AdamW(model.parameters(), lr=LR_MIN, weight_decay=1e-4)

    criterion = None
    if LABEL_SMOOTHING > 0:
        criterion = LabelSmoothingLoss(smoothing=LABEL_SMOOTHING)
        print(f"📊 Using label smoothing with α={LABEL_SMOOTHING}")
    
    # Mixed precision scaler
    if device.type.startswith("cuda"):
        try:
            from torch.amp import GradScaler
            scaler = GradScaler('cuda')
        except ImportError:
            from torch.cuda.amp import GradScaler
            scaler = GradScaler()
    else:
        scaler = None

    # Scheduler setup (restored from original)
    use_cosine = args.cosine
    
    if use_cosine:
        def get_lr(epoch):
            if epoch < WARMUP_EPOCHS:
                return LR_MIN + (LR_MAX - LR_MIN) * epoch / WARMUP_EPOCHS
            progress = (epoch - WARMUP_EPOCHS) / (EPOCHS - WARMUP_EPOCHS)
            return LR_MIN + (LR_MAX - LR_MIN) * 0.5 * (1 + math.cos(math.pi * progress))
        
        scheduler = optim.lr_scheduler.LambdaLR(optimiser, lambda epoch: get_lr(epoch) / LR_MIN)
        print(f"📈 Standard Cosine Annealing: {LR_MIN:.1e} → {LR_MAX:.1e} → {LR_MIN:.1e}")
    else:
        # SGDR (default)
        T_0 = args.sgdr_t0 if args.sgdr_t0 else max(10, len(train_loader) // 4)
        
        if args.penn_treebank and len(train_loader) < 200:
            T_0 = args.sgdr_t0 if args.sgdr_t0 else max(5, len(train_loader) // 6)
            T_mult = args.sgdr_mult if hasattr(args, 'sgdr_mult') else 1.5
            eta_min_ratio = 0.01
        else:
            T_mult = args.sgdr_mult
            eta_min_ratio = 0.1
        
        eta_min = LR_MAX * eta_min_ratio
        
        print(f"🔄 SGDR Scheduler (default):")
        print(f"   • First cycle: {T_0} steps")
        print(f"   • Cycle multiplier: {T_mult}x") 
        print(f"   • LR range: {eta_min:.1e} → {LR_MAX:.1e}")
        
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimiser, T_0=T_0, T_mult=int(T_mult) if T_mult == int(T_mult) else 2,
            eta_min=eta_min, last_epoch=-1
        )
    
    # Performance monitoring (restored from original)
    print(f"\n📊 COMPUTE NODE PERFORMANCE BASELINE:")
    print(f"   • Dataset size: {len(ds_train):,} train, {len(ds_val):,} val sentences")
    print(f"   • Vocabulary size: {len(vocab):,} tokens")
    print(f"   • Steps per epoch: {len(train_loader):,}")
    
    # Check which POS classes are present (restored from original)
    print(f"   • POS tag analysis:")
    sample_upos = []
    for i, batch in enumerate(train_loader):
        if i >= 5:
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
    
    print()

    # Training setup
    step_count = 0
    early_stopping = None
    if USE_EARLY_STOPPING:
        early_stopping = EarlyStopping(
            patience=args.patience,
            min_delta=args.min_delta,
            monitor=args.monitor,
            restore_best_weights=True,
            verbose=False
        )
    
    # SGDR cycle tracking (restored from original)
    if not use_cosine:
        sgdr_T_0 = args.sgdr_t0 if args.sgdr_t0 else max(10, len(train_loader) // 4)
        if args.penn_treebank and len(train_loader) < 200:
            sgdr_T_0 = args.sgdr_t0 if args.sgdr_t0 else max(5, len(train_loader) // 6)
            sgdr_T_mult = args.sgdr_mult if hasattr(args, 'sgdr_mult') else 1.5
        else:
            sgdr_T_mult = args.sgdr_mult
        
        sgdr_current_cycle_length = sgdr_T_0
        sgdr_cycle_step = 0
        sgdr_cycle_num = 1
        previous_lr = LR_MIN
    
    # Training loop (restored from original with full functionality)
    for epoch in range(1, MAX_EPOCHS + 1):
        if use_cosine:
            train_ppl, train_acc, _, _ = run_epoch(
                model, train_loader, optimiser, device, scaler, criterion, epoch, 
                detailed_analysis=False, calculate_f1=False
            )
            scheduler.step()
        else:
            # SGDR training
            if adaptive_batch_sizer is not None:
                train_ppl, train_acc, _, step_count, batch_changes = run_epoch_with_sgdr(
                    model, train_loader, optimiser, device, scaler, criterion, scheduler, step_count, epoch,
                    adaptive_batch_sizer, train_enc, collate, NUM_WORKERS_TRAIN, PIN_MEMORY, PREFETCH_FACTOR
                )
                
                if batch_changes:
                    for change in batch_changes[-1:]:
                        stats = change['stats']
                        noise_str = f"(noise ratio: {stats['noise_ratio']:.3f})" if stats.get('noise_ratio') else ""
                        print(f"   📦 Batch size: {change['old_size']} → {change['new_size']} {noise_str}")
                
                current_batch_size = adaptive_batch_sizer.get_current_batch_size()
                if current_batch_size != BATCH_SIZE:
                    BATCH_SIZE = current_batch_size
                    train_loader = create_adaptive_dataloader(
                        train_enc, adaptive_batch_sizer, collate, NUM_WORKERS_TRAIN, PIN_MEMORY, PREFETCH_FACTOR
                    )
            else:
                train_ppl, train_acc, _, step_count, _ = run_epoch_with_sgdr(
                    model, train_loader, optimiser, device, scaler, criterion, scheduler, step_count, epoch
                )
        
        # Validation with detailed analysis every 10 epochs
        detailed = (epoch % 10 == 0) or (epoch == MAX_EPOCHS)
        val_ppl, val_acc, f1_score_val, analysis = run_epoch(
            model, val_loader, None, device, None, criterion, epoch, 
            detailed_analysis=detailed, calculate_f1=CALCULATE_F1, f1_average=F1_AVERAGE
        )
        
        current_lr = optimiser.param_groups[0]['lr']
        
        # SGDR phase tracking (restored from original)
        if use_cosine:
            phase = "warmup" if epoch <= WARMUP_EPOCHS else "cosine"
        else:
            if epoch > 1:
                if current_lr > previous_lr * 1.5:
                    sgdr_cycle_num += 1
                    sgdr_cycle_step = 0
                    sgdr_current_cycle_length = int(sgdr_current_cycle_length * sgdr_T_mult)
                else:
                    sgdr_cycle_step += len(train_loader)
            else:
                sgdr_cycle_step += len(train_loader)
                
            cycle_progress = sgdr_cycle_step / sgdr_current_cycle_length if sgdr_current_cycle_length > 0 else 0
            
            if cycle_progress < 0.1:
                phase = "restart"
            elif cycle_progress < 0.3:
                phase = "early"
            elif cycle_progress < 0.7:
                phase = "mid"
            else:
                phase = "late"
                
            phase = f"{phase}-C{sgdr_cycle_num}"
            previous_lr = current_lr
        
        # Temperature display
        temp_val = model.temperature.item() if TEMP_SCALING else 1.0
        temp_str = f" | temp {temp_val:.3f}" if TEMP_SCALING and abs(temp_val - 1.0) > 0.01 else ""
        
        # Print progress (restored from original format)
        if CALCULATE_F1 and f1_score_val > 0:
            if args.monitor in ['macro_f1', 'weighted_f1']:
                print(f"epoch {epoch:02d} | "
                      f"train acc {train_acc*100:5.2f}% | "
                      f"{F1_AVERAGE[0].upper()}F1 {f1_score_val*100:5.2f}% (acc {val_acc*100:4.1f}%) | "
                      f"val ppl {val_ppl:4.2f} | "
                      f"lr {current_lr:.1e} ({phase}){temp_str}")
            else:
                print(f"epoch {epoch:02d} | "
                      f"train acc {train_acc*100:5.2f}% | "
                      f"val acc {val_acc*100:5.2f}% ({F1_AVERAGE[0].upper()}F1 {f1_score_val*100:4.1f}%) | "
                      f"val ppl {val_ppl:4.2f} | "
                      f"lr {current_lr:.1e} ({phase}){temp_str}")
        else:
            print(f"epoch {epoch:02d} | "
                f"train acc {train_acc*100:5.2f}% | "
                f"val acc {val_acc*100:5.2f}% | "
                f"val ppl {val_ppl:4.2f} | "
                f"lr {current_lr:.1e} ({phase}){temp_str}")
        
        # Check early stopping
        if early_stopping is not None:
            val_loss_for_es = val_ppl
            if early_stopping(epoch, val_loss_for_es, val_acc, val_ppl, f1_score_val, model):
                print(f"\n🛑 Training stopped early at epoch {epoch}")
                break
        
        # Periodic temperature calibration (restored from original)
        if TEMP_SCALING and CALCULATE_F1 and epoch % 20 == 0 and epoch > 20:
            print(f"\n🌡️  Calibrating temperature at epoch {epoch}...")
            old_temp = model.temperature.item()
            calibrate_temperature(model, val_loader, device, verbose=False)
            new_temp = model.temperature.item()
            if abs(new_temp - old_temp) > 0.01:
                print(f"   Temperature updated: {old_temp:.3f} → {new_temp:.3f}")
                print("   🔍 Re-evaluating with updated temperature...")
                val_ppl, val_acc, f1_score_val, _ = run_epoch(
                    model, val_loader, None, device, None, criterion, epoch, 
                    detailed_analysis=False, calculate_f1=CALCULATE_F1, f1_average=F1_AVERAGE
                )
                print(f"   📊 Results with new temperature: acc {val_acc*100:.2f}%, "
                      f"{F1_AVERAGE[0].upper()}F1 {f1_score_val*100:.2f}%")
            else:
                print(f"   Temperature unchanged: {old_temp:.3f}")
        
        # Print detailed analysis (FULLY RESTORED from original)
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
                
                # Precision analysis for confused predictions
                print("\n🎯 Precision Analysis for Most Confused Predictions:")
                
                predicted_totals = defaultdict(int)
                for true_idx in analysis['confusion_matrix']:
                    for pred_idx, count in analysis['confusion_matrix'][true_idx].items():
                        predicted_totals[pred_idx] += count
                
                for class_name, acc in sorted(problematic_classes, key=lambda x: x[1])[:3]:  # Top 3 worst
                    class_idx = UPOS_TAGS.index(class_name)
                    confusion_data = analysis['confusion_matrix'][class_idx]
                    
                    if confusion_data:
                        wrong_predictions = [(pred_idx, count) for pred_idx, count in confusion_data.items() 
                                           if pred_idx != class_idx]
                        top_wrong = sorted(wrong_predictions, key=lambda x: x[1], reverse=True)[:2]
                        
                        for pred_idx, count in top_wrong:
                            if pred_idx < len(UPOS_TAGS) and predicted_totals[pred_idx] > 0:
                                pred_name = UPOS_TAGS[pred_idx]
                                
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
                            for pred_idx, count in sorted_confusions[:8]:
                                if pred_idx < len(UPOS_TAGS):
                                    pred_name = UPOS_TAGS[pred_idx]
                                    pct = (count / total_verbs) * 100
                                    status = "CORRECT" if pred_idx == verb_idx else "WRONG"
                                    print(f"      VERB → {pred_name:8s}: {count:3d} ({pct:5.1f}%) [{status}]")
                            
                            verb_acc = analysis['per_class_accuracy'].get('VERB', 0)
                            aux_acc = analysis['per_class_accuracy'].get('AUX', 0)
                            print(f"   ⚖️  VERB accuracy: {verb_acc*100:.1f}% vs AUX accuracy: {aux_acc*100:.1f}%")
                            
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
                macro_f1_from_report = analysis['classification_report']['macro avg']['f1-score']
                weighted_f1_from_report = analysis['classification_report']['weighted avg']['f1-score']
                print(f"\n📋 Macro F1: {macro_f1_from_report*100:.2f}%")
                print(f"   Weighted F1: {weighted_f1_from_report*100:.2f}%")
                if CALCULATE_F1:
                    expected_f1 = weighted_f1_from_report if F1_AVERAGE == 'weighted' else macro_f1_from_report
                    if abs(f1_score_val - expected_f1) < 0.01:
                        print(f"   ✓ {F1_AVERAGE.title()} F1 calculation verified")
            print()

    # Final steps (restored from original)
    if early_stopping is not None:
        if early_stopping.best_weights is not None:
            early_stopping.restore_best(model, device)
            
        stats = early_stopping.get_stats()
        print(f"\n📊 Early Stopping Summary:")
        print(f"   • Training stopped at epoch: {stats['stopped_epoch']}")
        print(f"   • Best epoch: {stats['best_epoch']}")
        print(f"   • Best {stats['monitor']}: {stats['best_value']:.4f}")
        print(f"   • Patience used: {stats['patience_used']}/{early_stopping.patience}")
    
    # Final temperature calibration
    if TEMP_SCALING:
        print("\n🌡️  Final temperature calibration...")
        calibrate_temperature(model, val_loader, device, verbose=True)
        
        print("🔍 Re-evaluating with calibrated temperature...")
        cal_ppl, cal_acc, cal_f1_score, cal_analysis = run_epoch(
            model, val_loader, None, device, None, criterion, 
            epoch=MAX_EPOCHS, detailed_analysis=True, calculate_f1=CALCULATE_F1, f1_average=F1_AVERAGE
        )
        cal_f1_str = f", {F1_AVERAGE[0].upper()}F1 {cal_f1_score*100:.2f}%" if CALCULATE_F1 and cal_f1_score > 0 else ""
        print(f"📊 Calibrated results: acc {cal_acc*100:.2f}%{cal_f1_str}, ppl {cal_ppl:.2f}")

    # Save model (restored from original)
    if args.penn_treebank:
        model_name = "router_penn_wsj.pt"
    elif args.combined_penn:
        model_name = "router_combined_penn.pt"
    elif args.combine:
        model_name = "router_combined.pt"
    else:
        model_name = f"router_{args.treebank}.pt"
    
    torch.save(model.state_dict(), model_name)
    
    training_method = "early stopping" if USE_EARLY_STOPPING else "fixed epochs"
    print(f"✓ finished; weights saved to {model_name} (trained with {training_method})")
    
    if early_stopping and early_stopping.best_weights:
        print(f"💡 Model contains best weights from epoch {early_stopping.best_epoch}")

if __name__ == "__main__":
    main() 