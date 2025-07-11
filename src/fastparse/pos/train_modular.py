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
import json
import os
from datetime import datetime

# Import our modular components
from models.router import DepthWiseCNNRouter
from losses.label_smoothing import LabelSmoothingLoss
from losses.class_balanced import create_class_balanced_loss
from training.early_stopping import EarlyStopping
from training.adaptive_batch import AdaptiveBatchSizer, create_adaptive_dataloader
from training.temperature import calibrate_temperature
from data.penn_treebank import load_penn_treebank_data
from data.preprocessing import (
    build_vocab, encode_sent, augment_dataset, 
    calculate_batch_size, collate,
    encode_sent_with_attrs, collate_with_attrs  # Hash-based embedding support
)

# Constants from original script
EMB_DIM = 48
DW_KERNEL = 5
N_TAGS = 18
LR_MAX = 7e-2
LR_MIN = 1e-5
EPOCHS = 30
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

def create_model_directory():
    """Create models directory if it doesn't exist."""
    os.makedirs("models", exist_ok=True)
    return "models"

def save_model_config(model_name, args, vocab, dataset_info, architecture_info):
    """Create and save comprehensive model configuration JSON."""
    config = {
        "model_name": model_name,
        "description": _get_model_description(args),
        "created_at": datetime.now().isoformat(),
        "architecture": {
            "type": "DepthWiseCNNRouter",
            "emb_dim": args.hash_dim if args.hash_embed else EMB_DIM,
            "dw_kernel": DW_KERNEL,
            "n_tags": N_TAGS,
            "max_len": MAX_LEN,
            "use_two_layers": True,  # Current architecture has 2 layers
            "use_temperature_scaling": not args.no_temp_scaling,
            "use_hash_embed": args.hash_embed,
            "hash_dim": args.hash_dim if args.hash_embed else None,
            "num_buckets": args.num_buckets if args.hash_embed else None,
            "ngram_min": args.ngram_min if args.hash_embed else None,
            "ngram_max": args.ngram_max if args.hash_embed else None,
            "dropout_rate": 0.1,
            "activation": "ReLU",
            "normalization": "LayerNorm"
        },
        "vocabulary": {
            "size": len(vocab) if not args.hash_embed else None,
            "type": "hash_based" if args.hash_embed else _get_vocab_type(args),
            "treebanks": _get_treebanks_used(args),
            "pad_token": "<PAD>",
            "augmented": args.augment,
            "penn_treebank_included": args.penn_treebank or args.combined_penn,
            "hash_based": args.hash_embed
        },
        "training": {
            "dataset_size": dataset_info,
            "label_smoothing": LABEL_SMOOTHING if not args.no_label_smoothing else 0.0,
            "temperature_scaling": not args.no_temp_scaling,
            "lr_max": LR_MAX,
            "lr_min": LR_MIN,
            "epochs": EPOCHS,
            "warmup_epochs": WARMUP_EPOCHS,
            "scheduler": "SGDR" if not args.cosine else "cosine",
            "mixed_precision": True,
            "early_stopping": not args.fixed_epochs,
            "monitor_metric": args.monitor if not args.fixed_epochs else None,
            "patience": args.patience if not args.fixed_epochs else None
        },
        "pos_tags": {
            "tagset": "Universal Dependencies",
            "tags": UPOS_TAGS,
            "count": len(UPOS_TAGS)
        },
        "inference": {
            "default_batch_size": 512,
            "enable_temperature": not args.no_temp_scaling,
            "enable_amp": False  # Disabled for inference stability
        },
        "files": {
            "model_weights": f"{model_name}.pt",
            "config": f"{model_name}.json",
            "vocabulary": f"{model_name}_vocab.json",
            "training_log": f"{model_name}_training.json"
        }
    }
    return config

def save_vocabulary_json(vocab, model_dir, model_name):
    """Save vocabulary as a JSON file."""
    vocab_file = os.path.join(model_dir, f"{model_name}_vocab.json")
    
    # Create vocabulary mapping (both directions)
    vocab_data = {
        "vocab_size": len(vocab),
        "token_to_id": vocab,
        "id_to_token": {str(v): k for k, v in vocab.items()},
        "special_tokens": {
            "pad_token": "<PAD>",
            "pad_id": 0
        },
        "created_at": datetime.now().isoformat()
    }
    
    with open(vocab_file, 'w', encoding='utf-8') as f:
        json.dump(vocab_data, f, indent=2, ensure_ascii=False)
    
    return vocab_file

def save_training_results(model_dir, model_name, training_history, final_results, args):
    """Save comprehensive training results and metrics."""
    training_file = os.path.join(model_dir, f"{model_name}_training.json")
    
    training_data = {
        "training_completed_at": datetime.now().isoformat(),
        "command_line_args": vars(args),
        "final_results": final_results,
        "training_history": training_history,
        "hyperparameters": {
            "architecture": {
                "emb_dim": EMB_DIM,
                "dw_kernel": DW_KERNEL,
                "n_tags": N_TAGS,
                "max_len": MAX_LEN
            },
            "training": {
                "lr_max": LR_MAX,
                "lr_min": LR_MIN,
                "epochs": EPOCHS,
                "warmup_epochs": WARMUP_EPOCHS,
                "label_smoothing": LABEL_SMOOTHING if not args.no_label_smoothing else 0.0,
                "batch_size": getattr(args, 'final_batch_size', args.batch_size),
                "scheduler": "SGDR" if not args.cosine else "cosine"
            },
            "optimization": {
                "mixed_precision": True,
                "optimizer": "AdamW",
                "weight_decay": 1e-4,
                "compute_node_optimizations": args.compute_node
            }
        },
        "environment": {
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
    }
    
    with open(training_file, 'w') as f:
        json.dump(training_data, f, indent=2)
    
    return training_file

def _get_model_description(args):
    """Generate a descriptive model name based on training configuration."""
    if args.penn_treebank:
        desc = "Pure Penn Treebank WSJ POS Router"
    elif args.combined_penn:
        desc = "Combined UD + Penn Treebank POS Router"
    elif args.combine:
        desc = "Combined Universal Dependencies English POS Router"
    else:
        desc = f"Universal Dependencies {args.treebank.upper()} POS Router"
    
    # Add embedding type
    if args.hash_embed:
        desc += " (Hash Embeddings)"
    
    # Add training method
    if args.fixed_epochs:
        desc += f" (Fixed {EPOCHS} epochs)"
    else:
        desc += f" (Early stopping, {args.monitor})"
    
    # Add special features
    features = []
    if args.hash_embed:
        features.append(f"Hash-{args.hash_dim}D")
    if args.adaptive_batch:
        features.append("CABS")
    if not args.no_temp_scaling:
        features.append("TempScaling")
    if args.augment:
        features.append("Augmented")
    if args.cosine:
        features.append("Cosine")
    else:
        features.append("SGDR")
    
    if features:
        desc += f" [{', '.join(features)}]"
    
    return desc

def _get_vocab_type(args):
    """Determine vocabulary type from arguments."""
    if args.penn_treebank:
        return "penn_treebank"
    elif args.combined_penn:
        return "combined_ud_penn"
    elif args.combine:
        return "combined_ud"
    else:
        return "single_treebank"

def _get_treebanks_used(args):
    """Get list of treebanks used for training."""
    if args.penn_treebank:
        return ["penn_wsj"]
    elif args.combined_penn:
        return ["en_ewt", "en_gum", "en_lines", "en_partut", "penn_wsj"]
    elif args.combine:
        return ["en_ewt", "en_gum", "en_lines", "en_partut"]
    else:
        return [args.treebank]

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
    for batch_data in tqdm(loader, desc=desc, leave=True):
        inputs, upos, mask = batch_data
        
        # Handle both traditional (tensor) and hash-based (list) inputs
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.to(device, non_blocking=True)
        # For hash embeddings, inputs is a list and doesn't need GPU transfer
        
        upos = upos.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        
        if train and scaler is not None:
            # Mixed precision training
            with torch.amp.autocast('cuda'):
                logp = model(inputs, mask)
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
            logp = model(inputs, mask, use_temperature=use_temp)
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
    for batch_idx, batch_data in enumerate(pbar):
        inputs, upos, mask = batch_data
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
        # Handle both traditional (tensor) and hash-based (list) inputs
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.to(device, non_blocking=True)
        # For hash embeddings, inputs is a list and doesn't need GPU transfer
        
        upos = upos.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        
        if scaler is not None:
            # Mixed precision training
            with torch.amp.autocast('cuda'):
                logp = model(inputs, mask)
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
            logp = model(inputs, mask)
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
    
    # Output directory options
    parser.add_argument("--model-dir", default="models",
                        help="Directory to save model outputs (default: models)")
    parser.add_argument("--model-prefix", default=None,
                        help="Custom prefix for model files (default: auto-generated)")
    
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
    
    # Hash-based embedding arguments
    parser.add_argument("--hash-embed", action="store_true",
                        help="Use hash-based embeddings instead of vocabulary-based (spaCy-style)")
    parser.add_argument("--hash-dim", type=int, default=96,
                        help="Hash embedding dimension (default: 96, spaCy default)")
    parser.add_argument("--num-buckets", type=int, default=1048576,
                        help="Number of hash buckets (default: 1048576 = 2^20)")
    parser.add_argument("--ngram-min", type=int, default=3,
                        help="Minimum character n-gram length (default: 3)")
    parser.add_argument("--ngram-max", type=int, default=5,
                        help="Maximum character n-gram length (default: 5)")
    parser.add_argument("--class-balanced", action="store_true",
                        help="Use class-balanced loss with inverse log frequency weighting")
    
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

    # Create model output directory
    model_dir = args.model_dir
    os.makedirs(model_dir, exist_ok=True)
    print(f"📁 Model outputs will be saved to: {model_dir}/")

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
    if args.hash_embed:
        print("🎯 Using hash-based embeddings (vocabulary-free)")
        print(f"   • Hash dimension: {args.hash_dim}")
        print(f"   • Hash buckets: {args.num_buckets:,}")
        print(f"   • Character n-grams: {args.ngram_min}-{args.ngram_max}")
        vocab = {"<PAD>": 0}  # Minimal vocab for compatibility
    else:
        print("📚 Using vocabulary-based embeddings")
        vocab = build_vocab(ds_train)
        print(f"   • Vocabulary size: {len(vocab):,} tokens")
    
    if args.augment:
        ds_train = augment_dataset(ds_train, augment_factor=1.5)
        if not hasattr(ds_train, 'map'):
            ds_train = Dataset.from_list(ds_train)
    
    if args.hash_embed:
        # Hash-based encoding
        train_enc = ds_train.map(lambda ex: encode_sent_with_attrs(ex, args.ngram_min, args.ngram_max))
        val_enc = ds_val.map(lambda ex: encode_sent_with_attrs(ex, args.ngram_min, args.ngram_max))
        train_enc = train_enc.with_format("torch", columns=["attrs", "upos"], output_all_columns=True)
        val_enc = val_enc.with_format("torch", columns=["attrs", "upos"], output_all_columns=True)
    else:
        # Traditional vocabulary-based encoding
        train_enc = ds_train.map(lambda ex: encode_sent(ex, vocab))
        val_enc = ds_val.map(lambda ex: encode_sent(ex, vocab))
        train_enc = train_enc.with_format("torch", columns=["ids", "upos"], output_all_columns=True)
        val_enc = val_enc.with_format("torch", columns=["ids", "upos"], output_all_columns=True)

    # Batch sizing strategy (restored from original)
    adaptive_batch_sizer = None
    
    # Choose appropriate collate function
    collate_fn = collate_with_attrs if args.hash_embed else collate
    
    if args.adaptive_batch and not args.hash_embed:
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
            train_enc, adaptive_batch_sizer, collate_fn, NUM_WORKERS_TRAIN, PIN_MEMORY, PREFETCH_FACTOR
        )
        val_batch_size = min(args.pilot_batch_size, len(ds_val) // 10) if len(ds_val) > 100 else len(ds_val)
        val_loader = DataLoader(
            val_enc, batch_size=val_batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=NUM_WORKERS_VAL, pin_memory=PIN_MEMORY,
            prefetch_factor=PREFETCH_FACTOR, persistent_workers=True
        )
    elif args.adaptive_batch and args.hash_embed:
        print("⚠️  Adaptive batch sizing disabled with hash embeddings (not yet supported)")
    else:
        if args.batch_size:
            BATCH_SIZE = args.batch_size
        else:
            BATCH_SIZE = calculate_batch_size(len(ds_train))
        
        train_loader = DataLoader(
            train_enc, batch_size=BATCH_SIZE, shuffle=True,
            collate_fn=collate_fn, num_workers=NUM_WORKERS_TRAIN, pin_memory=PIN_MEMORY,
            prefetch_factor=PREFETCH_FACTOR, persistent_workers=True, drop_last=False
        )
        val_loader = DataLoader(
            val_enc, batch_size=BATCH_SIZE, shuffle=False,
            collate_fn=collate_fn, num_workers=NUM_WORKERS_VAL, pin_memory=PIN_MEMORY,
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
    if args.hash_embed:
        model = DepthWiseCNNRouter(
            use_hash_embed=True,
            hash_dim=args.hash_dim,
            num_buckets=args.num_buckets
        ).to(device)
        print(f"🧠 Hash-based model created:")
        print(f"   • Embedding dimension: {args.hash_dim}")
        print(f"   • Hash buckets: {args.num_buckets:,}")
        print(f"   • Memory usage: ~{args.num_buckets * args.hash_dim * 4 / 1024**2:.1f} MB")
    else:
        model = DepthWiseCNNRouter(len(vocab)).to(device)
        print(f"🧠 Vocabulary-based model created:")
        print(f"   • Vocabulary size: {len(vocab):,}")
        print(f"   • Embedding dimension: {EMB_DIM}")
    
    print(f"   • Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    optimiser = optim.AdamW(model.parameters(), lr=LR_MIN, weight_decay=1e-4)

    criterion = None
    if args.class_balanced:
        # Class-balanced loss with optional label smoothing
        smoothing = 0.0 if args.no_label_smoothing else LABEL_SMOOTHING
        criterion = create_class_balanced_loss(
            ds_train, 
            num_classes=N_TAGS, 
            smoothing=smoothing
        )
        print(f"⚖️  Using class-balanced loss (inverse log frequency)")
        if smoothing > 0:
            print(f"   • With label smoothing α={smoothing}")
        
        # Print class weights for debugging
        criterion.print_class_weights(UPOS_TAGS)
        
    elif LABEL_SMOOTHING > 0:
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
    if args.hash_embed:
        print(f"   • Embedding type: Hash-based (vocabulary-free)")
        print(f"   • Hash buckets: {args.num_buckets:,}")
    else:
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
    
    # Training loop tracking for JSON output
    training_history = {
        "epochs": [],
        "loss": [],
        "accuracy": [],
        "validation": [],
        "learning_rates": [],
        "early_stopping_info": None,
        "adaptive_batch_info": None
    }
    
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
                    adaptive_batch_sizer, train_enc, collate_fn, NUM_WORKERS_TRAIN, PIN_MEMORY, PREFETCH_FACTOR
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
                        train_enc, adaptive_batch_sizer, collate_fn, NUM_WORKERS_TRAIN, PIN_MEMORY, PREFETCH_FACTOR
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
        
        # Track training history for JSON output
        training_history["epochs"].append(epoch)
        training_history["loss"].append(train_ppl)
        training_history["accuracy"].append(train_acc)
        training_history["validation"].append({
            "loss": val_ppl,
            "accuracy": val_acc,
            "f1_score": f1_score_val if CALCULATE_F1 else None
        })
        training_history["learning_rates"].append(current_lr)
        
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

    # At the end of training, before model saving, collect final results
    final_results = {
        "training_method": "early stopping" if USE_EARLY_STOPPING else "fixed epochs",
        "total_epochs": epoch if 'epoch' in locals() else MAX_EPOCHS,
        "final_train_acc": train_acc if 'train_acc' in locals() else None,
        "final_val_acc": val_acc if 'val_acc' in locals() else None,
        "final_val_ppl": val_ppl if 'val_ppl' in locals() else None,
        "final_f1_score": f1_score_val if 'f1_score_val' in locals() and CALCULATE_F1 else None,
        "final_temperature": model.temperature.item() if TEMP_SCALING else 1.0,
        "final_batch_size": adaptive_batch_sizer.get_current_batch_size() if adaptive_batch_sizer else BATCH_SIZE
    }
    
    # Add early stopping info to training history
    if early_stopping is not None:
        training_history["early_stopping_info"] = early_stopping.get_stats()
        final_results.update(early_stopping.get_stats())
    
    # Add adaptive batch sizing info
    if adaptive_batch_sizer is not None:
        training_history["adaptive_batch_info"] = adaptive_batch_sizer.get_statistics()
        # Store final batch size for hyperparameters
        args.final_batch_size = adaptive_batch_sizer.get_current_batch_size()

    # Generate model name and save all files
    if args.model_prefix:
        model_base_name = args.model_prefix
    elif args.penn_treebank:
        model_base_name = "router_penn_wsj"
    elif args.combined_penn:
        model_base_name = "router_combined_penn"
    elif args.combine:
        model_base_name = "router_combined"
    else:
        model_base_name = f"router_{args.treebank}"
    
    # Add hash embedding suffix to model name
    if args.hash_embed:
        model_base_name += "_hash"
    
    # Add timestamp for uniqueness if desired
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # model_base_name = f"{model_base_name}_{timestamp}"
    
    model_path = os.path.join(model_dir, f"{model_base_name}.pt")
    
    # Save model weights
    torch.save(model.state_dict(), model_path)
    print(f"💾 Model weights saved: {model_path}")
    
    # Create and save comprehensive JSON files
    dataset_info = {
        "train_size": len(ds_train),
        "val_size": len(ds_val),
        "treebanks": _get_treebanks_used(args)
    }
    
    architecture_info = {
        "emb_dim": EMB_DIM,
        "n_tags": N_TAGS,
        "layers": 2
    }
    
    # Save model configuration JSON
    config = save_model_config(model_base_name, args, vocab, dataset_info, architecture_info)
    config_path = os.path.join(model_dir, f"{model_base_name}.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"📋 Model config saved: {config_path}")
    
    # Save vocabulary JSON (only for traditional embeddings)
    if not args.hash_embed:
        vocab_path = save_vocabulary_json(vocab, model_dir, model_base_name)
        print(f"📚 Vocabulary saved: {vocab_path}")
    else:
        print(f"📚 Hash-based embedding: No vocabulary file needed")
    
    # Save training results JSON
    training_path = save_training_results(model_dir, model_base_name, training_history, final_results, args)
    print(f"📊 Training results saved: {training_path}")
    
    # Summary
    training_method = "early stopping" if USE_EARLY_STOPPING else "fixed epochs"
    print(f"\n✅ Training complete! Files saved to {model_dir}/:")
    print(f"   🧠 Model weights: {model_base_name}.pt")
    print(f"   📋 Configuration: {model_base_name}.json") 
    print(f"   📚 Vocabulary: {model_base_name}_vocab.json")
    print(f"   📊 Training log: {model_base_name}_training.json")
    print(f"   🎯 Trained with: {training_method}")
    
    if early_stopping and early_stopping.best_weights:
        print(f"   ⭐ Contains best weights from epoch {early_stopping.best_epoch}")
    
    # Show model info
    print(f"\n📋 Model Summary:")
    print(f"   • Name: {config['model_name']}")
    print(f"   • Description: {config['description']}")
    if args.hash_embed:
        print(f"   • Embedding type: Hash-based (vocabulary-free)")
        print(f"   • Hash buckets: {args.num_buckets:,}")
        print(f"   • Hash dimension: {args.hash_dim}")
    else:
        print(f"   • Vocabulary size: {len(vocab):,} tokens")
        print(f"   • Embedding dimension: {EMB_DIM}")
    print(f"   • Final accuracy: {final_results.get('final_val_acc', 0)*100:.2f}%")
    if final_results.get('final_f1_score'):
        print(f"   • Final F1 score: {final_results['final_f1_score']*100:.2f}%")

if __name__ == "__main__":
    main() 