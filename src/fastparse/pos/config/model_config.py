"""
Model Configuration Management

Handles all aspects of model configuration, including:
- Model descriptions and naming
- Architecture configuration  
- Training parameter configuration
- Vocabulary type determination
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any

# Best command (7/12/25): python train_clean.py --adaptive-class-balanced --adaptive-cb-threshold 0.6 --adaptive-cb-weight-power 2.0 --adaptive-cb-max-ratio 8.0 --fixed-epochs --batch-size 512 --combined-penn
# w/ 300 epochs

# Constants (moved from train_modular.py)
EMB_DIM = 48  # Size of embeddings. Scales O(n^2)*number of layers
DW_KERNEL = 7  # Amount of context taken in. Scales O(dw)
N_TAGS = 18
LR_MAX = 7e-2
LR_MIN = 1e-4
EPOCHS = 30
WARMUP_EPOCHS = 6
MAX_LEN = 64
LABEL_SMOOTHING = 0.1

# Universal POS tag names
UPOS_TAGS = [
    "NOUN", "PUNCT", "ADP", "NUM", "SYM", "SCONJ", "ADJ", "PART", "DET", "CCONJ", 
    "PROPN", "PRON", "X", "_", "ADV", "INTJ", "VERB", "AUX"
]


def create_model_config(model_name: str, args: Any, vocab: Dict, dataset_info: Dict, final_results: Dict = None) -> Dict:
    """Create comprehensive model configuration dictionary."""
    config = {
        "model_name": model_name,
        "description": get_model_description(args),
        "created_at": datetime.now().isoformat(),
        "architecture": {
            "type": "DepthWiseCNNRouter",
            "emb_dim": args.hash_dim if args.hash_embed else EMB_DIM,
            "dw_kernel": args.dw_kernel,
            "n_tags": N_TAGS,
            "max_len": MAX_LEN,
            "use_two_layers": True,
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
            "type": "hash_based" if args.hash_embed else get_vocab_type(args),
            "treebanks": get_treebanks_used(args),
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
            "actual_epochs": final_results.get('total_epochs') if final_results else None,
            "warmup_epochs": WARMUP_EPOCHS,
            "scheduler": "SGDR" if not args.cosine else "cosine",
            "mixed_precision": True,
            "early_stopping": not args.fixed_epochs,
            "monitor_metric": args.monitor if not args.fixed_epochs else None,
            "patience": args.patience if not args.fixed_epochs else None,
            "timing": final_results.get('timing') if final_results else None
        },
        "pos_tags": {
            "tagset": "Universal Dependencies",
            "tags": UPOS_TAGS,
            "count": len(UPOS_TAGS)
        },
        "inference": {
            "default_batch_size": 512,
            "enable_temperature": not args.no_temp_scaling,
            "enable_amp": False
        },
        "files": {
            "model_weights": f"{model_name}.pt",
            "config": f"{model_name}.json",
            "vocabulary": f"{model_name}_vocab.json",
            "training_log": f"{model_name}_training.json"
        }
    }
    return config


def get_model_description(args: Any) -> str:
    """Generate descriptive model name based on training configuration."""
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
    if args.class_balanced:
        features.append("ClassBalanced")
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


def get_vocab_type(args: Any) -> str:
    """Determine vocabulary type from arguments."""
    if args.penn_treebank:
        return "penn_treebank"
    elif args.combined_penn:
        return "combined_ud_penn"
    elif args.combine:
        return "combined_ud"
    else:
        return "single_treebank"


def get_treebanks_used(args: Any) -> List[str]:
    """Get list of treebanks used for training."""
    if args.penn_treebank:
        return ["penn_wsj"]
    elif args.combined_penn:
        return ["en_ewt", "en_gum", "en_lines", "en_partut", "penn_wsj"]
    elif args.combine:
        return ["en_ewt", "en_gum", "en_lines", "en_partut"]
    else:
        return [args.treebank]


def generate_model_name(args: Any, final_results: Dict = None) -> str:
    """Generate model name based on configuration."""
    if args.model_prefix:
        base_name = args.model_prefix
    elif args.penn_treebank:
        base_name = "router_penn_wsj"
    elif args.combined_penn:
        base_name = "router_combined_penn"
    elif args.combine:
        base_name = "router_combined"
    else:
        base_name = f"router_{args.treebank}"
    
    # Add embedding dimension
    if args.hash_embed:
        base_name += f"_hash{args.hash_dim}d"
    else:
        base_name += f"_{EMB_DIM}d"
    
    # Add epoch count if final_results is provided
    if final_results and 'total_epochs' in final_results:
        base_name += f"_e{final_results['total_epochs']}"
    
    return base_name 