"""
Command Line Interface

Handles all command line argument parsing for the POS training script.
Clean separation of CLI concerns from training logic.
"""

import argparse
from typing import Any


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="Train POS tagger with intelligent batch sizing, early stopping, and advanced optimization techniques."
    )
    
    # Dataset options
    parser.add_argument("--treebank", default="en_ewt",
                        help="Any UD code accepted by datasets (e.g. en_ewt, en_gum, fr_sequoia)")
    parser.add_argument("--combine", action="store_true",
                        help="Combine multiple UD English treebanks ONLY (no Penn Treebank)")
    parser.add_argument("--penn-treebank", action="store_true",
                        help="Train on Penn Treebank WSJ ONLY (pure Penn Treebank training)")
    parser.add_argument("--combined-penn", action="store_true",
                        help="Combine UD treebanks WITH Penn Treebank (experimental)")
    parser.add_argument("--penn-path", type=str, default=None,
                        help="Path to full Penn Treebank directory (required for --penn-treebank or --combined-penn)")
    parser.add_argument("--augment", action="store_true",
                        help="Apply data augmentation techniques")
    
    # Training options
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override default batch size")
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
    
    # Output directory options
    parser.add_argument("--model-dir", default="models",
                        help="Directory to save model outputs (default: models)")
    parser.add_argument("--model-prefix", default=None,
                        help="Custom prefix for model files (default: auto-generated)")
    
    # Early stopping arguments
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
    
    # Training configuration
    parser.add_argument("--no-label-smoothing", action="store_true",
                        help="Disable label smoothing")
    parser.add_argument("--no-temp-scaling", action="store_true",
                        help="Disable temperature scaling")
    parser.add_argument("--class-balanced", action="store_true",
                        help="Use class-balanced loss with inverse log frequency weighting")
    parser.add_argument("--class-balanced-temperature", type=float, default=2.0,
                        help="Temperature for class-balanced loss (default: 2.0, higher = more moderate)")
    parser.add_argument("--class-balanced-scale", type=float, default=0.5,
                        help="Scale factor for class-balanced loss (default: 0.5, lower = more moderate)")
    parser.add_argument("--class-balanced-schedule", type=str, default=None,
                        choices=["accuracy", "epoch", "linear"],
                        help="Schedule type for class-balanced loss (default: None = immediate)")
    parser.add_argument("--class-balanced-threshold", type=float, default=0.8,
                        help="Accuracy threshold to activate class-balanced loss (default: 0.8)")
    parser.add_argument("--class-balanced-warmup", type=int, default=10,
                        help="Warmup epochs before activating class-balanced loss (default: 10)")
    parser.add_argument("--cosine", action="store_true",
                        help="Use cosine annealing instead of SGDR")
    parser.add_argument("--share", action="store_true",
                        help="If multiple GPUs are available, force use of cuda:1 (compute node sharing)")
    parser.add_argument("--gpu", type=int, default=None,
                        help="Force specific GPU ID (overrides auto-selection)")
    
    # SGDR scheduler options
    parser.add_argument("--sgdr-t0", type=int, default=10,
                        help="SGDR T_0 (initial restart period, default: 10)")
    parser.add_argument("--sgdr-t-mult", type=float, default=2.0,
                        help="SGDR T_mult (period multiplier, default: 2.0)")
    parser.add_argument("--sgdr-eta-min", type=float, default=1e-6,
                        help="SGDR eta_min (minimum learning rate, default: 1e-6)")
    
    # Compute node optimizations
    parser.add_argument("--compute-node", action="store_true",
                        help="Enable compute node optimizations (high worker count, etc.)")
    
    # Analysis and debugging
    parser.add_argument("--detailed-analysis", action="store_true",
                        help="Enable detailed per-class analysis during training")
    parser.add_argument("--save-checkpoints", action="store_true",
                        help="Save model checkpoints during training")
    parser.add_argument("--checkpoint-freq", type=int, default=10,
                        help="Checkpoint frequency in epochs (default: 10)")
    
    # Temperature calibration
    parser.add_argument("--temp-calibration-freq", type=int, default=25,
                        help="Temperature recalibration frequency in epochs (default: 25)")
    parser.add_argument("--temp-calibration-samples", type=int, default=1000,
                        help="Number of samples for temperature calibration (default: 1000)")
    
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
    
    return parser


def parse_args() -> Any:
    """Parse command line arguments and return parsed args."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Validate argument combinations
    validate_args(args)
    
    return args


def validate_args(args: Any) -> None:
    """Validate argument combinations and set defaults."""
    # Validate dataset selection
    dataset_options = [args.penn_treebank, args.combined_penn, args.combine]
    if sum(dataset_options) > 1:
        raise ValueError("Only one of --penn-treebank, --combined-penn, or --combine can be specified")
    
    # Set F1 average type
    if args.use_weighted_f1:
        args.f1_average = 'weighted'
    else:
        args.f1_average = 'macro'
    
    # Validate adaptive batch sizing parameters
    if args.adaptive_batch:
        if args.pilot_batch_size >= args.max_batch_adaptive:
            raise ValueError("Pilot batch size must be less than max batch size")
        if args.min_batch_size >= args.max_batch_adaptive:
            raise ValueError("Min batch size must be less than max batch size")
        if args.noise_threshold <= 0 or args.noise_threshold >= 1:
            raise ValueError("Noise threshold must be between 0 and 1")
        if args.variance_estimation_freq <= 0:
            raise ValueError("Variance estimation frequency must be positive")
    
    # Validate early stopping parameters
    if not args.fixed_epochs:
        if args.patience <= 0:
            raise ValueError("Patience must be positive")
        if args.min_delta < 0:
            raise ValueError("Min delta must be non-negative")
        if args.max_epochs <= 0:
            raise ValueError("Max epochs must be positive")
    
    # Validate SGDR parameters
    if not args.cosine:
        if args.sgdr_t0 <= 0:
            raise ValueError("SGDR T_0 must be positive")
        if args.sgdr_t_mult <= 0:
            raise ValueError("SGDR T_mult must be positive")
        if args.sgdr_eta_min < 0:
            raise ValueError("SGDR eta_min must be non-negative")
    
    # Validate temperature calibration
    if args.temp_calibration_freq <= 0:
        raise ValueError("Temperature calibration frequency must be positive")
    if args.temp_calibration_samples <= 0:
        raise ValueError("Temperature calibration samples must be positive")
    
    # Validate checkpoint settings
    if args.save_checkpoints and args.checkpoint_freq <= 0:
        raise ValueError("Checkpoint frequency must be positive")
    
    # Validate hash embedding parameters
    if args.hash_embed:
        if args.hash_dim <= 0:
            raise ValueError("Hash dimension must be positive")
        if args.num_buckets <= 0:
            raise ValueError("Number of hash buckets must be positive")
        if args.ngram_min <= 0:
            raise ValueError("Minimum n-gram length must be positive")
        if args.ngram_max <= 0:
            raise ValueError("Maximum n-gram length must be positive")
        if args.ngram_min > args.ngram_max:
            raise ValueError("Minimum n-gram length must be less than or equal to maximum")
    
    # Validate class-balanced loss parameters
    if args.class_balanced:
        if args.class_balanced_temperature <= 0:
            raise ValueError("Class-balanced temperature must be positive")
        if args.class_balanced_scale <= 0:
            raise ValueError("Class-balanced scale must be positive")


def print_args_summary(args: Any) -> None:
    """Print a summary of the parsed arguments."""
    print(f"\n{'='*60}")
    print(f"🔧 TRAINING CONFIGURATION")
    print(f"{'='*60}")
    
    # Dataset
    if args.penn_treebank:
        dataset_type = "Penn Treebank WSJ"
    elif args.combined_penn:
        dataset_type = "Combined UD + Penn Treebank"
    elif args.combine:
        dataset_type = "Combined UD English"
    else:
        dataset_type = f"UD {args.treebank}"
    
    print(f"📊 Dataset: {dataset_type}")
    print(f"🔄 Augmentation: {'Yes' if args.augment else 'No'}")
    
    # Training
    print(f"⚙️  Training Mode: {'Fixed epochs' if args.fixed_epochs else 'Early stopping'}")
    if not args.fixed_epochs:
        print(f"   Monitor: {args.monitor}")
        print(f"   Patience: {args.patience}")
        print(f"   Max epochs: {args.max_epochs}")
    
    # Batch sizing
    if args.adaptive_batch:
        print(f"📈 Adaptive Batch Sizing: Yes")
        print(f"   Pilot size: {args.pilot_batch_size}")
        print(f"   Min size: {args.min_batch_size}")
        print(f"   Max size: {args.max_batch_adaptive}")
        print(f"   Noise threshold: {args.noise_threshold}")
    else:
        print(f"📊 Batch Size: {args.batch_size or 'Auto'}")
    
    # Embeddings
    if args.hash_embed:
        print(f"🎯 Embeddings: Hash-based (vocabulary-free)")
        print(f"   Hash dimension: {args.hash_dim}")
        print(f"   Hash buckets: {args.num_buckets:,}")
        print(f"   Character n-grams: {args.ngram_min}-{args.ngram_max}")
    else:
        print(f"📚 Embeddings: Vocabulary-based")
    
    # Features
    print(f"🌡️  Temperature Scaling: {'No' if args.no_temp_scaling else 'Yes'}")
    print(f"🎯 Label Smoothing: {'No' if args.no_label_smoothing else 'Yes'}")
    print(f"⚖️  Class-Balanced Loss: {'Yes' if args.class_balanced else 'No'}")
    print(f"📈 Scheduler: {'Cosine' if args.cosine else 'SGDR'}")
    print(f"🖥️  Compute Node: {'Yes' if args.compute_node else 'No'}")
    
    # Output
    print(f"📁 Model Directory: {args.model_dir}")
    if args.model_prefix:
        print(f"🏷️  Model Prefix: {args.model_prefix}")
    
    print(f"{'='*60}") 