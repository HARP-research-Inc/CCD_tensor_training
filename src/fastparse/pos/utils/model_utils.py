"""
Model Utilities

Handles all model I/O operations including:
- Model saving and loading
- Vocabulary serialization
- Training results logging
- Directory management
"""

import json
import os
import torch
from datetime import datetime
from typing import Dict, Any, Optional


def create_model_directory(model_dir: str = "models") -> str:
    """Create models directory if it doesn't exist."""
    os.makedirs(model_dir, exist_ok=True)
    return model_dir


def save_model_config(config: Dict, model_dir: str, model_name: str) -> str:
    """Save model configuration to JSON file."""
    config_file = os.path.join(model_dir, f"{model_name}.json")
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    return config_file


def save_vocabulary_json(vocab: Dict, model_dir: str, model_name: str) -> str:
    """Save vocabulary as a JSON file."""
    vocab_file = os.path.join(model_dir, f"{model_name}_vocab.json")
    
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


def save_model_weights(model: torch.nn.Module, model_dir: str, model_name: str) -> str:
    """Save model weights to file."""
    model_file = os.path.join(model_dir, f"{model_name}.pt")
    torch.save(model.state_dict(), model_file)
    return model_file


def save_training_results(training_history: Dict, final_results: Dict, args: Any, 
                         model_dir: str, model_name: str) -> str:
    """Save comprehensive training results and metrics."""
    training_file = os.path.join(model_dir, f"{model_name}_training.json")
    
    training_data = {
        "training_completed_at": datetime.now().isoformat(),
        "command_line_args": vars(args),
        "final_results": final_results,
        "training_history": training_history,
        "hyperparameters": {
            "architecture": {
                "emb_dim": 48,
                "dw_kernel": 3,
                "n_tags": 18,
                "max_len": 64
            },
            "training": {
                "lr_max": 7e-2,
                "lr_min": 1e-4,
                "epochs": 100,
                "warmup_epochs": 3,
                "label_smoothing": 0.1 if not args.no_label_smoothing else 0.0,
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


def load_model_config(config_path: str) -> Dict:
    """Load model configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def load_vocabulary(vocab_path: str) -> Dict:
    """Load vocabulary from JSON file."""
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
    return vocab_data['token_to_id']


def save_all_model_artifacts(model: torch.nn.Module, config: Dict, vocab: Dict, 
                           training_history: Dict, final_results: Dict, args: Any,
                           model_dir: str, model_name: str) -> Dict[str, str]:
    """Save all model artifacts (weights, config, vocab, training log)."""
    artifacts = {}
    
    # Save model weights
    artifacts['model_weights'] = save_model_weights(model, model_dir, model_name)
    
    # Save configuration
    artifacts['config'] = save_model_config(config, model_dir, model_name)
    
    # Save vocabulary
    artifacts['vocabulary'] = save_vocabulary_json(vocab, model_dir, model_name)
    
    # Save training results
    artifacts['training_log'] = save_training_results(
        training_history, final_results, args, model_dir, model_name
    )
    
    return artifacts


def print_model_summary(model_name: str, artifacts: Dict[str, str], 
                       final_results: Dict, args: Any) -> None:
    """Print a summary of the trained model and saved artifacts."""
    print(f"\n{'='*60}")
    print(f"🎯 TRAINING COMPLETE: {model_name}")
    print(f"{'='*60}")
    
    # Results
    print(f"📊 Final Results:")
    print(f"   Accuracy: {final_results.get('accuracy', 0):.1%}")
    print(f"   F1 Score: {final_results.get('f1_score', 0):.3f}")
    print(f"   Perplexity: {final_results.get('perplexity', 0):.2f}")
    
    # Artifacts
    print(f"\n📁 Saved Artifacts:")
    for artifact_type, path in artifacts.items():
        print(f"   {artifact_type}: {path}")
    
    # Configuration
    print(f"\n⚙️  Configuration:")
    print(f"   Architecture: 48D, 2-layer CNN")
    print(f"   Temperature Scaling: {'Yes' if not args.no_temp_scaling else 'No'}")
    print(f"   Label Smoothing: {'Yes' if not args.no_label_smoothing else 'No'}")
    print(f"   Adaptive Batch: {'Yes' if args.adaptive_batch else 'No'}")
    
    print(f"\n✅ Model ready for inference!")
    print(f"💡 Use: python pos_inference.py --model {artifacts['model_weights']}")
    print(f"{'='*60}") 