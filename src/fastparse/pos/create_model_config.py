#!/usr/bin/env python3
"""
create_model_config.py

Utility script to generate model configuration JSON files.
"""

import json
import argparse

def create_config(model_name, description, architecture_params, vocabulary_params, pos_tags):
    """Create a model configuration dictionary."""
    config = {
        "model_name": model_name,
        "description": description,
        "architecture": architecture_params,
        "vocabulary": vocabulary_params,
        "pos_tags": pos_tags,
        "training_info": {
            "epochs": 80,
            "batch_size": 25911,
            "lr_max": 0.07,
            "lr_min": 0.0001,
            "warmup_epochs": 3,
            "label_smoothing": 0.1,
            "mixed_precision": True
        },
        "inference": {
            "default_batch_size": 512,
            "use_temperature": True,
            "enable_amp": False
        }
    }
    return config

def main():
    parser = argparse.ArgumentParser(description="Generate model configuration files")
    parser.add_argument("--name", required=True, help="Model name (e.g., router_combined)")
    parser.add_argument("--description", required=True, help="Model description")
    parser.add_argument("--emb-dim", type=int, default=48, help="Embedding dimension")
    parser.add_argument("--kernel-size", type=int, default=3, help="Convolution kernel size")
    parser.add_argument("--n-tags", type=int, default=18, help="Number of POS tags")
    parser.add_argument("--max-len", type=int, default=64, help="Maximum sequence length")
    parser.add_argument("--two-layer", action="store_true", help="Use second convolution layer")
    parser.add_argument("--no-temp-scaling", action="store_true", help="Disable temperature scaling")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    
    # Vocabulary options
    parser.add_argument("--vocab-type", choices=["single", "combined"], required=True,
                        help="Vocabulary type")
    parser.add_argument("--treebanks", nargs="+", required=True,
                        help="List of treebanks used")
    parser.add_argument("--expected-vocab-size", type=int,
                        help="Expected vocabulary size")
    
    # Output
    parser.add_argument("--output", help="Output JSON file (default: {name}.json)")
    
    args = parser.parse_args()
    
    # Default POS tags (Universal Dependencies)
    default_pos_tags = [
        "NOUN", "PUNCT", "ADP", "NUM", "SYM", "SCONJ", "ADJ", "PART", 
        "DET", "CCONJ", "PROPN", "PRON", "X", "ADV", "INTJ", "VERB", "AUX", "SPACE"
    ]
    
    # Architecture parameters
    arch_params = {
        "emb_dim": args.emb_dim,
        "dw_kernel": args.kernel_size,
        "n_tags": args.n_tags,
        "max_len": args.max_len,
        "use_second_conv_layer": args.two_layer,
        "use_temperature_scaling": not args.no_temp_scaling,
        "dropout_rate": args.dropout
    }
    
    # Vocabulary parameters
    vocab_params = {
        "type": args.vocab_type,
        "treebanks": args.treebanks,
        "pad_token": "<PAD>"
    }
    
    if args.expected_vocab_size:
        vocab_params["expected_vocab_size"] = args.expected_vocab_size
    
    # Create configuration
    config = create_config(
        args.name, 
        args.description, 
        arch_params, 
        vocab_params, 
        default_pos_tags
    )
    
    # Output file
    output_file = args.output or f"{args.name}.json"
    
    # Save configuration
    with open(output_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Configuration saved to: {output_file}")
    print(f"📋 Model: {config['model_name']}")
    print(f"🏗️  Architecture: {arch_params['emb_dim']}D, " +
          f"{'2-layer' if arch_params['use_second_conv_layer'] else '1-layer'} CNN")
    print(f"📚 Vocabulary: {vocab_params['type']} ({len(vocab_params['treebanks'])} treebanks)")
    print(f"🎯 Expected vocab size: {vocab_params.get('expected_vocab_size', 'unknown')}")

if __name__ == "__main__":
    main() 