#!/usr/bin/env python3
"""
Extract vocabulary from an existing model and save it as JSON.

This utility is needed for models trained before the modular training script
that automatically saves vocabulary JSON files.
"""

import torch
import json
import argparse
import os
from collections import defaultdict
from datasets import load_dataset, Dataset

# Import the data processing functions
from data.preprocessing import build_vocab
from data.penn_treebank import load_penn_treebank_data

def extract_vocab_from_config(config_path):
    """Extract vocabulary by recreating it from the config."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    vocab_config = config['vocabulary']
    treebanks = vocab_config['treebanks']
    vocab_type = vocab_config['type']
    
    print(f"📚 Extracting vocabulary for {vocab_type}: {treebanks}")
    
    # Rebuild vocabulary exactly as it was done during training
    vocab = {"<PAD>": 0}
    
    if vocab_type in ['combined_penn', 'combined_ud_penn']:
        # Process UD treebanks
        ud_treebanks = [tb for tb in treebanks if not tb.startswith('penn')]
        
        print("🔥 Loading UD treebanks...")
        for tb in ud_treebanks:
            try:
                ds_train_tb = load_dataset("universal_dependencies", tb, split="train", trust_remote_code=True)
                for ex in ds_train_tb:
                    for tok in ex["tokens"]:
                        if tok not in vocab:
                            vocab[tok] = len(vocab)
                print(f"  ✓ Processed UD {tb}: vocab size now {len(vocab)}")
            except Exception as e:
                print(f"  ❌ Failed to load UD {tb}: {e}")
        
        # Add Penn Treebank if specified
        if any(tb.startswith('penn') for tb in treebanks):
            print("🏛️  Loading Penn Treebank...")
            try:
                train_data, val_data, test_data = load_penn_treebank_data()
                
                # Convert to dataset format and use build_vocab
                combined_data = train_data + val_data
                ds_penn = Dataset.from_list(combined_data)
                
                for ex in ds_penn:
                    for tok in ex["tokens"]:
                        if tok not in vocab:
                            vocab[tok] = len(vocab)
                            
                print(f"  ✓ Added Penn Treebank: final vocab size {len(vocab)}")
            except Exception as e:
                print(f"  ❌ Failed to load Penn Treebank: {e}")
                
    elif vocab_type == 'combined_ud':
        print("🔥 Loading combined UD treebanks...")
        for tb in treebanks:
            try:
                ds_train_tb = load_dataset("universal_dependencies", tb, split="train", trust_remote_code=True)
                for ex in ds_train_tb:
                    for tok in ex["tokens"]:
                        if tok not in vocab:
                            vocab[tok] = len(vocab)
                print(f"  ✓ Processed {tb}: vocab size now {len(vocab)}")
            except Exception as e:
                print(f"  ❌ Failed to load {tb}: {e}")
                
    elif vocab_type == 'single_treebank':
        print(f"📚 Loading single treebank: {treebanks[0]}")
        ds_train = load_dataset("universal_dependencies", treebanks[0], split="train", trust_remote_code=True)
        vocab = build_vocab(ds_train)
        
    elif vocab_type == 'penn_treebank':
        print("🏛️  Loading Penn Treebank...")
        train_data, val_data, test_data = load_penn_treebank_data()
        combined_data = train_data + val_data
        ds_penn = Dataset.from_list(combined_data)
        vocab = build_vocab(ds_penn)
    
    print(f"🎯 Final vocabulary size: {len(vocab)}")
    return vocab

def save_vocab_json(vocab, output_path):
    """Save vocabulary as JSON file."""
    from datetime import datetime
    
    vocab_data = {
        "vocab_size": len(vocab),
        "token_to_id": vocab,
        "id_to_token": {str(v): k for k, v in vocab.items()},
        "special_tokens": {
            "pad_token": "<PAD>",
            "pad_id": 0
        },
        "created_at": datetime.now().isoformat(),
        "source": "extracted_from_existing_model"
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(vocab_data, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Vocabulary saved to: {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Extract vocabulary from existing model")
    parser.add_argument("model_path", help="Path to model .pt file")
    parser.add_argument("--config", help="Path to config .json file (auto-detected if not provided)")
    parser.add_argument("--output", help="Output vocabulary .json file (auto-generated if not provided)")
    
    args = parser.parse_args()
    
    # Auto-detect config file
    if args.config is None:
        config_path = args.model_path.replace('.pt', '.json')
        if not os.path.exists(config_path):
            print(f"❌ Config file not found: {config_path}")
            exit(1)
    else:
        config_path = args.config
    
    # Auto-generate output path
    if args.output is None:
        model_name = os.path.basename(args.model_path).replace('.pt', '')
        output_path = f"{model_name}_vocab.json"
    else:
        output_path = args.output
    
    print(f"🔍 Extracting vocabulary from: {args.model_path}")
    print(f"📋 Using config: {config_path}")
    print(f"💾 Output will be saved to: {output_path}")
    
    try:
        # Extract vocabulary
        vocab = extract_vocab_from_config(config_path)
        
        # Save as JSON
        save_vocab_json(vocab, output_path)
        
        print(f"\n✅ Successfully extracted vocabulary!")
        print(f"   📊 Vocabulary size: {len(vocab):,} tokens")
        print(f"   📁 Saved to: {output_path}")
        
        # Verify the vocab can be loaded
        print(f"\n🔍 Verifying saved vocabulary...")
        with open(output_path, 'r') as f:
            test_vocab = json.load(f)
        print(f"✓ Verification successful: {test_vocab['vocab_size']} tokens")
        
    except Exception as e:
        print(f"\n❌ Failed to extract vocabulary: {e}")
        exit(1)

if __name__ == "__main__":
    main() 