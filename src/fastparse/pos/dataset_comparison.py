#!/usr/bin/env python3
"""
Compare dataset sizes for different training options.
Shows how much data you get with different combinations.
"""

from datasets import load_dataset

def check_dataset_size(treebank_name):
    """Check size of a single treebank."""
    try:
        ds_train = load_dataset("universal_dependencies", treebank_name, split="train", trust_remote_code=True)
        ds_val = load_dataset("universal_dependencies", treebank_name, split="validation", trust_remote_code=True)
        return len(ds_train), len(ds_val)
    except Exception as e:
        return None, None

def main():
    print("📊 POS Tagger Dataset Size Comparison")
    print("=" * 50)
    
    # English treebanks
    english_treebanks = [
        ("en_ewt", "English Web Treebank"),
        ("en_gum", "Georgetown University Multilayer"),
        ("en_lines", "English LinES"),
        ("en_partut", "English ParTUT"),
        ("en_pronouns", "English Pronouns"),
        ("en_esl", "English ESL"),
    ]
    
    total_train = 0
    total_val = 0
    
    print("\n🔍 Individual English Treebanks:")
    print("-" * 40)
    print(f"{'Code':<12} {'Name':<30} {'Train':<8} {'Val':<8}")
    print("-" * 40)
    
    for code, name in english_treebanks:
        train_size, val_size = check_dataset_size(code)
        if train_size is not None:
            total_train += train_size
            total_val += val_size
            print(f"{code:<12} {name:<30} {train_size:<8} {val_size:<8}")
        else:
            print(f"{code:<12} {name:<30} {'ERROR':<8} {'ERROR':<8}")
    
    print("-" * 40)
    print(f"{'TOTAL':<12} {'Combined English':<30} {total_train:<8} {total_val:<8}")
    
    # Training time estimates
    print(f"\n⏱️  Training Time Estimates (approximate):")
    print("-" * 40)
    print(f"Single treebank (en_ewt): ~12,543 sentences")
    print(f"  • Training time: ~5-10 minutes")
    print(f"  • Epochs needed: 30-50")
    print(f"  • Expected accuracy: 84-87%")
    
    print(f"\nCombined English (--combine): ~{total_train:,} sentences")
    print(f"  • Training time: ~15-25 minutes")
    print(f"  • Epochs needed: 20-30")
    print(f"  • Expected accuracy: 87-90%")
    
    print(f"\nWith augmentation (--combine --augment): ~{int(total_train * 1.5):,} sentences")
    print(f"  • Training time: ~25-35 minutes")
    print(f"  • Epochs needed: 15-25")
    print(f"  • Expected accuracy: 88-91%")
    
    print(f"\n💡 Recommendations:")
    print("• Use --combine for best accuracy/time tradeoff")
    print("• Add --augment if you have extra time and want maximum accuracy")
    print("• Single treebank is fine for quick experimentation")
    
    print(f"\n🚀 Usage Examples:")
    print("python pos_router_train.py                    # Single treebank")
    print("python pos_router_train.py --combine          # All English treebanks")
    print("python pos_router_train.py --combine --augment # Maximum data")

if __name__ == "__main__":
    main() 