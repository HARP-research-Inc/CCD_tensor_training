#!/usr/bin/env python3
"""
Example script showing how to use the hash-based embedding system for POS tagging.

This demonstrates the drop-in replacement of vocabulary-based embeddings with
spaCy-style hash embeddings for better OOV handling and vocabulary-free training.
"""

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset

# Import the updated components
from models.router import DepthWiseCNNRouter
from data.preprocessing import encode_sent_with_attrs, collate_with_attrs, token_attrs

def main():
    """Demonstrate hash-based embedding usage."""
    
    print("🔧 Hash-based Embedding Demo")
    print("=" * 50)
    
    # 1. Load a small dataset for demonstration
    print("📚 Loading dataset...")
    ds = load_dataset("universal_dependencies", "en_ewt", split="train[:100]", trust_remote_code=True)
    
    # 2. Encode sentences with attributes instead of vocabulary IDs
    print("\n🏷️  Encoding sentences with token attributes...")
    encoded_ds = ds.map(lambda ex: encode_sent_with_attrs(ex, ngram_min=3, ngram_max=5))
    encoded_ds = encoded_ds.with_format("torch", columns=["attrs", "upos"], output_all_columns=True)
    
    # 3. Show example token attributes
    print("\n📝 Example token attributes:")
    example_tokens = ["Hello", "world", "!", "123", "spaCy-style"]
    for token in example_tokens:
        attrs = token_attrs(token)
        print(f"  '{token}' -> {attrs[:8]}...")  # Show first 8 attributes
    
    # 4. Create DataLoader with new collate function
    print("\n📦 Creating DataLoader with attribute-based collation...")
    dataloader = DataLoader(
        encoded_ds, 
        batch_size=4, 
        shuffle=False,
        collate_fn=collate_with_attrs
    )
    
    # 5. Create hash-based router model
    print("\n🧠 Creating hash-based router model...")
    model = DepthWiseCNNRouter(
        use_hash_embed=True,
        hash_dim=96,           # spaCy default
        num_buckets=1<<20      # 1M buckets
    )
    
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Embedding table size: {model.emb.num_buckets:,} buckets × {model.emb.dim} dim")
    print(f"   Memory usage: ~{model.emb.num_buckets * model.emb.dim * 4 / 1024**2:.1f} MB")
    
    # 6. Test forward pass
    print("\n🚀 Testing forward pass...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    for batch_idx, (attrs, upos, mask) in enumerate(dataloader):
        print(f"\n   Batch {batch_idx + 1}:")
        print(f"     Attributes: {len(attrs)} tokens")
        print(f"     Shape: {upos.shape}")
        
        # Move to device
        upos = upos.to(device)
        mask = mask.to(device)
        
        # Forward pass
        with torch.no_grad():
            logits = model(attrs, mask)
        
        print(f"     Output shape: {logits.shape}")
        print(f"     Sample predictions: {logits[0, :3].argmax(-1).tolist()}")
        
        if batch_idx >= 2:  # Only show first 3 batches
            break
    
    print("\n✅ Hash-based embedding demo completed!")
    print("\n🎯 Key advantages:")
    print("   • No vocabulary size limitations")
    print("   • Graceful OOV handling")
    print("   • O(1) lookup complexity")
    print("   • Consistent memory usage")
    print("   • spaCy-compatible feature extraction")
    
    print("\n📖 Usage in training:")
    print("   # Replace vocabulary-based encoding:")
    print("   train_enc = ds_train.map(lambda ex: encode_sent_with_attrs(ex))")
    print("   train_enc = train_enc.with_format('torch', columns=['attrs', 'upos'])")
    print("   ")
    print("   # Use attribute-based collation:")
    print("   train_loader = DataLoader(train_enc, collate_fn=collate_with_attrs)")
    print("   ")
    print("   # Create hash-based model:")
    print("   model = DepthWiseCNNRouter(use_hash_embed=True, hash_dim=96)")
    print("   ")
    print("   # Training loop remains unchanged!")

if __name__ == "__main__":
    main() 