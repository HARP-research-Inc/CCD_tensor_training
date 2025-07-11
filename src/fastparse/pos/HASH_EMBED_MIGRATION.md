# Hash-based Embedding Migration Guide

This guide shows how to migrate from vocabulary-based to hash-based embeddings in your POS router training.

## 🎯 Quick Migration

### 1. Update Imports

```python
# OLD: Import only traditional preprocessing
from data.preprocessing import build_vocab, encode_sent, collate

# NEW: Import both traditional and hash-based preprocessing
from data.preprocessing import (
    build_vocab, encode_sent, collate,           # Traditional
    encode_sent_with_attrs, collate_with_attrs,  # Hash-based
    token_attrs                                  # For inspection
)
```

### 2. Update Data Preprocessing

```python
# OLD: Vocabulary-based encoding
vocab = build_vocab(ds_train)
train_enc = ds_train.map(lambda ex: encode_sent(ex, vocab))
train_enc = train_enc.with_format("torch", columns=["ids", "upos"])
train_loader = DataLoader(train_enc, batch_size=512, collate_fn=collate)

# NEW: Hash-based encoding (vocabulary-free)
train_enc = ds_train.map(lambda ex: encode_sent_with_attrs(ex))
train_enc = train_enc.with_format("torch", columns=["attrs", "upos"])
train_loader = DataLoader(train_enc, batch_size=512, collate_fn=collate_with_attrs)
```

### 3. Update Model Creation

```python
# OLD: Vocabulary-based model
model = DepthWiseCNNRouter(len(vocab))

# NEW: Hash-based model
model = DepthWiseCNNRouter(
    use_hash_embed=True,
    hash_dim=96,           # spaCy default
    num_buckets=1<<20      # 1M buckets
)
```

### 4. Training Loop (Unchanged!)

```python
# Training loop remains exactly the same
for attrs, upos, mask in train_loader:  # attrs instead of ids
    logits = model(attrs, mask)  # model automatically handles hash embeddings
    loss = criterion(logits.transpose(1,2), upos)
    # ... rest of training unchanged
```

## 📋 Complete Migration Example

Here's a complete before/after for `train_modular.py`:

### Before (Lines 673-685):
```python
# Data processing
vocab = build_vocab(ds_train)
if args.augment:
    ds_train = augment_dataset(ds_train, augment_factor=1.5)
    if not hasattr(ds_train, 'map'):
        ds_train = Dataset.from_list(ds_train)

train_enc = ds_train.map(lambda ex: encode_sent(ex, vocab))
val_enc = ds_val.map(lambda ex: encode_sent(ex, vocab))
train_enc = train_enc.with_format("torch", columns=["ids", "upos"])
val_enc = val_enc.with_format("torch", columns=["ids", "upos"])
```

### After (Hash-based):
```python
# Data processing (vocabulary-free)
if args.augment:
    ds_train = augment_dataset(ds_train, augment_factor=1.5)
    if not hasattr(ds_train, 'map'):
        ds_train = Dataset.from_list(ds_train)

train_enc = ds_train.map(lambda ex: encode_sent_with_attrs(ex))
val_enc = ds_val.map(lambda ex: encode_sent_with_attrs(ex))
train_enc = train_enc.with_format("torch", columns=["attrs", "upos"])
val_enc = val_enc.with_format("torch", columns=["attrs", "upos"])
```

### Before (Lines 718-726):
```python
train_loader = DataLoader(
    train_enc, batch_size=BATCH_SIZE, shuffle=True,
    collate_fn=collate, num_workers=NUM_WORKERS_TRAIN
)
val_loader = DataLoader(
    val_enc, batch_size=BATCH_SIZE, shuffle=False,
    collate_fn=collate, num_workers=NUM_WORKERS_VAL
)
```

### After (Hash-based):
```python
train_loader = DataLoader(
    train_enc, batch_size=BATCH_SIZE, shuffle=True,
    collate_fn=collate_with_attrs, num_workers=NUM_WORKERS_TRAIN
)
val_loader = DataLoader(
    val_enc, batch_size=BATCH_SIZE, shuffle=False,
    collate_fn=collate_with_attrs, num_workers=NUM_WORKERS_VAL
)
```

### Before (Line 1445):
```python
model = DepthWiseCNNRouter(len(vocab)).to(device)
```

### After (Hash-based):
```python
model = DepthWiseCNNRouter(
    use_hash_embed=True, 
    hash_dim=96, 
    num_buckets=1<<20
).to(device)
```

## 🚀 Command Line Usage

Add these flags to your training script:

```bash
# Enable hash embeddings
python train_modular.py --hash-embed

# Customize hash parameters
python train_modular.py --hash-embed --hash-dim 128 --num-buckets 2097152

# Traditional vocabulary-based (default)
python train_modular.py  # no --hash-embed flag
```

## 🔧 Configuration Options

| Parameter | Description | Default | spaCy Default |
|-----------|-------------|---------|---------------|
| `hash_dim` | Embedding dimension | 96 | 96 |
| `num_buckets` | Hash table size | 1M (2^20) | 1M |
| `ngram_min` | Min char n-gram | 3 | 3 |
| `ngram_max` | Max char n-gram | 5 | 5 |
| `dropout` | Embedding dropout | 0.1 | 0.1 |

## 🎯 Benefits

✅ **Vocabulary-free**: No more vocabulary size explosions  
✅ **OOV handling**: Graceful handling of unseen words  
✅ **Memory efficient**: Fixed 96MB embedding table (1M × 96 × 4 bytes)  
✅ **spaCy compatible**: Same feature extraction as spaCy  
✅ **Drop-in replacement**: Minimal code changes required  

## 🔍 Debugging

### Check token attributes:
```python
from data.preprocessing import token_attrs
attrs = token_attrs("Hello")
print(f"'Hello' -> {attrs}")
# Output: ['hello', 'hel', 'llo', 'Xxxx', 'ngram_3_hel', 'ngram_3_ell', ...]
```

### Inspect batch:
```python
for attrs, upos, mask in train_loader:
    print(f"Batch size: {len(attrs) // mask.shape[1]}")
    print(f"Sequence length: {mask.shape[1]}")
    print(f"Sample attributes: {attrs[0][:5]}")  # First token's attributes
    break
```

## 📈 Performance Expectations

- **Memory**: ~96MB for embedding table (vs. variable for vocabulary)
- **Speed**: Similar to vocabulary-based (O(1) lookup)
- **Accuracy**: Expected 1-2% improvement due to better OOV handling
- **Training time**: Negligible difference 