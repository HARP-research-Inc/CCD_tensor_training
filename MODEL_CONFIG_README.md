# Model Configuration System

The POS inference system now uses JSON configuration files to specify model architecture and parameters. This makes it easy to support different model variants without editing code.

## 📋 Configuration Files

Each model should have a corresponding `.json` config file with the same base name:
- `router_combined.pt` → `router_combined.json`
- `router_en_gum.pt` → `router_en_gum.json`

## 🏗️ Configuration Structure

```json
{
  "model_name": "router_combined",
  "description": "Combined English treebanks POS router",
  "architecture": {
    "emb_dim": 48,
    "dw_kernel": 3,
    "n_tags": 18,
    "max_len": 64,
    "use_second_conv_layer": false,
    "use_temperature_scaling": true,
    "dropout_rate": 0.1
  },
  "vocabulary": {
    "type": "combined",
    "treebanks": ["en_ewt", "en_gum", "en_lines", "en_partut", "en_pronouns", "en_esl"],
    "expected_vocab_size": 30894,
    "pad_token": "<PAD>"
  },
  "pos_tags": [
    "NOUN", "PUNCT", "ADP", "NUM", "SYM", "SCONJ", "ADJ", "PART", 
    "DET", "CCONJ", "PROPN", "PRON", "X", "ADV", "INTJ", "VERB", "AUX", "SPACE"
  ]
}
```

## 🚀 Usage

### Basic Inference
```bash
# Auto-detects config file
python pos_inference.py --model router_combined.pt

# Explicit config file
python pos_inference.py --model my_model.pt --config my_config.json

# Interactive mode
python pos_inference.py --model router_combined.pt --text "Hello world!"
```

### Benchmarking
```bash
# Compare with NLTK and spaCy
python pos_inference.py --model router_combined.pt --benchmark

# Batch processing
python pos_inference.py --model router_combined.pt --batch --num-sentences 5000
```

## 🛠️ Creating Configurations

Use the included utility to generate config files:

```bash
# Combined model
python create_model_config.py \
  --name router_combined \
  --description "Combined English treebanks POS router" \
  --vocab-type combined \
  --treebanks en_ewt en_gum en_lines en_partut en_pronouns en_esl \
  --expected-vocab-size 30894

# Single treebank model  
python create_model_config.py \
  --name router_en_gum \
  --description "Single EN-GUM treebank POS router" \
  --vocab-type single \
  --treebanks en_gum \
  --expected-vocab-size 15000

# Two-layer model with custom parameters
python create_model_config.py \
  --name router_large \
  --description "Large two-layer POS router" \
  --emb-dim 64 \
  --two-layer \
  --vocab-type combined \
  --treebanks en_ewt en_gum \
  --expected-vocab-size 25000
```

## 📊 Architecture Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `emb_dim` | Embedding dimension | 48 |
| `dw_kernel` | Convolution kernel size | 3 |
| `n_tags` | Number of POS tags | 18 |
| `max_len` | Maximum sequence length | 64 |
| `use_second_conv_layer` | Enable second conv layer | false |
| `use_temperature_scaling` | Enable temperature scaling | true |
| `dropout_rate` | Dropout rate | 0.1 |

## 📚 Vocabulary Types

### Combined Vocabulary
```json
{
  "type": "combined",
  "treebanks": ["en_ewt", "en_gum", "en_lines", "en_partut", "en_pronouns", "en_esl"]
}
```
Loads all specified treebanks to build a large combined vocabulary.

### Single Vocabulary  
```json
{
  "type": "single",
  "treebanks": ["en_gum"]
}
```
Loads only one treebank for vocabulary building.

## 🔧 Troubleshooting

### Vocabulary Size Mismatch
If you get a vocabulary size error, check:
1. Config file has correct `treebanks` list
2. `expected_vocab_size` matches your training
3. All treebanks are available

### Architecture Mismatch
If state_dict loading fails:
1. Check `use_second_conv_layer` setting
2. Verify `emb_dim` matches training
3. Ensure `n_tags` is correct

### Creating Config for Existing Model
```bash
# Find vocabulary size by trying to load the model
python -c "
import torch
state = torch.load('your_model.pt', map_location='cpu')
print('Embedding size:', state['emb.weight'].shape)
print('Output size:', state['lin.weight'].shape)
"
```

## 🎯 Benefits

- ✅ **No Code Changes**: Support new models without editing inference code
- ✅ **Version Control**: Track model configurations alongside weights  
- ✅ **Reproducibility**: Exact architecture specifications
- ✅ **Flexibility**: Easy to experiment with different architectures
- ✅ **Documentation**: Self-documenting model specifications 