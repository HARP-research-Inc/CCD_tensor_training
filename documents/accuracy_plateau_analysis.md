# Accuracy Plateau Analysis & Solutions

## Problem Analysis

Your model hit an accuracy plateau at **86% training / 81% validation** after epoch 12. Here's why this happened:

### 1. **Learning Rate Never Switched** 🚨
- **Issue**: LR threshold was set to 98% (unrealistic for this model)
- **Effect**: Model stuck with high LR (4e-2) causing training instability
- **Solution**: ✅ Lowered thresholds to 85% → 90% with multi-stage schedule

### 2. **Model Capacity Limitations** 📏
- **Issue**: Original model was too simple (~64K params)
  - Single conv layer
  - No regularization
  - No normalization
- **Solution**: ✅ Enhanced architecture with:
  - 2 conv layers (double capacity)
  - Layer normalization
  - Dropout regularization
  - Better feature extraction

### 3. **Overfitting Signs** 📈
- **Issue**: 5% gap between train (86%) and val (81%) accuracy
- **Solution**: ✅ Added weight decay (1e-4) to AdamW optimizer

### 4. **Dataset Complexity** 🌍
- **Issue**: Training on 5 different treebanks with annotation inconsistencies
- **Challenge**: Different linguistic styles and annotation standards
- **Potential Solution**: 🔄 Test single high-quality treebank (en_ewt)

## Improvements Made

### Learning Rate Schedule
```python
# OLD: Single unrealistic threshold
schedule_max = 0.98  # Never triggered!

# NEW: Multi-stage realistic schedule
schedule_first = 0.85   # 4e-2 → 2e-2 at 85%
schedule_second = 0.90  # 2e-2 → 1e-2 at 90%
```

### Enhanced Model Architecture
```python
# OLD: Simple model
emb → conv → linear (64K params)

# NEW: Enhanced model with regularization
emb → dropout → conv1 → norm1 → dropout1 → 
      conv2 → norm2 → dropout2 → linear (~120K params)
```

### Weight Decay Regularization
```python
# OLD: No regularization
opt = optim.AdamW(model.parameters(), lr=LR_HIGH)

# NEW: L2 regularization to prevent overfitting
opt = optim.AdamW(model.parameters(), lr=LR_HIGH, weight_decay=1e-4)
```

## Expected Improvements

1. **Better Convergence**: LR schedule will trigger fine-tuning at 85%
2. **Higher Accuracy**: Enhanced model has 2x capacity for complex patterns
3. **Less Overfitting**: Weight decay + dropout should reduce train/val gap
4. **Stable Training**: Layer normalization improves gradient flow

## Next Steps

1. **Test Enhanced Model**: Train with new architecture and compare results
2. **Analyze Dataset**: Check annotation quality across treebanks
3. **Single Treebank Test**: Try training on just en_ewt (highest quality)
4. **Cosine Annealing**: Implement smoother LR schedule if needed

## Performance Predictions

- **Expected Training Accuracy**: 92-95% (up from 86%)
- **Expected Validation Accuracy**: 87-90% (up from 81%)
- **Convergence**: Should reach fine-tuning phase by epoch 15-20

## Training Command

```bash
# Test the enhanced model
python pos_router_train.py --combine
```

The model should now break through the plateau and achieve significantly better accuracy with proper fine-tuning! 