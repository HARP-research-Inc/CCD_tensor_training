#!/usr/bin/env python3
"""
Temperature scaling for model calibration.

This module implements temperature scaling to improve probability calibration
of neural network outputs, especially important for uncertainty estimation.
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

def calibrate_temperature(model, val_loader, device, verbose=True):
    """
    Calibrate temperature parameter using validation set.
    
    This function optimizes the temperature parameter to improve calibration
    of the model's output probabilities using the validation data.
    
    Args:
        model: Neural network model with temperature parameter
        val_loader: Validation data loader  
        device: Device to run calibration on
        verbose: Whether to print calibration progress
    """
    if verbose:
        print("🌡️  Calibrating temperature for better probability calibration...")
    
    model.eval()
    logits_list = []
    targets_list = []
    
    # Collect logits and targets
    with torch.no_grad():
        loader_iter = tqdm(val_loader, desc="Collecting logits", leave=False) if verbose else val_loader
        for ids, upos, mask in loader_iter:
            ids = ids.to(device, non_blocking=True)
            upos = upos.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            
            # Get logits before softmax (without temperature)
            x = model.emb(ids)
            x = model.emb_dropout(x)
            x = x.transpose(1, 2)
            x = model.pw1(model.act1(model.dw1(x)))
            x = x.transpose(1, 2)
            x = model.norm1(x)
            x = model.dropout1(x)
            x = x.transpose(1, 2)
            x = model.pw2(model.act2(model.dw2(x)))
            x = x.transpose(1, 2)
            x = model.norm2(x)
            x = model.dropout2(x)
            logits = model.lin(x)
            
            # Collect valid positions
            valid_mask = mask & (upos != -100)
            if valid_mask.any():
                logits_list.append(logits[valid_mask])
                targets_list.append(upos[valid_mask])
    
    if not logits_list:
        if verbose:
            print("No valid data for temperature calibration")
        return
    
    all_logits = torch.cat(logits_list)
    all_targets = torch.cat(targets_list)
    
    # Optimize temperature
    temp_optimizer = optim.LBFGS([model.temperature], lr=0.01, max_iter=50)  # Fewer iterations for periodic calibration
    
    def temperature_loss():
        temp_optimizer.zero_grad()
        scaled_logits = all_logits / model.temperature
        loss = F.cross_entropy(scaled_logits, all_targets)
        loss.backward()
        return loss
    
    temp_optimizer.step(temperature_loss)
    
    if verbose:
        print(f"📊 Optimal temperature: {model.temperature.item():.4f}") 