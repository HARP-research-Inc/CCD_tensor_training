#!/usr/bin/env python3
"""
Label smoothing loss for better model calibration.

This module implements label smoothing cross-entropy loss, which helps
prevent overconfident predictions and improves model calibration.
"""

import torch
import torch.nn as nn

class LabelSmoothingLoss(nn.Module):
    """Label smoothing loss for better calibration."""
    
    def __init__(self, smoothing=0.1, ignore_index=-100):
        """
        Initialize label smoothing loss.
        
        Args:
            smoothing: Label smoothing factor (default: 0.1)
            ignore_index: Index to ignore in loss calculation (default: -100)
        """
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        
    def forward(self, pred, target):
        """
        Compute label smoothing loss.
        
        Args:
            pred: [B, C, T] log probabilities
            target: [B, T] target labels
            
        Returns:
            loss: Scalar loss value (sum, not mean, for consistency with F.nll_loss)
        """
        B, C, T = pred.shape
        pred = pred.transpose(1, 2).contiguous().view(-1, C)  # [B*T, C]
        target = target.view(-1)  # [B*T]
        
        # Create one-hot with label smoothing
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (C - 1))
        
        # Only apply to non-ignored indices
        mask = target != self.ignore_index
        if mask.any():
            true_dist[mask] = true_dist[mask].scatter_(1, target[mask].unsqueeze(1), 1.0 - self.smoothing)
        
        # Compute loss only on non-ignored tokens
        loss = -true_dist * pred
        loss = loss.sum(dim=1)
        # Return sum (not mean) to be consistent with F.nll_loss(reduction="sum") for perplexity calculation
        loss = loss[mask].sum() if mask.any() else torch.tensor(0.0, device=pred.device)
        
        return loss 