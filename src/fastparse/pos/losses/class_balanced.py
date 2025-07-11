#!/usr/bin/env python3
"""
Class-Balanced Loss for POS Tagging

Implements inverse log frequency weighting to handle class imbalance in POS tagging.
Rare tags like SYM, X, INTJ get higher weights so the optimizer doesn't ignore them.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import Counter
from typing import List, Dict, Optional

class ClassBalancedLoss(nn.Module):
    """
    Class-balanced loss using inverse log frequency weighting.
    
    Formula: weight_i = 1 / log(1 + freq_i)
    
    This up-weights rare classes so the optimizer pays attention to forgotten tags
    like SYM, X, INTJ instead of treating them as noise.
    """
    
    def __init__(self, class_frequencies: Dict[int, int], num_classes: int = 18, 
                 smoothing: float = 0.0, ignore_index: int = -100):
        """
        Initialize class-balanced loss.
        
        Args:
            class_frequencies: Dictionary mapping class_id -> frequency count
            num_classes: Total number of classes (default: 18 for Universal POS)
            smoothing: Label smoothing factor (default: 0.0)
            ignore_index: Index to ignore in loss computation (default: -100)
        """
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        
        # Compute class weights using inverse log frequency
        self.class_weights = self._compute_class_weights(class_frequencies, num_classes)
        
        # Register as buffer so it moves with the model to GPU
        self.register_buffer('weights', self.class_weights)
        
    def _compute_class_weights(self, class_frequencies: Dict[int, int], num_classes: int) -> torch.Tensor:
        """
        Compute class weights using inverse log frequency formula.
        
        Args:
            class_frequencies: Dictionary mapping class_id -> frequency count
            num_classes: Total number of classes
            
        Returns:
            Tensor of class weights [num_classes]
        """
        weights = torch.ones(num_classes)
        
        for class_id, freq in class_frequencies.items():
            if 0 <= class_id < num_classes and freq > 0:
                # Inverse log frequency weighting
                weights[class_id] = 1.0 / math.log(1 + freq)
        
        # Handle any missing classes (give them high weight)
        for i in range(num_classes):
            if i not in class_frequencies:
                weights[i] = 1.0  # High weight for unseen classes
        
        # Normalize weights so they sum to num_classes (keeps loss scale similar)
        weights = weights * num_classes / weights.sum()
        
        return weights
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute class-balanced loss.
        
        Args:
            logits: Model predictions [batch_size, seq_len, num_classes] or [batch_size, num_classes]
            targets: Ground truth labels [batch_size, seq_len] or [batch_size]
            
        Returns:
            Class-balanced loss value
        """
        # Reshape for sequence labeling if needed
        if logits.dim() == 3:  # [B, T, C]
            logits = logits.reshape(-1, self.num_classes)  # [B*T, C]
            targets = targets.reshape(-1)  # [B*T]
        
        if self.smoothing > 0.0:
            # Apply label smoothing with class weights
            return self._label_smoothed_nll_loss(logits, targets)
        else:
            # Standard weighted NLL loss
            log_probs = F.log_softmax(logits, dim=-1)
            return F.nll_loss(log_probs, targets, weight=self.weights, ignore_index=self.ignore_index)
    
    def _label_smoothed_nll_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute label-smoothed NLL loss with class weights.
        
        Args:
            logits: Model predictions [N, C]
            targets: Ground truth labels [N]
            
        Returns:
            Label-smoothed class-balanced loss
        """
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Create smoothed target distribution
        smooth_targets = torch.zeros_like(log_probs)
        smooth_targets.fill_(self.smoothing / (self.num_classes - 1))
        
        # Set true class probability
        valid_mask = targets != self.ignore_index
        smooth_targets[valid_mask, targets[valid_mask]] = 1.0 - self.smoothing
        
        # Apply class weights to the smoothed targets (ensure same device)
        weights_on_device = self.weights.to(smooth_targets.device)
        weighted_smooth_targets = smooth_targets * weights_on_device.unsqueeze(0)
        
        # Compute weighted cross-entropy
        loss = -(weighted_smooth_targets * log_probs).sum(dim=-1)
        
        # Only consider valid targets
        if valid_mask.any():
            loss = loss[valid_mask].mean()
        else:
            loss = loss.mean()
        
        return loss
    
    def get_class_weights(self) -> torch.Tensor:
        """Get the computed class weights."""
        return self.weights.clone()
    
    def print_class_weights(self, class_names: Optional[List[str]] = None):
        """Print class weights for debugging."""
        print("📊 Class-Balanced Loss Weights:")
        weights = self.weights.cpu().numpy()
        
        if class_names is None:
            class_names = [f"Class_{i}" for i in range(len(weights))]
        
        # Sort by weight (highest first)
        sorted_indices = sorted(range(len(weights)), key=lambda i: weights[i], reverse=True)
        
        for i in sorted_indices:
            print(f"   {class_names[i]:8s}: {weights[i]:.4f}")


def compute_class_frequencies(dataset, pos_field: str = 'upos') -> Dict[int, int]:
    """
    Compute class frequencies from a dataset.
    
    Args:
        dataset: Dataset with POS tags
        pos_field: Field name containing POS tags (default: 'upos')
        
    Returns:
        Dictionary mapping class_id -> frequency count
    """
    class_counts = Counter()
    
    for example in dataset:
        pos_tags = example[pos_field]
        for tag in pos_tags:
            if isinstance(tag, torch.Tensor):
                tag = tag.item()
            class_counts[tag] += 1
    
    return dict(class_counts)


def create_class_balanced_loss(dataset, num_classes: int = 18, smoothing: float = 0.0, 
                             pos_field: str = 'upos', ignore_index: int = -100) -> ClassBalancedLoss:
    """
    Create a class-balanced loss from a dataset.
    
    Args:
        dataset: Training dataset with POS tags
        num_classes: Number of POS classes (default: 18)
        smoothing: Label smoothing factor (default: 0.0)
        pos_field: Field name containing POS tags (default: 'upos')
        ignore_index: Index to ignore in loss computation (default: -100)
        
    Returns:
        ClassBalancedLoss instance
    """
    print("📊 Computing class frequencies for balanced loss...")
    
    # Compute class frequencies
    class_frequencies = compute_class_frequencies(dataset, pos_field)
    
    # Print frequency statistics
    total_tokens = sum(class_frequencies.values())
    print(f"   Total tokens: {total_tokens:,}")
    print(f"   Classes found: {len(class_frequencies)}/{num_classes}")
    
    # Show most/least frequent classes
    sorted_freqs = sorted(class_frequencies.items(), key=lambda x: x[1], reverse=True)
    print(f"   Most frequent: Class {sorted_freqs[0][0]} ({sorted_freqs[0][1]:,} tokens)")
    print(f"   Least frequent: Class {sorted_freqs[-1][0]} ({sorted_freqs[-1][1]:,} tokens)")
    
    # Create loss
    loss_fn = ClassBalancedLoss(class_frequencies, num_classes, smoothing, ignore_index)
    
    print("✓ Class-balanced loss created")
    return loss_fn 