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
from .label_smoothing import LabelSmoothingLoss

class ClassBalancedLoss(nn.Module):
    """
    Class-balanced loss using inverse log frequency weighting with temperature scaling.
    
    Formula: weight_i = (1 / log(1 + freq_i)) ** (1/temperature) * scale
    
    This up-weights rare classes so the optimizer pays attention to forgotten tags
    like SYM, X, INTJ instead of treating them as noise, but with moderation.
    """
    
    def __init__(self, class_frequencies: Dict[int, int], num_classes: int = 18, 
                 smoothing: float = 0.0, ignore_index: int = -100,
                 temperature: float = 2.0, scale: float = 0.5):
        """
        Initialize class-balanced loss.
        
        Args:
            class_frequencies: Dictionary mapping class_id -> frequency count
            num_classes: Total number of classes (default: 18 for Universal POS)
            smoothing: Label smoothing factor (default: 0.0)
            ignore_index: Index to ignore in loss computation (default: -100)
            temperature: Temperature to moderate the weights (default: 2.0, higher = more moderate)
            scale: Overall scaling factor for weights (default: 0.5, lower = more moderate)
        """
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.temperature = temperature
        self.scale = scale
        
        # Compute class weights using inverse log frequency with temperature
        self.class_weights = self._compute_class_weights(class_frequencies, num_classes)
        
        # Register as buffer so it moves with the model to GPU
        self.register_buffer('weights', self.class_weights)
        
    def _compute_class_weights(self, class_frequencies: Dict[int, int], num_classes: int) -> torch.Tensor:
        """
        Compute class weights using inverse log frequency formula with temperature scaling.
        
        Args:
            class_frequencies: Dictionary mapping class_id -> frequency count
            num_classes: Total number of classes
            
        Returns:
            Tensor of class weights [num_classes]
        """
        weights = torch.ones(num_classes)
        
        for class_id, freq in class_frequencies.items():
            if 0 <= class_id < num_classes and freq > 0:
                # Inverse log frequency weighting with temperature scaling
                base_weight = 1.0 / math.log(1 + freq)
                # Apply temperature to moderate the weights
                weights[class_id] = (base_weight ** (1.0 / self.temperature)) * self.scale
        
        # Handle any missing classes (give them moderate weight)
        for i in range(num_classes):
            if i not in class_frequencies:
                weights[i] = self.scale  # Moderate weight for unseen classes
        
        # Normalize weights so they sum to num_classes (keeps loss scale similar)
        weights = weights * num_classes / weights.sum()
        
        return weights
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute class-balanced loss.
        
        Args:
            logits: Model predictions [batch_size, seq_len, num_classes] or [batch_size, num_classes]
                    NOTE: These are already log probabilities from the model!
            targets: Ground truth labels [batch_size, seq_len] or [batch_size]
            
        Returns:
            Class-balanced loss value (sum reduction to match baseline)
        """
        # Reshape for sequence labeling if needed
        if logits.dim() == 3:  # [B, T, C]
            logits = logits.reshape(-1, self.num_classes)  # [B*T, C]
            targets = targets.reshape(-1)  # [B*T]
        
        if self.smoothing > 0.0:
            # Apply label smoothing with class weights
            return self._label_smoothed_nll_loss(logits, targets)
        else:
            # Standard weighted NLL loss with sum reduction
            # NOTE: logits are already log probabilities, so we don't apply log_softmax!
            weights_on_device = self.weights.to(logits.device)
            return F.nll_loss(logits, targets, weight=weights_on_device, 
                            ignore_index=self.ignore_index, reduction="sum")
    
    def _label_smoothed_nll_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute label-smoothed NLL loss with class weights.
        
        Args:
            logits: Model predictions [N, C] - already log probabilities!
            targets: Ground truth labels [N]
            
        Returns:
            Label-smoothed class-balanced loss (sum reduction to match baseline)
        """
        # logits are already log probabilities from the model
        log_probs = logits
        
        # Create one-hot with label smoothing (exactly like baseline)
        true_dist = torch.zeros_like(log_probs)
        true_dist.fill_(self.smoothing / (self.num_classes - 1))
        
        # Only apply to non-ignored indices
        mask = targets != self.ignore_index
        if mask.any():
            # Set the true class probability (exactly like baseline)
            true_dist[mask] = true_dist[mask].scatter_(
                1, targets[mask].unsqueeze(1), 1.0 - self.smoothing
            )
            
            # Compute loss exactly like baseline: -true_dist * log_probs
            loss = -true_dist * log_probs
            loss = loss.sum(dim=1)  # Sum over classes
            
            # Apply class weights AFTER computing the loss
            # Weight each sample's loss by the weight of its true class
            weights_on_device = self.weights.to(log_probs.device)
            sample_weights = weights_on_device[targets[mask]]
            
            # Apply weights and sum
            weighted_loss = loss[mask] * sample_weights
            final_loss = weighted_loss.sum()
        else:
            final_loss = torch.tensor(0.0, device=log_probs.device)
        
        return final_loss
    
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


class ScheduledClassBalancedLoss(nn.Module):
    """
    Scheduled class-balanced loss that gradually introduces class weights.
    
    The loss starts as standard cross-entropy and gradually transitions to 
    class-balanced loss as training progresses, allowing the model to first
    learn general patterns before focusing on rare classes.
    """
    
    def __init__(self, class_frequencies: Dict[int, int], num_classes: int = 18, 
                 smoothing: float = 0.0, ignore_index: int = -100,
                 temperature: float = 2.0, scale: float = 0.5,
                 schedule_type: str = "accuracy", threshold: float = 0.8,
                 warmup_epochs: int = 10):
        """
        Initialize scheduled class-balanced loss.
        
        Args:
            class_frequencies: Dictionary mapping class_id -> frequency count
            num_classes: Total number of classes (default: 18 for Universal POS)
            smoothing: Label smoothing factor (default: 0.0)
            ignore_index: Index to ignore in loss computation (default: -100)
            temperature: Temperature to moderate the weights (default: 2.0)
            scale: Overall scaling factor for weights (default: 0.5)
            schedule_type: "accuracy", "epoch", or "linear" (default: "accuracy")
            threshold: Accuracy threshold to start applying weights (default: 0.8)
            warmup_epochs: Epochs before starting to apply weights (default: 10)
        """
        super().__init__()
        self.schedule_type = schedule_type
        self.threshold = threshold
        self.warmup_epochs = warmup_epochs
        self.current_weight_scale = 0.0  # Start with no class weighting
        
        # Create the underlying class-balanced loss
        self.cb_loss = ClassBalancedLoss(
            class_frequencies, num_classes, smoothing, 
            ignore_index, temperature, scale
        )
        
        # Also keep a standard loss for blending
        self.standard_loss = LabelSmoothingLoss(smoothing, ignore_index) if smoothing > 0 else None
        
    def update_schedule(self, epoch: int = None, accuracy: float = None):
        """
        Update the weight scaling based on training progress.
        
        Args:
            epoch: Current epoch number
            accuracy: Current training or validation accuracy
        """
        if self.schedule_type == "accuracy" and accuracy is not None:
            # Gradually increase weight influence as accuracy improves
            if accuracy >= self.threshold:
                # Linear ramp from threshold to threshold + 0.1
                self.current_weight_scale = min(1.0, (accuracy - self.threshold) / 0.1)
            else:
                self.current_weight_scale = 0.0
                
        elif self.schedule_type == "epoch" and epoch is not None:
            # Start applying weights after warmup
            if epoch >= self.warmup_epochs:
                # Linear ramp over 10 epochs
                self.current_weight_scale = min(1.0, (epoch - self.warmup_epochs) / 10)
            else:
                self.current_weight_scale = 0.0
                
        elif self.schedule_type == "linear" and epoch is not None:
            # Simple linear schedule
            self.current_weight_scale = min(1.0, epoch / 50)
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute scheduled class-balanced loss.
        
        Args:
            logits: Model predictions [batch_size, num_classes, seq_len] 
                    NOTE: Already transposed to [B, C, T] by trainer!
            targets: Ground truth labels [batch_size, seq_len]
            
        Returns:
            Blended loss value
        """
        # If no weighting yet, use standard loss
        if self.current_weight_scale == 0.0:
            if self.standard_loss is not None:
                # The standard label smoothing loss expects [B, C, T] format
                # which is what we already have
                return self.standard_loss(logits, targets)
            else:
                # Use standard NLL loss - need to transpose back and flatten
                logits_for_nll = logits.transpose(1, 2)  # [B, C, T] -> [B, T, C]
                logits_flat = logits_for_nll.reshape(-1, logits_for_nll.size(-1))
                targets_flat = targets.reshape(-1)
                return F.nll_loss(logits_flat, targets_flat, ignore_index=self.cb_loss.ignore_index, reduction="sum")
        
        # If full weighting, use class-balanced loss
        elif self.current_weight_scale >= 1.0:
            # Class-balanced loss expects [B, T, C] so transpose back
            return self.cb_loss(logits.transpose(1, 2), targets)
        
        # Otherwise, blend the two losses
        else:
            # Compute both losses
            cb_loss = self.cb_loss(logits.transpose(1, 2), targets)  # Expects [B, T, C]
            
            if self.standard_loss is not None:
                standard_loss = self.standard_loss(logits, targets)  # Expects [B, C, T]
            else:
                logits_for_nll = logits.transpose(1, 2)  # [B, C, T] -> [B, T, C]
                logits_flat = logits_for_nll.reshape(-1, logits_for_nll.size(-1))
                targets_flat = targets.reshape(-1)
                standard_loss = F.nll_loss(logits_flat, targets_flat, 
                                          ignore_index=self.cb_loss.ignore_index, reduction="sum")
            
            # Blend based on current scale
            return (1 - self.current_weight_scale) * standard_loss + self.current_weight_scale * cb_loss
    
    def get_schedule_info(self) -> str:
        """Get current schedule status as a string."""
        return f"CB-weight: {self.current_weight_scale:.2f}"


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
                             pos_field: str = 'upos', ignore_index: int = -100,
                             temperature: float = 2.0, scale: float = 0.5) -> ClassBalancedLoss:
    """
    Create a class-balanced loss from a dataset.
    
    Args:
        dataset: Training dataset with POS tags
        num_classes: Number of POS classes (default: 18)
        smoothing: Label smoothing factor (default: 0.0)
        pos_field: Field name containing POS tags (default: 'upos')
        ignore_index: Index to ignore in loss computation (default: -100)
        temperature: Temperature to moderate the weights (default: 2.0, higher = more moderate)
        scale: Overall scaling factor for weights (default: 0.5, lower = more moderate)
        
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
    
    # Create loss with temperature and scale
    loss_fn = ClassBalancedLoss(class_frequencies, num_classes, smoothing, ignore_index, temperature, scale)
    
    print(f"✓ Class-balanced loss created (temperature={temperature}, scale={scale})")
    return loss_fn 


def create_scheduled_class_balanced_loss(dataset, num_classes: int = 18, smoothing: float = 0.0, 
                                       pos_field: str = 'upos', ignore_index: int = -100,
                                       temperature: float = 2.0, scale: float = 0.5,
                                       schedule_type: str = "accuracy", threshold: float = 0.8,
                                       warmup_epochs: int = 10) -> ScheduledClassBalancedLoss:
    """
    Create a scheduled class-balanced loss from a dataset.
    
    Args:
        dataset: Training dataset with POS tags
        num_classes: Number of POS classes (default: 18)
        smoothing: Label smoothing factor (default: 0.0)
        pos_field: Field name containing POS tags (default: 'upos')
        ignore_index: Index to ignore in loss computation (default: -100)
        temperature: Temperature to moderate the weights (default: 2.0)
        scale: Overall scaling factor for weights (default: 0.5)
        schedule_type: "accuracy", "epoch", or "linear" (default: "accuracy")
        threshold: Accuracy threshold to start applying weights (default: 0.8)
        warmup_epochs: Epochs before starting to apply weights (default: 10)
        
    Returns:
        ScheduledClassBalancedLoss instance
    """
    print("📊 Computing class frequencies for scheduled balanced loss...")
    
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
    
    # Create scheduled loss
    loss_fn = ScheduledClassBalancedLoss(
        class_frequencies, num_classes, smoothing, ignore_index, 
        temperature, scale, schedule_type, threshold, warmup_epochs
    )
    
    print(f"✓ Scheduled class-balanced loss created")
    print(f"   • Schedule: {schedule_type}")
    if schedule_type == "accuracy":
        print(f"   • Will activate at {threshold*100:.0f}% accuracy")
    elif schedule_type == "epoch":
        print(f"   • Will activate after epoch {warmup_epochs}")
    
    return loss_fn 