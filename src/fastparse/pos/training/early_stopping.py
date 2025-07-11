#!/usr/bin/env python3
"""
Early stopping implementation for training.

This module provides early stopping functionality to prevent overfitting
and automatically stop training when validation metrics stop improving.
"""

import torch

class EarlyStopping:
    """
    Early stopping to stop training when a monitored metric has stopped improving.
    
    Args:
        patience: Number of epochs with no improvement to wait before stopping
        min_delta: Minimum change to qualify as an improvement
        monitor: Metric to monitor ('val_loss', 'val_acc', 'val_ppl', 'macro_f1', 'weighted_f1')
        mode: 'min' for metrics that should decrease, 'max' for metrics that should increase
        restore_best_weights: Whether to restore model weights from the best epoch
        verbose: Whether to print early stopping messages
    """
    
    def __init__(self, patience=5, min_delta=1e-4, monitor='val_loss', mode='auto', 
                 restore_best_weights=True, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        
        # Auto-determine mode based on metric name
        if mode == 'auto':
            if 'acc' in monitor or 'f1' in monitor:
                self.mode = 'max'  # Accuracy and F1 should increase
            else:
                self.mode = 'min'  # Loss/perplexity should decrease
        else:
            self.mode = mode
            
        self.best = float('inf') if self.mode == 'min' else float('-inf')
        self.best_epoch = 0
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
        
        if self.verbose:
            print(f"🛑 Early stopping enabled:")
            print(f"   • Monitor: {self.monitor} ({'minimize' if self.mode == 'min' else 'maximize'})")
            print(f"   • Patience: {self.patience} epochs")
            print(f"   • Min delta: {self.min_delta}")
            print(f"   • Restore best weights: {self.restore_best_weights}")
    
    def __call__(self, epoch, val_loss, val_acc, val_ppl, f1_score_val, model):
        """
        Check if training should stop and update best weights.
        
        Args:
            epoch: Current epoch number
            val_loss: Validation loss (used as perplexity proxy)
            val_acc: Validation accuracy
            val_ppl: Validation perplexity
            f1_score_val: Validation F1 score
            model: Model to save weights from
            
        Returns:
            bool: True if training should stop, False otherwise
        """
        # Get current metric value
        if self.monitor == 'val_loss':
            current = val_loss
        elif self.monitor == 'val_acc':
            current = val_acc
        elif self.monitor == 'val_ppl':
            current = val_ppl
        elif self.monitor == 'macro_f1' or self.monitor == 'weighted_f1':
            current = f1_score_val
        else:
            raise ValueError(f"Unknown monitor metric: {self.monitor}")
        
        # Check if current metric is better than best
        if self.mode == 'min':
            improved = current < (self.best - self.min_delta)
        else:
            improved = current > (self.best + self.min_delta)
        
        if improved:
            self.best = current
            self.best_epoch = epoch
            self.wait = 0
            
            # Save best weights
            if self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
            if self.verbose:
                direction = "↓" if self.mode == 'min' else "↑"
                print(f"   🎯 New best {self.monitor}: {current:.4f} {direction} (epoch {epoch})")
        else:
            self.wait += 1
            
        # Check if we should stop
        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            if self.verbose:
                print(f"   ⏹️  Early stopping triggered!")
                print(f"   📊 Best {self.monitor}: {self.best:.4f} at epoch {self.best_epoch}")
                print(f"   ⏱️  No improvement for {self.patience} epochs")
            return True
            
        return False
    
    def restore_best(self, model, device):
        """Restore the best weights to the model."""
        if self.best_weights is not None:
            # Move weights back to device
            best_weights_device = {k: v.to(device) for k, v in self.best_weights.items()}
            model.load_state_dict(best_weights_device)
            if self.verbose:
                print(f"   🔄 Restored best weights from epoch {self.best_epoch}")
        
    def get_stats(self):
        """Get early stopping statistics."""
        return {
            'stopped_epoch': self.stopped_epoch,
            'best_epoch': self.best_epoch,
            'best_value': self.best,
            'patience_used': self.wait,
            'monitor': self.monitor
        } 