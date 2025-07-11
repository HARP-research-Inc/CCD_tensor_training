#!/usr/bin/env python3
"""
Adaptive Batch Sizing implementation using CABS and Small-B-Early strategies.

This module implements CABS (Coupled Adaptive Batch Size) and related techniques
for better generalization through adaptive batch size selection based on gradient noise.
"""

import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

class AdaptiveBatchSizer:
    """
    Implements CABS (Coupled Adaptive Batch Size) and Small-B-Early strategies.
    
    Based on Friedlander & Schmidt (2012) and generalization theory from Hardt et al. (2016).
    """
    
    def __init__(self, 
                 min_batch_size=128, 
                 max_batch_size=2048, 
                 noise_threshold=0.1, 
                 pilot_batch_size=512,
                 small_batch_early=True,
                 variance_estimation_freq=5):
        """
        Initialize adaptive batch sizer.
        
        Args:
            min_batch_size: Minimum allowed batch size
            max_batch_size: Maximum allowed batch size
            noise_threshold: Noise threshold θ for CABS algorithm
            pilot_batch_size: Initial batch size for gradient estimation
            small_batch_early: Whether to start with small batches for exploration
            variance_estimation_freq: How often to re-estimate gradient statistics
        """
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.noise_threshold = noise_threshold  # θ in the paper
        self.pilot_batch_size = pilot_batch_size
        self.small_batch_early = small_batch_early
        self.variance_estimation_freq = variance_estimation_freq
        
        # State tracking
        self.current_batch_size = min_batch_size if small_batch_early else pilot_batch_size
        self.step_count = 0
        self.gradient_variance = None
        self.gradient_norm = None
        self.batch_size_history = []
        
        print(f"🎯 Adaptive Batch Sizing (CABS) enabled:")
        print(f"   • Range: {min_batch_size} → {max_batch_size}")
        print(f"   • Noise threshold θ: {noise_threshold}")
        print(f"   • Small-batch-early: {small_batch_early}")
        print(f"   • Variance estimation freq: every {variance_estimation_freq} steps")
    
    def estimate_gradient_statistics(self, model, data_loader, device, criterion=None, max_samples=None):
        """
        Estimate gradient variance Σ(w) and gradient norm ||∇F(w)|| using a pilot batch.
        
        Returns:
            gradient_variance: Sample variance of gradients
            gradient_norm: Norm of mean gradient
        """
        model.train()
        gradients = []
        
        # Use limited samples for efficiency
        sample_count = 0
        max_samples = max_samples or min(self.pilot_batch_size * 4, 512)
        
        for ids, upos, mask in data_loader:
            if sample_count >= max_samples:
                break
                
            ids = ids.to(device, non_blocking=True)
            upos = upos.to(device, non_blocking=True) 
            mask = mask.to(device, non_blocking=True)
            
            # Get gradients for each sample in the batch
            batch_size = ids.size(0)
            for i in range(min(batch_size, max_samples - sample_count)):
                model.zero_grad()
                
                # Single sample forward pass
                single_ids = ids[i:i+1]
                single_upos = upos[i:i+1]
                single_mask = mask[i:i+1]
                
                logp = model(single_ids, single_mask)
                if criterion is not None:
                    loss = criterion(logp.transpose(1,2), single_upos)
                else:
                    loss = F.nll_loss(logp.transpose(1,2), single_upos, 
                                    reduction="sum", ignore_index=-100)
                
                loss.backward()
                
                # Collect gradient vector
                grad_vector = []
                for param in model.parameters():
                    if param.grad is not None:
                        grad_vector.append(param.grad.view(-1))
                
                if grad_vector:
                    grad_flat = torch.cat(grad_vector)
                    gradients.append(grad_flat.detach())
                
                sample_count += 1
                if sample_count >= max_samples:
                    break
        
        if not gradients:
            return None, None
        
        # Stack gradients and compute statistics
        gradients = torch.stack(gradients)  # [N, D]
        mean_gradient = gradients.mean(dim=0)  # [D]
        
        # Compute sample variance: (1/N) * sum_i ||g_i - mean_g||^2
        centered_gradients = gradients - mean_gradient.unsqueeze(0)
        gradient_variance = (centered_gradients.norm(dim=1) ** 2).mean().item()
        gradient_norm = mean_gradient.norm().item()
        
        return gradient_variance, gradient_norm
    
    def compute_cabs_batch_size(self):
        """
        Compute CABS batch size: B = ceil(Σ(w) / (θ * ||∇F(w)||^2))
        """
        if self.gradient_variance is None or self.gradient_norm is None:
            return self.current_batch_size
        
        if self.gradient_norm < 1e-8:  # Avoid division by zero
            return self.current_batch_size
        
        # B_CABS = ceil(Σ(w) / (θ * ||∇F(w)||^2))
        optimal_batch_size = math.ceil(
            self.gradient_variance / (self.noise_threshold * self.gradient_norm ** 2)
        )
        
        # Clamp to valid range
        optimal_batch_size = max(self.min_batch_size, 
                               min(optimal_batch_size, self.max_batch_size))
        
        return optimal_batch_size
    
    def update_batch_size(self, model, data_loader, device, criterion=None, epoch=None):
        """
        Update batch size based on CABS algorithm.
        """
        self.step_count += 1
        
        # Re-estimate gradient statistics periodically
        if (self.step_count - 1) % self.variance_estimation_freq == 0:
            grad_var, grad_norm = self.estimate_gradient_statistics(
                model, data_loader, device, criterion, max_samples=256
            )
            
            if grad_var is not None and grad_norm is not None:
                self.gradient_variance = grad_var
                self.gradient_norm = grad_norm
        
        # Compute new batch size
        new_batch_size = self.compute_cabs_batch_size()
        
        # Apply Small-B-Early strategy: grow batch size gradually over epochs
        if self.small_batch_early and epoch is not None:
            # Gradually increase minimum batch size over epochs
            early_epochs = 20  # First 20 epochs use small batches
            if epoch <= early_epochs:
                growth_factor = epoch / early_epochs
                min_size_adjusted = int(self.min_batch_size + 
                                       growth_factor * (self.pilot_batch_size - self.min_batch_size))
                new_batch_size = max(min_size_adjusted, min(new_batch_size, self.pilot_batch_size))
        
        # Track batch size changes
        if new_batch_size != self.current_batch_size:
            self.batch_size_history.append({
                'step': self.step_count,
                'epoch': epoch,
                'old_size': self.current_batch_size,
                'new_size': new_batch_size,
                'grad_var': self.gradient_variance,
                'grad_norm': self.gradient_norm,
                'noise_ratio': (self.gradient_variance / (self.gradient_norm ** 2)) if self.gradient_norm > 1e-8 else None
            })
            
            self.current_batch_size = new_batch_size
        
        return self.current_batch_size
    
    def get_current_batch_size(self):
        """Get current batch size."""
        return self.current_batch_size
    
    def get_statistics(self):
        """Get current gradient statistics."""
        return {
            'batch_size': self.current_batch_size,
            'gradient_variance': self.gradient_variance,
            'gradient_norm': self.gradient_norm,
            'noise_ratio': (self.gradient_variance / (self.gradient_norm ** 2)) if 
                          (self.gradient_norm and self.gradient_norm > 1e-8) else None,
            'step_count': self.step_count,
            'batch_size_changes': len(self.batch_size_history)
        }

def create_adaptive_dataloader(dataset, batch_sizer, collate_fn, num_workers, pin_memory, prefetch_factor):
    """
    Create a DataLoader that can be recreated with different batch sizes.
    """
    return DataLoader(
        dataset, 
        batch_size=batch_sizer.get_current_batch_size(),
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=True,
        drop_last=False
    ) 