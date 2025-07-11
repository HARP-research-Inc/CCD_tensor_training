"""
Main Training Orchestrator

Handles the complete training pipeline with a clean, focused interface.
This is the main training orchestration class that replaces the monolithic main function.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report
from collections import defaultdict
from typing import Dict, Any, Tuple, Optional, List

from models.router import DepthWiseCNNRouter
from losses.label_smoothing import LabelSmoothingLoss
from training.early_stopping import EarlyStopping
from training.adaptive_batch import AdaptiveBatchSizer, create_adaptive_dataloader
from training.temperature import calibrate_temperature
from data.penn_treebank import load_penn_treebank_data
from data.preprocessing import (
    build_vocab, encode_sent, augment_dataset, calculate_batch_size, collate,
    encode_sent_with_attrs, collate_with_attrs  # Hash-based embedding support
)
from config.model_config import *
from utils.model_utils import save_all_model_artifacts


class POSTrainer:
    """Main POS training orchestrator class."""
    
    def __init__(self, args: Any):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.criterion = None
        self.early_stopping = None
        self.adaptive_batch_sizer = None
        
        # Training state
        self.training_history = []
        self.step_count = 0
        self.best_metric = 0.0
        self.best_model_state = None
        
        # Data
        self.vocab = None
        self.train_dataset = None
        self.val_dataset = None
        self.train_loader = None
        self.val_loader = None
        
        # Configuration
        self.model_name = None
        self.model_config = None
        
    def setup_data(self) -> Dict[str, Any]:
        """Setup datasets and data loaders."""
        print("🔍 Setting up data...")
        
        # Load datasets based on configuration
        if self.args.penn_treebank:
            print("📚 Loading Penn Treebank WSJ data...")
            self.train_dataset, self.val_dataset = load_penn_treebank_data()
        elif self.args.combined_penn:
            print("🔥 Loading Combined UD + Penn Treebank data...")
            self.train_dataset, self.val_dataset = self._load_combined_penn_data()
        elif self.args.combine:
            print("🔥 Loading Combined UD English data...")
            self.train_dataset, self.val_dataset = self._load_combined_ud_data()
        else:
            print(f"📚 Loading UD {self.args.treebank} data...")
            self.train_dataset, self.val_dataset = self._load_single_treebank_data()
        
        # Build vocabulary or setup hash embeddings
        if self.args.hash_embed:
            print("🎯 Setting up hash-based embeddings (vocabulary-free)...")
            print(f"   • Hash dimension: {self.args.hash_dim}")
            print(f"   • Hash buckets: {self.args.num_buckets:,}")
            print(f"   • Character n-grams: {self.args.ngram_min}-{self.args.ngram_max}")
            self.vocab = {"<PAD>": 0}  # Minimal vocab for compatibility
        else:
            print("🏗️  Building vocabulary...")
            self.vocab = build_vocab(self.train_dataset)
            print(f"   • Vocabulary size: {len(self.vocab):,} tokens")
        
        # Apply augmentation if requested
        if self.args.augment:
            print("🔄 Applying data augmentation...")
            self.train_dataset = augment_dataset(self.train_dataset, augment_factor=1.5)
            from datasets import Dataset
            if not hasattr(self.train_dataset, 'map'):
                self.train_dataset = Dataset.from_list(self.train_dataset)
        
        # Encode datasets based on embedding type
        if self.args.hash_embed:
            # Hash-based encoding
            train_enc = self.train_dataset.map(lambda ex: encode_sent_with_attrs(ex, self.args.ngram_min, self.args.ngram_max))
            val_enc = self.val_dataset.map(lambda ex: encode_sent_with_attrs(ex, self.args.ngram_min, self.args.ngram_max))
            train_enc = train_enc.with_format("torch", columns=["attrs", "upos"], output_all_columns=True)
            val_enc = val_enc.with_format("torch", columns=["attrs", "upos"], output_all_columns=True)
        else:
            # Traditional vocabulary-based encoding
            train_enc = self.train_dataset.map(lambda ex: encode_sent(ex, self.vocab))
            val_enc = self.val_dataset.map(lambda ex: encode_sent(ex, self.vocab))
            train_enc = train_enc.with_format("torch", columns=["ids", "upos"], output_all_columns=True)
            val_enc = val_enc.with_format("torch", columns=["ids", "upos"], output_all_columns=True)
        
        # Store encoded datasets for adaptive batch sizing
        self.train_dataset = train_enc
        self.val_dataset = val_enc
        
        # Calculate batch size
        if self.args.batch_size:
            batch_size = self.args.batch_size
        else:
            batch_size = calculate_batch_size(len(self.train_dataset))
        
        # Determine worker counts
        num_workers_train = 48 if self.args.compute_node else 4
        num_workers_val = 16 if self.args.compute_node else 2
        prefetch_factor = 4 if self.args.compute_node else 2
        
        # Choose appropriate collate function
        collate_fn = collate_with_attrs if self.args.hash_embed else collate
        
        # Setup data loaders
        if self.args.adaptive_batch:
            print("📈 Setting up adaptive batch sizing...")
            self.adaptive_batch_sizer = AdaptiveBatchSizer(
                min_batch_size=self.args.min_batch_size,
                max_batch_size=self.args.max_batch_adaptive,
                noise_threshold=self.args.noise_threshold,
                pilot_batch_size=self.args.pilot_batch_size,
                small_batch_early=self.args.small_batch_early,
                variance_estimation_freq=self.args.variance_estimation_freq
            )
            batch_size = self.adaptive_batch_sizer.get_current_batch_size()
            print(f"📦 Adaptive batch sizing: starts at {batch_size}")
            
            self.train_loader = create_adaptive_dataloader(
                train_enc, self.adaptive_batch_sizer, collate_fn,
                num_workers_train, True, prefetch_factor
            )
            val_batch_size = min(self.args.pilot_batch_size, len(val_enc) // 10) if len(val_enc) > 100 else len(val_enc)
            self.val_loader = DataLoader(
                val_enc, batch_size=val_batch_size, shuffle=False,
                collate_fn=collate_fn,
                num_workers=num_workers_val, pin_memory=True,
                prefetch_factor=prefetch_factor, persistent_workers=True
            )
        else:
            self.train_loader = DataLoader(
                train_enc, batch_size=batch_size, shuffle=True,
                collate_fn=collate_fn,
                num_workers=num_workers_train, pin_memory=True,
                prefetch_factor=prefetch_factor, persistent_workers=True, drop_last=False
            )
            self.val_loader = DataLoader(
                val_enc, batch_size=batch_size, shuffle=False,
                collate_fn=collate_fn,
                num_workers=num_workers_val, pin_memory=True,
                prefetch_factor=prefetch_factor, persistent_workers=True
            )
        
        # Return dataset info
        dataset_info = {
            "train_size": len(self.train_dataset),
            "val_size": len(self.val_dataset),
            "vocab_size": len(self.vocab),
            "batch_size": batch_size,
            "treebanks": get_treebanks_used(self.args)
        }
        
        print(f"✅ Data setup complete:")
        print(f"   Train: {dataset_info['train_size']:,} samples")
        print(f"   Val: {dataset_info['val_size']:,} samples")
        print(f"   Vocab: {dataset_info['vocab_size']:,} tokens")
        print(f"   Batch size: {dataset_info['batch_size']}")
        
        return dataset_info
    
    def setup_model(self) -> None:
        """Setup model, optimizer, scheduler, and other training components."""
        print("🏗️  Setting up model...")
        
        # Create model
        if self.args.hash_embed:
            self.model = DepthWiseCNNRouter(
                use_hash_embed=True,
                hash_dim=self.args.hash_dim,
                num_buckets=self.args.num_buckets
            ).to(self.device)
            print(f"🧠 Hash-based model created:")
            print(f"   • Embedding dimension: {self.args.hash_dim}")
            print(f"   • Hash buckets: {self.args.num_buckets:,}")
            print(f"   • Memory usage: ~{self.args.num_buckets * self.args.hash_dim * 4 / 1024**2:.1f} MB")
        else:
            self.model = DepthWiseCNNRouter(len(self.vocab)).to(self.device)
            print(f"🧠 Vocabulary-based model created:")
            print(f"   • Vocabulary size: {len(self.vocab):,}")
            print(f"   • Embedding dimension: {EMB_DIM}")
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=LR_MAX, 
            weight_decay=1e-4
        )
        
        # Setup scheduler
        if self.args.cosine:
            # Cosine annealing with warmup
            def get_lr(epoch):
                if epoch < WARMUP_EPOCHS:
                    return LR_MIN + (LR_MAX - LR_MIN) * epoch / WARMUP_EPOCHS
                progress = (epoch - WARMUP_EPOCHS) / (EPOCHS - WARMUP_EPOCHS)
                return LR_MIN + (LR_MAX - LR_MIN) * 0.5 * (1 + math.cos(math.pi * progress))
            
            self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lambda epoch: get_lr(epoch) / LR_MIN)
            print(f"📈 Standard Cosine Annealing: {LR_MIN:.1e} → {LR_MAX:.1e} → {LR_MIN:.1e}")
        else:
            # SGDR (default)
            T_0 = self.args.sgdr_t0
            
            # Auto-detect T_0 if not provided
            if not T_0 and hasattr(self, 'train_loader'):
                T_0 = max(10, len(self.train_loader) // 4)
                if self.args.penn_treebank and len(self.train_loader) < 200:
                    T_0 = max(5, len(self.train_loader) // 6)
            elif not T_0:
                T_0 = 10  # Default value
            
            T_mult = self.args.sgdr_t_mult
            eta_min_ratio = 0.01 if self.args.penn_treebank and hasattr(self, 'train_loader') and len(self.train_loader) < 200 else 0.1
            eta_min = LR_MAX * eta_min_ratio
            
            print(f"🔄 SGDR Scheduler (default):")
            print(f"   • First cycle: {T_0} steps")
            print(f"   • Cycle multiplier: {T_mult}x")
            print(f"   • LR range: {eta_min:.1e} → {LR_MAX:.1e}")
            
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, 
                T_0=T_0, 
                T_mult=int(T_mult) if T_mult == int(T_mult) else 2,
                eta_min=eta_min, 
                last_epoch=-1
            )
        
        # Setup mixed precision
        self.scaler = GradScaler()
        
        # Setup loss function
        if not self.args.no_label_smoothing:
            self.criterion = LabelSmoothingLoss(smoothing=LABEL_SMOOTHING)
            print(f"📊 Using label smoothing with α={LABEL_SMOOTHING}")
        else:
            self.criterion = None
        
        # Setup early stopping
        if not self.args.fixed_epochs:
            self.early_stopping = EarlyStopping(
                patience=self.args.patience,
                min_delta=self.args.min_delta,
                mode='max' if 'f1' in self.args.monitor or 'acc' in self.args.monitor else 'min',
                verbose=True
            )
        
        print(f"✅ Model setup complete:")
        print(f"   Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"   Optimizer: AdamW")
        print(f"   Scheduler: {'Cosine' if self.args.cosine else 'SGDR'}")
        print(f"   Loss: {'Label Smoothing' if not self.args.no_label_smoothing else 'NLL'}")
        print(f"   Early Stopping: {'No' if self.args.fixed_epochs else 'Yes'}")
    
    def train_epoch(self, epoch: int) -> Tuple[float, float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss, total_tok, correct = 0.0, 0, 0
        
        # Handle SGDR vs standard training
        if not self.args.cosine:
            return self._train_epoch_sgdr(epoch)
        else:
            return self._train_epoch_standard(epoch)
    
    def validate_epoch(self, epoch: int) -> Tuple[float, float, float, Dict]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss, total_tok, correct = 0.0, 0, 0
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Val Epoch {epoch}", leave=True)
            for batch_data in pbar:
                inputs, upos, mask = batch_data
                
                # Handle both traditional (tensor) and hash-based (list) inputs
                if isinstance(inputs, torch.Tensor):
                    inputs = inputs.to(self.device, non_blocking=True)
                # For hash embeddings, inputs is a list and doesn't need GPU transfer
                
                upos = upos.to(self.device, non_blocking=True)
                mask = mask.to(self.device, non_blocking=True)
                
                # Apply temperature scaling during validation
                use_temp = not self.args.no_temp_scaling
                logp = self.model(inputs, mask, use_temperature=use_temp)
                
                if self.criterion is not None:
                    loss = self.criterion(logp.transpose(1,2), upos)
                else:
                    loss = F.nll_loss(logp.transpose(1,2), upos, reduction="sum", ignore_index=-100)
                
                total_loss += loss.item()
                total_tok += mask.sum().item()
                pred = logp.argmax(-1)
                correct += ((pred == upos) & mask).sum().item()
                
                # Collect predictions for F1 calculation
                if not self.args.no_f1:
                    valid_mask = mask & (upos != -100)
                    if valid_mask.any():
                        all_preds.extend(pred[valid_mask].cpu().numpy())
                        all_targets.extend(upos[valid_mask].cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / total_tok if total_tok > 0 else float('inf')
        ppl = math.exp(avg_loss) if avg_loss < 100 else float('inf')
        acc = correct / total_tok if total_tok > 0 else 0.0
        
        # Calculate F1 score
        f1 = 0.0
        if not self.args.no_f1 and all_preds:
            try:
                f1 = f1_score(all_targets, all_preds, average=self.args.f1_average, zero_division=0)
            except Exception as e:
                print(f"Warning: Could not calculate F1: {e}")
        
        # Detailed analysis
        analysis = {}
        if self.args.detailed_analysis and all_preds:
            try:
                present_classes = sorted(set(all_targets + all_preds))
                present_class_names = [UPOS_TAGS[i] for i in present_classes if i < len(UPOS_TAGS)]
                report = classification_report(
                    all_targets, all_preds, 
                    labels=present_classes,
                    target_names=present_class_names, 
                    output_dict=True, 
                    zero_division=0
                )
                analysis['classification_report'] = report
            except Exception as e:
                print(f"Warning: Could not generate detailed analysis: {e}")
        
        return ppl, acc, f1, analysis
    
    def train(self) -> Dict[str, Any]:
        """Main training loop."""
        print("🚀 Starting training...")
        
        max_epochs = EPOCHS if self.args.fixed_epochs else self.args.max_epochs
        
        # SGDR cycle tracking for phase display
        if not self.args.cosine:
            sgdr_T_0 = self.args.sgdr_t0 if self.args.sgdr_t0 else max(10, len(self.train_loader) // 4)
            if self.args.penn_treebank and len(self.train_loader) < 200:
                sgdr_T_0 = self.args.sgdr_t0 if self.args.sgdr_t0 else max(5, len(self.train_loader) // 6)
                sgdr_T_mult = self.args.sgdr_t_mult if hasattr(self.args, 'sgdr_t_mult') else 1.5
            else:
                sgdr_T_mult = self.args.sgdr_t_mult
            
            sgdr_current_cycle_length = sgdr_T_0
            sgdr_cycle_step = 0
            sgdr_cycle_num = 1
            previous_lr = LR_MIN
        
        for epoch in range(1, max_epochs + 1):
            # Training
            train_ppl, train_acc, train_f1 = self.train_epoch(epoch)
            
            # Validation
            val_ppl, val_acc, val_f1, analysis = self.validate_epoch(epoch)
            
            # Step scheduler (for cosine)
            if self.args.cosine:
                self.scheduler.step()
            
            # Temperature calibration
            if (not self.args.no_temp_scaling and 
                epoch % self.args.temp_calibration_freq == 0):
                print(f"🌡️  Recalibrating temperature...")
                calibrate_temperature(self.model, self.val_loader, self.device, verbose=False)
            
            # Record metrics
            epoch_metrics = {
                'epoch': epoch,
                'train_ppl': train_ppl,
                'train_acc': train_acc,
                'train_f1': train_f1,
                'val_ppl': val_ppl,
                'val_acc': val_acc,
                'val_f1': val_f1,
                'lr': self.optimizer.param_groups[0]['lr']
            }
            
            if self.args.detailed_analysis:
                epoch_metrics['analysis'] = analysis
            
            self.training_history.append(epoch_metrics)
            
            # SGDR phase tracking
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.args.cosine:
                phase = "warmup" if epoch <= WARMUP_EPOCHS else "cosine"
            else:
                if epoch > 1:
                    if current_lr > previous_lr * 1.5:
                        sgdr_cycle_num += 1
                        sgdr_cycle_step = 0
                        sgdr_current_cycle_length = int(sgdr_current_cycle_length * sgdr_T_mult)
                    else:
                        sgdr_cycle_step += len(self.train_loader)
                else:
                    sgdr_cycle_step += len(self.train_loader)
                    
                cycle_progress = sgdr_cycle_step / sgdr_current_cycle_length if sgdr_current_cycle_length > 0 else 0
                
                if cycle_progress < 0.1:
                    phase = "restart"
                elif cycle_progress < 0.3:
                    phase = "early"
                elif cycle_progress < 0.7:
                    phase = "mid"
                else:
                    phase = "late"
                    
                phase = f"{phase}-C{sgdr_cycle_num}"
                previous_lr = current_lr
            
            # Temperature display
            temp_val = self.model.temperature.item() if hasattr(self.model, 'temperature') and not self.args.no_temp_scaling else 1.0
            temp_str = f" | temp {temp_val:.3f}" if not self.args.no_temp_scaling and abs(temp_val - 1.0) > 0.01 else ""
            
            # Print progress (restored from original format)
            if not self.args.no_f1 and val_f1 > 0:
                if self.args.monitor in ['macro_f1', 'weighted_f1']:
                    print(f"epoch {epoch:02d} | "
                          f"train acc {train_acc*100:5.2f}% | "
                          f"{self.args.f1_average[0].upper()}F1 {val_f1*100:5.2f}% (acc {val_acc*100:4.1f}%) | "
                          f"val ppl {val_ppl:4.2f} | "
                          f"lr {current_lr:.1e} ({phase}){temp_str}")
                else:
                    print(f"epoch {epoch:02d} | "
                          f"train acc {train_acc*100:5.2f}% | "
                          f"val acc {val_acc*100:5.2f}% ({self.args.f1_average[0].upper()}F1 {val_f1*100:4.1f}%) | "
                          f"val ppl {val_ppl:4.2f} | "
                          f"lr {current_lr:.1e} ({phase}){temp_str}")
            else:
                print(f"epoch {epoch:02d} | "
                      f"train acc {train_acc*100:5.2f}% | "
                      f"val acc {val_acc*100:5.2f}% | "
                      f"val ppl {val_ppl:4.2f} | "
                      f"lr {current_lr:.1e} ({phase}){temp_str}")
            
            # Early stopping check
            if not self.args.fixed_epochs:
                metric_value = self._get_monitor_metric(epoch_metrics)
                
                # Call early stopping with proper parameters
                val_loss_for_es = val_ppl
                if self.early_stopping(epoch, val_loss_for_es, val_acc, val_ppl, val_f1, self.model):
                    print(f"🛑 Early stopping triggered at epoch {epoch}")
                    break
                
                # Track best metric
                if metric_value > self.best_metric:
                    self.best_metric = metric_value
            
            # Save checkpoints
            if (self.args.save_checkpoints and 
                epoch % self.args.checkpoint_freq == 0):
                self._save_checkpoint(epoch)
        
        # Restore best model if using early stopping
        if not self.args.fixed_epochs and self.early_stopping is not None:
            if self.early_stopping.best_weights is not None:
                self.early_stopping.restore_best(self.model, self.device)
                
            stats = self.early_stopping.get_stats()
            print(f"\n📊 Early Stopping Summary:")
            print(f"   • Training stopped at epoch: {stats['stopped_epoch']}")
            print(f"   • Best epoch: {stats['best_epoch']}")
            print(f"   • Best {stats['monitor']}: {stats['best_value']:.4f}")
            print(f"   • Patience used: {stats['patience_used']}/{self.early_stopping.patience}")
        
        # Final temperature calibration
        if not self.args.no_temp_scaling:
            print("\n🌡️  Final temperature calibration...")
            calibrate_temperature(self.model, self.val_loader, self.device, verbose=True)
            
            print("🔍 Re-evaluating with calibrated temperature...")
            cal_ppl, cal_acc, cal_f1_score, cal_analysis = self.validate_epoch(max_epochs)
            cal_f1_str = f", {self.args.f1_average[0].upper()}F1 {cal_f1_score*100:.2f}%" if not self.args.no_f1 and cal_f1_score > 0 else ""
            print(f"📊 Calibrated results: acc {cal_acc*100:.2f}%{cal_f1_str}, ppl {cal_ppl:.2f}")
            
            # Update final values with calibrated results
            val_ppl, val_acc, val_f1 = cal_ppl, cal_acc, cal_f1_score
        
        # Final results
        final_results = {
            'training_method': 'early stopping' if not self.args.fixed_epochs else 'fixed epochs',
            'total_epochs': len(self.training_history),
            'final_train_acc': train_acc if 'train_acc' in locals() else None,
            'final_val_acc': val_acc if 'val_acc' in locals() else None,
            'final_val_ppl': val_ppl if 'val_ppl' in locals() else None,
            'final_f1_score': val_f1 if 'val_f1' in locals() and not self.args.no_f1 else None,
            'final_temperature': self.model.temperature.item() if hasattr(self.model, 'temperature') and not self.args.no_temp_scaling else 1.0,
            'final_batch_size': self.adaptive_batch_sizer.get_current_batch_size() if self.adaptive_batch_sizer else self.args.batch_size,
            'best_metric': self.best_metric,
            'perplexity': val_ppl,
            'accuracy': val_acc,
            'f1_score': val_f1
        }
        
        # Add early stopping info
        if self.early_stopping is not None:
            final_results.update(self.early_stopping.get_stats())
        
        print(f"✅ Training complete!")
        print(f"   Final accuracy: {val_acc:.1%}")
        print(f"   Final F1: {val_f1:.3f}")
        print(f"   Best {self.args.monitor}: {self.best_metric:.3f}")
        
        return final_results
    
    def save_model(self, final_results: Dict, model_dir: str) -> Dict[str, str]:
        """Save all model artifacts."""
        print("💾 Saving model artifacts...")
        
        # Generate model name and config
        self.model_name = generate_model_name(self.args)
        self.model_config = create_model_config(
            self.model_name, self.args, self.vocab, 
            {"train_size": len(self.train_dataset), "val_size": len(self.val_dataset)}
        )
        
        # Save all artifacts
        artifacts = save_all_model_artifacts(
            self.model, self.model_config, self.vocab,
            self.training_history, final_results, self.args,
            model_dir, self.model_name
        )
        
        print(f"✅ Model saved: {self.model_name}")
        return artifacts
    
    def _load_single_treebank_data(self):
        """Load single treebank data."""
        from datasets import load_dataset
        
        ds_train = load_dataset("universal_dependencies", self.args.treebank, 
                               split="train", trust_remote_code=True)
        ds_val = load_dataset("universal_dependencies", self.args.treebank, 
                             split="validation", trust_remote_code=True)
        print(f"📊 Single treebank {self.args.treebank}: {len(ds_train)} train, {len(ds_val)} val sentences")
        
        return ds_train, ds_val
    
    def _load_combined_ud_data(self):
        """Load combined UD data."""
        from datasets import load_dataset, Dataset
        
        print("🔥 Combined UD Training Mode (NO Penn Treebank)")
        english_treebanks = ["en_ewt", "en_gum", "en_lines", "en_partut"]
        combined_train = []
        combined_val = []
        
        for tb in english_treebanks:
            try:
                ds_train_tb = load_dataset("universal_dependencies", tb, split="train", trust_remote_code=True)
                ds_val_tb = load_dataset("universal_dependencies", tb, split="validation", trust_remote_code=True)
                combined_train.extend(list(ds_train_tb))
                combined_val.extend(list(ds_val_tb))
                print(f"  ✓ Loaded {tb}: {len(ds_train_tb)} train, {len(ds_val_tb)} val sentences")
            except Exception as e:
                print(f"  ❌ Failed to load {tb}: {e}")
        
        ds_train = Dataset.from_list(combined_train)
        ds_val = Dataset.from_list(combined_val)
        print(f"🎯 Combined dataset: {len(ds_train)} train, {len(ds_val)} val sentences")
        
        return ds_train, ds_val
    
    def _load_combined_penn_data(self):
        """Load combined UD + Penn data."""
        from datasets import load_dataset, Dataset
        
        # Load UD first
        english_treebanks = ["en_ewt", "en_gum", "en_lines", "en_partut"]
        combined_train = []
        combined_val = []
        
        for tb in english_treebanks:
            try:
                ds_train_tb = load_dataset("universal_dependencies", tb, split="train", trust_remote_code=True)
                ds_val_tb = load_dataset("universal_dependencies", tb, split="validation", trust_remote_code=True)
                combined_train.extend(list(ds_train_tb))
                combined_val.extend(list(ds_val_tb))
                print(f"  ✓ Loaded UD {tb}: {len(ds_train_tb)} train, {len(ds_val_tb)} val sentences")
            except Exception as e:
                print(f"  ❌ Failed to load UD {tb}: {e}")
        
        # Add Penn Treebank
        penn_train, penn_val, penn_test = load_penn_treebank_data(getattr(self.args, 'penn_path', None))
        combined_train.extend(penn_train)
        combined_val.extend(penn_val)
        print(f"  ✓ Added Penn Treebank: {len(penn_train)} train, {len(penn_val)} val sentences")
        
        ds_train = Dataset.from_list(combined_train)
        ds_val = Dataset.from_list(combined_val)
        print(f"🎯 Combined UD+Penn dataset: {len(ds_train)} train, {len(ds_val)} val sentences")
        
        return ds_train, ds_val
    
    def _create_standard_dataloader(self, dataset, batch_size, shuffle=True):
        """Create standard PyTorch DataLoader."""
        return DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            collate_fn=lambda batch: collate(batch, self.vocab),
            num_workers=48 if self.args.compute_node else 4,
            pin_memory=True,
            prefetch_factor=4 if self.args.compute_node else 2
        )
    
    def _train_epoch_standard(self, epoch: int) -> Tuple[float, float, float]:
        """Standard training epoch with cosine annealing."""
        self.model.train()
        total_loss, total_tok, correct = 0.0, 0, 0
        
        all_preds = []
        all_targets = []
        
        pbar = tqdm(self.train_loader, desc=f"Train Epoch {epoch}", leave=True)
        for batch_data in pbar:
            inputs, upos, mask = batch_data
            
            # Handle both traditional (tensor) and hash-based (list) inputs
            if isinstance(inputs, torch.Tensor):
                inputs = inputs.to(self.device, non_blocking=True)
            # For hash embeddings, inputs is a list and doesn't need GPU transfer
            
            upos = upos.to(self.device, non_blocking=True)
            mask = mask.to(self.device, non_blocking=True)
            
            if self.scaler is not None:
                # Mixed precision training
                with torch.cuda.amp.autocast():
                    logp = self.model(inputs, mask)
                    if self.criterion is not None:
                        loss = self.criterion(logp.transpose(1,2), upos)
                    else:
                        loss = F.nll_loss(logp.transpose(1,2), upos, reduction="sum", ignore_index=-100)
                
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard training
                logp = self.model(inputs, mask)
                if self.criterion is not None:
                    loss = self.criterion(logp.transpose(1,2), upos)
                else:
                    loss = F.nll_loss(logp.transpose(1,2), upos, reduction="sum", ignore_index=-100)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            total_tok += mask.sum().item()
            pred = logp.argmax(-1)
            correct += ((pred == upos) & mask).sum().item()
            
            # Collect predictions for F1 if needed
            if not self.args.no_f1:
                valid_mask = mask & (upos != -100)
                if valid_mask.any():
                    all_preds.extend(pred[valid_mask].cpu().numpy())
                    all_targets.extend(upos[valid_mask].cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / total_tok if total_tok > 0 else float('inf')
        ppl = math.exp(avg_loss) if avg_loss < 100 else float('inf')
        acc = correct / total_tok if total_tok > 0 else 0.0
        
        # Calculate F1 score
        f1 = 0.0
        if not self.args.no_f1 and all_preds:
            try:
                f1 = f1_score(all_targets, all_preds, average=self.args.f1_average, zero_division=0)
            except Exception as e:
                print(f"Warning: Could not calculate F1: {e}")
        
        return ppl, acc, f1
    
    def _train_epoch_sgdr(self, epoch: int) -> Tuple[float, float, float]:
        """SGDR training epoch with step-based learning rate scheduling."""
        self.model.train()
        total_loss, total_tok, correct = 0.0, 0, 0
        current_loader = self.train_loader
        
        # Track batch size changes for reporting
        batch_size_changes = []
        
        # Create progress bar with epoch and batch size info
        initial_batch_size = self.adaptive_batch_sizer.get_current_batch_size() if self.adaptive_batch_sizer else 512
        desc = f"Train Epoch {epoch} [BS={initial_batch_size}]" if self.adaptive_batch_sizer else f"Train Epoch {epoch}"
        
        # Create progress bar object that we can update
        pbar = tqdm(current_loader, desc=desc, leave=True)
        for batch_idx, batch_data in enumerate(pbar):
            inputs, upos, mask = batch_data
            # Update batch size with adaptive batch sizer
            if self.adaptive_batch_sizer is not None:
                old_batch_size = self.adaptive_batch_sizer.get_current_batch_size()
                new_batch_size = self.adaptive_batch_sizer.update_batch_size(
                    self.model, current_loader, self.device, self.criterion, epoch
                )
                
                # Recreate DataLoader if batch size changed significantly
                if abs(new_batch_size - old_batch_size) > 0.1 * old_batch_size and batch_idx < len(current_loader) - 1:
                    batch_size_changes.append({
                        'batch_idx': batch_idx,
                        'old_size': old_batch_size,
                        'new_size': new_batch_size,
                        'stats': self.adaptive_batch_sizer.get_statistics()
                    })
                    
                    # Update progress bar description with new batch size
                    pbar.set_description(f"Train Epoch {epoch} [BS={new_batch_size}]")
            
            # Non-blocking transfers to GPU
            # Handle both traditional (tensor) and hash-based (list) inputs
            if isinstance(inputs, torch.Tensor):
                inputs = inputs.to(self.device, non_blocking=True)
            # For hash embeddings, inputs is a list and doesn't need GPU transfer
            
            upos = upos.to(self.device, non_blocking=True)
            mask = mask.to(self.device, non_blocking=True)
            
            if self.scaler is not None:
                # Mixed precision training
                with torch.cuda.amp.autocast():
                    logp = self.model(inputs, mask)
                    if self.criterion is not None:
                        loss = self.criterion(logp.transpose(1,2), upos)
                    else:
                        loss = F.nll_loss(logp.transpose(1,2), upos, reduction="sum", ignore_index=-100)
                
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard training
                logp = self.model(inputs, mask)
                if self.criterion is not None:
                    loss = self.criterion(logp.transpose(1,2), upos)
                else:
                    loss = F.nll_loss(logp.transpose(1,2), upos, reduction="sum", ignore_index=-100)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # SGDR: Step the scheduler after each batch
            self.scheduler.step()
            self.step_count += 1

            total_loss += loss.item()
            total_tok += mask.sum().item()
            pred = logp.argmax(-1)
            correct += ((pred == upos) & mask).sum().item()

        # Calculate metrics
        avg_loss = total_loss / total_tok if total_tok > 0 else float('inf')
        ppl = math.exp(avg_loss) if avg_loss < 100 else float('inf')
        acc = correct / total_tok if total_tok > 0 else 0.0
        
        # Report batch size changes
        if batch_size_changes:
            for change in batch_size_changes[-1:]:  # Show last change
                stats = change['stats']
                noise_str = f"(noise ratio: {stats['noise_ratio']:.3f})" if stats.get('noise_ratio') else ""
                print(f"   📦 Batch size: {change['old_size']} → {change['new_size']} {noise_str}")
        
        # Update data loader if batch size changed
        if self.adaptive_batch_sizer:
            current_batch_size = self.adaptive_batch_sizer.get_current_batch_size()
            if current_batch_size != initial_batch_size:
                # Recreate the dataloader with new batch size
                num_workers_train = 48 if self.args.compute_node else 4
                prefetch_factor = 4 if self.args.compute_node else 2
                self.train_loader = create_adaptive_dataloader(
                    self.train_dataset, self.adaptive_batch_sizer, collate,
                    num_workers_train, True, prefetch_factor
                )
        
        return ppl, acc, 0.0  # F1 calculation is not done during SGDR training
    
    def _get_monitor_metric(self, metrics: Dict) -> float:
        """Get the metric value for monitoring."""
        if self.args.monitor == 'val_loss':
            return -metrics['val_ppl']  # Negative because we want to maximize
        elif self.args.monitor == 'val_acc':
            return metrics['val_acc']
        elif self.args.monitor == 'val_ppl':
            return -metrics['val_ppl']
        elif self.args.monitor in ['macro_f1', 'weighted_f1']:
            return metrics['val_f1']
        else:
            return metrics['val_acc']
    
    def _save_checkpoint(self, epoch: int):
        """Save model checkpoint."""
        checkpoint_path = f"checkpoint_epoch_{epoch}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'training_history': self.training_history,
            'best_metric': self.best_metric
        }, checkpoint_path)
        print(f"📁 Saved checkpoint: {checkpoint_path}") 