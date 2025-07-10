#!/usr/bin/env python3
"""
Test script to verify GPU optimizations are working correctly.
Shows before/after GPU utilization improvements.
"""

import torch
import time
import subprocess
import psutil

def check_gpu_utilization():
    """Check current GPU utilization using nvidia-smi if available."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            gpu_util, mem_used, mem_total = result.stdout.strip().split(", ")
            return float(gpu_util), float(mem_used), float(mem_total)
    except Exception:
        pass
    return None, None, None

def benchmark_training_step():
    """Benchmark a single training step to show optimization effects."""
    print("🔍 GPU Optimization Test")
    print("=" * 50)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    if device == "cpu":
        print("❌ No GPU available - cannot test GPU optimizations")
        return
    
    # Enable optimizations
    torch.backends.cudnn.benchmark = True
    
    # Create a dummy model similar to our POS tagger
    from pos_router_train import DepthWiseCNNRouter
    
    vocab_size = 10000
    model = DepthWiseCNNRouter(vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler()
    
    # Create dummy batch data
    batch_size = 2048
    seq_len = 32
    
    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {seq_len}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Warm up
    print("\n🔥 Warming up...")
    for _ in range(5):
        ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
        upos = torch.randint(0, 18, (batch_size, seq_len), device=device)
        
        with torch.cuda.amp.autocast():
            logp = model(ids, mask)
            loss = torch.nn.functional.nll_loss(
                logp.transpose(1,2), upos, reduction="sum", ignore_index=-100
            )
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
    torch.cuda.synchronize()
    
    # Benchmark
    print("\n⚡ Benchmarking optimized training...")
    start_time = time.time()
    gpu_util_before, mem_before, mem_total = check_gpu_utilization()
    
    num_steps = 20
    for step in range(num_steps):
        ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
        upos = torch.randint(0, 18, (batch_size, seq_len), device=device)
        
        with torch.cuda.amp.autocast():
            logp = model(ids, mask)
            loss = torch.nn.functional.nll_loss(
                logp.transpose(1,2), upos, reduction="sum", ignore_index=-100
            )
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        if step == 10:  # Check GPU utilization mid-training
            gpu_util_during, mem_during, _ = check_gpu_utilization()
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    gpu_util_after, mem_after, _ = check_gpu_utilization()
    
    total_time = end_time - start_time
    steps_per_sec = num_steps / total_time
    tokens_per_sec = (batch_size * seq_len * num_steps) / total_time
    
    print(f"\n📊 BENCHMARK RESULTS:")
    print(f"Training steps: {num_steps}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Steps/sec: {steps_per_sec:.1f}")
    print(f"Tokens/sec: {tokens_per_sec:,.0f}")
    
    if gpu_util_before is not None:
        print(f"\n🎯 GPU UTILIZATION:")
        print(f"Before: {gpu_util_before}%")
        print(f"During: {gpu_util_during}%")
        print(f"After: {gpu_util_after}%")
        print(f"Memory used: {mem_during:.0f}/{mem_total:.0f} MB ({mem_during/mem_total*100:.1f}%)")
        
        if gpu_util_during > 50:
            print("✅ GPU utilization looks good!")
        elif gpu_util_during > 20:
            print("⚠️  GPU utilization is moderate - consider larger batch size")
        else:
            print("❌ Low GPU utilization - check if optimizations are working")
    else:
        print("❌ Could not check GPU utilization (nvidia-smi not available)")
    
    print(f"\n💡 OPTIMIZATION STATUS:")
    print(f"✅ cuDNN benchmark: {torch.backends.cudnn.benchmark}")
    print(f"✅ Mixed precision: AMP enabled")
    print(f"✅ Device: {device}")
    print(f"✅ Model on GPU: {next(model.parameters()).device}")

if __name__ == "__main__":
    benchmark_training_step() 