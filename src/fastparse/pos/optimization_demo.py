#!/usr/bin/env python3
"""
Demo script showing automatic batch size optimization functionality.
"""

import subprocess
import sys
import json
import os

def run_optimization():
    """Run batch size optimization."""
    cmd = [sys.executable, "pos_inference.py", "--optimize-batch-size"]
    
    print("🎯 Running batch size optimization...")
    print("This will test different batch sizes to find the optimal one for your GPU")
    print("=" * 60)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("❌ Optimization timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"❌ Error running optimization: {e}")
        return False

def show_config(config_file="batch_config.json"):
    """Display the saved configuration."""
    if not os.path.exists(config_file):
        print(f"❌ No config file found: {config_file}")
        return
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        print(f"\n📋 SAVED CONFIGURATION:")
        print("=" * 40)
        print(f"Config file: {config_file}")
        print(f"Timestamp: {config.get('timestamp', 'Unknown')}")
        print(f"GPU: {config.get('gpu_info', 'Unknown')}")
        print(f"GPU Memory: {config.get('gpu_memory', 'Unknown')}")
        print(f"Optimal batch size: {config.get('optimal_batch_size', 'Unknown'):,}")
        print(f"Peak throughput: {config.get('optimal_throughput', 0):.0f} tokens/sec")
        
        if 'all_results' in config:
            print(f"\nTop performing batch sizes:")
            for i, result in enumerate(config['all_results'][:3], 1):
                print(f"  {i}. Batch {result['batch_size']:,}: {result['throughput']:.0f} tokens/sec")
        
    except Exception as e:
        print(f"❌ Error reading config: {e}")

def test_optimized_batch():
    """Test using the optimized batch size."""
    cmd = [sys.executable, "pos_inference.py", "--batch", "--num-sentences", "2000"]
    
    print("\n🚀 Testing with optimized batch size...")
    print("This should automatically use the optimal batch size we just found")
    print("=" * 60)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("❌ Test timed out after 2 minutes")
        return False
    except Exception as e:
        print(f"❌ Error running test: {e}")
        return False

def main():
    print("🎯 BATCH SIZE OPTIMIZATION DEMO")
    print("=" * 50)
    print("This demo will:")
    print("1. Find the optimal batch size for your GPU")
    print("2. Save it to a configuration file")
    print("3. Show how future runs automatically use it")
    
    response = input("\nWould you like to run the optimization? (y/n): ").strip().lower()
    if response != 'y':
        print("👋 Exiting...")
        return
    
    # Step 1: Run optimization
    print(f"\n{'='*60}")
    print("STEP 1: BATCH SIZE OPTIMIZATION")
    print(f"{'='*60}")
    
    success = run_optimization()
    if not success:
        print("❌ Optimization failed")
        return
    
    # Step 2: Show saved config
    print(f"\n{'='*60}")
    print("STEP 2: CONFIGURATION RESULTS")
    print(f"{'='*60}")
    
    show_config()
    
    # Step 3: Test using optimized batch size
    print(f"\n{'='*60}")
    print("STEP 3: AUTOMATIC OPTIMIZATION IN ACTION")
    print(f"{'='*60}")
    
    response = input("\nTest the optimized configuration? (y/n): ").strip().lower()
    if response == 'y':
        test_optimized_batch()
    
    # Summary
    print(f"\n{'='*60}")
    print("🎉 OPTIMIZATION DEMO COMPLETE!")
    print(f"{'='*60}")
    print("✅ Your GPU's optimal batch size has been found and saved")
    print("✅ Future runs will automatically use this optimal configuration")
    print("✅ No need to manually specify batch sizes anymore!")
    
    print(f"\n💡 WHAT HAPPENS NOW:")
    print("• All future --batch commands will use the optimal batch size")
    print("• Your configuration is saved in batch_config.json")
    print("• Re-run --optimize-batch-size if you upgrade your GPU")
    print("• Manual batch sizes (--batch-size X) still override the optimization")
    
    print(f"\n🚀 TRY THESE COMMANDS:")
    print("python pos_inference.py --batch --num-sentences 5000")
    print("python pos_inference.py --stress-test")
    print("python pos_inference.py --batch --benchmark --num-sentences 1000")

if __name__ == "__main__":
    main() 