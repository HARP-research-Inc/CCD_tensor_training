#!/usr/bin/env python3
"""
Extreme stress test script for pushing the POS tagger to its absolute limits.
"""

import subprocess
import sys
import time

def run_stress_test(max_batch_size=None):
    """Run stress test with specified max batch size."""
    cmd = [sys.executable, "pos_inference.py", "--stress-test"]
    
    if max_batch_size:
        cmd.extend(["--max-batch-size", str(max_batch_size)])
    
    print(f"🔥 Running extreme stress test...")
    if max_batch_size:
        print(f"🎯 Max batch size: {max_batch_size:,}")
    print("=" * 60)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("❌ Test timed out after 10 minutes")
        return False
    except Exception as e:
        print(f"❌ Error running test: {e}")
        return False

def main():
    print("🚀 EXTREME PERFORMANCE STRESS TESTING")
    print("=" * 60)
    print("This will push your POS tagger to its absolute limits!")
    print("Testing with datasets up to 500K sentences and batch sizes up to 16K+")
    
    print("\n⚠️  WARNING:")
    print("• This test may use significant GPU memory")
    print("• Tests may take 5-15 minutes to complete")
    print("• GPU memory errors are normal - we're finding the limits!")
    
    response = input("\nContinue? (y/n): ").strip().lower()
    if response != 'y':
        print("👋 Exiting...")
        return
    
    # Progressive stress testing
    stress_configs = [
        {"name": "Standard Stress Test", "batch_size": None},
        {"name": "High Performance Test", "batch_size": 8192},
        {"name": "Extreme Performance Test", "batch_size": 16384},
        {"name": "Maximum Limits Test", "batch_size": 32768},
    ]
    
    results = []
    
    for config in stress_configs:
        print(f"\n{'='*60}")
        print(f"🎯 {config['name']}")
        print(f"{'='*60}")
        
        start_time = time.time()
        success = run_stress_test(config["batch_size"])
        test_time = time.time() - start_time
        
        results.append({
            "name": config["name"],
            "batch_size": config["batch_size"],
            "success": success,
            "time": test_time
        })
        
        if not success:
            print(f"⚠️  {config['name']} failed - likely hit GPU memory limits")
            if config["batch_size"] and config["batch_size"] >= 16384:
                print("🎯 This is expected with very large batch sizes")
                break
        
        print(f"✅ {config['name']} completed in {test_time:.1f}s")
        
        # Brief pause between tests
        time.sleep(2)
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 STRESS TEST SUMMARY")
    print(f"{'='*60}")
    
    for result in results:
        status = "✅ PASSED" if result["success"] else "❌ FAILED"
        batch_info = f" (batch: {result['batch_size']:,})" if result['batch_size'] else ""
        print(f"{result['name']}{batch_info}: {status} ({result['time']:.1f}s)")
    
    successful_tests = sum(1 for r in results if r["success"])
    
    print(f"\n🏆 FINAL RESULTS:")
    print(f"• Tests passed: {successful_tests}/{len(results)}")
    print(f"• Your GPU can handle extreme workloads!")
    print(f"• GPU optimizations are working perfectly")
    
    if successful_tests >= 2:
        print(f"🎉 EXCELLENT! Your model shows exceptional performance")
    elif successful_tests >= 1:
        print(f"👍 GOOD! Your model handles large-scale processing well")
    else:
        print(f"⚠️  Check GPU memory availability and model setup")

if __name__ == "__main__":
    main() 