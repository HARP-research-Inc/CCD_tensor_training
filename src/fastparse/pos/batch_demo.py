#!/usr/bin/env python3
"""
Demo script showing the large-scale batch processing functionality.
"""

import subprocess
import sys

def run_batch_test(num_sentences=1000, batch_size=512, benchmark=True):
    """Run a batch test with specified parameters."""
    cmd = [
        sys.executable, "pos_inference.py",
        "--batch",
        "--num-sentences", str(num_sentences),
        "--batch-size", str(batch_size),
    ]
    
    if benchmark:
        cmd.append("--benchmark")
    
    print(f"🚀 Running batch test: {num_sentences} sentences, batch size {batch_size}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("❌ Test timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"❌ Error running test: {e}")
        return False

def main():
    print("📊 POS Tagger Batch Processing Demo")
    print("=" * 50)
    
    print("\n🔍 Available batch testing options:")
    print("-" * 40)
    print("1. Quick test (500 sentences, our model only)")
    print("2. Medium test (1000 sentences, with benchmarks)")
    print("3. Large test (2000 sentences, with benchmarks)")
    print("4. Massive test (5000 sentences, our model only)")
    print("5. Custom test")
    print("6. Show usage examples")
    
    while True:
        try:
            choice = input("\nSelect option (1-6, or 'q' to quit): ").strip()
            
            if choice.lower() == 'q':
                break
            elif choice == '1':
                run_batch_test(500, 512, False)
            elif choice == '2':
                run_batch_test(1000, 512, True)
            elif choice == '3':
                run_batch_test(2000, 512, True)
            elif choice == '4':
                run_batch_test(5000, 1024, False)
            elif choice == '5':
                try:
                    num_sent = int(input("Number of sentences (default 1000): ") or "1000")
                    batch_size = int(input("Batch size (default 512): ") or "512")
                    benchmark = input("Include benchmarks? (y/n, default y): ").lower() != 'n'
                    run_batch_test(num_sent, batch_size, benchmark)
                except ValueError:
                    print("❌ Invalid input, please enter numbers")
            elif choice == '6':
                show_usage_examples()
            else:
                print("❌ Invalid choice")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break

def show_usage_examples():
    """Show usage examples for batch testing."""
    print("\n📖 Batch Testing Usage Examples:")
    print("=" * 50)
    
    examples = [
        {
            "desc": "Quick test with 500 sentences (our model only)",
            "cmd": "python pos_inference.py --batch --num-sentences 500"
        },
        {
            "desc": "Benchmark comparison with 1000 sentences", 
            "cmd": "python pos_inference.py --batch --num-sentences 1000 --benchmark"
        },
        {
            "desc": "Large-scale test with custom batch size",
            "cmd": "python pos_inference.py --batch --num-sentences 2000 --batch-size 1024"
        },
        {
            "desc": "Maximum performance test (5000 sentences)",
            "cmd": "python pos_inference.py --batch --num-sentences 5000 --batch-size 1024"
        },
        {
            "desc": "Use different treebank for test data",
            "cmd": "python pos_inference.py --batch --treebank en_gum --num-sentences 1000"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['desc']}")
        print(f"   {example['cmd']}")
    
    print(f"\n💡 Performance Tips:")
    print("• Use larger batch sizes (1024+) for maximum GPU utilization")
    print("• Add --benchmark to compare with NLTK and spaCy")
    print("• Our model is typically 3-5x faster than alternatives")
    print("• Expect 2000-5000+ tokens/sec on modern GPUs")

if __name__ == "__main__":
    main() 