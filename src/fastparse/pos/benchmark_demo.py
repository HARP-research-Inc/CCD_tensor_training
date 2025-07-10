#!/usr/bin/env python3
"""
Demonstration script for the POS tagger benchmark functionality.
Shows accuracy and speed comparisons with NLTK and spaCy.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pos_inference import POSPredictor, print_comparison_results

def main():
    print("🚀 POS Tagger Benchmark Demo")
    print("=" * 50)
    
    # Test sentences with expected tags for accuracy testing
    test_cases = [
        {
            "text": "The quick brown fox jumps over the lazy dog.",
            "expected": ["DET", "ADJ", "ADJ", "NOUN", "VERB", "ADP", "DET", "ADJ", "NOUN", "PUNCT"]
        },
        {
            "text": "I love artificial intelligence and machine learning.",
            "expected": ["PRON", "VERB", "ADJ", "NOUN", "CCONJ", "NOUN", "NOUN", "PUNCT"]
        },
        {
            "text": "OpenAI has developed remarkable language models.",
            "expected": ["PROPN", "AUX", "VERB", "ADJ", "NOUN", "NOUN", "PUNCT"]
        },
        {
            "text": "She quickly ran to the store yesterday.",
            "expected": ["PRON", "ADV", "VERB", "ADP", "DET", "NOUN", "ADV", "PUNCT"]
        }
    ]
    
    # Initialize predictor
    try:
        predictor = POSPredictor("router_en_gum.pt", "en_gum")
    except FileNotFoundError:
        print("❌ Model file 'router_en_gum.pt' not found!")
        print("Please train the model first using pos_router_train.py")
        return
    
    print("\n🏆 Running benchmark comparisons...")
    
    total_our_time = 0
    total_nltk_time = 0
    total_spacy_time = 0
    total_tokens = 0
    
    our_correct = 0
    nltk_correct = 0
    spacy_correct = 0
    total_accuracy_tokens = 0
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🔍 Test Case {i}/{len(test_cases)}")
        print("-" * 40)
        
        # Create expected tags tuple
        tokens = predictor.tokenize(test_case["text"])
        expected_tags = list(zip(tokens, test_case["expected"])) if len(tokens) == len(test_case["expected"]) else None
        
        # Run comparison
        results = predictor.compare_with_baselines(test_case["text"], expected_tags)
        print_comparison_results(results)
        
        # Aggregate statistics
        total_tokens += len(tokens)
        if "our_model" in results:
            total_our_time += results["our_model"]["time_ms"]
            if "accuracy" in results["our_model"]:
                our_correct += results["our_model"]["correct"]
                total_accuracy_tokens += results["our_model"]["total"]
        
        if "nltk" in results and "time_ms" in results["nltk"]:
            total_nltk_time += results["nltk"]["time_ms"]
            if "accuracy" in results["nltk"]:
                nltk_correct += results["nltk"]["correct"]
        
        if "spacy" in results and "time_ms" in results["spacy"]:
            total_spacy_time += results["spacy"]["time_ms"]
            if "accuracy" in results["spacy"]:
                spacy_correct += results["spacy"]["correct"]
    
    # Print overall summary
    print(f"\n{'='*80}")
    print("📊 OVERALL BENCHMARK SUMMARY")
    print(f"{'='*80}")
    print(f"Total tokens processed: {total_tokens}")
    print(f"Total sentences: {len(test_cases)}")
    
    print(f"\n⏱️  SPEED COMPARISON:")
    print(f"{'Model':<12} {'Total Time (ms)':<15} {'Avg Time/Token (ms)':<20} {'Tokens/sec':<12}")
    print("-" * 70)
    
    if total_our_time > 0:
        our_avg = total_our_time / total_tokens
        our_speed = total_tokens / (total_our_time / 1000)
        print(f"{'Our Model':<12} {total_our_time:<15.1f} {our_avg:<20.3f} {our_speed:<12.0f}")
    
    if total_nltk_time > 0:
        nltk_avg = total_nltk_time / total_tokens
        nltk_speed = total_tokens / (total_nltk_time / 1000)
        print(f"{'NLTK':<12} {total_nltk_time:<15.1f} {nltk_avg:<20.3f} {nltk_speed:<12.0f}")
    
    if total_spacy_time > 0:
        spacy_avg = total_spacy_time / total_tokens
        spacy_speed = total_tokens / (total_spacy_time / 1000)
        print(f"{'spaCy':<12} {total_spacy_time:<15.1f} {spacy_avg:<20.3f} {spacy_speed:<12.0f}")
    
    print(f"\n🎯 ACCURACY COMPARISON:")
    print(f"{'Model':<12} {'Correct':<8} {'Total':<8} {'Accuracy':<10}")
    print("-" * 40)
    
    if total_accuracy_tokens > 0:
        our_acc = our_correct / total_accuracy_tokens
        print(f"{'Our Model':<12} {our_correct:<8} {total_accuracy_tokens:<8} {our_acc*100:<10.1f}%")
        
        nltk_acc = nltk_correct / total_accuracy_tokens
        print(f"{'NLTK':<12} {nltk_correct:<8} {total_accuracy_tokens:<8} {nltk_acc*100:<10.1f}%")
        
        spacy_acc = spacy_correct / total_accuracy_tokens
        print(f"{'spaCy':<12} {spacy_correct:<8} {total_accuracy_tokens:<8} {spacy_acc*100:<10.1f}%")
    
    print(f"\n💡 INSIGHTS:")
    if total_our_time > 0 and total_nltk_time > 0:
        speedup_nltk = total_nltk_time / total_our_time
        print(f"• Our model is {speedup_nltk:.1f}x faster than NLTK")
    
    if total_our_time > 0 and total_spacy_time > 0:
        speedup_spacy = total_spacy_time / total_our_time
        print(f"• Our model is {speedup_spacy:.1f}x faster than spaCy")
    
    if total_accuracy_tokens > 0:
        print(f"• Our model accuracy: {our_acc*100:.1f}%")
        print(f"• Model size: Tiny CNN (64D embeddings, 3-width kernel)")
        print(f"• Model parameters: ~{(len(predictor.vocab) * 64 + 64*64*3 + 64*64 + 64*18) / 1000:.1f}K")

if __name__ == "__main__":
    main() 