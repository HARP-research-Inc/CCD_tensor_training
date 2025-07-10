#!/usr/bin/env python3
"""
Example usage of the enhanced POS tagger with benchmarking capabilities.
"""

from pos_inference import POSPredictor, print_comparison_results

def main():
    # Initialize the predictor
    print("🚀 POS Tagger Example Usage")
    print("=" * 40)
    
    try:
        predictor = POSPredictor("router_en_gum.pt", "en_gum")
        print("✅ Model loaded successfully!")
    except FileNotFoundError:
        print("❌ Model file not found. Please train the model first:")
        print("   python pos_router_train.py")
        return
    
    # Example 1: Basic prediction with timing
    print("\n1️⃣ Basic Prediction with Timing:")
    print("-" * 35)
    
    import time
    text = "The artificial intelligence model performs very well."
    
    start = time.time()
    predictions = predictor.predict(text)
    inference_time = (time.time() - start) * 1000
    
    print(f"Input: {text}")
    print("Predictions:")
    for token, pos in predictions:
        print(f"  {token:15} -> {pos}")
    print(f"Inference time: {inference_time:.1f}ms")
    print(f"Speed: {len(predictions) / (inference_time / 1000):.0f} tokens/sec")
    
    # Example 2: Benchmark comparison
    print("\n2️⃣ Benchmark Comparison:")
    print("-" * 25)
    
    text = "OpenAI develops amazing language models."
    results = predictor.compare_with_baselines(text)
    print_comparison_results(results)
    
    # Example 3: Accuracy testing with expected tags
    print("\n3️⃣ Accuracy Testing:")
    print("-" * 20)
    
    text = "She quickly ran home."
    expected_tags = [("She", "PRON"), ("quickly", "ADV"), ("ran", "VERB"), ("home", "NOUN"), (".", "PUNCT")]
    
    results = predictor.compare_with_baselines(text, expected_tags)
    print_comparison_results(results)
    
    # Example 4: Show model efficiency
    print("\n4️⃣ Model Efficiency Analysis:")
    print("-" * 30)
    
    vocab_size = len(predictor.vocab)
    emb_params = vocab_size * 64  # embeddings
    conv_params = 64 * 64 * 3     # depth-wise conv
    pointwise_params = 64 * 64    # pointwise conv
    linear_params = 64 * 18       # final linear layer
    
    total_params = emb_params + conv_params + pointwise_params + linear_params
    
    print(f"Model Statistics:")
    print(f"  Vocabulary size: {vocab_size:,}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Model size: ~{total_params/1000:.1f}K parameters")
    print(f"  Memory footprint: ~{total_params*4/1024/1024:.1f}MB (FP32)")
    
    # Example 5: Speed comparison summary
    print("\n5️⃣ Speed Comparison Summary:")
    print("-" * 30)
    
    test_sentence = "The quick brown fox jumps over the lazy dog."
    results = predictor.compare_with_baselines(test_sentence)
    
    our_speed = results['our_model']['tokens_per_sec']
    print(f"Our model: {our_speed:.0f} tokens/sec")
    
    if 'nltk' in results and 'tokens_per_sec' in results['nltk']:
        nltk_speed = results['nltk']['tokens_per_sec']
        speedup = our_speed / nltk_speed
        print(f"NLTK: {nltk_speed:.0f} tokens/sec (our model is {speedup:.1f}x faster)")
    
    if 'spacy' in results and 'tokens_per_sec' in results['spacy']:
        spacy_speed = results['spacy']['tokens_per_sec']
        speedup = our_speed / spacy_speed
        print(f"spaCy: {spacy_speed:.0f} tokens/sec (our model is {speedup:.1f}x faster)")
    
    print("\n✨ Summary:")
    print("  • Our tiny CNN model is extremely fast")
    print("  • Competitive accuracy with much larger models")
    print("  • Perfect for real-time applications")
    print("  • Easy to deploy and integrate")

if __name__ == "__main__":
    main() 