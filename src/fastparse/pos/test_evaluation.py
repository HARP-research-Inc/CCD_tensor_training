#!/usr/bin/env python3
# test_evaluation.py
#
# Comprehensive evaluation of the POS tagger with various test cases

from pos_inference import POSPredictor
import time

# Test sentences with expected POS tags (roughly)
test_cases = [
    # Basic sentences
    ("The cat sits on the mat.", [
        ("The", "DET"), ("cat", "NOUN"), ("sits", "VERB"), ("on", "ADP"), 
        ("the", "DET"), ("mat", "NOUN"), (".", "PUNCT")
    ]),
    ("Bob likes Mary.", [
        ("Bob", "PROPN"), ("likes", "VERB"), ("Mary", "PROPN"), (".", "PUNCT")
    ]),
    ("I once met a man from Peru.", [
        ("I", "PRON"), ("once", "ADV"), ("met", "VERB"), ("a", "DET"), 
        ("man", "NOUN"), ("from", "ADP"), ("Peru", "PROPN"), (".", "PUNCT")
    ]),
    
    # More complex sentences
    ("The quick brown fox jumps over the lazy dog.", [
        ("The", "DET"), ("quick", "ADJ"), ("brown", "ADJ"), ("fox", "NOUN"),
        ("jumps", "VERB"), ("over", "ADP"), ("the", "DET"), ("lazy", "ADJ"), 
        ("dog", "NOUN"), (".", "PUNCT")
    ]),
    ("She is running quickly through the park.", [
        ("She", "PRON"), ("is", "AUX"), ("running", "VERB"), ("quickly", "ADV"),
        ("through", "ADP"), ("the", "DET"), ("park", "NOUN"), (".", "PUNCT")
    ]),
    ("They have been working hard all day.", [
        ("They", "PRON"), ("have", "AUX"), ("been", "AUX"), ("working", "VERB"),
        ("hard", "ADV"), ("all", "DET"), ("day", "NOUN"), (".", "PUNCT")
    ]),
    
    # Tricky cases
    ("Dr. Smith works at IBM.", [
        ("Dr", "NOUN"), (".", "PUNCT"), ("Smith", "PROPN"), ("works", "VERB"),
        ("at", "ADP"), ("IBM", "PROPN"), (".", "PUNCT")
    ]),
    ("It's a beautiful day, isn't it?", [
        ("It", "PRON"), ("'", "PUNCT"), ("s", "AUX"), ("a", "DET"), ("beautiful", "ADJ"),
        ("day", "NOUN"), (",", "PUNCT"), ("isn", "AUX"), ("'", "PUNCT"), ("t", "PART"),
        ("it", "PRON"), ("?", "PUNCT")
    ]),
]

def evaluate_model():
    """Run comprehensive evaluation of the POS tagger."""
    print("🔍 Loading POS Tagger...")
    predictor = POSPredictor("router_en_gum.pt")
    
    print("\n" + "="*80)
    print("COMPREHENSIVE POS TAGGER EVALUATION")
    print("="*80)
    
    total_tokens = 0
    correct_tokens = 0
    sentence_scores = []
    
    for i, (sentence, expected) in enumerate(test_cases, 1):
        print(f"\n📝 Test Case {i}: {sentence}")
        print("-" * 60)
        
        # Get predictions
        start_time = time.time()
        predictions = predictor.predict(sentence)
        inference_time = time.time() - start_time
        
        # Compare with expected (if available)
        expected_dict = {token: pos for token, pos in expected} if expected else {}
        
        sentence_correct = 0
        sentence_total = len(predictions)
        
        print("Predictions vs Expected:")
        for token, pred_pos in predictions:
            expected_pos = expected_dict.get(token, "?")
            is_correct = pred_pos == expected_pos if expected_pos != "?" else "?"
            
            status = "✓" if is_correct == True else "❌" if is_correct == False else "?"
            print(f"  {token:12} -> {pred_pos:6} (expected: {expected_pos:6}) {status}")
            
            if is_correct == True:
                sentence_correct += 1
                correct_tokens += 1
            elif is_correct == False:
                pass  # counted as incorrect
            total_tokens += 1
        
        # Calculate sentence accuracy
        if expected:
            sentence_acc = sentence_correct / sentence_total * 100
            sentence_scores.append(sentence_acc)
            print(f"\n   Sentence accuracy: {sentence_acc:.1f}% ({sentence_correct}/{sentence_total})")
        
        print(f"   Inference time: {inference_time*1000:.1f}ms")
    
    # Overall statistics
    print("\n" + "="*80)
    print("📊 OVERALL RESULTS")
    print("="*80)
    
    if sentence_scores:
        overall_acc = correct_tokens / total_tokens * 100
        avg_sentence_acc = sum(sentence_scores) / len(sentence_scores)
        
        print(f"Token-level accuracy:    {overall_acc:.1f}% ({correct_tokens}/{total_tokens})")
        print(f"Average sentence acc:    {avg_sentence_acc:.1f}%")
        print(f"Best sentence:           {max(sentence_scores):.1f}%")
        print(f"Worst sentence:          {min(sentence_scores):.1f}%")
    
    print(f"Total sentences tested:  {len(test_cases)}")
    print(f"Total tokens processed:  {total_tokens}")

def quick_test():
    """Quick test with a few sentences."""
    predictor = POSPredictor("router_en_gum.pt")
    
    quick_sentences = [
        "I once met a man from Peru.",
        "The quick brown fox jumps.",
        "She is running quickly.",
        "Dr. Smith works at IBM."
    ]
    
    print("\n🚀 QUICK TEST RESULTS:")
    print("-" * 40)
    
    for sentence in quick_sentences:
        predictions = predictor.predict(sentence)
        print(f"\n{sentence}")
        for token, pos in predictions:
            print(f"  {token:10} -> {pos}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        quick_test()
    else:
        evaluate_model() 