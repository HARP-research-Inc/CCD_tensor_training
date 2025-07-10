#!/usr/bin/env python3
# pos_inference.py
#
# Inference script for the trained POS router model

import torch
import torch.nn as nn
from datasets import load_dataset
import argparse
import re
import time
import nltk
from collections import defaultdict

# Same hyperparameters as training
EMB_DIM = 64
DW_KERNEL = 3
N_TAGS = 18  # Model trained with 18 classes (0-17) to match dataset
MAX_LEN = 64

# Correct POS tag mapping from dataset (18 tags: 0-17)
POS_TAGS = [
    "NOUN", "PUNCT", "ADP", "NUM", "SYM", "SCONJ", "ADJ", "PART", 
    "DET", "CCONJ", "PROPN", "PRON", "X", "_", "ADV", "INTJ", "VERB", "AUX"
]

class DepthWiseCNNRouter(nn.Module):
    """Same model architecture as training."""
    def __init__(self, vocab_size: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, EMB_DIM, padding_idx=0)
        self.dw = nn.Conv1d(
            EMB_DIM, EMB_DIM, kernel_size=DW_KERNEL,
            padding=DW_KERNEL // 2,
            groups=EMB_DIM, bias=True
        )
        self.pw = nn.Conv1d(EMB_DIM, EMB_DIM, kernel_size=1)
        self.act = nn.ReLU()
        self.lin = nn.Linear(EMB_DIM, N_TAGS)

    def forward(self, token_ids, mask):
        x = self.emb(token_ids)
        x = x.transpose(1, 2)
        x = self.pw(self.act(self.dw(x)))
        x = x.transpose(1, 2)
        logits = self.lin(x)
        logits = logits.masked_fill(~mask.unsqueeze(-1), -1e4)
        return torch.log_softmax(logits, dim=-1)

def build_vocab_from_dataset(treebank="en_gum"):
    """Build the same vocabulary as used in training."""
    print(f"Loading {treebank} dataset to rebuild vocabulary...")
    ds_train = load_dataset("universal_dependencies", treebank, split="train", trust_remote_code=True)
    
    vocab = {"<PAD>": 0}
    for ex in ds_train:
        for tok in ex["tokens"]:
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab

class POSPredictor:
    def __init__(self, model_path, treebank="en_gum"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Rebuild vocabulary (same as training)
        self.vocab = build_vocab_from_dataset(treebank)
        
        # Load model
        self.model = DepthWiseCNNRouter(len(self.vocab)).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        print(f"✓ Model loaded on {self.device}")
        print(f"✓ Vocabulary size: {len(self.vocab)}")

    def tokenize(self, text):
        """Lightweight tokenization that separates punctuation properly."""
        # Split on punctuation while keeping the punctuation as separate tokens
        # This handles cases like "peru." -> ["peru", "."]
        text = text.strip()
        
        # Simple regex to split on word boundaries and punctuation
        tokens = re.findall(r"\w+|[^\w\s]", text)
        
        return [token for token in tokens if token.strip()]
    
    def compare_with_baselines(self, text, expected_tags=None):
        """Compare our model with NLTK and spaCy baselines."""
        tokens = self.tokenize(text)
        if not tokens:
            return {}
            
        results = {"text": text, "tokens": tokens}
        
        # Our model prediction with timing
        start_time = time.time()
        our_predictions = self.predict(text)
        our_time = (time.time() - start_time) * 1000  # Convert to ms
        
        results["our_model"] = {
            "predictions": our_predictions,
            "time_ms": our_time,
            "tokens_per_sec": len(tokens) / (our_time / 1000) if our_time > 0 else 0
        }
        
        # NLTK baseline
        try:
            # Ensure required NLTK data is available
            try:
                nltk.data.find('taggers/averaged_perceptron_tagger')
            except LookupError:
                print("Downloading NLTK POS tagger...")
                nltk.download('averaged_perceptron_tagger', quiet=True)
            
            start_time = time.time()
            nltk_tagged = nltk.pos_tag(tokens)
            nltk_time = (time.time() - start_time) * 1000
            
            # Convert NLTK tags to Universal POS (simplified mapping)
            nltk_universal = []
            for word, tag in nltk_tagged:
                # Simple mapping from Penn Treebank to Universal POS
                if tag.startswith('N'):
                    uni_tag = 'NOUN' if not tag.startswith('NNP') else 'PROPN'
                elif tag.startswith('V'):
                    uni_tag = 'AUX' if tag in ['VBZ', 'VBP', 'VBD', 'VB'] and word.lower() in ['is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had'] else 'VERB'
                elif tag.startswith('J'):
                    uni_tag = 'ADJ'
                elif tag.startswith('R'):
                    uni_tag = 'ADV'
                elif tag in ['DT', 'PDT', 'WDT']:
                    uni_tag = 'DET'
                elif tag in ['IN']:
                    uni_tag = 'ADP'
                elif tag in ['PRP', 'PRP$', 'WP', 'WP$']:
                    uni_tag = 'PRON'
                elif tag in ['CC']:
                    uni_tag = 'CCONJ'
                elif tag in ['CD']:
                    uni_tag = 'NUM'
                elif tag in ['.', ',', ':', ';', '!', '?', '``', "''"]:
                    uni_tag = 'PUNCT'
                else:
                    uni_tag = 'X'
                nltk_universal.append((word, uni_tag))
            
            results["nltk"] = {
                "predictions": nltk_universal,
                "time_ms": nltk_time,
                "tokens_per_sec": len(tokens) / (nltk_time / 1000) if nltk_time > 0 else 0
            }
        except Exception as e:
            results["nltk"] = {"error": str(e)}
        
        # spaCy baseline (if available)
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
            
            start_time = time.time()
            doc = nlp(text)
            spacy_time = (time.time() - start_time) * 1000
            
            spacy_predictions = [(token.text, token.pos_) for token in doc]
            
            results["spacy"] = {
                "predictions": spacy_predictions,
                "time_ms": spacy_time,
                "tokens_per_sec": len(tokens) / (spacy_time / 1000) if spacy_time > 0 else 0
            }
        except Exception as e:
            results["spacy"] = {"error": f"spaCy not available: {e}"}
        
        # Calculate accuracy if expected tags provided
        if expected_tags:
            expected_dict = {token: pos for token, pos in expected_tags}
            
            for model_name in ["our_model", "nltk", "spacy"]:
                if model_name in results and "predictions" in results[model_name]:
                    predictions = results[model_name]["predictions"]
                    correct = sum(1 for token, pred_pos in predictions 
                                if expected_dict.get(token) == pred_pos)
                    total = len(predictions)
                    results[model_name]["accuracy"] = correct / total if total > 0 else 0
                    results[model_name]["correct"] = correct
                    results[model_name]["total"] = total
        
        return results

    def predict(self, text):
        """Predict POS tags for input text."""
        tokens = self.tokenize(text)
        if not tokens:
            return []
        
        # Convert to token IDs
        token_ids = [self.vocab.get(tok, 0) for tok in tokens][:MAX_LEN]
        
        # Create tensors
        ids = torch.tensor([token_ids]).to(self.device)
        mask = torch.ones_like(ids, dtype=torch.bool)
        
        # Get predictions
        with torch.no_grad():
            logp = self.model(ids, mask)
            pred_ids = logp.argmax(-1).squeeze(0).cpu().numpy()
        
        # Convert to POS tags
        predictions = []
        for i, (token, pred_id) in enumerate(zip(tokens, pred_ids)):
            if i < len(token_ids):  # Only for actual tokens
                pos_tag = POS_TAGS[pred_id] if pred_id < len(POS_TAGS) else "X"
                predictions.append((token, pos_tag))
        
        return predictions

def print_comparison_results(results):
    """Print formatted comparison results."""
    print(f"\n{'='*80}")
    print(f"🏆 POS TAGGER COMPARISON")
    print(f"{'='*80}")
    print(f"Input: {results['text']}")
    print(f"Tokens: {len(results['tokens'])}")
    
    # Performance table
    print(f"\n📊 PERFORMANCE COMPARISON:")
    print(f"{'Model':<12} {'Time (ms)':<10} {'Tokens/sec':<12} {'Accuracy':<10} {'Status'}")
    print("-" * 60)
    
    for model_name in ["our_model", "nltk", "spacy"]:
        if model_name in results:
            data = results[model_name]
            if "error" in data:
                print(f"{model_name:<12} {'N/A':<10} {'N/A':<12} {'N/A':<10} {data['error']}")
            else:
                time_str = f"{data['time_ms']:.1f}"
                speed_str = f"{data['tokens_per_sec']:.0f}"
                acc_str = f"{data.get('accuracy', 0)*100:.1f}%" if 'accuracy' in data else "N/A"
                status = "✓"
                print(f"{model_name:<12} {time_str:<10} {speed_str:<12} {acc_str:<10} {status}")
    
    # Detailed predictions
    print(f"\n🔍 DETAILED PREDICTIONS:")
    for model_name in ["our_model", "nltk", "spacy"]:
        if model_name in results and "predictions" in results[model_name]:
            print(f"\n{model_name.upper()}:")
            predictions = results[model_name]["predictions"]
            for token, pos in predictions:
                print(f"  {token:15} -> {pos}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="router_en_gum.pt", 
                        help="Path to saved model weights")
    parser.add_argument("--treebank", default="en_gum",
                        help="Treebank used for training (to rebuild vocab)")
    parser.add_argument("--text", type=str,
                        help="Text to analyze (if not provided, enters interactive mode)")
    parser.add_argument("--benchmark", action="store_true",
                        help="Compare with NLTK and spaCy baselines")
    parser.add_argument("--expected", type=str, nargs="*",
                        help="Expected POS tags for accuracy calculation (space-separated)")
    args = parser.parse_args()

    # Initialize predictor
    predictor = POSPredictor(args.model, args.treebank)

    if args.text:
        if args.benchmark:
            # Parse expected tags if provided
            expected_tags = None
            if args.expected:
                tokens = predictor.tokenize(args.text)
                if len(args.expected) == len(tokens):
                    expected_tags = list(zip(tokens, args.expected))
                else:
                    print(f"Warning: Expected {len(tokens)} tags, got {len(args.expected)}")
            
            # Run benchmark comparison
            results = predictor.compare_with_baselines(args.text, expected_tags)
            print_comparison_results(results)
        else:
            # Simple prediction mode
            start_time = time.time()
            predictions = predictor.predict(args.text)
            inference_time = (time.time() - start_time) * 1000
            
            print(f"\nInput: {args.text}")
            print("POS predictions:")
            for token, pos in predictions:
                print(f"  {token:15} -> {pos}")
            print(f"\nInference time: {inference_time:.1f}ms")
            print(f"Speed: {len(predictions) / (inference_time / 1000):.0f} tokens/sec")
    else:
        # Interactive mode
        print("\n" + "="*60)
        print("Interactive POS Tagger")
        print("Enter text to analyze (or 'quit' to exit)")
        if args.benchmark:
            print("🏆 Benchmark mode: comparing with NLTK and spaCy")
        print("="*60)
        
        while True:
            try:
                text = input("\n> ").strip()
                if text.lower() in ['quit', 'exit', 'q']:
                    break
                if not text:
                    continue
                
                if args.benchmark:
                    # Run benchmark comparison
                    results = predictor.compare_with_baselines(text)
                    print_comparison_results(results)
                else:
                    # Simple prediction mode
                    start_time = time.time()
                    predictions = predictor.predict(text)
                    inference_time = (time.time() - start_time) * 1000
                    
                    print("\nPOS predictions:")
                    for token, pos in predictions:
                        print(f"  {token:15} -> {pos}")
                    print(f"\nInference time: {inference_time:.1f}ms")
                    print(f"Speed: {len(predictions) / (inference_time / 1000):.0f} tokens/sec")
                    
            except KeyboardInterrupt:
                break
        
        print("\nGoodbye!")

if __name__ == "__main__":
    main() 