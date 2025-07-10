#!/usr/bin/env python3
# pos_inference.py
#
# Inference script for the trained POS router model

import torch
import torch.nn as nn
from datasets import load_dataset
import argparse
import re

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="router_en_gum.pt", 
                        help="Path to saved model weights")
    parser.add_argument("--treebank", default="en_gum",
                        help="Treebank used for training (to rebuild vocab)")
    parser.add_argument("--text", type=str,
                        help="Text to analyze (if not provided, enters interactive mode)")
    args = parser.parse_args()

    # Initialize predictor
    predictor = POSPredictor(args.model, args.treebank)

    if args.text:
        # Single prediction
        predictions = predictor.predict(args.text)
        print(f"\nInput: {args.text}")
        print("POS predictions:")
        for token, pos in predictions:
            print(f"  {token:15} -> {pos}")
    else:
        # Interactive mode
        print("\n" + "="*50)
        print("Interactive POS Tagger")
        print("Enter text to analyze (or 'quit' to exit)")
        print("="*50)
        
        while True:
            try:
                text = input("\n> ").strip()
                if text.lower() in ['quit', 'exit', 'q']:
                    break
                if not text:
                    continue
                    
                predictions = predictor.predict(text)
                print("\nPOS predictions:")
                for token, pos in predictions:
                    print(f"  {token:15} -> {pos}")
                    
            except KeyboardInterrupt:
                break
        
        print("\nGoodbye!")

if __name__ == "__main__":
    main() 