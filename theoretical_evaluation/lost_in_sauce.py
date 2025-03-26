import sys
import math
import torch
import spacy
import os
from tqdm import tqdm
from datasets import load_dataset
from typing import Tuple
from multiprocessing import Pool, cpu_count
from functools import partial

# Flags to decide whether we do BERT-only or not:
BERT_ONLY = "--bert" in sys.argv
# We'll still load spacy if not BERT_ONLY.

##############################################################################
# 0) GPU Setup (Optional)
##############################################################################
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

##############################################################################
# 1) SpaCy Setup: parse with multi-process + disable NER for speed
##############################################################################
if not BERT_ONLY:
    # Load spaCy English, but disable the 'ner' component for speed
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    # Optionally set a larger max_length if needed (some very long articles)
    # e.g. nlp.max_length = 2_000_000

##############################################################################
# 2) DisCoCirc Setup
##############################################################################
def get_token_rank(token):
    """Same rank logic you used before, e.g. check if token is a verb with direct obj, etc."""
    if token.pos_ in {"NOUN", "PROPN", "PRON"}:
        return 1
    if token.pos_ in {"ADJ", "ADV", "DET"}:
        return 2
    if token.pos_ in {"ADP", "CCONJ", "SCONJ"}:
        return 3
    if token.pos_ == "INTJ":
        return 1
    if token.pos_ == "AUX":
        return 2
    if token.pos_ == "VERB":
        has_dobj = any(child.dep_ in {"dobj", "obj"} for child in token.children)
        has_iobj = any(child.dep_ == "iobj" for child in token.children)
        if has_dobj and has_iobj:
            return 4  # ditransitive
        elif has_dobj:
            return 3  # transitive
        else:
            return 2  # intransitive
    return 1

def compute_discocirc_of_doc(doc, d=300, disco_factor=20.0):
    """Compute the total DisCoCirc naive & optimized FLOPs for one spaCy Doc."""
    if BERT_ONLY:
        # If in BERT-only mode, skip DisCoCirc computations
        return 0.0, 0.0
    
    total_naive = 0.0
    total_opt = 0.0
    for token in doc:
        rank = get_token_rank(token)
        naive = float(d**rank)
        opt = naive / disco_factor
        total_naive += naive
        total_opt += opt
    return total_naive, total_opt

##############################################################################
# 3) BERT Setup: Theoretical FLOP Calculation
##############################################################################
from transformers import BertTokenizer

BERT_TOKENIZER = BertTokenizer.from_pretrained("bert-base-uncased")

def process_document(args):
    """Process a single document for parallel execution"""
    doc_text, bert_factor, disco_factor, d, is_bert_only = args
    try:
        if is_bert_only:
            naive_disc, opt_disc = 0.0, 0.0
        else:
            # Create a new spaCy doc for this process
            doc = nlp(doc_text)
            naive_disc, opt_disc = compute_discocirc_of_doc(doc, d=d, disco_factor=disco_factor)

        # BERT formula:
        naive_bert, _, opt_bert = estimate_bert_complexity_for_doc(doc_text, bert_factor)
        
        return naive_disc, opt_disc, naive_bert, opt_bert
    except Exception as e:
        print(f"Warning: Error processing document: {str(e)}")
        return 0.0, 0.0, 0.0, 0.0

def estimate_bert_complexity_for_doc(doc_text: str, bert_optim_factor=5.0) -> Tuple[float, int, float]:
    """
    The same 'symbolic' formula for BERT flops:
      - no real forward pass
      - seq_len**2 * hidden_size for self-attn, etc.
    """
    try:
        tokens = BERT_TOKENIZER.encode(
            doc_text, 
            add_special_tokens=True, 
            truncation=True,       # enable truncation
            max_length=512,        # default BERT limit
            return_tensors=None    # return list instead of tensor
        )

        seq_len = len(tokens)

        num_layers = 12
        hidden_size = 768
        intermediate_size = 3072  # typically 4 * hidden_size

        # The formula:
        self_attention_flops = 2.0 * (seq_len ** 2) * hidden_size
        feed_forward_flops = 2.0 * seq_len * hidden_size * intermediate_size
        flops_per_layer = self_attention_flops + feed_forward_flops
        naive_flops = flops_per_layer * num_layers
        optimized_flops = naive_flops / bert_optim_factor
        return naive_flops, seq_len, optimized_flops
    except Exception as e:
        print(f"Warning: Error processing document: {str(e)}")
        return 0.0, 0, 0.0

##############################################################################
# 4) Main
##############################################################################
def main():
    # Parameters
    disco_factor = 20.0   # For DisCoCirc
    bert_factor = 5.0     # For BERT
    d = 300

    # If user passes "--wikitext", stream WikiText-103
    if "--wikitext" in sys.argv:
        print("Loading WikiText-103 from Hugging Face in memory (be sure you have enough RAM).")
        dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")
        # Note: Here we use non-streaming mode to get a list. This can be large but you said 192GB is fine.

        # Convert all training examples to a list of doc strings
        articles = [ex["text"] for ex in tqdm(dataset, desc="Reading WikiText dataset")]
        docs = articles  # We'll process these in parallel
    else:
        # Default: load "the_raven.txt" or fallback text
        file_path = os.path.join(os.path.dirname(__file__), "the_raven.txt")
        if os.path.exists(file_path):
            print(f"Reading file: {file_path}")
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        else:
            text = "The big dog quickly chased a ball in the yard."
        docs = [text]
    
    ############################################################################
    # Parallel Processing Setup
    ############################################################################
    # Determine number of processes (use 75% of available CPU cores)
    n_processes = max(1, int(cpu_count() * 0.75))
    print(f"\nUsing {n_processes} processes for parallel computation")
    
    # Prepare arguments for parallel processing
    process_args = [(doc, bert_factor, disco_factor, d, BERT_ONLY) for doc in docs]
    
    # Initialize multiprocessing pool
    with Pool(processes=n_processes) as pool:
        # Process documents in parallel with progress bar
        results = list(tqdm(
            pool.imap(process_document, process_args),
            total=len(docs),
            desc="Processing documents"
        ))
    
    ############################################################################
    # Summation of results
    ############################################################################
    disc_naive_sum = 0.0
    disc_opt_sum = 0.0
    bert_naive_sum = 0.0
    bert_opt_sum = 0.0
    doc_count = len(docs)

    for naive_disc, opt_disc, naive_bert, opt_bert in results:
        # Accumulate
        disc_naive_sum += naive_disc
        disc_opt_sum += opt_disc
        bert_naive_sum += naive_bert
        bert_opt_sum += opt_bert

    print("\n=== Final Results ===")
    print(f"Processed {doc_count} docs/articles.")

    if BERT_ONLY:
        print("-- BERT-only mode --")
        print(f"BERT naive total FLOPs:     {bert_naive_sum:,.0f}")
        print(f"BERT optimized total FLOPs: {bert_opt_sum:,.0f}")
        return

    # Otherwise, show both DisCoCirc + BERT
    print("\n-- DisCoCirc (Aggregated) --")
    print(f"Naive Total FLOPs:     {disc_naive_sum:,.2f}")
    print(f"Optimized Total FLOPs: {disc_opt_sum:,.2f}")

    print("\n-- BERT-base (Aggregated) --")
    print(f"Naive Total FLOPs:     {bert_naive_sum:,.0f}")
    print(f"Optimized Total FLOPs: {bert_opt_sum:,.0f}")

    # Percentage improvement of DisCoCirc over BERT
    if bert_naive_sum > 0:
        naive_improv = 100.0 * (bert_naive_sum - disc_naive_sum) / bert_naive_sum
    else:
        naive_improv = 0.0
    
    if bert_opt_sum > 0:
        opt_improv = 100.0 * (bert_opt_sum - disc_opt_sum) / bert_opt_sum
    else:
        opt_improv = 0.0

    print("\n-- Improvement of DisCoCirc over BERT (Aggregated) --")
    print(f"Naive: DisCoCirc uses {naive_improv:.2f}% fewer FLOPs than BERT-base")
    print(f"Optimized: DisCoCirc uses {opt_improv:.2f}% fewer FLOPs than BERT-base")

if __name__ == "__main__":
    main()
