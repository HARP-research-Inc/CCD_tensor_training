import spacy
from transformers import BertTokenizer
from tqdm import tqdm  # for progress bars
import sys
import os
import concurrent.futures
import multiprocessing
import math
import torch
from typing import Optional, Tuple

# Flag to decide whether to use WikiText-103.
USE_WIKITEXT = "--wikitext" in sys.argv

if USE_WIKITEXT:
    try:
        from datasets import load_dataset
    except ImportError:
        print("Please install the huggingface datasets library: pip install datasets")
        sys.exit(1)

# Check for GPU availability
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

###############################################################################
# 1) Load spaCy model (download "en_core_web_sm" if not installed)
###############################################################################
nlp = spacy.load("en_core_web_sm")

###############################################################################
# 2) DisCoCirc Complexity Estimation (functions unchanged)
###############################################################################
def get_token_rank(token):
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

def estimate_discocirc_complexity(sentence, d=300, disco_factor=1.0, verbose=False):
    doc = nlp(sentence)
    total_naive = 0.0
    total_optimized = 0.0
    breakdown = []
    for token in doc:
        rank = get_token_rank(token)
        naive_cost = float(d ** rank)
        optimized_cost = naive_cost / disco_factor
        total_naive += naive_cost
        total_optimized += optimized_cost
        if verbose:
            breakdown.append((token.text, token.pos_, rank, naive_cost, optimized_cost))
    return total_naive, total_optimized, breakdown

###############################################################################
# 3) BERT Complexity Estimation (GPU-accelerated)
###############################################################################
def estimate_bert_complexity(sentence: str, model_name: str = "bert-base-uncased", 
                           bert_optim_factor: float = 1.0) -> Tuple[float, int, float]:
    """
    GPU-accelerated BERT complexity estimation.
    Returns: (naive_flops, seq_len, optimized_flops)
    """
    tokenizer = BertTokenizer.from_pretrained(model_name)
    # Move tokenizer to GPU if available
    if hasattr(tokenizer, 'to') and DEVICE.type == 'cuda':
        tokenizer.to(DEVICE)
    
    # Tokenize and move to GPU if available
    tokens = tokenizer.encode(sentence, add_special_tokens=True)
    if DEVICE.type == 'cuda':
        tokens = torch.tensor(tokens, device=DEVICE)
    
    seq_len = len(tokens)
    num_layers = 12
    hidden_size = 768
    intermediate_size = 3072  # typically 4 * hidden_size
    
    # Calculate FLOPs (GPU-accelerated if available)
    if DEVICE.type == 'cuda':
        seq_len_tensor = torch.tensor(seq_len, device=DEVICE)
        hidden_size_tensor = torch.tensor(hidden_size, device=DEVICE)
        intermediate_size_tensor = torch.tensor(intermediate_size, device=DEVICE)
        
        self_attention_flops = 2.0 * (seq_len_tensor ** 2) * hidden_size_tensor
        feed_forward_flops = 2.0 * seq_len_tensor * hidden_size_tensor * intermediate_size_tensor
        flops_per_layer = self_attention_flops + feed_forward_flops
        naive_flops = flops_per_layer * num_layers
        optimized_flops = naive_flops / bert_optim_factor
        
        # Move results back to CPU for return
        naive_flops = naive_flops.item()
        optimized_flops = optimized_flops.item()
    else:
        # CPU fallback
        self_attention_flops = 2.0 * (seq_len ** 2) * hidden_size
        feed_forward_flops = 2.0 * seq_len * hidden_size * intermediate_size
        flops_per_layer = self_attention_flops + feed_forward_flops
        naive_flops = flops_per_layer * num_layers
        optimized_flops = naive_flops / bert_optim_factor
    
    return naive_flops, seq_len, optimized_flops

###############################################################################
# 4) Corpus Analysis and Parallel Sentence Processing
###############################################################################
def get_optimal_workers():
    """Determine optimal number of worker processes based on CPU cores."""
    cpu_count = multiprocessing.cpu_count()
    # Use 75% of available cores to leave some headroom for system processes
    return max(1, math.floor(cpu_count * 0.75))

def chunk_text_into_sentences(text, max_chunk=None, batch_size=1000):
    if max_chunk is None:
        max_chunk = nlp.max_length - 1000  # safe margin

    if len(text) > max_chunk:
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""
        for para in paragraphs:
            if len(current_chunk) + len(para) < max_chunk:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = para + "\n\n"
        if current_chunk:
            chunks.append(current_chunk)
        
        # Process chunks in parallel
        sentences = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=get_optimal_workers()) as executor:
            future_to_chunk = {executor.submit(process_chunk, chunk): chunk for chunk in chunks}
            for future in tqdm(concurrent.futures.as_completed(future_to_chunk), 
                             total=len(chunks), 
                             desc="Processing text chunks"):
                sentences.extend(future.result())
        return sentences
    else:
        doc = nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

def process_chunk(chunk):
    """Process a single chunk of text into sentences."""
    doc = nlp(chunk)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

def process_sentence_batch(sentences: list, d: float, disco_factor: float, bert_optim_factor: float) -> list:
    """Process a batch of sentences with GPU acceleration for BERT."""
    results = []
    for sentence in sentences:
        # DisCoCirc processing remains on CPU
        disc_naive, disc_opt, _ = estimate_discocirc_complexity(sentence, d=d, disco_factor=disco_factor, verbose=False)
        # BERT processing uses GPU if available
        bert_naive, seq_len, bert_opt = estimate_bert_complexity(sentence, bert_optim_factor=bert_optim_factor)
        results.append((disc_naive, disc_opt, bert_naive, bert_opt, seq_len))
    return results

def estimate_corpus_complexity_from_sentences(sentences, d=300, disco_factor=1.0, bert_optim_factor=1.0, use_progress_bar=True):
    corpus_results = {
        "num_sentences": len(sentences),
        "discocirc_total_naive": 0.0,
        "discocirc_total_optimized": 0.0,
        "bert_total_naive": 0.0,
        "bert_total_optimized": 0.0,
        "sentence_details": []
    }
    
    # Calculate optimal batch size based on number of sentences
    batch_size = max(1, len(sentences) // (get_optimal_workers() * 4))
    batches = [sentences[i:i + batch_size] for i in range(0, len(sentences), batch_size)]
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=get_optimal_workers()) as executor:
        futures = [
            executor.submit(process_sentence_batch, batch, d, disco_factor, bert_optim_factor)
            for batch in batches
        ]
        
        if use_progress_bar:
            futures = list(tqdm(concurrent.futures.as_completed(futures), 
                              total=len(futures), 
                              desc="Processing sentence batches"))
        else:
            futures = list(concurrent.futures.as_completed(futures))
            
        for future in futures:
            batch_results = future.result()
            for disc_naive, disc_opt, bert_naive, bert_opt, seq_len in batch_results:
                corpus_results["discocirc_total_naive"] += disc_naive
                corpus_results["discocirc_total_optimized"] += disc_opt
                corpus_results["bert_total_naive"] += bert_naive
                corpus_results["bert_total_optimized"] += bert_opt
                corpus_results["sentence_details"].append({
                    "discocirc_naive": disc_naive,
                    "discocirc_optimized": disc_opt,
                    "bert_naive": bert_naive,
                    "bert_optimized": bert_opt,
                    "token_count": seq_len
                })
    
    return corpus_results

###############################################################################
# 5) Helper: Load Text File
###############################################################################
def load_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

###############################################################################
# 6) Main: Choose Corpus and Run Analysis
###############################################################################
if __name__ == "__main__":
    disco_factor = 20.0     
    bert_optim_factor = 5.0 
    d = 300               
    
    if USE_WIKITEXT:
        print("Loading WikiText-103 using Hugging Face datasets in streaming mode...")
        dataset = load_dataset("wikitext", "wikitext-103-v1", split="train", streaming=True)
        all_sentences = []
        # Using a total count of 28,475 examples for the progress bar (adjust if needed).
        for example in tqdm(dataset, total=28475, desc="Processing WikiText examples"):
            example_text = example["text"]
            sents = chunk_text_into_sentences(example_text)
            all_sentences.extend(sents)
        sentences = all_sentences
    else:
        file_path = os.path.join(os.path.dirname(__file__), "the_raven.txt")
        if os.path.exists(file_path):
            print(f"Loading text file: {file_path}")
            text = load_text_file(file_path)
        else:
            text = "The big dog quickly chased a ball in the yard."
        sentences = chunk_text_into_sentences(text)
    
    corpus_results = estimate_corpus_complexity_from_sentences(
        sentences, d=d, disco_factor=disco_factor, bert_optim_factor=bert_optim_factor,
        use_progress_bar=True
    )
    
    print("\n" + "="*60)
    print("[Corpus Complexity Analysis (Optimized and Unoptimized)]")
    print(f"Number of sentences: {corpus_results['num_sentences']}")
    print("\n-- DisCoCirc (Aggregated) --")
    print(f"Naive Total FLOPs:     {corpus_results['discocirc_total_naive']:,.2f}")
    print(f"Optimized Total FLOPs: {corpus_results['discocirc_total_optimized']:,.2f}")
    print("\n-- BERT-base (Aggregated) --")
    print(f"Naive Total FLOPs:     {corpus_results['bert_total_naive']:,.0f}")
    print(f"Optimized Total FLOPs: {corpus_results['bert_total_optimized']:,.0f}")
    
    naive_improvement = 100 * (corpus_results['bert_total_naive'] - corpus_results['discocirc_total_naive']) / corpus_results['bert_total_naive']
    optimized_improvement = 100 * (corpus_results['bert_total_optimized'] - corpus_results['discocirc_total_optimized']) / corpus_results['bert_total_optimized']
    
    print("\n-- Improvement of DisCoCirc over BERT (Aggregated) --")
    print(f"Naive:     DisCoCirc uses {naive_improvement:.2f}% fewer FLOPs than BERT-base")
    print(f"Optimized: DisCoCirc uses {optimized_improvement:.2f}% fewer FLOPs than BERT-base")
