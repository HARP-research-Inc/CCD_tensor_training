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
from functools import partial

# Flags for different modes
USE_WIKITEXT = "--wikitext" in sys.argv
BERT_ONLY   = "--bert"     in sys.argv
CPU_ONLY    = "--cpu"      in sys.argv   # <-- ADDED FLAG

if USE_WIKITEXT:
    try:
        from datasets import load_dataset
    except ImportError:
        print("Please install the huggingface datasets library: pip install datasets")
        sys.exit(1)

###############################################################################
# Enhanced GPU diagnostics and initialization
###############################################################################
print("\n=== GPU Diagnostics ===")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Decide on device, checking for --cpu override
if CPU_ONLY:
    print("User requested --cpu, forcing CPU usage.")
    DEVICE = torch.device("cpu")
else:
    # Force CUDA device selection if available
    if torch.cuda.is_available():
        cuda_devices = [i for i in range(torch.cuda.device_count())]
        if cuda_devices:
            nvidia_device = cuda_devices[0]
            print(f"Found NVIDIA GPU at device {nvidia_device}")
            print(f"Device name: {torch.cuda.get_device_name(nvidia_device)}")
            print(f"CUDA version: {torch.version.cuda}")
            
            torch.cuda.set_device(nvidia_device)
            DEVICE = torch.device(f"cuda:{nvidia_device}")
            
            # Force CUDA initialization
            torch.cuda.init()
            
            # Test CUDA with a small computation
            test_tensor = torch.rand(5, 5).cuda()
            test_result = torch.matmul(test_tensor, test_tensor.t())
            print("CUDA test computation successful!")
            print(f"Test tensor device: {test_tensor.device}")
            print(f"Test result device: {test_result.device}")
        else:
            print("No CUDA devices found!")
            DEVICE = torch.device("cpu")
    else:
        print("CUDA is not available, falling back to CPU")
        DEVICE = torch.device("cpu")

print(f"Using device: {DEVICE}")
print("=====================\n")

# Initialize BERT tokenizer once
BERT_TOKENIZER = BertTokenizer.from_pretrained("bert-base-uncased")

###############################################################################
# 1) Load spaCy model (download "en_core_web_sm" if not installed)
###############################################################################
if not BERT_ONLY:
    nlp = spacy.load("en_core_web_sm")

###############################################################################
# 2) DisCoCirc Complexity Estimation
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
    """Compute naive and optimized FLOPs for DisCoCirc, if not in BERT-only mode."""
    if BERT_ONLY:
        return 0.0, 0.0, []  # no DisCoCirc computations in BERT-only mode
    
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
# 3) BERT Complexity Estimation (GPU-accelerated unless --cpu is used)
###############################################################################
def estimate_bert_complexity(
    sentence: str, 
    model_name: str = "bert-base-uncased", 
    bert_optim_factor: float = 1.0
) -> Tuple[float, int, float]:
    """
    GPU-accelerated BERT complexity estimation (symbolic, not a real forward pass).
    Returns: (naive_flops, seq_len, optimized_flops)
    """
    # Use the global tokenizer
    tokens = BERT_TOKENIZER.encode(sentence, add_special_tokens=True)
    
    if DEVICE.type == 'cuda':
        # Move tokens to GPU
        tokens = torch.tensor(tokens, device=DEVICE)
    
    seq_len = len(tokens)
    num_layers = 12
    hidden_size = 768
    intermediate_size = 3072  # typically 4 * hidden_size
    
    # Calculate FLOPs (symbolic formula)
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
    """Determine number of worker processes based on CPU cores."""
    cpu_count = multiprocessing.cpu_count()
    return max(1, math.floor(cpu_count * 0.75))

def process_chunk(chunk):
    """Process a chunk of text into sentences (spaCy)."""
    doc = nlp(chunk)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

def chunk_text_into_sentences(text, max_chunk=None):
    """
    Splits text into sentences using spaCy. If the text length exceeds nlp.max_length,
    first splits into smaller chunks (double newlines). 
    """
    if BERT_ONLY:
        # For BERT-only mode, just do a basic split
        sents = []
        for line in text.split('\n'):
            subs = line.split('.')
            for sub in subs:
                sub = sub.strip()
                if sub:
                    sents.append(sub)
        return sents
    
    if max_chunk is None:
        max_chunk = nlp.max_length - 1000

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
        
        sentences = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=get_optimal_workers()) as executor:
            fut_to_chunk = {executor.submit(process_chunk, c): c for c in chunks}
            for fut in concurrent.futures.as_completed(fut_to_chunk):
                sentences.extend(fut.result())
        return sentences
    else:
        doc = nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

def process_sentence_batch(
    sentences: list, d: float, disco_factor: float, bert_optim_factor: float
) -> list:
    """Process a batch of sentences, computing (disc_naive, disc_opt, bert_naive, bert_opt, seq_len)."""
    results = []
    for sentence in sentences:
        # DisCoCirc processing remains on CPU
        disc_naive, disc_opt, _ = estimate_discocirc_complexity(
            sentence, d=d, disco_factor=disco_factor, verbose=False
        )
        # BERT processing uses GPU if available (unless --cpu)
        bert_naive, seq_len, bert_opt = estimate_bert_complexity(
            sentence, bert_optim_factor=bert_optim_factor
        )
        results.append((disc_naive, disc_opt, bert_naive, bert_opt, seq_len))
    return results

def estimate_corpus_complexity_from_sentences(
    sentences: list, 
    d: float=300, 
    disco_factor: float=1.0, 
    bert_optim_factor: float=1.0, 
    use_progress_bar: bool=True
) -> dict:
    """
    Process sentences in parallel using multiprocessing. 
    Returns aggregated complexity results.
    """
    corpus_results = {
        "num_sentences": len(sentences),
        "discocirc_total_naive": 0.0,
        "discocirc_total_optimized": 0.0,
        "bert_total_naive": 0.0,
        "bert_total_optimized": 0.0,
        "sentence_details": []
    }
    
    # Determine optimal number of workers (75% of CPU cores)
    n_workers = get_optimal_workers()
    print(f"\nUsing {n_workers} worker processes for parallel processing")
    
    # Split sentences into chunks for parallel processing
    chunk_size = max(1, len(sentences) // (n_workers * 4))  # Divide into smaller chunks for better load balancing
    sentence_chunks = [sentences[i:i + chunk_size] for i in range(0, len(sentences), chunk_size)]
    
    # Create a partial function with the fixed arguments
    process_batch = partial(process_sentence_batch, d=d, disco_factor=disco_factor, bert_optim_factor=bert_optim_factor)
    
    # Process chunks in parallel
    with multiprocessing.Pool(processes=n_workers) as pool:
        if use_progress_bar:
            chunk_results = list(tqdm(
                pool.imap(process_batch, sentence_chunks),
                total=len(sentence_chunks),
                desc="Processing sentence chunks"
            ))
        else:
            chunk_results = pool.map(process_batch, sentence_chunks)
    
    # Aggregate results from all chunks
    for chunk_result in chunk_results:
        for disc_naive, disc_opt, bert_naive, bert_opt, seq_len in chunk_result:
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
# 6) Main Execution
###############################################################################
if __name__ == "__main__":
    disco_factor = 20.0     
    bert_optim_factor = 5.0 
    d = 300               
    
    if USE_WIKITEXT:
        print("Loading WikiText-103 using Hugging Face datasets in streaming mode...")

        # We'll track a third progress bar for tokens
        # We know training has ~28k articles, so one bar for articles is up to 28475
        # Let's set a second bar for approximate tokens. We'll guess 101,425,671 tokens in training.
        from datasets import load_dataset
        dataset = load_dataset("wikitext", "wikitext-103-v1", split="train", streaming=True)
        
        total_articles = 28475
        total_tokens_est = 101_425_671  # Approx for the entire training set
        article_pbar = tqdm(total=total_articles, desc="WikiText Articles")
        token_pbar = tqdm(total=total_tokens_est, desc="WikiText Tokens Processed")
        
        all_sentences = []
        article_count = 0
        
        for example in dataset:
            article_count += 1
            article_pbar.update(1)
            
            example_text = example["text"]
            
            ex_tokens = len(example_text.split())
            token_pbar.update(ex_tokens)
            
            sents = chunk_text_into_sentences(example_text)
            all_sentences.extend(sents)
        
        article_pbar.close()
        token_pbar.close()
        
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
    
    # DisCoCirc results
    if BERT_ONLY:
        print("\n-- DisCoCirc --")
        print("N/A (BERT-only mode)")
    else:
        print("\n-- DisCoCirc (Aggregated) --")
        print(f"Naive Total FLOPs:     {corpus_results['discocirc_total_naive']:,.2f}")
        print(f"Optimized Total FLOPs: {corpus_results['discocirc_total_optimized']:,.2f}")
    
    # BERT results
    print("\n-- BERT-base (Aggregated) --")
    print(f"Naive Total FLOPs:     {corpus_results['bert_total_naive']:,.0f}")
    print(f"Optimized Total FLOPs: {corpus_results['bert_total_optimized']:,.0f}")
    
    # If not in BERT-only mode, show comparative improvement
    if not BERT_ONLY:
        naive_improvement = 100 * (
            corpus_results['bert_total_naive'] - corpus_results['discocirc_total_naive']
        ) / corpus_results['bert_total_naive'] if corpus_results['bert_total_naive'] else 0
        
        optimized_improvement = 100 * (
            corpus_results['bert_total_optimized'] - corpus_results['discocirc_total_optimized']
        ) / corpus_results['bert_total_optimized'] if corpus_results['bert_total_optimized'] else 0
        
        print("\n-- Improvement of DisCoCirc over BERT (Aggregated) --")
        print(f"Naive:     DisCoCirc uses {naive_improvement:.2f}% fewer FLOPs than BERT-base")
        print(f"Optimized: DisCoCirc uses {optimized_improvement:.2f}% fewer FLOPs than BERT-base")
    else:
        print("\n-- BERT-only mode: DisCoCirc comparison not available --")
