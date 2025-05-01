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
import time

# Add datasets import at the top
try:
    from datasets import load_dataset
except ImportError:
    print("Warning: datasets package not installed. WikiText functionality will be disabled.")
    load_dataset = None

# Flags for different modes
USE_WIKITEXT = "--wikitext" in sys.argv and load_dataset is not None
BERT_ONLY   = "--bert"     in sys.argv
DISCO_ONLY  = "--disco"    in sys.argv
CPU_ONLY    = "--cpu"      in sys.argv
CPU_BERT    = "--cpu-bert" in sys.argv
BENCHMARK   = "--benchmark" in sys.argv
USE_FORK    = "--fork"     in sys.argv  # Use fork instead of spawn (can be faster but less reliable)

# Process worker limits
MAX_WORKERS = None
for arg in sys.argv:
    if arg.startswith("--workers="):
        try:
            MAX_WORKERS = int(arg.split("=")[1])
        except (ValueError, IndexError):
            print("Warning: Invalid value for --workers flag.")

# Initialize global variables to be set in initialize_gpu()
DEVICE = None
BERT_TOKENIZER = None

def initialize_gpu():
    """Initialize GPU and set up global variables. Only run in main process."""
    global DEVICE, BERT_TOKENIZER
    
    print("\n=== GPU Diagnostics ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    # Decide on device, checking for --cpu and --cpu-bert overrides
    if CPU_ONLY:
        print("User requested --cpu, forcing CPU usage for all operations.")
        DEVICE = torch.device("cpu")
    elif CPU_BERT:
        print("User requested --cpu-bert, forcing CPU usage for BERT operations only.")
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
# === PATCH START : accurate FLOP models =====================================
###############################################################################
import math

# --- DisCoCirc helpers -------------------------------------------------------­
def get_token_rank(token):
    """Return the multilinear order k for each word-token."""
    if token.pos_ in {"NOUN", "PROPN", "PRON"}:
        return 1
    if token.pos_ in {"ADJ", "ADV", "DET", "AUX"}:
        return 2
    if token.pos_ in {"ADP", "CCONJ", "SCONJ"}:
        return 3                # preps / conjunctions
    if token.pos_ == "VERB":     # de-trans / trans / di-trans
        has_dobj = any(c.dep_ in {"obj", "dobj"} for c in token.children)
        has_iobj = any(c.dep_ == "iobj"          for c in token.children)
        return 4 if (has_dobj and has_iobj) else 3 if has_dobj else 2
    return 1                     # fallback / INTJ / SYM / etc.

def estimate_discocirc_complexity(sentence: str,
                                  d: int = 768
                                  ) -> float:
    """
    FLOPs for *full-rank* DisCoCirc evaluation on one sentence.
      • Each rank-k operator   ⇒   d**k  multiplies + adds  (≈ 2·d**k FLOPs)
        We count the dominant term only (no *2) for readability.
    Returns a single float (total FLOPs).
    """
    global nlp
    if nlp is None:
        nlp = spacy.load("en_core_web_sm")
    
    doc = nlp(sentence)
    return sum(d ** get_token_rank(tok) for tok in doc)

def estimate_cp_discocirc_complexity(sentence: str,
                                     d: int = 768,
                                     R: int = 50
                                     ) -> float:
    """
    FLOPs for a CP-decomposed DisCoCirc where each k-linear map
        V ≈  Σ_{r=1..R}  a_r ⊗ b_r^(1) ⊗ … ⊗ b_r^(k)
    gives cost  R · (k+2) · d        (see derivation in previous answer).
    """
    global nlp
    if nlp is None:
        nlp = spacy.load("en_core_web_sm")
    
    doc = nlp(sentence)
    return sum(R * (get_token_rank(tok) + 2) * d for tok in doc)

# --- BERT helpers ------------------------------------------------------------­
MAC = 2                            # multiply+add counted as 2 FLOPs

def _layer_vanilla(L: int, H: int) -> int:
    """One encoder layer, classical kernels (no fusions)."""
    proj_qkv  = 3 * MAC * L * H * H
    proj_out  =     MAC * L * H * H
    attn_mat  = 2 * MAC * L * L * H          #  QKᵀ  +  αV
    softmax   =       7 * L * L              #  scale + exp + norm
    ffw       = 2 * MAC * L * H * 4 * H      #  two dense layers
    gelu      =       8 * L * 4 * H
    norm      =       2 * 8 * L * H          #  mean + var + scale + shift
    res       =       2 * L * H
    return (proj_qkv + proj_out + attn_mat +
            softmax  + ffw      + gelu +
            norm     + res)

def _layer_flash(L: int, H: int) -> int:
    """Encoder layer with FlashAttention-2 & fused GELU/LayerNorm."""
    proj_qkv  = 3 * MAC * L * H * H          # fused GEMM; FLOPs same
    proj_out  =     MAC * L * H * H
    attn_mat  =     MAC * L * L * H          # QKᵀ+αV in one pass (½ cost)
    ffw       = 2 * MAC * L * H * 4 * H
    norm      =       2 * 8 * L * H          # still need µ,σ
    res       =       2 * L * H
    return proj_qkv + proj_out + attn_mat + ffw + norm + res

def estimate_bert_complexity(sentence: str,
                             vanilla: bool = True,
                             layers: int = 12,
                             hidden: int = 768
                             ) -> Tuple[float, int]:
    """
    Symbolic FLOP count for BERT-base forward pass.
    Returns (total_flops, seq_len).
    """
    global BERT_TOKENIZER
    if BERT_TOKENIZER is None:
        BERT_TOKENIZER = BertTokenizer.from_pretrained("bert-base-uncased")
    
    tokens = BERT_TOKENIZER.encode(sentence, add_special_tokens=True)
    L = len(tokens)
    per_layer = _layer_vanilla(L, hidden) if vanilla else _layer_flash(L, hidden)
    return per_layer * layers, L
###############################################################################
# === PATCH END ===============================================================
###############################################################################

###############################################################################
# 4) Corpus Analysis and Parallel Sentence Processing
###############################################################################
def get_optimal_workers():
    """Determine number of worker processes based on CPU cores."""
    cpu_count = multiprocessing.cpu_count()
    optimal = max(1, math.floor(cpu_count * 0.75))
    
    # Apply MAX_WORKERS constraint if specified
    if MAX_WORKERS is not None:
        optimal = min(optimal, MAX_WORKERS)
        print(f"Limiting to {optimal} workers due to --workers flag")
    
    return optimal

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
    sentences: list, d: float, cp_rank: float, bert_optim_factor: float
) -> list:
    """Process a batch of sentences, computing complexity metrics."""
    # Initialize tokenizer in worker process if needed
    global BERT_TOKENIZER, nlp
    if BERT_TOKENIZER is None and not DISCO_ONLY:
        BERT_TOKENIZER = BertTokenizer.from_pretrained("bert-base-uncased")
    
    # Initialize spaCy in worker process if needed
    if nlp is None and not BERT_ONLY:
        nlp = spacy.load("en_core_web_sm")
    
    results = []
    for sentence in sentences:
        # DisCoCirc processing remains on CPU
        if not BERT_ONLY:
            disc_naive = estimate_discocirc_complexity(sentence, d=d)
            disc_cp = estimate_cp_discocirc_complexity(sentence, d=d, R=int(cp_rank))
        else:
            disc_naive, disc_cp = 0.0, 0.0
        
        # BERT processing uses GPU if available (unless --cpu)
        if not DISCO_ONLY:
            bert_vanilla, seq_len = estimate_bert_complexity(sentence, vanilla=True, hidden=d)
            bert_flash, _ = estimate_bert_complexity(sentence, vanilla=False, hidden=d)
        else:
            bert_vanilla, bert_flash, seq_len = 0.0, 0.0, 0
            
        results.append({
            "disc_naive": disc_naive,
            "disc_cp": disc_cp,
            "bert_naive": bert_vanilla,
            "bert_opt": bert_flash,
            "token_count": seq_len
        })
    return results

def estimate_corpus_complexity_from_sentences(
    sentences: list, 
    d: float=300, 
    cp_rank: float=50,  # Changed from disco_factor to cp_rank
    bert_optim_factor: float=1.0, 
    use_progress_bar: bool=True
) -> dict:
    """
    Process sentences in parallel using multiprocessing. 
    Returns aggregated complexity results.
    """
    corpus_results = {
        "num_sentences": len(sentences),
        "discocirc_naive": 0.0,  # Updated key names to match new version
        "discocirc_cp": 0.0,
        "bert_vanilla": 0.0,
        "bert_flash": 0.0,
        "sentence_details": []
    }
    
    # Determine optimal number of workers (75% of CPU cores)
    n_workers = get_optimal_workers()
    print(f"\nUsing {n_workers} worker processes for parallel processing")
    
    # Split sentences into chunks for parallel processing using improved formula
    total_sentences = len(sentences)
    
    # Advanced optimization for very high core counts
    # Scale more aggressively with higher core counts
    # For 8 cores: ~100 sentences
    # For 16 cores: ~200 sentences
    # For 32 cores: ~400 sentences
    # For 64 cores: ~800 sentences
    # For 128 cores: ~1600 sentences
    base_chunk = 100
    # Exponential scaling with worker count
    chunk_multiplier = math.log2(n_workers) / math.log2(8)  # Scaled relative to 8 cores
    chunk_size = int(base_chunk * chunk_multiplier)
    chunk_size = max(50, min(2000, chunk_size))
    
    # Make sure we don't have too many or too few chunks
    max_chunks = n_workers * 4  # Aim for around 4 chunks per worker
    min_chunks = n_workers  # At least one chunk per worker
    
    total_chunks = max(1, total_sentences // chunk_size)
    if total_chunks < min_chunks:
        chunk_size = max(50, total_sentences // min_chunks)
    elif total_chunks > max_chunks:
        chunk_size = max(50, total_sentences // max_chunks)
        
    print(f"Using chunk size of {chunk_size} sentences per worker")
    
    sentence_chunks = [sentences[i:i + chunk_size] for i in range(0, len(sentences), chunk_size)]
    
    # Create a partial function with the fixed arguments
    process_batch = partial(process_sentence_batch, d=d, cp_rank=cp_rank, bert_optim_factor=bert_optim_factor)
    
    # Process chunks in parallel
    with multiprocessing.Pool(processes=n_workers) as pool:
        if use_progress_bar:
            # Determine which algorithms are being processed
            if BERT_ONLY:
                desc = "Processing BERT"
            elif DISCO_ONLY:
                desc = "Processing DisCoCirc"
            else:
                desc = "Processing BERT+DisCoCirc"
                
            # Enhanced progress bar with more information
            completed_sentences = 0
            disc_total_naive = 0.0
            disc_total_cp = 0.0
            bert_total_vanilla = 0.0
            bert_total_flash = 0.0
            
            pbar = tqdm(total=total_sentences, desc=desc, position=0)
            
            for chunk_result in pool.imap(process_batch, sentence_chunks):
                # Update metrics for this chunk
                for result in chunk_result:
                    disc_total_naive += result["disc_naive"]
                    disc_total_cp += result["disc_cp"]
                    bert_total_vanilla += result["bert_naive"]
                    bert_total_flash += result["bert_opt"]
                    completed_sentences += 1
                
                # Update progress bar with detailed metrics
                pbar.update(len(chunk_result))
                
                # Create a dynamic status message based on which algorithms are active
                status_parts = []
                if not BERT_ONLY:
                    status_parts.append(f"DisCoCirc: {disc_total_naive:.2e}/{disc_total_cp:.2e}")
                if not DISCO_ONLY:
                    status_parts.append(f"BERT: {bert_total_vanilla:.2e}/{bert_total_flash:.2e}")
                
                status = " | ".join(status_parts)
                pbar.set_postfix_str(f"{completed_sentences}/{total_sentences} sents | {status}")
                
                # Add to final results
                for result in chunk_result:
                    corpus_results["discocirc_naive"] += result["disc_naive"]
                    corpus_results["discocirc_cp"] += result["disc_cp"]
                    corpus_results["bert_vanilla"] += result["bert_naive"]
                    corpus_results["bert_flash"] += result["bert_opt"]
                    corpus_results["sentence_details"].append({
                        "discocirc_naive": result["disc_naive"],
                        "discocirc_cp": result["disc_cp"],
                        "bert_vanilla": result["bert_naive"],
                        "bert_flash": result["bert_opt"],
                        "token_count": result["token_count"]
                    })
            
            pbar.close()
        else:
            chunk_results = pool.map(process_batch, sentence_chunks)
            # Aggregate results from all chunks if no progress bar
            for chunk_result in chunk_results:
                for result in chunk_result:
                    corpus_results["discocirc_naive"] += result["disc_naive"]
                    corpus_results["discocirc_cp"] += result["disc_cp"]
                    corpus_results["bert_vanilla"] += result["bert_naive"]
                    corpus_results["bert_flash"] += result["bert_opt"]
                    corpus_results["sentence_details"].append({
                        "discocirc_naive": result["disc_naive"],
                        "discocirc_cp": result["disc_cp"],
                        "bert_vanilla": result["bert_naive"],
                        "bert_flash": result["bert_opt"],
                        "token_count": result["token_count"]
                    })
    
    return corpus_results

###############################################################################
# 5) Helper: Load Text File
###############################################################################
def load_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

def benchmark_processing_methods(sample_size=100):
    """Run a quick benchmark to determine fastest processing methods."""
    global BERT_TOKENIZER, DEVICE
    
    # Make sure we have initialized everything
    if BERT_TOKENIZER is None or DEVICE is None:
        initialize_gpu()
        
    print("\n=== Running Benchmark ===")
    
    # Load a small sample of data
    if USE_WIKITEXT:
        dataset = load_dataset("wikitext", "wikitext-103-v1", split="train", streaming=True)
        sample_texts = []
        for i, example in enumerate(dataset):
            if i >= sample_size:
                break
            sample_texts.append(example["text"])
    else:
        file_path = os.path.join(os.path.dirname(__file__), "the_raven.txt")
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        else:
            text = "The big dog quickly chased a ball in the yard."
        sample_texts = [text]
    
    # Test both CPU and GPU if available
    devices = ["cpu"]
    if torch.cuda.is_available() and not CPU_ONLY and not DISCO_ONLY:
        devices.append("cuda")
    
    print("\nBenchmarking BERT processing on different devices...")
    bert_results = {}
    for device in devices:
        print(f"\nTesting on {device.upper()}")
        bert_times = []
        bert_pbar = tqdm(total=len(sample_texts), desc=f"BERT ({device.upper()})", position=1, leave=False)
        for text in sample_texts:
            start_time = time.time()
            # Process the entire text, including tokenization and FLOP calculation
            tokens = BERT_TOKENIZER.encode(text, add_special_tokens=True)
            if device == "cuda":
                tokens = torch.tensor(tokens, device=torch.device("cuda"))
            seq_len = len(tokens)
            # Calculate FLOPs (symbolic formula)
            num_layers = 12
            hidden_size = 768
            intermediate_size = 3072
            self_attention_flops = 2.0 * (seq_len ** 2) * hidden_size
            feed_forward_flops = 2.0 * seq_len * hidden_size * intermediate_size
            flops_per_layer = self_attention_flops + feed_forward_flops
            naive_flops = flops_per_layer * num_layers
            optimized_flops = naive_flops / 5.0
            bert_times.append(time.time() - start_time)
            bert_pbar.update(1)
        bert_pbar.close()
        bert_results[device] = {
            "times": bert_times,
            "avg": sum(bert_times) / len(bert_times),
            "min": min(bert_times),
            "max": max(bert_times),
            "total": sum(bert_times)
        }
    
    # Benchmark DisCoCirc processing (always on CPU)
    print("\nBenchmarking DisCoCirc processing...")
    disco_times = []
    disco_pbar = tqdm(total=len(sample_texts), desc="DisCoCirc", position=2, leave=False)
    for text in sample_texts:
        start_time = time.time()
        doc = nlp(text)
        total_naive = 0.0
        total_opt = 0.0
        for token in doc:
            rank = get_token_rank(token)
            naive = float(300**rank)
            opt = naive / 20.0
            total_naive += naive
            total_opt += opt
        disco_times.append(time.time() - start_time)
        disco_pbar.update(1)
    disco_pbar.close()
    
    # Print benchmark results
    print("\n=== Benchmark Results ===")
    if not DISCO_ONLY:
        print("\nBERT Processing:")
        for device in devices:
            results = bert_results[device]
            print(f"\n{device.upper()}:")
            print(f"  Average time: {results['avg']:.4f} seconds")
            print(f"  Min time:     {results['min']:.4f} seconds")
            print(f"  Max time:     {results['max']:.4f} seconds")
            print(f"  Total time:   {results['total']:.4f} seconds")
    
    if not BERT_ONLY:
        print("\nDisCoCirc Processing:")
        print(f"  Average time: {sum(disco_times)/len(disco_times):.4f} seconds")
        print(f"  Min time:     {min(disco_times):.4f} seconds")
        print(f"  Max time:     {max(disco_times):.4f} seconds")
        print(f"  Total time:   {sum(disco_times):.4f} seconds")
    
    # Determine optimal settings based on fastest device
    if not DISCO_ONLY:
        fastest_device = min(devices, key=lambda d: bert_results[d]["avg"])
    else:
        fastest_device = "cpu"  # Always use CPU in DisCoCirc-only mode
    
    # Calculate optimal chunk size based on CPU cores - improved for high core counts
    cpu_count = multiprocessing.cpu_count()
    
    # Advanced optimization for very high core counts
    # Scale more aggressively with higher core counts
    # For 8 cores: ~50 sentences
    # For 16 cores: ~100 sentences
    # For 32 cores: ~200 sentences
    # For 64 cores: ~400 sentences
    # For 128 cores: ~800 sentences
    base_chunk = 50
    # Exponential scaling with core count
    chunk_multiplier = math.log2(cpu_count) / math.log2(8)  # Scaled relative to 8 cores
    suggested_chunk_size = int(base_chunk * chunk_multiplier)
    optimal_chunk_size = max(50, min(1000, suggested_chunk_size))
    
    # Adjust based on timing results
    # If DisCoCirc is much slower than BERT, use larger chunks to amortize overhead
    if sum(disco_times) > 5 * sum(bert_results[fastest_device]["times"]):
        optimal_chunk_size = min(1000, optimal_chunk_size * 2)
    
    optimal_settings = {
        "use_cpu": fastest_device == "cpu",
        "parallel_chunk_size": optimal_chunk_size
    }
    
    print("\n=== Optimal Settings ===")
    print(f"Fastest device: {fastest_device.upper()}")
    print(f"Using CPU: {optimal_settings['use_cpu']}")
    print(f"Parallel chunk size: {optimal_settings['parallel_chunk_size']} (based on {cpu_count} CPU cores)")
    print(f"Multiprocessing mode: {'fork' if USE_FORK else 'spawn'} (use --fork flag for possibly faster processing)")
    
    return optimal_settings

###############################################################################
# 6) Main Execution  ––  UPDATED FOR NEW COMPLEXITY FORMULAS
###############################################################################
if __name__ == "__main__":
    # ------------------------------------------------------------------ config
    H_DIM   = 384          # unified embedding dimension for all models
    CP_RANK = 50           # CP decomposition rank for DisCoCirc

    # ------------------------------------------------------------------ init
    if not DISCO_ONLY:
        initialize_gpu()                 # sets DEVICE & tokenizer
    else:
        print("\nRunning in DisCoCirc-only mode, skipping GPU init")

    cpu_count = multiprocessing.cpu_count()
    effective_workers = get_optimal_workers()
    print(f"\nSystem has {cpu_count} CPU cores; using {effective_workers} workers")

    # ------------------------------------------------------------------ BENCH
    if BENCHMARK:
        optimal = benchmark_processing_methods()
        CPU_ONLY = optimal["use_cpu"]
        print(f"\nBenchmark-selected device: {'CPU' if CPU_ONLY else 'GPU'}")

    # ================================================================= WIKITEXT
    if USE_WIKITEXT:
        print("\nLoading WikiText-103 (streaming)…")
        dataset = load_dataset("wikitext", "wikitext-103-v1",
                               split="train", streaming=True)

        corpus_results = {
            "num_sentences": 0,
            "discocirc_naive": 0.0,
            "discocirc_cp": 0.0,
            "bert_vanilla": 0.0,
            "bert_flash": 0.0,
        }

        batch_size = 100
        current_sent = []
        n_workers = get_optimal_workers()
        print(f"Using {n_workers} worker processes")

        # process_sentence_batch now uses H_DIM & CP_RANK
        proc_fn = partial(process_sentence_batch, d=H_DIM, cp_rank=CP_RANK)

        # safe spawn unless --fork
        if not USE_FORK:
            multiprocessing.set_start_method("spawn", force=True)

        with multiprocessing.Pool(n_workers) as pool, \
             tqdm(total=28_475, desc="Articles") as art_pbar:

            for example in dataset:
                art_pbar.update(1)
                current_sent.extend(chunk_text_into_sentences(example["text"]))

                if len(current_sent) >= batch_size:
                    for res in pool.apply(proc_fn, (current_sent,)):
                        corpus_results["discocirc_naive"] += res["disc_naive"]
                        corpus_results["discocirc_cp"] += res["disc_cp"]
                        corpus_results["bert_vanilla"] += res["bert_naive"]
                        corpus_results["bert_flash"] += res["bert_opt"]
                        corpus_results["num_sentences"] += 1
                    current_sent.clear()

            # flush remainder
            if current_sent:
                for res in pool.apply(proc_fn, (current_sent,)):
                    corpus_results["discocirc_naive"] += res["disc_naive"]
                    corpus_results["discocirc_cp"] += res["disc_cp"]
                    corpus_results["bert_vanilla"] += res["bert_naive"]
                    corpus_results["bert_flash"] += res["bert_opt"]
                    corpus_results["num_sentences"] += 1
    # ================================================================== FILE
    else:
        file_path = os.path.join(os.path.dirname(__file__), "the_raven.txt")
        text = load_text_file(file_path) if os.path.exists(file_path) \
               else "The big dog quickly chased a ball in the yard."
        sentences = chunk_text_into_sentences(text)

        corpus_results = estimate_corpus_complexity_from_sentences(
            sentences,
            d=H_DIM,
            cp_rank=CP_RANK,
            use_progress_bar=True,
        )

    # ---------------------------------------------------------------- summary
    print("\n" + "=" * 60)
    print("[Corpus-level FLOP Totals]")
    print(f"Sentences processed : {corpus_results['num_sentences']:,}")
    print(f"Full-rank DisCoCirc : {corpus_results['discocirc_naive']:,.2e}")
    print(f"CP-rank-{CP_RANK}   : {corpus_results['discocirc_cp']:,.2e}")
    print(f"BERT vanilla        : {corpus_results['bert_vanilla']:,.2e}")
    print(f"BERT FlashAttention : {corpus_results['bert_flash']:,.2e}")

    if not (BERT_ONLY or DISCO_ONLY):
        naive_gain = 100 * (corpus_results["bert_vanilla"]
                            - corpus_results["discocirc_naive"]
                           ) / corpus_results["bert_vanilla"]
        cp_gain = 100 * (corpus_results["bert_flash"]
                          - corpus_results["discocirc_cp"]
                         ) / corpus_results["bert_flash"]
        print("\nDisCoCirc vs BERT:")
        print(f"  Full-rank saves : {naive_gain:6.2f}% FLOPs over vanilla BERT")
        print(f"  CP-rank saves   : {cp_gain:6.2f}% FLOPs over Flash-BERT")
