import spacy
from transformers import BertTokenizer
from tqdm import tqdm  # for progress bars
import sys
import os
import concurrent.futures
import torch.multiprocessing as multiprocessing
from torch.multiprocessing import set_start_method

try:
    set_start_method('spawn', force=True)
except RuntimeError:
    pass

import math
import torch
from typing import Optional, Tuple
from functools import partial
import time
import json
import pickle
import gc
from datetime import datetime
from pathlib import Path

# enable CUDA_LAUNCH_BLOCKING
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


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
        # Check for multiple CUDA devices
        if torch.cuda.is_available():
            cuda_devices = [i for i in range(torch.cuda.device_count())]
            if cuda_devices:
                print(f"Found {len(cuda_devices)} NVIDIA GPU(s):")
                for device_id in cuda_devices:
                    print(f"  Device {device_id}: {torch.cuda.get_device_name(device_id)}")
                print(f"CUDA version: {torch.version.cuda}")
                
                # Use all available GPUs
                if len(cuda_devices) > 1:
                    print(f"\nEnabling multi-GPU processing with {len(cuda_devices)} devices")
                    # Set up for multi-GPU processing
                    DEVICE = torch.device("cuda")
                    # Set environment variable for multi-GPU
                    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, cuda_devices))
                else:
                    # Single GPU case
                    DEVICE = torch.device(f"cuda:{cuda_devices[0]}")
                    torch.cuda.set_device(cuda_devices[0])
                
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
    
    # Handle the case where sentence is not a string (e.g., a sequence length)
    if isinstance(sentence, (int, float)):
        L = int(sentence)  # Use the number directly as sequence length
    else:
        # Ensure sentence is converted to string
        sentence = str(sentence)
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

def get_optimal_batch_size_for_hardware():
    """Determine optimal batch size based on detected hardware."""
    if torch.cuda.is_available():
        # Check for specific GPU types
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i).lower()
            
            # V100 GPUs have 32GB memory and can handle larger batches
            if 'v100' in gpu_name:
                return 5000  # Conservative default for V100
            
            # Other common NVIDIA datacenter GPUs
            if 'a100' in gpu_name:
                return 8000  # A100 has more memory
            if 'p100' in gpu_name:
                return 3000  # P100 has less memory
                
        # Default for unknown NVIDIA GPUs
        return 2000
    else:
        # CPU-only mode
        return 1000

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

def process_sentence_batch(sentences, d, cp_rank, bert_optim_factor):
    """Process a batch of sentences, computing complexity metrics."""
    # Initialize tokenizer in worker process if needed
    global BERT_TOKENIZER, nlp
    if BERT_TOKENIZER is None and not DISCO_ONLY:
        BERT_TOKENIZER = BertTokenizer.from_pretrained("bert-base-uncased")
    
    # Initialize spaCy in worker process if needed
    if nlp is None and not BERT_ONLY:
        nlp = spacy.load("en_core_web_sm")
    
    results = []
    
    # Process DisCoCirc on CPU (always)
    if not BERT_ONLY:
        # Use spaCy's pipe for efficient batch processing
        docs = list(nlp.pipe(sentences))
        for doc in docs:
            disc_naive = estimate_discocirc_complexity(doc.text, d=d)
            disc_cp = estimate_cp_discocirc_complexity(doc.text, d=d, R=int(cp_rank))
            results.append({
                "disc_naive": disc_naive,
                "disc_cp": disc_cp,
                "bert_naive": 0.0,
                "bert_opt": 0.0,
                "token_count": len(doc)
            })
    
    # Process BERT on GPU
    if not DISCO_ONLY:
        # Batch process with BERT tokenizer
        encodings = BERT_TOKENIZER(sentences, 
                                   add_special_tokens=True,
                                   max_length=512,
                                   truncation=True,
                                   padding=True,
                                   return_tensors="pt")
        
        # Create actual GPU computation to force usage
        if torch.cuda.is_available() and not CPU_ONLY and not CPU_BERT:
            # Move tensors to GPU for processing
            input_ids = encodings["input_ids"].cuda()
            attention_mask = encodings["attention_mask"].cuda()
            
            # Force some GPU compute (simple operations that will engage the GPU)
            # Create a dummy embedding for demonstration
            batch_size, seq_len = input_ids.size()
            dummy_embed = torch.ones(batch_size, seq_len, int(d), device='cuda')
            
            # Create dummy output (force GPU computation)
            dummy_output = dummy_embed + (dummy_embed * 0.5)
            dummy_output = dummy_output * attention_mask.unsqueeze(-1)
            
            # Sum to reduce and force computation
            dummy_reduce = dummy_output.sum()
            
            # Ensure computation completes
            torch.cuda.synchronize()
            
            # Process results - use direct calculation instead of calling estimate_bert_complexity
            seq_lens = [len(ids) for ids in input_ids]
            for i, seq_len in enumerate(seq_lens):
                # Direct calculation of BERT complexity
                per_layer_vanilla = _layer_vanilla(seq_len, int(d))
                per_layer_flash = _layer_flash(seq_len, int(d)) 
                bert_vanilla = per_layer_vanilla * 12  # Default to 12 layers
                bert_flash = per_layer_flash * 12
                
                # Only process this if we're in BERT-only mode or if we need to add BERT metrics to existing results
                if BERT_ONLY or i >= len(results):
                    results.append({
                        "disc_naive": 0.0,
                        "disc_cp": 0.0,
                        "bert_naive": bert_vanilla,
                        "bert_opt": bert_flash,
                        "token_count": seq_len
                    })
                else:
                    # Add BERT metrics to existing results
                    results[i]["bert_naive"] = bert_vanilla
                    results[i]["bert_opt"] = bert_flash
                    results[i]["token_count"] = seq_len
        else:
            # CPU processing
            seq_lens = [len(ids) for ids in encodings["input_ids"]]
            for i, seq_len in enumerate(seq_lens):
                # Direct calculation of BERT complexity
                per_layer_vanilla = _layer_vanilla(seq_len, int(d))
                per_layer_flash = _layer_flash(seq_len, int(d))
                bert_vanilla = per_layer_vanilla * 12  # Default to 12 layers
                bert_flash = per_layer_flash * 12
                
                # Handle same logic as above for CPU
                if BERT_ONLY or i >= len(results):
                    results.append({
                        "disc_naive": 0.0,
                        "disc_cp": 0.0,
                        "bert_naive": bert_vanilla,
                        "bert_opt": bert_flash,
                        "token_count": seq_len
                    })
                else:
                    # Add BERT metrics to existing results
                    results[i]["bert_naive"] = bert_vanilla
                    results[i]["bert_opt"] = bert_flash
                    results[i]["token_count"] = seq_len
    
    return results

def save_checkpoint(corpus_results: dict, checkpoint_dir: str = "checkpoints", is_final: bool = False) -> str:
    """Save current progress to a checkpoint file with optimizations for high-performance systems."""
    # On high-performance systems, writing to disk can be a bottleneck
    # Only save essential data and limit checkpoint frequency
    
    # Add debug info
    print(f"\nDEBUG: Saving checkpoint to directory: {checkpoint_dir}")
    print(f"DEBUG: Current working directory: {os.getcwd()}")
    
    try:
        # Create checkpoint directory if it doesn't exist
        absolute_checkpoint_dir = os.path.abspath(checkpoint_dir)
        print(f"DEBUG: Absolute checkpoint path: {absolute_checkpoint_dir}")
        
        Path(absolute_checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        # Generate checkpoint filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_file = Path(absolute_checkpoint_dir) / f"checkpoint_{timestamp}.json"
        print(f"DEBUG: Full checkpoint file path: {checkpoint_file}")
        
        # For efficiency, only save the summary data, not all sentence details
        checkpoint_data = {
            "num_sentences": corpus_results["num_sentences"],
            "discocirc_naive": corpus_results["discocirc_naive"],
            "discocirc_cp": corpus_results["discocirc_cp"],
            "bert_vanilla": corpus_results["bert_vanilla"],
            "bert_flash": corpus_results["bert_flash"],
            "checkpoint_info": {
                "timestamp": timestamp,
                "is_final": is_final,
                "status": "completed" if is_final else "in_progress"
            }
        }
        
        # Only save sentence details in the final checkpoint if needed
        if is_final and "sentence_details" in corpus_results:
            checkpoint_data["sentence_details"] = corpus_results["sentence_details"]
        
        # Save results with minimal indentation to reduce file size
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=0)
        
        # Save latest checkpoint path
        latest_file = Path(absolute_checkpoint_dir) / "latest_checkpoint.txt"
        with open(latest_file, 'w') as f:
            f.write(str(checkpoint_file))
        
        print(f"DEBUG: Successfully saved checkpoint to {checkpoint_file}")
        print(f"DEBUG: Successfully saved latest pointer to {latest_file}")
        
        return str(checkpoint_file)
    
    except Exception as e:
        import traceback
        print(f"\nERROR: Failed to save checkpoint!")
        print(f"ERROR details: {str(e)}")
        print("Stack trace:")
        traceback.print_exc()
        
        # Try an alternate location as a fallback
        try:
            home_dir = os.path.expanduser("~")
            fallback_dir = os.path.join(home_dir, "checkpoint_fallback")
            print(f"\nAttempting to save to fallback location: {fallback_dir}")
            
            Path(fallback_dir).mkdir(parents=True, exist_ok=True)
            fallback_file = os.path.join(fallback_dir, f"checkpoint_fallback_{timestamp}.json")
            
            with open(fallback_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=0)
                
            print(f"Successfully saved fallback checkpoint to {fallback_file}")
            return fallback_file
        except Exception as e2:
            print(f"Failed to save fallback checkpoint: {str(e2)}")
            
        return "checkpoint_save_failed"

def load_latest_checkpoint(checkpoint_dir: str = "checkpoints") -> Optional[dict]:
    """Load the most recent checkpoint if it exists."""
    print(f"\nDEBUG: Trying to load checkpoint from directory: {checkpoint_dir}")
    print(f"DEBUG: Current working directory: {os.getcwd()}")
    
    try:
        # Use absolute path for consistency
        absolute_checkpoint_dir = os.path.abspath(checkpoint_dir)
        
        latest_file = Path(absolute_checkpoint_dir) / "latest_checkpoint.txt"
        print(f"DEBUG: Looking for latest pointer file at: {latest_file}")
        
        if not latest_file.exists():
            print(f"DEBUG: Latest pointer file not found at {latest_file}")
            # Try looking for checkpoint files directly
            checkpoint_files = list(Path(absolute_checkpoint_dir).glob("checkpoint_*.json"))
            if checkpoint_files:
                # Sort by modification time (newest first)
                checkpoint_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                latest_checkpoint_path = str(checkpoint_files[0])
                print(f"DEBUG: Found checkpoint file directly: {latest_checkpoint_path}")
            else:
                print("DEBUG: No checkpoint files found in directory")
                return None
        else:
            print(f"DEBUG: Latest pointer file found at {latest_file}")
            with open(latest_file, 'r') as f:
                latest_checkpoint_path = f.read().strip()
            print(f"DEBUG: Latest checkpoint path from pointer: {latest_checkpoint_path}")
        
        checkpoint_path = Path(latest_checkpoint_path)
        if not checkpoint_path.exists():
            print(f"DEBUG: Checkpoint file not found at {checkpoint_path}")
            return None
        
        print(f"DEBUG: Loading checkpoint from {checkpoint_path}")
        with open(checkpoint_path, 'r') as f:
            checkpoint_data = json.load(f)
        
        print(f"DEBUG: Successfully loaded checkpoint with {checkpoint_data.get('num_sentences', 0)} sentences")
        return checkpoint_data
    
    except Exception as e:
        import traceback
        print(f"\nERROR: Failed to load checkpoint!")
        print(f"ERROR details: {str(e)}")
        print("Stack trace:")
        traceback.print_exc()
        
        # Try fallback location
        try:
            home_dir = os.path.expanduser("~")
            fallback_dir = os.path.join(home_dir, "checkpoint_fallback")
            print(f"\nAttempting to load from fallback location: {fallback_dir}")
            
            if os.path.exists(fallback_dir):
                fallback_files = list(Path(fallback_dir).glob("checkpoint_fallback_*.json"))
                if fallback_files:
                    # Sort by modification time (newest first)
                    fallback_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                    fallback_path = fallback_files[0]
                    print(f"Found fallback checkpoint at {fallback_path}")
                    
                    with open(fallback_path, 'r') as f:
                        checkpoint_data = json.load(f)
                    
                    print(f"Successfully loaded fallback checkpoint with {checkpoint_data.get('num_sentences', 0)} sentences")
                    return checkpoint_data
            
            print("No fallback checkpoints found")
        except Exception as e2:
            print(f"Failed to load fallback checkpoint: {str(e2)}")
        
        return None

def was_run_completed(checkpoint_dir: str = "checkpoints") -> bool:
    """Check if the last run was completed successfully."""
    try:
        # Use absolute path for consistency
        absolute_checkpoint_dir = os.path.abspath(checkpoint_dir)
        
        latest_checkpoint = load_latest_checkpoint(absolute_checkpoint_dir)
        if latest_checkpoint is None:
            return False
        
        # Check if the last checkpoint was marked as final
        checkpoint_info = latest_checkpoint.get("checkpoint_info", {})
        return checkpoint_info.get("is_final", False)
    except Exception as e:
        print(f"Error checking run completion status: {e}")
        return False

def estimate_corpus_complexity_from_sentences(
    sentences: list, 
    d: float=300, 
    cp_rank: float=50,
    bert_optim_factor: float=1.0, 
    use_progress_bar: bool=True,
    checkpoint_interval: int=100,  # Save every N sentences
    checkpoint_dir: str="checkpoints",
    resume: bool=True,
    batch_size: Optional[int]=None,
    output_file: Optional[str]=None
) -> dict:
    """
    Process sentences in parallel using multiprocessing. 
    Returns aggregated complexity results.
    """
    # Start timing for total runtime
    start_time = time.time()
    
    total_sentences = len(sentences)
    
    # Try to load from checkpoint if resuming
    if resume:
        corpus_results = load_latest_checkpoint(checkpoint_dir)
        if corpus_results is not None:
            # Check if the previous run was completed
            if was_run_completed(checkpoint_dir):
                print("\nPrevious run was completed successfully.")
                if corpus_results["num_sentences"] >= total_sentences:
                    print("All sentences already processed!")
                    return corpus_results
                else:
                    print("Processing additional sentences...")
            else:
                print(f"\nResuming incomplete run with {corpus_results['num_sentences']} sentences processed")
            
            # Filter out already processed sentences
            sentences = sentences[corpus_results['num_sentences']:]
            if not sentences:
                print("All sentences already processed!")
                return corpus_results
        else:
            corpus_results = {
                "num_sentences": 0,
                "discocirc_naive": 0.0,
                "discocirc_cp": 0.0,
                "bert_vanilla": 0.0,
                "bert_flash": 0.0,
                "sentence_details": []
            }
    else:
        corpus_results = {
            "num_sentences": 0,
            "discocirc_naive": 0.0,
            "discocirc_cp": 0.0,
            "bert_vanilla": 0.0,
            "bert_flash": 0.0,
            "sentence_details": []
        }
    
    # Determine optimal number of workers (75% of CPU cores)
    n_workers = get_optimal_workers()
    print(f"\nUsing {n_workers} worker processes for parallel processing")
    
    # Split sentences into chunks for parallel processing
    total_sentences = len(sentences)
    base_chunk = 100
    
    # If we have a specified batch size, use it
    if batch_size is not None:
        # Use provided batch size
        pass
    else:
        # Get optimal batch size from benchmark if available
        batch_size = base_chunk
        
        # Scale batch size based on worker count
        chunk_multiplier = math.log2(n_workers) / math.log2(8)
        batch_size = int(base_chunk * chunk_multiplier)
        batch_size = max(50, min(2000, batch_size))
    
    max_chunks = n_workers * 4
    min_chunks = n_workers
    
    total_chunks = max(1, total_sentences // batch_size)
    if total_chunks < min_chunks:
        batch_size = max(50, total_sentences // min_chunks)
    elif total_chunks > max_chunks:
        batch_size = max(50, total_sentences // max_chunks)
        
    print(f"Using batch size of {batch_size} sentences per worker")
    
    sentence_chunks = [sentences[i:i + batch_size] for i in range(0, len(sentences), batch_size)]
    
    # Create a partial function with the fixed arguments
    process_batch = partial(process_sentence_batch, 
                           d=d, 
                           cp_rank=cp_rank,
                           bert_optim_factor=bert_optim_factor)
    
    # Get start time for rate calculations
    start_time = time.time()
    
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
            completed_sentences = corpus_results["num_sentences"]
            disc_total_naive = corpus_results["discocirc_naive"]
            disc_total_cp = corpus_results["discocirc_cp"]
            bert_total_vanilla = corpus_results["bert_vanilla"]
            bert_total_flash = corpus_results["bert_flash"]
            
            pbar = tqdm(total=total_sentences, initial=completed_sentences, desc=desc, position=0)
            
            for chunk_result in pool.imap(process_batch, sentence_chunks):
                # Update metrics for this chunk
                for result in chunk_result:
                    disc_total_naive += result["disc_naive"]
                    disc_total_cp += result["disc_cp"]
                    bert_total_vanilla += result["bert_naive"]
                    bert_total_flash += result["bert_opt"]
                    completed_sentences += 1
                    
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
                    corpus_results["num_sentences"] += 1
                
                # Update progress bar with detailed metrics
                pbar.update(len(chunk_result))
                
                # Create a dynamic status message based on which algorithms are active
                status_parts = []
                if not BERT_ONLY:
                    status_parts.append(f"DisCoCirc: {disc_total_naive:.2e}/{disc_total_cp:.2e}")
                if not DISCO_ONLY:
                    status_parts.append(f"BERT: {bert_total_vanilla:.2e}/{bert_total_flash:.2e}")
                
                # Calculate processing rate
                elapsed_time = time.time() - start_time
                rate = completed_sentences / elapsed_time
                status_parts.append(f"Rate: {rate:.1f} sent/s")
                
                status = " | ".join(status_parts)
                pbar.set_postfix_str(f"{completed_sentences}/{total_sentences} sents | {status}")
                
                # Save checkpoint periodically
                if completed_sentences % checkpoint_interval == 0:
                    print(f"\nDEBUG: Creating checkpoint at {completed_sentences} sentences")
                    checkpoint_file = save_checkpoint(corpus_results, checkpoint_dir)
                    print(f"\nSaved checkpoint to {checkpoint_file}")
            
            pbar.close()
        else:
            # Similar processing without progress bar
            for chunk_result in pool.map(process_batch, sentence_chunks):
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
                    corpus_results["num_sentences"] += 1
                
                # Save checkpoint periodically
                if corpus_results["num_sentences"] % checkpoint_interval == 0:
                    print(f"\nDEBUG: Creating checkpoint at {corpus_results['num_sentences']} sentences")
                    checkpoint_file = save_checkpoint(corpus_results, checkpoint_dir)
                    print(f"\nSaved checkpoint to {checkpoint_file}")
    
    # Save final checkpoint with completion status
    final_checkpoint = save_checkpoint(corpus_results, checkpoint_dir, is_final=True)
    print(f"\nSaved final checkpoint to {final_checkpoint}")
    
    # Calculate final runtime and save results to file
    end_time = time.time()
    corpus_results["runtime"] = end_time - start_time
    corpus_results["h_dim"] = d
    corpus_results["cp_rank"] = cp_rank
    
    # Save to output file
    save_results_to_file(corpus_results, output_file)
    
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
    
    # BERT model parameters
    BERT_NUM_LAYERS = 12
    BERT_HIDDEN_SIZE = 768
    BERT_INTERMEDIATE_SIZE = 3072
        
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
    
    # First find the optimal batch size for CPU
    print("\nFinding optimal batch size for CPU...")
    
    def test_batch_size(batch_size, device="cpu"):
        """Test a specific batch size and return processing rate."""
        print(f"\nTesting batch size: {batch_size} on {device.upper()}")
        start_time = time.time()
        
        # Process in batches
        current_batch = []
        total_sentences = 0
        
        for text in sample_texts:
            sentences = chunk_text_into_sentences(text)
            current_batch.extend(sentences)
            
            if len(current_batch) >= batch_size:
                # Process batch
                if not BERT_ONLY and device == "cpu":
                    # Use spaCy's pipe for faster processing
                    docs = list(nlp.pipe(current_batch))
                    for doc in docs:
                        for token in doc:
                            rank = get_token_rank(token)
                            naive = float(300**rank)
                            opt = naive / 20.0
                
                if not DISCO_ONLY:
                    # Batch process with BERT tokenizer
                    encodings = BERT_TOKENIZER(current_batch, 
                                            add_special_tokens=True,
                                            max_length=512,
                                            truncation=True,
                                            padding=True,
                                            return_tensors="pt")
                    # Move to GPU or keep on CPU based on device
                    if device != "cpu" and torch.cuda.is_available():
                        device_id = int(device.split(':')[1]) if ':' in device else 0
                        encodings = {k: v.cuda(device_id) for k, v in encodings.items()}
                    
                    # Calculate FLOPs for the batch
                    seq_lens = [len(ids) for ids in encodings["input_ids"]]
                    for seq_len in seq_lens:
                        # Calculate FLOPs manually instead of calling the function
                        per_layer_vanilla = _layer_vanilla(seq_len, BERT_HIDDEN_SIZE)
                        per_layer_flash = _layer_flash(seq_len, BERT_HIDDEN_SIZE)
                        bert_vanilla = per_layer_vanilla * BERT_NUM_LAYERS
                        bert_flash = per_layer_flash * BERT_NUM_LAYERS
                
                total_sentences += len(current_batch)
                current_batch.clear()
        
        # Process remaining sentences
        if current_batch:
            if not BERT_ONLY and device == "cpu":
                docs = list(nlp.pipe(current_batch))
                for doc in docs:
                    for token in doc:
                        rank = get_token_rank(token)
                        naive = float(300**rank)
                        opt = naive / 20.0
            
            if not DISCO_ONLY:
                encodings = BERT_TOKENIZER(current_batch, 
                                        add_special_tokens=True,
                                        max_length=512,
                                        truncation=True,
                                        padding=True,
                                        return_tensors="pt")
                if device != "cpu" and torch.cuda.is_available():
                    device_id = int(device.split(':')[1]) if ':' in device else 0
                    encodings = {k: v.cuda(device_id) for k, v in encodings.items()}
                
                seq_lens = [len(ids) for ids in encodings["input_ids"]]
                for seq_len in seq_lens:
                    # Calculate FLOPs manually instead of calling the function
                    per_layer_vanilla = _layer_vanilla(seq_len, BERT_HIDDEN_SIZE)
                    per_layer_flash = _layer_flash(seq_len, BERT_HIDDEN_SIZE)
                    bert_vanilla = per_layer_vanilla * BERT_NUM_LAYERS
                    bert_flash = per_layer_flash * BERT_NUM_LAYERS
            
            total_sentences += len(current_batch)
        
        elapsed_time = time.time() - start_time
        sentences_per_second = total_sentences / elapsed_time
        
        print(f"  Processed {total_sentences} sentences in {elapsed_time:.2f} seconds")
        print(f"  Rate: {sentences_per_second:.1f} sentences/second")
        
        return sentences_per_second
    
    # Function to find optimal batch size for a device using binary search
    def find_optimal_batch_size(device):
        print(f"\nFinding optimal batch size for {device.upper()}...")
        
        # Binary search parameters
        min_batch = 100
        max_batch = 50000  # Start with a high upper bound
        best_rate = 0
        best_batch = min_batch
        tolerance = 0.01  # 1% improvement threshold
        
        # Initial test at min_batch
        current_rate = test_batch_size(min_batch, device)
        best_rate = current_rate
        
        # Binary search loop
        while max_batch - min_batch > 100:  # Continue until batch sizes are close
            mid_batch = (min_batch + max_batch) // 2
            mid_rate = test_batch_size(mid_batch, device)
            
            if mid_rate > best_rate * (1 + tolerance):
                # Found better rate, search higher
                best_rate = mid_rate
                best_batch = mid_batch
                min_batch = mid_batch
            else:
                # No significant improvement, search lower
                max_batch = mid_batch
        
        # Final test at best batch size
        final_rate = test_batch_size(best_batch, device)
        if final_rate > best_rate:
            best_rate = final_rate
        else:
            best_batch = (best_batch + min_batch) // 2
            best_rate = test_batch_size(best_batch, device)
        
        print(f"\nOptimal batch size for {device.upper()}: {best_batch} (rate: {best_rate:.1f} sentences/second)")
        return {"batch_size": best_batch, "rate": best_rate}
    
    # Find optimal batch sizes for each device
    optimal_settings = {}
    
    # Always benchmark CPU
    optimal_settings["cpu"] = find_optimal_batch_size("cpu")
    
    # Benchmark each GPU if available
    if torch.cuda.is_available() and not CPU_ONLY and not DISCO_ONLY:
        for i in range(torch.cuda.device_count()):
            device = f"cuda:{i}"
            optimal_settings[device] = find_optimal_batch_size(device)
    
    # Benchmark DisCoCirc processing (always on CPU)
    print("\nBenchmarking DisCoCirc processing...")
    disco_times = []
    disco_pbar = tqdm(total=len(sample_texts), desc="DisCoCirc", position=2, leave=False)
    
    # Use the CPU optimal batch size for DisCoCirc
    batch_size = optimal_settings["cpu"]["batch_size"]
    current_batch = []
    
    for text in sample_texts:
        current_batch.append(text)
        
        if len(current_batch) >= batch_size:
            start_time = time.time()
            # Process batch
            docs = list(nlp.pipe(current_batch))
            for doc in docs:
                total_naive = 0.0
                total_opt = 0.0
                for token in doc:
                    rank = get_token_rank(token)
                    naive = float(300**rank)
                    opt = naive / 20.0
                    total_naive += naive
                    total_opt += opt
                disco_times.append(time.time() - start_time)
            disco_pbar.update(len(current_batch))
            current_batch.clear()
    
    # Process remaining texts
    if current_batch:
        start_time = time.time()
        docs = list(nlp.pipe(current_batch))
        for doc in docs:
            total_naive = 0.0
            total_opt = 0.0
            for token in doc:
                rank = get_token_rank(token)
                naive = float(300**rank)
                opt = naive / 20.0
                total_naive += naive
                total_opt += opt
            disco_times.append(time.time() - start_time)
            disco_pbar.update(len(current_batch))
    
    disco_pbar.close()
    
    # Print benchmark results
    print("\n=== Benchmark Results ===")
    if not DISCO_ONLY:
        print("\nBERT Processing:")
        for device in optimal_settings.keys():
            settings = optimal_settings[device]
            print(f"\n{device.upper()}:")
            print(f"  Optimal batch size: {settings['batch_size']}")
            print(f"  Processing rate: {settings['rate']:.1f} sentences/second")
    
    if not BERT_ONLY:
        print("\nDisCoCirc Processing:")
        print(f"  Average time: {sum(disco_times)/len(disco_times):.4f} seconds")
        print(f"  Min time:     {min(disco_times):.4f} seconds")
        print(f"  Max time:     {max(disco_times):.4f} seconds")
        print(f"  Total time:   {sum(disco_times):.4f} seconds")
    
    # Determine optimal settings based on fastest device
    if not DISCO_ONLY:
        fastest_device = max(optimal_settings.keys(), key=lambda d: optimal_settings[d]["rate"])
    else:
        fastest_device = "cpu"  # Always use CPU in DisCoCirc-only mode
    
    optimal_result = {
        "use_cpu": "cpu" in fastest_device,
        "fastest_device": fastest_device,
        "device_settings": optimal_settings,
        "batch_size": optimal_settings[fastest_device]["batch_size"]
    }
    
    print("\n=== Optimal Settings ===")
    print(f"Fastest device: {fastest_device.upper()}")
    print(f"Using CPU: {optimal_result['use_cpu']}")
    print(f"Optimal batch size: {optimal_result['batch_size']}")
    print(f"Multiprocessing mode: {'fork' if USE_FORK else 'spawn'} (use --fork flag for possibly faster processing)")
    
    return optimal_result

def scan_article(article):
    """Process a single article and return its statistics."""
    # Simple sentence splitting by periods
    text = article["text"]
    sentences = []
    for line in text.split('\n'):
        for sent in line.split('.'):
            sent = sent.strip()
            if sent:  # Only add non-empty sentences
                sentences.append(sent)
    
    total_sentences = len(sentences)
    total_tokens = 0
    
    # Count tokens using simple whitespace splitting for all modes
    for sent in sentences:
        total_tokens += len(sent.split())
    
    return {
        "sentences": total_sentences,
        "tokens": total_tokens
    }

def scan_wikitext_size(cache_file="wikitext_scan_cache.json"):
    """Scan WikiText-103 to get total size and article count, with caching."""
    # Check if we have a cached result already
    if os.path.exists(cache_file):
        print(f"\nLoading cached WikiText-103 statistics from {cache_file}...")
        try:
            with open(cache_file, 'r') as f:
                cached_stats = json.load(f)
            print("\n=== WikiText-103 Dataset Statistics (Cached) ===")
            print(f"Total articles: {cached_stats['total_articles']:,}")
            print(f"Total sentences: {cached_stats['total_sentences']:,}")
            print(f"Total tokens: {cached_stats['total_tokens']:,}")
            print(f"Average sentences per article: {cached_stats['total_sentences']/cached_stats['total_articles']:.1f}")
            print(f"Average tokens per sentence: {cached_stats['total_tokens']/cached_stats['total_sentences']:.1f}")
            return cached_stats
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error reading cache file: {e}. Will rescan dataset.")
    
    print("\nScanning WikiText-103 dataset size (this may take a while)...")
    
    # First get the total number of articles from the dataset info
    dataset = load_dataset("wikitext", "wikitext-103-v1")
    total_articles = len(dataset["train"])
    print(f"Total articles in WikiText-103: {total_articles:,}")
    
    # Now stream through to get detailed stats
    dataset = load_dataset("wikitext", "wikitext-103-v1", split="train", streaming=True)
    
    # Initialize counters
    total_sentences = 0
    total_tokens = 0
    processed_articles = 0
    start_time = time.time()
    
    # Use a larger batch size for processing
    batch_size = 1000
    current_batch = []
    
    # Process articles in batches
    with tqdm(total=total_articles, desc="Scanning articles", unit="articles") as pbar:
        for article in dataset:
            current_batch.append(article)
            processed_articles += 1
            
            if len(current_batch) >= batch_size or processed_articles == total_articles:
                # Process batch
                batch_sentences = []
                
                # Simple sentence splitting by periods
                for art in current_batch:
                    text = art["text"]
                    for line in text.split('\n'):
                        for sent in line.split('.'):
                            sent = sent.strip()
                            if sent:  # Only add non-empty sentences
                                batch_sentences.append(sent)
                
                # Count tokens using simple whitespace splitting
                total_tokens += sum(len(sent.split()) for sent in batch_sentences)
                total_sentences += len(batch_sentences)
                pbar.update(len(current_batch))
                
                # Print intermediate stats
                if processed_articles % 10000 == 0:
                    elapsed_time = time.time() - start_time
                    rate = processed_articles / elapsed_time
                    print(f"\nIntermediate stats:")
                    print(f"  Articles processed: {processed_articles:,}/{total_articles:,}")
                    print(f"  Total sentences: {total_sentences:,}")
                    print(f"  Total tokens: {total_tokens:,}")
                    print(f"  Progress: {(processed_articles/total_articles)*100:.1f}%")
                    print(f"  Processing rate: {rate:.1f} articles/second")
                    
                    # Save intermediate cache in case of interruption
                    intermediate_stats = {
                        "total_articles": total_articles,
                        "total_sentences": total_sentences,
                        "total_tokens": total_tokens,
                        "processed_articles": processed_articles,
                        "is_complete": False,
                        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
                    }
                    with open(cache_file + ".partial", 'w') as f:
                        json.dump(intermediate_stats, f, indent=2)
                
                # Clear batch
                current_batch.clear()
    
    # Calculate stats
    stats = {
        "total_articles": total_articles,
        "total_sentences": total_sentences,
        "total_tokens": total_tokens,
        "is_complete": True,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
    }
    
    # Save to cache file
    print(f"\nSaving WikiText-103 statistics to cache file {cache_file}...")
    with open(cache_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("\n=== WikiText-103 Dataset Statistics ===")
    print(f"Total articles: {total_articles:,}")
    print(f"Total sentences: {total_sentences:,}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Average sentences per article: {total_sentences/total_articles:.1f}")
    print(f"Average tokens per sentence: {total_tokens/total_sentences:.1f}")
    
    return stats

###############################################################################
# 7) Results Saving  
###############################################################################
def save_results_to_file(corpus_results, output_file=None):
    """Save the final results to a human-readable file."""
    if output_file is None:
        # Generate a default filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"complexity_results_{timestamp}.txt"
    
    with open(output_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("[Corpus-level FLOP Totals]\n")
        f.write(f"Sentences processed : {corpus_results['num_sentences']:,}\n")
        f.write(f"Full-rank DisCoCirc : {corpus_results['discocirc_naive']:,.2e}\n")
        f.write(f"CP-rank-{corpus_results.get('cp_rank', 50)}   : {corpus_results['discocirc_cp']:,.2e}\n")
        f.write(f"BERT vanilla        : {corpus_results['bert_vanilla']:,.2e}\n")
        f.write(f"BERT FlashAttention : {corpus_results['bert_flash']:,.2e}\n")

        # Add comparison of DisCoCirc vs BERT if both are calculated
        if not (BERT_ONLY or DISCO_ONLY):
            naive_gain = 100 * (corpus_results["bert_vanilla"] - 
                               corpus_results["discocirc_naive"]) / corpus_results["bert_vanilla"]
            cp_gain = 100 * (corpus_results["bert_flash"] - 
                            corpus_results["discocirc_cp"]) / corpus_results["bert_flash"]
            f.write("\nDisCoCirc vs BERT:\n")
            f.write(f"  Full-rank saves : {naive_gain:6.2f}% FLOPs over vanilla BERT\n")
            f.write(f"  CP-rank saves   : {cp_gain:6.2f}% FLOPs over Flash-BERT\n")
        
        # Add system information
        f.write("\n" + "=" * 60 + "\n")
        f.write("[System Information]\n")
        f.write(f"Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # GPU information
        if torch.cuda.is_available():
            f.write(f"GPU Count: {torch.cuda.device_count()}\n")
            for i in range(torch.cuda.device_count()):
                f.write(f"  Device {i}: {torch.cuda.get_device_name(i)}\n")
            f.write(f"CUDA Version: {torch.version.cuda}\n")
        else:
            f.write("GPU: None (CPU-only)\n")
        
        # Processing information
        f.write(f"PyTorch Version: {torch.__version__}\n")
        f.write(f"Embedding Dimension: {corpus_results.get('h_dim', 768)}\n")
        f.write(f"CP-rank: {corpus_results.get('cp_rank', 50)}\n")
        
        if 'runtime' in corpus_results:
            f.write(f"Total Runtime: {corpus_results['runtime']:.2f} seconds\n")
            f.write(f"Processing Speed: {corpus_results['num_sentences']/corpus_results['runtime']:.2f} sentences/second\n")
    
    print(f"\nResults saved to {output_file}")
    return output_file

###############################################################################
# Memory Management Helper
###############################################################################
def manage_memory(force_gc=False):
    """
    Monitor and manage memory to avoid periodic halts.
    Returns current memory usage statistics.
    """
    # Force garbage collection if requested
    if force_gc:
        gc.collect()
    
    mem_stats = {}
    
    # Get CPU memory stats
    try:
        import psutil
        process = psutil.Process()
        mem_stats['cpu_percent'] = process.cpu_percent()
        mem_info = process.memory_info()
        mem_stats['ram_used_mb'] = mem_info.rss / (1024 * 1024)
        mem_stats['ram_percent'] = process.memory_percent()
    except ImportError:
        mem_stats['cpu_ram'] = "psutil not available"
    
    # Get CUDA memory stats if available
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            try:
                mem_stats[f'cuda_{i}_allocated_mb'] = torch.cuda.memory_allocated(i) / (1024 * 1024)
                mem_stats[f'cuda_{i}_reserved_mb'] = torch.cuda.memory_reserved(i) / (1024 * 1024)
                # Get memory stats from nvidia-smi if possible
                try:
                    import subprocess
                    result = subprocess.check_output(['nvidia-smi', f'--id={i}', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'])
                    used, total = map(int, result.decode('utf-8').strip().split(','))
                    mem_stats[f'cuda_{i}_used_mb'] = used
                    mem_stats[f'cuda_{i}_total_mb'] = total
                    mem_stats[f'cuda_{i}_percent'] = used / total * 100
                except:
                    pass
            except:
                mem_stats[f'cuda_{i}'] = "Error getting memory stats"
    
    return mem_stats

###############################################################################
# 6) Main Execution  ––  UPDATED FOR NEW COMPLEXITY FORMULAS
###############################################################################
if __name__ == "__main__":
    # Add new command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Complexity estimation with checkpointing')
    parser.add_argument('--checkpoint-interval', type=int, default=420,
                       help='Save checkpoint every N sentences (default: 420)')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints (default: checkpoints)')
    parser.add_argument('--no-resume', action='store_true',
                       help='Do not resume from checkpoint')
    parser.add_argument('--force-resume', action='store_true',
                       help='Force resume even if previous run was completed')
    parser.add_argument('--wikitext', action='store_true',
                       help='Use WikiText-103 dataset')
    parser.add_argument('--bert', action='store_true',
                       help='Run in BERT-only mode')
    parser.add_argument('--disco', action='store_true',
                       help='Run in DisCoCirc-only mode')
    parser.add_argument('--cpu', action='store_true',
                       help='Force CPU usage for all operations')
    parser.add_argument('--cpu-bert', action='store_true',
                       help='Force CPU usage for BERT operations only')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run benchmarks to determine optimal settings')
    parser.add_argument('--fork', action='store_true',
                       help='Use fork instead of spawn for multiprocessing')
    parser.add_argument('--workers', type=int,
                       help='Number of worker processes to use')
    parser.add_argument('--batch-size', type=int,
                       help='Batch size for processing (default: determined by benchmark)')
    parser.add_argument('--gpu-id', type=int, default=None,
                       help='Specific GPU ID to use (default: use all available GPUs)')
    parser.add_argument('--scan', action='store_true',
                       help='Scan WikiText-103 size before processing')
    parser.add_argument('--scan-only', action='store_true',
                       help='Only scan WikiText-103 size without processing')
    parser.add_argument('--scan-cache-file', type=str, default='wikitext_scan_cache.json',
                       help='File to cache WikiText scan results (default: wikitext_scan_cache.json)')
    parser.add_argument('--output-file', type=str, default=None,
                       help='File to save final results (default: auto-generated filename)')
    args = parser.parse_args()

    # Update global flags based on arguments
    USE_WIKITEXT = args.wikitext and load_dataset is not None
    BERT_ONLY = args.bert
    DISCO_ONLY = args.disco
    CPU_ONLY = args.cpu
    CPU_BERT = args.cpu_bert
    BENCHMARK = args.benchmark
    USE_FORK = args.fork
    if args.workers is not None:
        MAX_WORKERS = args.workers
    if args.gpu_id is not None:
        torch.cuda.set_device(args.gpu_id)

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
    batch_size = None
    if BENCHMARK:
        optimal = benchmark_processing_methods()
        CPU_ONLY = optimal["use_cpu"]
        print(f"\nBenchmark-selected device: {'CPU' if CPU_ONLY else optimal['fastest_device']}")
        
        # Use bench-determined batch size unless user specified one
        if args.batch_size is None:
            batch_size = optimal["batch_size"]
            print(f"Using benchmark-determined batch size: {batch_size}")
        else:
            batch_size = args.batch_size
            print(f"Using user-specified batch size: {batch_size}")
        
        # If specific GPU was determined to be fastest, use it
        if not optimal["use_cpu"] and ":" in optimal["fastest_device"]:
            gpu_id = int(optimal["fastest_device"].split(':')[1])
            print(f"Setting fastest GPU ({gpu_id}) as primary device")
            torch.cuda.set_device(gpu_id)
    else:
        batch_size = args.batch_size if args.batch_size is not None else get_optimal_batch_size_for_hardware()
        print(f"Using {'default' if args.batch_size is None else 'specified'} batch size: {batch_size}")

    # ------------------------------------------------------------------ WIKITEXT
    if USE_WIKITEXT:
        # Optionally scan the dataset size
        dataset_stats = None
        if args.scan or args.scan_only:
            print("\nScanning WikiText-103 dataset size...")
            dataset_stats = scan_wikitext_size(cache_file=args.scan_cache_file)
            
            if args.scan_only:
                print("\nScan complete. Exiting as requested.")
                sys.exit(0)
        else:
            # Try to load from cache first
            if os.path.exists(args.scan_cache_file):
                print(f"\nLoading cached dataset statistics from {args.scan_cache_file}...")
                try:
                    with open(args.scan_cache_file, 'r') as f:
                        dataset_stats = json.load(f)
                    print("Using cached dataset statistics")
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Error reading cache file: {e}. Using estimates instead.")
                    # No cache file exists
                    print("\nNo scan cache found. Using estimated dataset size (use --scan for accurate size)")
                    dataset_stats = {
                        "total_articles": 1801350,  # Known size from dataset info
                        "total_sentences": 4700000,  # Rough estimate
                        "total_tokens": 103000000   # Rough estimate (103M tokens)
                    }
        
        # Use the dataset stats for better progress reporting
        total_articles = dataset_stats["total_articles"]
        total_sentences = dataset_stats["total_sentences"]
        
        print("\nLoading WikiText-103 (streaming)…")
        dataset = load_dataset("wikitext", "wikitext-103-v1",
                               split="train", streaming=True)

        corpus_results = {
            "num_sentences": 0,
            "discocirc_naive": 0.0,
            "discocirc_cp": 0.0,
            "bert_vanilla": 0.0,
            "bert_flash": 0.0,
            "dataset_stats": dataset_stats
        }

        # Initialize the sentences array
        current_sentences = []
        
        # Start timing
        start_time = time.time()
        last_checkpoint_time = start_time
        sentences_since_checkpoint = 0
        total_sentences_processed = 0

        # Configure multiprocessing for optimal performance
        if USE_FORK and (CPU_ONLY or DISCO_ONLY):
            # Only use fork if no CUDA operations are needed
            print("Using 'fork' start method for multiprocessing")
            multiprocessing.set_start_method("fork", force=True)
        else:
            # Always use spawn when using CUDA (fork doesn't work with CUDA)
            print("Using 'spawn' start method for multiprocessing (required for CUDA)")
            multiprocessing.set_start_method("spawn", force=True)

        # process_sentence_batch now uses H_DIM & CP_RANK
        proc_fn = partial(process_sentence_batch, 
                         d=H_DIM, 
                         cp_rank=CP_RANK,
                         bert_optim_factor=1.0)

        print(f"\nRunning with batch size: {batch_size}, workers: {effective_workers}")
        print(f"Using checkpointing interval: every {args.checkpoint_interval} sentences")
        
        # For V100s, use larger chunksize for better GPU utilization
        optimal_chunksize = max(5, batch_size // (4 * effective_workers))

        with multiprocessing.Pool(effective_workers) as pool, \
             tqdm(total=total_articles, desc="Articles") as art_pbar:

            # Memory management variables
            last_mem_check = time.time()
            mem_check_interval = 60  # Check memory every 60 seconds
            processed_since_gc = 0
            gc_threshold = 2000      # Garbage collect after this many sentences
            mem_stats = {}
            
            for example in dataset:
                art_pbar.update(1)
                current_sentences.extend(chunk_text_into_sentences(example["text"]))

                if len(current_sentences) >= batch_size:
                    # Periodically check memory and potentially clean up
                    now = time.time()
                    if (now - last_mem_check) > mem_check_interval or processed_since_gc > gc_threshold:
                        mem_stats = manage_memory(force_gc=(processed_since_gc > gc_threshold))
                        last_mem_check = now
                        processed_since_gc = 0
                        
                        # Print memory stats if there might be an issue
                        if any(('percent' in k and v > 80) for k, v in mem_stats.items() if isinstance(v, (int, float))):
                            print("\nHigh memory usage detected:")
                            for k, v in mem_stats.items():
                                if isinstance(v, float):
                                    print(f"  {k}: {v:.1f}")
                                else:
                                    print(f"  {k}: {v}")
                            print("Triggering garbage collection...")
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                    
                    # Process in larger batches, with optimal chunksize for GPUs
                    try:
                        results = pool.apply(proc_fn, (current_sentences,))
                        for res in results:
                            corpus_results["discocirc_naive"] += res["disc_naive"]
                            corpus_results["discocirc_cp"] += res["disc_cp"]
                            corpus_results["bert_vanilla"] += res["bert_naive"]
                            corpus_results["bert_flash"] += res["bert_opt"]
                            corpus_results["num_sentences"] += 1
                            sentences_since_checkpoint += 1
                            processed_since_gc += 1
                            total_sentences_processed += 1
                        
                            # Print sentence count status regularly
                            if total_sentences_processed % 100 == 0:
                                print(f"\nDEBUG: Sentences processed: {total_sentences_processed}, since last checkpoint: {sentences_since_checkpoint}")
                                print(f"DEBUG: Checkpoint will be created at {args.checkpoint_interval} sentences")
                                
                            # Log checkpoint conditions if we're getting close
                            if sentences_since_checkpoint >= args.checkpoint_interval * 0.9 and sentences_since_checkpoint % 10 == 0:
                                print(f"\nDEBUG: Getting close to checkpoint! {sentences_since_checkpoint}/{args.checkpoint_interval} sentences")
                        
                            # Create a checkpoint when we've processed enough sentences (no waiting time)
                            if sentences_since_checkpoint >= args.checkpoint_interval:
                                print(f"\nDEBUG: Creating checkpoint at {sentences_since_checkpoint} sentences")
                                checkpoint_file = save_checkpoint(corpus_results, args.checkpoint_dir)
                                print(f"\nSaved checkpoint to {checkpoint_file}")
                                
                                # Print memory stats along with checkpoint
                                mem_stats = manage_memory(force_gc=False)
                                print("\nMemory usage at checkpoint:")
                                for k, v in mem_stats.items():
                                    if isinstance(v, float):
                                        print(f"  {k}: {v:.1f}")
                                    else:
                                        print(f"  {k}: {v}")
                                
                                # Print current processing rate
                                elapsed_time = time.time() - start_time
                                rate = corpus_results["num_sentences"] / elapsed_time
                                print(f"Processing rate: {rate:.1f} sentences/second")
                                print(f"Progress: {corpus_results['num_sentences']:,}/{total_sentences:,} sentences")
                                
                                # Reset counters
                                last_checkpoint_time = time.time()
                                sentences_since_checkpoint = 0
                        
                        current_sentences.clear()
                        
                    except Exception as e:
                        print(f"\nError processing batch: {e}")
                        # Try to recover by clearing memory and reducing batch size
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        print("Recovered from error. Continuing with a smaller batch...")
                        
                        # Reduce batch size temporarily
                        if len(current_sentences) > 1000:
                            print(f"Reducing batch from {len(current_sentences)} to 1000 sentences")
                            reduced_batch = current_sentences[:1000]
                            try:
                                results = pool.apply(proc_fn, (reduced_batch,))
                                for res in results:
                                    corpus_results["discocirc_naive"] += res["disc_naive"]
                                    corpus_results["discocirc_cp"] += res["disc_cp"]
                                    corpus_results["bert_vanilla"] += res["bert_naive"]
                                    corpus_results["bert_flash"] += res["bert_opt"]
                                    corpus_results["num_sentences"] += 1
                                
                                # Keep remainder
                                current_sentences = current_sentences[1000:]
                            except:
                                # If still failing, just skip this batch
                                print("Still failing, skipping batch")
                                current_sentences.clear()
                        else:
                            # If batch is already small, just skip it
                            print("Skipping problematic batch")
                            current_sentences.clear()

            # flush remainder
            if current_sentences:
                results = pool.apply(proc_fn, (current_sentences,))
                for res in results:
                    corpus_results["discocirc_naive"] += res["disc_naive"]
                    corpus_results["discocirc_cp"] += res["disc_cp"]
                    corpus_results["bert_vanilla"] += res["bert_naive"]
                    corpus_results["bert_flash"] += res["bert_opt"]
                    corpus_results["num_sentences"] += 1
            
            # Save final checkpoint
            final_checkpoint = save_checkpoint(corpus_results, args.checkpoint_dir, is_final=True)
            print(f"\nSaved final checkpoint to {final_checkpoint}")
            
            # Calculate final runtime and save results to file
            end_time = time.time()
            corpus_results["runtime"] = end_time - start_time
            corpus_results["h_dim"] = H_DIM
            corpus_results["cp_rank"] = CP_RANK
            
            # Save to output file
            save_results_to_file(corpus_results, args.output_file)
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
            checkpoint_interval=args.checkpoint_interval,
            checkpoint_dir=args.checkpoint_dir,
            resume=not args.no_resume,
            batch_size=batch_size,
            output_file=args.output_file
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

    save_results_to_file(corpus_results, args.output_file)
