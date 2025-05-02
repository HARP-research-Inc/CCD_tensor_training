# complexity_estimator.py – streamlined & robust
"""
Estimate FLOPs for DisCoCirc / BERT on large corpora.
Highlights of this clean rewrite
-------------------------------------------------
* Modular layout (utils, gpu, complexity, io, main)
* VRAM‑aware batch‑size guess + binary back‑off on CUDA OOM
* torch.nn.DataParallel support out‑of‑the‑box when >1 GPU
* Checkpoint‑on‑fail – never lose > slice
* Optional synthetic GPU load behind --synthetic flag
* Works identical for WikiText streaming or plain files
* Progress tracking with ETA display
* Resume from checkpoint functionality
* Preprocessing cache for faster repeated runs
"""

from __future__ import annotations

import argparse, json, math, os, sys, time, gc, subprocess, re, hashlib
from pathlib import Path
from datetime import datetime, timedelta
from functools import partial
from typing import List, Dict, Tuple, Optional
import multiprocessing as mp

import torch
import spacy
from transformers import BertTokenizer
from tqdm import tqdm

# ---------------------------------------------------------------------------
# ---------------------------  ── CONSTANTS  ──  ----------------------------
# ---------------------------------------------------------------------------

# WikiText-103 dataset statistics
WIKITEXT_STATS = {
    "articles": 1801350,
    "sentences": 4548336,
    "tokens": 97989460
}

# Cache directory for preprocessed text
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# ---------------------------  ── CACHE  ──  ------------------------------
# ---------------------------------------------------------------------------

def get_cache_key(text: str, dataset_name: str = "", chunk_id: int = 0) -> str:
    """Generate a cache key for the text."""
    if dataset_name:
        # For dataset chunks, use deterministic ID
        return f"{dataset_name}_{chunk_id}"
    else:
        # For arbitrary text, use hash
        return hashlib.md5(text.encode()).hexdigest()

def save_to_cache(sentences: List[str], cache_key: str):
    """Save preprocessed sentences to cache."""
    cache_file = CACHE_DIR / f"{cache_key}.json"
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(sentences, f)
    return cache_file

def load_from_cache(cache_key: str) -> Optional[List[str]]:
    """Load preprocessed sentences from cache if available."""
    cache_file = CACHE_DIR / f"{cache_key}.json"
    if cache_file.exists():
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to load cache: {e}")
    return None

def get_cache_stats():
    """Return statistics about the cache."""
    files = list(CACHE_DIR.glob("*.json"))
    total_size = sum(f.stat().st_size for f in files)
    return {
        "num_files": len(files),
        "total_size_mb": total_size / (1024 * 1024),
        "last_modified": max((f.stat().st_mtime for f in files), default=0)
    }

# ---------------------------------------------------------------------------
# ---------------------------  ── GPU / DEVICE  ──  -------------------------
# ---------------------------------------------------------------------------

CUDA_LAUNCH_BLOCKING = os.getenv("CUDA_LAUNCH_BLOCKING", "1")
os.environ["CUDA_LAUNCH_BLOCKING"] = CUDA_LAUNCH_BLOCKING  # ensure sync on errors

DEVICE: torch.device | None = None
BERT_TOKENIZER: BertTokenizer | None = None
NLPROC: spacy.language.Language | None = None


def _vram_min_mb() -> int:
    """Return min total VRAM (MiB) across visible GPUs, or 0 if none."""
    try:
        out = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=memory.total",
            "--format=csv,noheader,nounits",
        ])
        return min(map(int, out.decode().strip().splitlines()))
    except Exception:  # noqa: BLE001
        return 0


def guess_batch_size() -> int:
    """Heuristic: ~1.2 MiB / sentence @ 512 tok padded; leave 20 % headroom."""
    vram = _vram_min_mb()
    if vram:
        est = int(vram / 1.2)
        return max(256, min(est, 4096))
    return 1000  # CPU default


def init_device(force_cpu: bool = False, specific_gpu: Optional[int] = None):
    global DEVICE, BERT_TOKENIZER

    if force_cpu or not torch.cuda.is_available():
        DEVICE = torch.device("cpu")
    else:
        if specific_gpu is not None:
            torch.cuda.set_device(specific_gpu)
        DEVICE = torch.device("cuda")
        if torch.cuda.device_count() > 1:
            print(f"[GPU] DataParallel across {torch.cuda.device_count()} devices")
    print("Using", DEVICE)

    BERT_TOKENIZER = BertTokenizer.from_pretrained("bert-base-uncased")


# ---------------------------------------------------------------------------
# ---------------------------  ── COMPLEXITY  ──  ---------------------------
# ---------------------------------------------------------------------------

MAC = 2  # mul+add


def _layer_vanilla(L: int, H: int = 768) -> int:
    proj_qkv, proj_out = 3 * MAC * L * H * H, MAC * L * H * H
    attn_mat = 2 * MAC * L * L * H
    softmax, ffw, gelu = 7 * L * L, 2 * MAC * L * H * 4 * H, 8 * L * 4 * H
    norm_res = 2 * 8 * L * H + 2 * L * H
    return proj_qkv + proj_out + attn_mat + softmax + ffw + gelu + norm_res


def _layer_flash(L: int, H: int = 768) -> int:
    proj_qkv, proj_out = 3 * MAC * L * H * H, MAC * L * H * H
    attn_mat = MAC * L * L * H
    ffw = 2 * MAC * L * H * 4 * H
    norm_res = 2 * 8 * L * H + 2 * L * H
    return proj_qkv + proj_out + attn_mat + ffw + norm_res

# -------- spacy helpers ----------------------------------------------------

POS_RANK = {
    1: {"NOUN", "PROPN", "PRON"},
    2: {"ADJ", "ADV", "DET", "AUX"},
    3: {"ADP", "CCONJ", "SCONJ"},
}

def token_rank(tok) -> int:
    if tok.pos_ == "VERB":
        has_d = any(c.dep_ in {"obj", "dobj"} for c in tok.children)
        has_i = any(c.dep_ == "iobj" for c in tok.children)
        return 4 if (has_d and has_i) else 3 if has_d else 2
    for k, v in POS_RANK.items():
        if tok.pos_ in v:
            return k
    return 1


def discocirc_flops(sent: str, d: int = 768) -> float:
    doc = NLPROC(sent)
    return sum(d ** token_rank(t) for t in doc)


def discocirc_cp_flops(sent: str, d: int = 768, R: int = 50) -> float:
    doc = NLPROC(sent)
    return sum(R * (token_rank(t) + 2) * d for t in doc)


def bert_flops(sent: str | int, flash: bool = False) -> Tuple[float, int]:
    if isinstance(sent, int):
        L = sent
    else:
        tokens = BERT_TOKENIZER.encode(str(sent), add_special_tokens=True)
        L = len(tokens)
    per_layer = _layer_flash(L) if flash else _layer_vanilla(L)
    return per_layer * 12, L

# ---------------------------------------------------------------------------
# ---------------------------  ── SENTENCE SPLIT  ──  -----------------------
# ---------------------------------------------------------------------------

def sentences_spacy(text: str, use_cache: bool = True, cache_key: str = "", 
                   dataset_name: str = "", chunk_id: int = 0, fast_mode: bool = False) -> List[str]:
    """Split text into sentences using spaCy with caching support."""
    if not text or len(text.strip()) == 0:
        return []
    
    # Generate cache key if not provided
    if use_cache and not cache_key:
        cache_key = get_cache_key(text, dataset_name, chunk_id)
        
    # Try to load from cache first
    if use_cache:
        cached_sentences = load_from_cache(cache_key)
        if cached_sentences:
            return cached_sentences
    
    # For very long texts, use a more efficient approach
    if len(text) > 100000 or fast_mode:  # 100KB threshold or fast mode
        if len(text) > 100000:
            print(f"[INFO] Processing large text ({len(text)/1024:.1f} KB)")
        
        # Fast mode: simple paragraph and sentence splitting
        if fast_mode:
            sentences = []
            paragraphs = text.split('\n')
            for para in paragraphs:
                if para.strip():
                    # Simple sentence splitting by punctuation
                    for sent in re.split(r'(?<=[.!?])\s+', para):
                        if sent.strip():
                            sentences.append(sent.strip())
            if use_cache:
                save_to_cache(sentences, cache_key)
            return sentences
        
        # Regular processing for large text: split by newlines first
        paragraphs = text.split('\n')
        sentences = []
        for para in tqdm(paragraphs, desc="Parsing paragraphs", leave=False):
            if para.strip():
                try:
                    doc = NLPROC(para)
                    sentences.extend([s.text.strip() for s in doc.sents if s.text.strip()])
                except Exception as e:
                    print(f"[WARN] Error parsing paragraph: {e}")
                    # Fallback to simple sentence splitting
                    sentences.extend([s.strip() for s in para.split('.') if s.strip()])
        
        # Cache the results
        if use_cache:
            save_to_cache(sentences, cache_key)
        return sentences
    
    # Normal case for smaller texts
    doc = NLPROC(text)
    sentences = [s.text.strip() for s in doc.sents if s.text.strip()]
    
    # Cache the results
    if use_cache:
        save_to_cache(sentences, cache_key)
    return sentences

# fallback splitter for BERT‑only mode
def crude_split(text: str) -> List[str]:
    """Simple sentence splitter that doesn't require SpaCy."""
    sents = []
    for line in text.split("\n"):
        if not line.strip():
            continue
        for sub in line.split('.'):
            sub = sub.strip()
            if sub:
                sents.append(sub)
    return sents

# ---------------------------------------------------------------------------
# ---------------------------  ── WORKER  ──  -------------------------------
# ---------------------------------------------------------------------------


def process_batch(
    sents: List[str],
    d: int,
    cp_rank: int,
    do_disco: bool,
    do_bert: bool,
) -> List[Dict[str, float]]:
    """Return metrics for each sentence in *sents*. This runs in worker."""
    global NLPROC, BERT_TOKENIZER
    if NLPROC is None and do_disco:
        NLPROC = spacy.load("en_core_web_sm")
    if BERT_TOKENIZER is None and do_bert:
        BERT_TOKENIZER = BertTokenizer.from_pretrained("bert-base-uncased")

    out = []
    if do_disco:
        docs = list(NLPROC.pipe(sents))
    else:
        docs = [None] * len(sents)  # placeholder

    for idx, sent in enumerate(sents):
        doc = docs[idx]
        disc_n, disc_cp = (0.0, 0.0)
        if do_disco:
            disc_n = discocirc_flops(doc.text, d)
            disc_cp = discocirc_cp_flops(doc.text, d, cp_rank)
        bert_v, bert_f = (0.0, 0.0)
        tok = 0
        if do_bert:
            bert_v, tok = bert_flops(sent, flash=False)
            bert_f, _ = bert_flops(tok, flash=True)
        out.append(
            {
                "disc_naive": disc_n,
                "disc_cp": disc_cp,
                "bert_naive": bert_v,
                "bert_opt": bert_f,
                "token_count": tok,
            }
        )
    return out

# ---------------------------------------------------------------------------
# ---------------------------  ── CHECKPOINT  ──  ---------------------------
# ---------------------------------------------------------------------------

CKPT_DIR = Path("checkpoints")
CKPT_DIR.mkdir(exist_ok=True)


def format_elapsed_time(seconds):
    """Format seconds into readable time format."""
    return str(timedelta(seconds=int(seconds)))


def format_flops(flops):
    """Format FLOP counts in human readable form."""
    if flops < 1e9:
        return f"{flops/1e6:.2f} MFLOPs"
    elif flops < 1e12:
        return f"{flops/1e9:.2f} GFLOPs"
    else:
        return f"{flops/1e12:.2f} TFLOPs"


def save_ckpt(stats: Dict, final: bool = False, dataset_position: int = None):
    """Save checkpoint with processing stats and position."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = CKPT_DIR / f"checkpoint_{ts}.json"
    
    # Ensure dataset_position is saved
    data = {**stats}
    if dataset_position is not None:
        # Store in both places for backward compatibility
        data["processed_examples"] = dataset_position
        if "checkpoint_info" not in data:
            data["checkpoint_info"] = {}
        data["checkpoint_info"]["dataset_position"] = dataset_position
    
    # Add timestamp
    if "checkpoint_info" not in data:
        data["checkpoint_info"] = {}
    data["checkpoint_info"]["timestamp"] = ts
    data["checkpoint_info"]["is_final"] = final
    
    # Save to file
    with open(fname, "w") as f:
        json.dump(data, f, indent=2)
    
    # Update latest checkpoint pointer
    (CKPT_DIR / "latest_checkpoint.txt").write_text(str(fname))
    return fname


def load_latest_checkpoint():
    """Load the most recent checkpoint if available."""
    latest_path = CKPT_DIR / "latest_checkpoint.txt"
    if not latest_path.exists():
        return None
    
    try:
        ckpt_path = Path(latest_path.read_text().strip())
        if ckpt_path.exists():
            print(f"Loading checkpoint: {ckpt_path}")
            with open(ckpt_path, 'r') as f:
                checkpoint = json.load(f)
                # Debug info about position
                pos1 = checkpoint.get("processed_examples", None)
                pos2 = checkpoint.get("checkpoint_info", {}).get("dataset_position", None)
                print(f"Checkpoint positions: processed_examples={pos1}, dataset_position={pos2}")
                return checkpoint
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
    
    return None


# ---------------------------------------------------------------------------
# ---------------------------  ── MAIN LOOP  ──  ---------------------------
# ---------------------------------------------------------------------------

def run_stream(
    dataset_iter,
    batch_est: int,
    d: int,
    cp_rank: int,
    do_disco: bool,
    do_bert: bool,
    ckpt_every: int,
    total_examples: int = None,
    resume_position: int = 0,
    est_runtime_minutes: int = None,
    use_cache: bool = True,
    fast_mode: bool = False,
    checkpoint_stats: Dict = None
):
    stats = {
        "num_sentences": 0,
        "discocirc_naive": 0.0,
        "discocirc_cp": 0.0,
        "bert_vanilla": 0.0,
        "bert_flash": 0.0,
        "start_time": time.time(),
        "processed_examples": resume_position,
        "last_update_time": time.time()
    }
    
    # Restore stats from checkpoint if available
    if checkpoint_stats:
        for key in ["num_sentences", "discocirc_naive", "discocirc_cp", "bert_vanilla", "bert_flash"]:
            if key in checkpoint_stats:
                stats[key] = checkpoint_stats[key]
        print(f"Restored stats from checkpoint: {stats['num_sentences']} sentences processed")

    # Skip to resume position if needed - IMPORTANT PART FOR RESUME FUNCTIONALITY
    if resume_position > 0:
        # Explicitly check what kind of dataset we're working with
        if hasattr(dataset_iter, '__iter__') and not hasattr(dataset_iter, '__getitem__'):
            # It's a regular iterator/generator, need to consume items
            print(f"Skipping to position {resume_position} in streaming dataset...")
            try:
                iterator = iter(dataset_iter)
                for _ in tqdm(range(resume_position), desc="Skipping", unit="examples"):
                    next(iterator)
                # Reset the iterator to the current position
                dataset_iter = iterator
                print(f"Successfully skipped {resume_position} examples")
            except StopIteration:
                print("Warning: Resume position exceeds dataset length")
        elif isinstance(dataset_iter, list):
            # It's a list, so we can slice it
            print(f"Slicing list dataset to start at position {resume_position}")
            if resume_position < len(dataset_iter):
                dataset_iter = dataset_iter[resume_position:]
            else:
                print("Warning: Resume position exceeds dataset length")
                dataset_iter = []
        else:
            print(f"Warning: Unknown dataset type, can't skip to position {resume_position}")
    
    workers = max(1, math.floor(mp.cpu_count() * 0.75))
    process_fn = partial(process_batch, d=d, cp_rank=cp_rank, do_disco=do_disco, do_bert=do_bert)
    
    print(f"Starting processing with {workers} workers")
    print("=" * 80)
    progress_bar = None
    if total_examples:
        progress_bar = tqdm(total=total_examples, initial=resume_position, 
                          desc="Processing", unit="examples")

    with mp.Pool(workers) as pool:
        current: List[str] = []
        since_ckpt = 0
        last_status_time = time.time()
        examples_processed = resume_position
        
        try:
            for ex in dataset_iter:
                examples_processed += 1
                stats["processed_examples"] = examples_processed
                
                # Progress reporting
                if progress_bar:
                    progress_bar.update(1)
                elif time.time() - last_status_time > 10:  # Status update every 10 seconds
                    elapsed = time.time() - stats["start_time"]
                    sentences_per_sec = stats["num_sentences"] / max(1, elapsed)
                    avg_bert_vanilla = stats["bert_vanilla"] / max(1, stats["num_sentences"])
                    avg_bert_flash = stats["bert_flash"] / max(1, stats["num_sentences"])
                    
                    # Create percentage bar if total is known
                    percent_str = ""
                    if total_examples:
                        percent = min(100, examples_processed / total_examples * 100)
                        bar_width = 30
                        filled_width = int(bar_width * percent / 100)
                        bar = "█" * filled_width + "░" * (bar_width - filled_width)
                        percent_str = f"\n[{bar}] {percent:.1f}% "
                        
                        # Add ETA if we have enough data
                        if sentences_per_sec > 0:
                            remaining_examples = total_examples - examples_processed
                            eta_seconds = remaining_examples / (examples_processed / elapsed)
                            percent_str += f"(ETA: {format_elapsed_time(eta_seconds)})"
                    # If we don't know total examples but have an estimated runtime, show progress based on time
                    elif est_runtime_minutes:
                        est_runtime_seconds = est_runtime_minutes * 60
                        percent = min(100, elapsed / est_runtime_seconds * 100)
                        bar_width = 30
                        filled_width = int(bar_width * percent / 100)
                        bar = "█" * filled_width + "░" * (bar_width - filled_width)
                        percent_str = f"\n[{bar}] {percent:.1f}% (time-based estimate) "
                        
                        # Add ETA based on estimated runtime
                        if percent > 0:
                            eta_seconds = max(0, est_runtime_seconds - elapsed)
                            percent_str += f"(ETA: {format_elapsed_time(eta_seconds)})"
                    
                    print(f"Processed: {examples_processed} examples, {stats['num_sentences']} sentences{percent_str}")
                    print(f"Rate: {sentences_per_sec:.2f} sentences/sec")
                    print(f"Elapsed: {format_elapsed_time(elapsed)}")
                    if do_bert:
                        print(f"Avg BERT complexity: {format_flops(avg_bert_vanilla)} vanilla, {format_flops(avg_bert_flash)} flash")
                    print("=" * 40)
                    last_status_time = time.time()
                
                # Verbose processing status
                if time.time() - stats.get("last_update_time", 0) > 2:
                    # Give feedback during sentence parsing to show progress
                    print(f"[Processing example {examples_processed}/{total_examples if total_examples else '?'}, text size: {len(ex['text'])/1024:.1f} KB]", end="\r", flush=True)
                    stats["last_update_time"] = time.time()
                
                # Sentence parsing with caching
                try:
                    cache_key = get_cache_key(ex["text"], "wikitext", examples_processed)
                    new_sents = sentences_spacy(
                        ex["text"], 
                        use_cache=use_cache, 
                        cache_key=cache_key,
                        dataset_name="wikitext", 
                        chunk_id=examples_processed,
                        fast_mode=fast_mode
                    )
                    
                    if len(new_sents) == 0 and len(ex["text"].strip()) > 0:
                        # Fallback if spaCy failed
                        print(f"[WARN] SpaCy failed to split text into sentences, falling back to crude split")
                        new_sents = crude_split(ex["text"])
                        # Cache the fallback result too
                        if use_cache:
                            save_to_cache(new_sents, cache_key)
                            
                    current.extend(new_sents)
                except Exception as e:
                    print(f"[ERROR] Failed to process example {examples_processed}: {e}")
                    print("Attempting to continue with next example...")
                    continue
                
                # Process sentences in batches
                while len(current) >= batch_est:
                    slice_size = len(current)
                    while slice_size:
                        try:
                            part = current[:slice_size]
                            for res in pool.apply(process_fn, (part,)):
                                stats["discocirc_naive"] += res["disc_naive"]
                                stats["discocirc_cp"] += res["disc_cp"]
                                stats["bert_vanilla"] += res["bert_naive"]
                                stats["bert_flash"] += res["bert_opt"]
                                stats["num_sentences"] += 1
                                since_ckpt += 1
                            current = current[slice_size:]
                            break  # success
                        except RuntimeError as e:
                            if "out of memory" in str(e).lower():
                                torch.cuda.empty_cache()
                                slice_size //= 2
                                print(f"[OOM] retrying with {slice_size} sents")
                                if slice_size < 64:
                                    print("[OOM] Giving up slice – checkpoint & skip")
                                    save_ckpt(stats, dataset_position=examples_processed)
                                    current = current[slice_size:]
                                    break
                            else:
                                raise

                    if since_ckpt >= ckpt_every:
                        save_ckpt(stats, dataset_position=examples_processed)
                        since_ckpt = 0
                        
            # flush remainder
            if current:
                print(f"Processing remaining {len(current)} sentences...")
                for res in pool.apply(process_fn, (current,)):
                    stats["discocirc_naive"] += res["disc_naive"]
                    stats["discocirc_cp"] += res["disc_cp"]
                    stats["bert_vanilla"] += res["bert_naive"]
                    stats["bert_flash"] += res["bert_opt"]
                    stats["num_sentences"] += 1
                    
        except KeyboardInterrupt:
            print("\nProcess interrupted. Saving checkpoint...")
            save_ckpt(stats, dataset_position=examples_processed)
            if progress_bar:
                progress_bar.close()
            print("Checkpoint saved. You can resume later using --resume")
            sys.exit(0)
            
    stats["end_time"] = time.time()
    stats["total_runtime"] = stats["end_time"] - stats["start_time"]
    
    if progress_bar:
        progress_bar.close()
        
    ckpt_file = save_ckpt(stats, final=True, dataset_position=examples_processed)
    print(f"Final checkpoint saved to {ckpt_file}")
    return stats

# ---------------------------------------------------------------------------
# ---------------------------  ── PREPROCESSING  ──  -----------------------
# ---------------------------------------------------------------------------

def preprocess_dataset_chunk(chunk):
    """Process a chunk of dataset examples using SpaCy."""
    global NLPROC
    if NLPROC is None:
        NLPROC = spacy.load("en_core_web_sm", disable=["ner"])
    
    chunk_id, examples = chunk
    all_sentences = []
    
    for idx, example in enumerate(examples):
        text = example["text"]
        cache_key = get_cache_key(text, "wikitext", chunk_id * 1000 + idx)
        
        # Try to get from cache first
        cached = load_from_cache(cache_key)
        if cached:
            all_sentences.extend(cached)
            continue
            
        # Process with SpaCy and cache
        try:
            sentences = sentences_spacy(text, use_cache=True, cache_key=cache_key, fast_mode=False)
            all_sentences.extend(sentences)
        except Exception as e:
            print(f"[ERROR] Failed to process example in chunk {chunk_id}: {e}")
            # Fallback to crude split
            sentences = crude_split(text)
            all_sentences.extend(sentences)
            # Still cache the result
            save_to_cache(sentences, cache_key)
    
    return all_sentences

def preprocess_dataset(dataset, num_workers=None, chunk_size=10, fast_mode=False):
    """Preprocess an entire dataset in parallel."""
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)
    
    # Convert streaming dataset to list and chunk it
    print(f"[PREPROCESS] Collecting dataset examples...")
    all_examples = []
    for i, example in enumerate(tqdm(dataset, desc="Collecting examples")):
        all_examples.append(example)
        # Limit collection for debugging/testing
        if i >= 100000:  # Adjust as needed
            break
    
    total_examples = len(all_examples)
    print(f"[PREPROCESS] Processing {total_examples} examples with {num_workers} workers")
    
    # Split into chunks
    chunks = []
    for i in range(0, total_examples, chunk_size):
        chunk_id = i // chunk_size
        chunk_examples = all_examples[i:i+chunk_size]
        chunks.append((chunk_id, chunk_examples))
    
    # Process chunks in parallel
    all_sentences = []
    with mp.Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(preprocess_dataset_chunk, chunks),
            total=len(chunks),
            desc="Processing chunks"
        ))
        for sentences in results:
            all_sentences.extend(sentences)
    
    print(f"[PREPROCESS] Extracted {len(all_sentences)} sentences")
    return all_sentences

# ---------------------------------------------------------------------------
# ---------------------------  ── CLI  ──  ----------------------------------
# ---------------------------------------------------------------------------

def cli():
    p = argparse.ArgumentParser()
    p.add_argument("file", nargs="?", help="plain‑text file (defaults to Edgar Poe Raven)")
    p.add_argument("--wikitext", action="store_true", help="use WikiText-103 dataset")
    p.add_argument("--bert", action="store_true", help="only compute BERT complexity")
    p.add_argument("--disco", action="store_true", help="only compute DisCoCirc complexity")
    p.add_argument("--cpu", action="store_true", help="force CPU mode")
    p.add_argument("--gpu", type=int, help="specify GPU device number")
    p.add_argument("--batch", type=int, help="specify batch size (auto by default)")
    p.add_argument("--ckpt", type=int, default=500, help="checkpoint frequency")
    p.add_argument("--synthetic", action="store_true", help="keep dummy GPU ops to show utilisation")
    p.add_argument("--resume", action="store_true", help="resume from latest checkpoint")
    p.add_argument("--total", type=int, help="total examples in dataset (for progress bar)")
    p.add_argument("--est-runtime", type=int, help="estimated runtime in minutes (for progress bar when total is unknown)")
    p.add_argument("--dataset", choices=["wikitext-2", "wikitext-103-v1"], default="wikitext-103-v1", 
                  help="HuggingFace dataset version to use")
    p.add_argument("--fast", action="store_true", help="use faster text processing (less accurate but much quicker)")
    p.add_argument("--sample", type=float, help="process only a percentage of dataset (0.0-1.0)")
    p.add_argument("--preprocess", action="store_true", help="preprocess dataset in parallel before running analysis")
    p.add_argument("--use-cache", action="store_true", help="use cached preprocessing results")
    p.add_argument("--clear-cache", action="store_true", help="clear preprocessing cache before running")
    p.add_argument("--workers", type=int, help="number of worker processes for preprocessing")
    return p.parse_args()


if __name__ == "__main__":
    args = cli()
    do_bert = not args.disco
    do_disco = not args.bert

    # Clear cache if requested
    if args.clear_cache:
        import shutil
        print(f"Clearing preprocessing cache...")
        if CACHE_DIR.exists():
            shutil.rmtree(CACHE_DIR)
        CACHE_DIR.mkdir(exist_ok=True)
    
    # Print cache stats if it exists and we're using it
    if args.use_cache and CACHE_DIR.exists():
        stats = get_cache_stats()
        if stats["num_files"] > 0:
            print(f"Cache stats: {stats['num_files']} files, {stats['total_size_mb']:.2f} MB")
            last_modified = datetime.fromtimestamp(stats["last_modified"])
            print(f"Last modified: {last_modified}")

    init_device(force_cpu=args.cpu, specific_gpu=args.gpu)
    NLPROC = spacy.load("en_core_web_sm", disable=["ner"] if args.fast else []) if do_disco else None

    batch_est = args.batch or guess_batch_size()
    print("Initial batch estimate:", batch_est)
    
    # Check for checkpoint to resume from
    resume_position = 0
    checkpoint_stats = {}
    if args.resume:
        checkpoint = load_latest_checkpoint()
        if checkpoint:
            # Restore previous statistics
            checkpoint_stats = {
                "num_sentences": checkpoint.get("num_sentences", 0),
                "discocirc_naive": checkpoint.get("discocirc_naive", 0.0),
                "discocirc_cp": checkpoint.get("discocirc_cp", 0.0),
                "bert_vanilla": checkpoint.get("bert_vanilla", 0.0),
                "bert_flash": checkpoint.get("bert_flash", 0.0),
            }
            
            # IMPORTANT: Determine resume position correctly
            # First check processed_examples (directly in the root)
            if "processed_examples" in checkpoint:
                resume_position = checkpoint["processed_examples"]
            # Then check checkpoint_info.dataset_position
            elif "checkpoint_info" in checkpoint and "dataset_position" in checkpoint["checkpoint_info"]:
                resume_position = checkpoint["checkpoint_info"]["dataset_position"]
            
            print(f"Restored stats: {checkpoint_stats['num_sentences']} sentences processed")
            print(f"Will resume from position {resume_position}")
        else:
            print("No checkpoint found, starting from beginning")

    if args.wikitext:
        try:
            from datasets import load_dataset
        except ImportError:  # noqa: D401
            print("datasets not installed – install or use plain file mode"); sys.exit(1)
        
        print(f"Loading dataset: {args.dataset}")
        
        # Set default total examples based on dataset version
        if args.dataset == "wikitext-103-v1":
            total_examples = args.total or WIKITEXT_STATS["articles"]
        else:
            # Default fallback if we don't have exact stats
            total_examples = args.total or 36718
        
        # Handle streaming dataset and resuming position
        if args.resume and resume_position > 0:
            print(f"Will use non-streaming dataset for better resume handling")
            ds = load_dataset("wikitext", args.dataset, split="train", streaming=False)
            
            # Apply sampling if requested
            if args.sample and 0.0 < args.sample <= 1.0:
                sample_size = int(len(ds) * args.sample)
                print(f"Sampling {args.sample:.1%} of dataset ({sample_size:,} examples)")
                
                # Use deterministic sampling
                import random
                random.seed(42)  # For reproducibility
                indices = random.sample(range(len(ds)), sample_size)
                ds = ds.select(indices)
                
                # Update total if needed
                if total_examples > sample_size:
                    total_examples = sample_size
            
            # Convert to list for precise resuming
            print("Converting dataset to list for precise resuming...")
            ds = list(ds)
            
            # Resume from specific position
            if resume_position < len(ds):
                print(f"Will resume from example {resume_position} of {len(ds)}")
                total_examples = len(ds)
            else:
                print(f"Warning: Resume position {resume_position} exceeds dataset length {len(ds)}")
                resume_position = 0
        else:
            # Regular streaming approach
            ds = load_dataset("wikitext", args.dataset, split="train", streaming=True)
            
            # Apply sampling if requested
            if args.sample and 0.0 < args.sample <= 1.0:
                sample_size = int(total_examples * args.sample)
                print(f"Sampling {args.sample:.1%} of dataset ({sample_size:,} examples)")
                total_examples = sample_size
                
                # Apply sampling to streaming dataset
                import random
                random.seed(42)  # For reproducibility
                ds = ds.filter(lambda _, idx: random.random() < args.sample, with_indices=True)
        
        # Print dataset statistics
        if args.dataset == "wikitext-103-v1":
            print(f"WikiText-103 statistics:")
            print(f"  - Articles: {WIKITEXT_STATS['articles']:,}")
            print(f"  - Sentences: {WIKITEXT_STATS['sentences']:,}")
            print(f"  - Tokens: {WIKITEXT_STATS['tokens']:,}")
            print(f"Will process approximately {total_examples:,} examples")
        
        # Special case for fast mode
        if args.fast:
            print("[FAST MODE] Using simplified processing for speed")
        
        # Preprocessing mode - process whole dataset upfront
        if args.preprocess:
            print(f"[PREPROCESS] Preprocessing entire dataset...")
            num_workers = args.workers or max(1, mp.cpu_count() - 1)
            
            # Preprocess and get all sentences
            all_sentences = preprocess_dataset(
                ds, 
                num_workers=num_workers, 
                chunk_size=10,
                fast_mode=args.fast
            )
            
            print(f"[PREPROCESS] Running analysis on {len(all_sentences)} preprocessed sentences")
            
            # Create a simple dataset from the preprocessed sentences
            preprocessed_dataset = [{"text": " ".join(all_sentences)}]
            
            # Run on the preprocessed sentences
            stats = run_stream(
                preprocessed_dataset, 
                batch_est, 768, 50, 
                do_disco, do_bert, args.ckpt,
                use_cache=args.use_cache,
                fast_mode=args.fast,
                checkpoint_stats=checkpoint_stats if args.resume else None
            )
        else:
            # Regular streaming mode
            stats = run_stream(
                ds, batch_est, 768, 50, 
                do_disco, do_bert, args.ckpt, 
                total_examples=total_examples, 
                resume_position=resume_position,
                est_runtime_minutes=args.est_runtime,
                use_cache=args.use_cache,
                fast_mode=args.fast,
                checkpoint_stats=checkpoint_stats if args.resume else None
            )
    else:
        text = (
            Path(args.file).read_text(encoding="utf-8") if args.file else
            "Once upon a midnight dreary, while I pondered weak and weary."
        )
        sents = sentences_spacy(text, use_cache=args.use_cache, fast_mode=args.fast) if do_disco else crude_split(text)
        stats = run_stream(
            [{"text": " ".join(sents)}], batch_est, 768, 50, do_disco, do_bert, args.ckpt,
            est_runtime_minutes=args.est_runtime,
            use_cache=args.use_cache,
            fast_mode=args.fast,
            checkpoint_stats=checkpoint_stats if args.resume else None
        )

    print("\n=== Summary ===")
    print(f"Total runtime: {format_elapsed_time(stats.get('total_runtime', 0))}")
    print(f"Sentences processed: {stats['num_sentences']:,}")
    if stats['num_sentences'] > 0:
        print(f"Average DisCoCirc complexity (naive): {format_flops(stats['discocirc_naive']/stats['num_sentences'])}")
        print(f"Average DisCoCirc complexity (CP): {format_flops(stats['discocirc_cp']/stats['num_sentences'])}")
        print(f"Average BERT complexity (vanilla): {format_flops(stats['bert_vanilla']/stats['num_sentences'])}")
        print(f"Average BERT complexity (flash): {format_flops(stats['bert_flash']/stats['num_sentences'])}")
    print(json.dumps({**stats, "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}, indent=2))
