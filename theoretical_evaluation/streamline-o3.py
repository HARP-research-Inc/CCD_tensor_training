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
"""

from __future__ import annotations

import argparse, json, math, os, sys, time, gc, subprocess, re
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

def sentences_spacy(text: str) -> List[str]:
    doc = NLPROC(text)
    return [s.text.strip() for s in doc.sents if s.text.strip()]

# fallback splitter for BERT‑only mode

def crude_split(text: str) -> List[str]:
    sents = []
    for line in text.split("\n"):
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
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = CKPT_DIR / f"checkpoint_{ts}.json"
    data = {
        **stats,
        "checkpoint_info": {
            "timestamp": ts,
            "is_final": final,
            "dataset_position": dataset_position
        },
    }
    json.dump(data, open(fname, "w"), indent=2)
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
                return json.load(f)
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
    est_runtime_minutes: int = None
):
    stats = {
        "num_sentences": 0,
        "discocirc_naive": 0.0,
        "discocirc_cp": 0.0,
        "bert_vanilla": 0.0,
        "bert_flash": 0.0,
        "start_time": time.time(),
        "processed_examples": resume_position
    }

    # Skip to resume position if needed
    if resume_position > 0:
        print(f"Resuming from position {resume_position}")
        for _ in range(resume_position):
            try:
                next(dataset_iter)
            except StopIteration:
                print("Warning: Resume position exceeds dataset length")
                break

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
                
                current.extend(sentences_spacy(ex["text"]))
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
    return p.parse_args()


if __name__ == "__main__":
    args = cli()
    do_bert = not args.disco
    do_disco = not args.bert

    init_device(force_cpu=args.cpu, specific_gpu=args.gpu)
    NLPROC = spacy.load("en_core_web_sm") if do_disco else None

    batch_est = args.batch or guess_batch_size()
    print("Initial batch estimate:", batch_est)
    
    # Check for checkpoint to resume from
    resume_position = 0
    if args.resume:
        checkpoint = load_latest_checkpoint()
        if checkpoint:
            # Restore previous statistics
            stats = {
                "num_sentences": checkpoint.get("num_sentences", 0),
                "discocirc_naive": checkpoint.get("discocirc_naive", 0.0),
                "discocirc_cp": checkpoint.get("discocirc_cp", 0.0),
                "bert_vanilla": checkpoint.get("bert_vanilla", 0.0),
                "bert_flash": checkpoint.get("bert_flash", 0.0),
            }
            
            # Get resume position
            resume_position = checkpoint.get("checkpoint_info", {}).get("dataset_position", 0)
            print(f"Restored stats: {stats['num_sentences']} sentences processed")
            print(f"Will resume from position {resume_position}")
        else:
            print("No checkpoint found, starting from beginning")

    if args.wikitext:
        try:
            from datasets import load_dataset
        except ImportError:  # noqa: D401
            print("datasets not installed – install or use plain file mode"); sys.exit(1)
        
        print(f"Loading dataset: {args.dataset}")
        ds = load_dataset("wikitext", args.dataset, split="train", streaming=True)
        
        # Use the correct total based on the dataset version
        if args.dataset == "wikitext-103-v1":
            total_examples = args.total or WIKITEXT_STATS["articles"]
            print(f"WikiText-103 statistics:")
            print(f"  - Articles: {WIKITEXT_STATS['articles']:,}")
            print(f"  - Sentences: {WIKITEXT_STATS['sentences']:,}")
            print(f"  - Tokens: {WIKITEXT_STATS['tokens']:,}")
        else:
            # Default fallback if we don't have exact stats
            total_examples = args.total or 36718
            
        stats = run_stream(ds, batch_est, 768, 50, do_disco, do_bert, args.ckpt, 
                          total_examples=total_examples, resume_position=resume_position,
                          est_runtime_minutes=args.est_runtime)
    else:
        text = (
            Path(args.file).read_text(encoding="utf-8") if args.file else
            "Once upon a midnight dreary, while I pondered weak and weary."
        )
        sents = sentences_spacy(text) if do_disco else crude_split(text)
        stats = run_stream(
            [{"text": " ".join(sents)}], batch_est, 768, 50, do_disco, do_bert, args.ckpt,
            est_runtime_minutes=args.est_runtime
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
