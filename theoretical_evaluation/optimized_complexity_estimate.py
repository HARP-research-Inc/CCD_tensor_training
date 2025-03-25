import spacy
from transformers import BertTokenizer
from tqdm import tqdm  # for progress bar
import sys
import os
import concurrent.futures

# Optional: use pardata for WikiText-103 if flag is provided.
USE_WIKITEXT = "--wikitext" in sys.argv

if USE_WIKITEXT:
    import pardata

###############################################################################
# 1) Load spaCy model (download "en_core_web_sm" if not installed)
###############################################################################
nlp = spacy.load("en_core_web_sm")

###############################################################################
# 2) DisCoCirc Complexity Estimation
###############################################################################
def get_token_rank(token):
    """
    Return the multilinear 'rank' (order) for this token/operator in a 
    simplified DisCoCirc-style grammar.
      - NOUN/PROPN/PRON => rank 1
      - ADJ/ADV/DET => rank 2
      - ADP, CCONJ, SCONJ => rank 3
      - AUX => rank 2
      - VERB => rank 2 (intransitive), 3 (transitive), or 4 (ditransitive)
      - INTJ and others => rank 1 as fallback
    """
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
    """
    For a given sentence, compute:
      - naive cost = sum(d^rank) over tokens
      - optimized cost = naive cost / disco_factor
    Returns: (total_naive, total_optimized, breakdown)
    """
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
# 3) BERT Complexity Estimation
###############################################################################
def estimate_bert_complexity(sentence, model_name="bert-base-uncased", bert_optim_factor=1.0):
    """
    Estimate the FLOPs for processing 'sentence' with BERT-base using a rough formula:
      - Self-attention: 2 * (L^2 * H) per layer.
      - Feed-forward: 2 * L * H * (4H) per layer.
    Then multiply by number of layers and apply an optimization factor.
    Returns: (naive_flops, seq_len, optimized_flops)
    """
    tokenizer = BertTokenizer.from_pretrained(model_name)
    tokens = tokenizer.encode(sentence, add_special_tokens=True)
    seq_len = len(tokens)
    # BERT-base hyperparameters:
    num_layers = 12
    hidden_size = 768
    intermediate_size = 3072  # usually 4*hidden_size
    self_attention_flops = 2.0 * (seq_len ** 2) * hidden_size
    feed_forward_flops = 2.0 * seq_len * hidden_size * intermediate_size
    flops_per_layer = self_attention_flops + feed_forward_flops
    naive_flops = flops_per_layer * num_layers
    optimized_flops = naive_flops / bert_optim_factor
    return naive_flops, seq_len, optimized_flops

###############################################################################
# 4) Corpus Analysis and Parallel Sentence Processing
###############################################################################
def chunk_text_into_sentences(text):
    """
    Uses spaCy's sentence segmentation.
    Returns a list of sentence strings.
    """
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

def process_sentence(sentence, d, disco_factor, bert_optim_factor):
    """
    Process a single sentence and return:
      (disc_naive, disc_opt, bert_naive, bert_opt, seq_len)
    """
    disc_naive, disc_opt, _ = estimate_discocirc_complexity(sentence, d=d, disco_factor=disco_factor, verbose=False)
    bert_naive, seq_len, bert_opt = estimate_bert_complexity(sentence, bert_optim_factor=bert_optim_factor)
    return disc_naive, disc_opt, bert_naive, bert_opt, seq_len

def estimate_corpus_complexity_parallel(text, d=300, disco_factor=1.0, bert_optim_factor=1.0, use_progress_bar=True, max_workers=4):
    """
    Splits text into sentences and processes them in parallel.
    Aggregates naive and optimized FLOP counts for both DisCoCirc and BERT.
    """
    sentences = chunk_text_into_sentences(text)
    corpus_results = {
        "num_sentences": len(sentences),
        "discocirc_total_naive": 0.0,
        "discocirc_total_optimized": 0.0,
        "bert_total_naive": 0.0,
        "bert_total_optimized": 0.0,
        "sentence_details": []  # Optional per-sentence breakdown
    }
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_sentence, sent, d, disco_factor, bert_optim_factor) for sent in sentences]
        if use_progress_bar:
            futures = list(tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing sentences"))
        else:
            futures = list(concurrent.futures.as_completed(futures))
        for future in futures:
            disc_naive, disc_opt, bert_naive, bert_opt, seq_len = future.result()
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
    """
    Reads the entire content of a text file.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

###############################################################################
# 6) Main: Choose Corpus and Run Analysis
###############################################################################
if __name__ == "__main__":
    # Optimization factors.
    disco_factor = 20.0     # Simulated low-rank approximations in DisCoCirc.
    bert_optim_factor = 5.0 # Simulated optimizations in BERT.
    d = 300  # Embedding dimension for DisCoCirc
    
    # Decide which corpus to use.
    if USE_WIKITEXT:
        print("Loading WikiText-103 from pardata...")
        dataset = pardata.load_dataset('wikitext103')
        # For demonstration, we'll use a subset (e.g., first 100 examples) from the train split.
        # Adjust as needed.
        texts = [item["text"] for item in dataset["train"][:100]]
        text = "\n".join(texts)
    else:
        # Default: Use "the_raven.txt" in the same directory.
        file_path = os.path.join(os.path.dirname(__file__), "the_raven.txt")
        if os.path.exists(file_path):
            print(f"Loading text file: {file_path}")
            text = load_text_file(file_path)
        else:
            # Fallback sample text.
            text = "The big dog quickly chased a ball in the yard."
    
    # Use parallel processing for corpus analysis.
    corpus_results = estimate_corpus_complexity_parallel(
        text, d=d, disco_factor=disco_factor, bert_optim_factor=bert_optim_factor,
        use_progress_bar=True, max_workers=4
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
    
    # Calculate the percentage improvement of DisCoCirc over BERT.
    naive_improvement = 100 * (corpus_results['bert_total_naive'] - corpus_results['discocirc_total_naive']) / corpus_results['bert_total_naive']
    optimized_improvement = 100 * (corpus_results['bert_total_optimized'] - corpus_results['discocirc_total_optimized']) / corpus_results['bert_total_optimized']
    
    print("\n-- Improvement of DisCoCirc over BERT (Aggregated) --")
    print(f"Naive:     DisCoCirc uses {naive_improvement:.2f}% fewer FLOPs than BERT-base")
    print(f"Optimized: DisCoCirc uses {optimized_improvement:.2f}% fewer FLOPs than BERT-base")
