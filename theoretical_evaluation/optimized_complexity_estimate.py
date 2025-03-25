import spacy
from transformers import BertTokenizer
from tqdm import tqdm  # progress bar for corpus processing
import sys
import os

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
    Estimate DisCoCirc complexity for a sentence.
      - Each token's naive cost is computed as d^r (r = rank).
      - Optimized cost = (naive cost) / disco_factor.
    
    If verbose is True, also returns a per-token breakdown.
    
    Returns:
      total_naive (float): Sum of naive costs.
      total_optimized (float): Sum of optimized costs.
      breakdown (list of tuples): Each tuple is 
         (token_text, pos, rank, naive_cost, optimized_cost)
         if verbose; otherwise, an empty list.
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
    Estimate the FLOPs for processing 'sentence' with a standard BERT-base.
    
    Uses a rough formula:
      - Self-attention: 2 * (L^2 * H) per layer.
      - Feed-forward: 2 * L * H * (4H) per layer.
      - Total per layer is the sum, multiplied by number of layers.
    Then applies an optimization factor.
    
    Returns:
      naive_flops (float): Unoptimized FLOPs.
      seq_len (int): Tokenized sequence length.
      optimized_flops (float): FLOPs after applying bert_optim_factor.
    """
    tokenizer = BertTokenizer.from_pretrained(model_name)
    tokens = tokenizer.encode(sentence, add_special_tokens=True)
    seq_len = len(tokens)
    
    # BERT-base hyperparameters:
    num_layers = 12
    hidden_size = 768
    intermediate_size = 3072  # typically 4 * hidden_size
    
    self_attention_flops = 2.0 * (seq_len ** 2) * hidden_size
    feed_forward_flops = 2.0 * seq_len * hidden_size * intermediate_size
    flops_per_layer = self_attention_flops + feed_forward_flops
    naive_flops = flops_per_layer * num_layers
    optimized_flops = naive_flops / bert_optim_factor
    
    return naive_flops, seq_len, optimized_flops

###############################################################################
# 4) Corpus Analysis Functions
###############################################################################
def chunk_text_into_sentences(text):
    """
    Uses spaCy's sentence segmentation to chunk text into sentences.
    Returns a list of sentence strings.
    """
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

def estimate_corpus_complexity(text, d=300, disco_factor=1.0, bert_optim_factor=1.0, verbose=False, use_progress_bar=True):
    """
    Process a large text by splitting it into sentences.
    For each sentence, compute both DisCoCirc and BERT complexities (naive and optimized).
    If use_progress_bar is True, shows a progress bar.
    Returns aggregated results as a dictionary.
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
    
    iter_sentences = tqdm(sentences, desc="Processing sentences") if use_progress_bar else sentences
    
    for sent in iter_sentences:
        d_naive, d_opt, _ = estimate_discocirc_complexity(sent, d=d, disco_factor=disco_factor, verbose=verbose)
        b_naive, seq_len, b_opt = estimate_bert_complexity(sent, bert_optim_factor=bert_optim_factor)
        corpus_results["discocirc_total_naive"] += d_naive
        corpus_results["discocirc_total_optimized"] += d_opt
        corpus_results["bert_total_naive"] += b_naive
        corpus_results["bert_total_optimized"] += b_opt
        
        corpus_results["sentence_details"].append({
            "sentence": sent,
            "discocirc_naive": d_naive,
            "discocirc_optimized": d_opt,
            "bert_naive": b_naive,
            "bert_optimized": b_opt,
            "token_count": seq_len
        })
    
    return corpus_results

###############################################################################
# 5) Helper: Load Text File
###############################################################################
def load_text_file(file_path):
    """
    Reads the entire content of a text file and returns it as a string.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

###############################################################################
# 6) Example Usage
###############################################################################
if __name__ == "__main__":
    # Optimization factors.
    disco_factor = 20.0     # Factor simulating low-rank approximations in DisCoCirc.
    bert_optim_factor = 5.0 # Factor simulating kernel-level or structural optimizations in BERT.
    
    # Determine the file path to process.
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        # Default to "the_raven.txt" located in the same directory as this script.
        file_path = os.path.join(os.path.dirname(__file__), "the_raven.txt")
    
    if os.path.exists(file_path):
        print(f"Loading text file: {file_path}")
        text = load_text_file(file_path)
        # Process the file as a large corpus.
        corpus_results = estimate_corpus_complexity(
            text, d=300, disco_factor=disco_factor, bert_optim_factor=bert_optim_factor, verbose=False, use_progress_bar=True
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
    else:
        # If the file doesn't exist, use a fallback sample sentence.
        sentence = "The big dog quickly chased a ball in the yard."
        disc_naive, disc_opt, disc_breakdown = estimate_discocirc_complexity(
            sentence, d=300, disco_factor=disco_factor, verbose=False
        )
        bert_naive, seq_len, bert_opt = estimate_bert_complexity(
            sentence, bert_optim_factor=bert_optim_factor
        )
        
        print("[Single Sentence Complexity (Optimized and Unoptimized)]")
        print(f"Sentence: {sentence}")
        print("\n-- DisCoCirc --")
        print(f"Naive Total FLOPs:     {disc_naive:,.2f}")
        print(f"Optimized Total FLOPs: {disc_opt:,.2f}")
        
        print("\n-- BERT-base --")
        print(f"Tokenized length (with special tokens): {seq_len}")
        print(f"Naive Total FLOPs (forward pass): {bert_naive:,.0f}")
        print(f"Optimized Total FLOPs (forward pass): {bert_opt:,.0f}")
