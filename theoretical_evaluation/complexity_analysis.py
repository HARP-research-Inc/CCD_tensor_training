import spacy
from transformers import BertTokenizer

###############################################################################
# 1) Load a spaCy model (download "en_core_web_sm" if not installed)
###############################################################################
nlp = spacy.load("en_core_web_sm")

###############################################################################
# 2) DisCoCirc Complexity Estimation
###############################################################################
def get_token_rank(token):
    """
    Return the multilinear 'rank' (order) we assign to this token/operator
    in a simplified DisCoCirc-style grammar:
      - NOUN/PRON/PROPN => rank 1
      - ADJ/ADV => rank 2
      - DET => rank 2
      - ADP (preposition) => rank 3
      - CONJ, SCONJ => rank 3
      - AUX => rank 2
      - VERB => rank 2 (intransitive), 3 (transitive), or 4 (ditransitive)
      - INTJ => rank 1
      - Else => rank 1 as fallback
    """
    if token.pos_ in {"NOUN", "PROPN", "PRON"}:
        return 1
    if token.pos_ == "ADJ":
        return 2
    if token.pos_ == "ADV":
        return 2
    if token.pos_ == "DET":
        return 2
    if token.pos_ in {"ADP"}:
        return 3
    if token.pos_ in {"CCONJ", "SCONJ"}:
        return 3
    if token.pos_ == "INTJ":
        return 1
    if token.pos_ == "AUX":
        return 2
    
    if token.pos_ == "VERB":
        # Check for direct and indirect objects
        has_dobj = False
        has_iobj = False
        for child in token.children:
            if child.dep_ in {"dobj", "obj"}:
                has_dobj = True
            if child.dep_ == "iobj":
                has_iobj = True
        
        if has_dobj and has_iobj:
            return 4  # ditransitive
        elif has_dobj:
            return 3  # transitive
        else:
            return 2  # intransitive
    
    # Fallback for anything else
    return 1

def estimate_discocirc_complexity(sentence, d=300):
    """
    Estimate a naive DisCoCirc complexity for each token:
      - rank-r operator => ~ d^r FLOPs
    Returns total_flops, plus a breakdown list of (token, pos, rank, cost).
    """
    doc = nlp(sentence)
    total_flops = 0
    breakdown = []
    
    for token in doc:
        rank = get_token_rank(token)
        cost = d ** rank  # naive multilinear cost
        breakdown.append((token.text, token.pos_, rank, cost))
        total_flops += cost
    
    return total_flops, breakdown

###############################################################################
# 3) BERT Complexity Estimation
###############################################################################
def estimate_bert_complexity(sentence, model_name="bert-base-uncased"):
    """
    Approximate the FLOPs for processing 'sentence' with a standard BERT-base:
      - BERT-base typical configuration:
          - L layers = 12
          - Hidden size H = 768
          - Intermediate size (feed-forward) ~ 3072
          - #Attention heads = 12
      - We'll estimate:
          FLOPs_per_layer ~ 2 * (L^2 * H)  [Self-attention: QK + V, merges, etc.]
                           + 2 * (L * H * 4H) [FFN with 2 projections if int. dim=4H]
        Then multiply by #layers.
      - We'll get actual token length (L) by tokenizing with the specified BERT tokenizer.
      - This is a known rough formula from "On the Parameterization and Complexity of Transformers."
    
    In practice, specialized kernels & parallelism can shift real values, but this is a standard reference.
    """
    # 3A) Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_name)
    
    # 3B) Tokenize
    tokens = tokenizer.encode(sentence, add_special_tokens=True)
    seq_len = len(tokens)  # L
    
    # 3C) "Standard BERT-base" approximate hyperparams
    num_layers = 12
    hidden_size = 768
    intermediate_size = 3072  # typically 4 * hidden_size
    
    # 3D) Approximate cost per layer
    #  - Self-attention: O(L^2 * H). 
    #    There's often a factor of ~2 or 3 for Q,K,V, but we’ll approximate 2 * L^2 * H.
    #  - Feed-forward: O(L * H * intermediate_size) * factor 2 (forward + projection back)
    #    ~ 2 * L * H * 4H = 8 * L * H^2, if intermediate_size=4H
    # So we combine them:
    self_attention_flops = 2 * (seq_len**2) * hidden_size
    feed_forward_flops = 2 * seq_len * hidden_size * intermediate_size
    flops_per_layer = self_attention_flops + feed_forward_flops
    
    # Multiply by number of transformer layers
    total_flops = flops_per_layer * num_layers
    
    return total_flops, seq_len

###############################################################################
# 4) Example usage: Compare DisCoCirc vs BERT-base on a single sentence
###############################################################################
if __name__ == "__main__":
    sentence = """The big dog quickly chased a ball in the yard."""
    
    # DisCoCirc estimate:
    discocirc_total, discocirc_breakdown = estimate_discocirc_complexity(sentence, d=300)
    
    print("[DisCoCirc Complexity]")
    print(f"Sentence: {sentence}")
    print(f"Estimated total FLOPs (naive): {discocirc_total:,}")
    print("Breakdown by token:")
    for tok, pos, rank, cost in discocirc_breakdown:
        print(f"  Token='{tok}', POS={pos}, Rank={rank}, Cost={cost:,}")
    
    print("\n" + "="*60 + "\n")
    
    # BERT-base estimate:
    bert_total, L = estimate_bert_complexity(sentence, model_name="bert-base-uncased")
    print("[BERT-base Complexity Approximation]")
    print(f"Sentence: {sentence}")
    print(f"Tokenized length (with special tokens) = {L}")
    print(f"Estimated total FLOPs (forward pass): {bert_total:,.0f}")
