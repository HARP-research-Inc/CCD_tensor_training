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
MAC_FLOP = 2  # multiply + add

def flops_transformer_layer(L: int, H: int) -> int:
    """Forward FLOPs for one encoder layer (BERT style)."""
    proj_qkv = 3 * MAC_FLOP * L * H * H
    proj_out  =     MAC_FLOP * L * H * H
    attn_matmul = 2 * MAC_FLOP * L * L * H          # QKᵀ + αV
    scale_softmax = L * L + 6 * L * L               # scale + softmax
    feedforward   = 2 * MAC_FLOP * L * H * 4*H      # Linear1 + Linear2
    gelu          = 8 * L * 4 * H
    layernorm     = 2 * 8 * L * H
    residual_add  = 2 * L * H
    return (proj_qkv + proj_out + attn_matmul +
            scale_softmax + feedforward + gelu +
            layernorm + residual_add)

def bert_forward_flops(sentence: str,
                       model_name="bert-base-uncased",
                       layers=12, hidden=768):
    tok = BertTokenizer.from_pretrained(model_name)
    L = len(tok.encode(sentence, add_special_tokens=True))
    per_layer = flops_transformer_layer(L, hidden)
    return per_layer * layers, L

###############################################################################
# 4) Example usage: Compare DisCoCirc vs BERT-base on a single sentence
###############################################################################
if __name__ == "__main__":
    sentence = """The big dog quickly chased a ball in the yard."""
    sentence = """Jack loves Diane."""
    
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
    bert_total, L = bert_forward_flops(sentence, model_name="bert-base-uncased")
    print("[BERT-base Complexity Approximation]")
    print(f"Sentence: {sentence}")
    print(f"Tokenized length (with special tokens) = {L}")
    print(f"Estimated total FLOPs (forward pass): {bert_total:,.0f}")
