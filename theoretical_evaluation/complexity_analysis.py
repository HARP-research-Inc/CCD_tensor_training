import spacy
from transformers import BertTokenizer

###############################################################################
# 1) Load a spaCy model (download "en_core_web_sm" if not installed)
###############################################################################
nlp = spacy.load("en_core_web_sm")

###############################################################################
# 2) DisCoCirc Complexity Estimation (full-rank vs CP-rank-R)
###############################################################################
def get_token_rank(token):
    """
    Return the multilinear 'rank' we assign to this token/operator.
    (same mapping you already had)
    """
    if token.pos_ in {"NOUN", "PROPN", "PRON"}:
        return 1
    if token.pos_ in {"ADJ", "ADV", "DET", "AUX"}:
        return 2
    if token.pos_ in {"ADP", "CCONJ", "SCONJ"}:
        return 3
    if token.pos_ == "VERB":
        # ditransitive / transitive / intransitive
        has_dobj = any(child.dep_ in {"dobj", "obj"} for child in token.children)
        has_iobj = any(child.dep_ == "iobj"            for child in token.children)
        return 4 if (has_dobj and has_iobj) else 3 if has_dobj else 2
    if token.pos_ == "INTJ":
        return 1
    return 1               # fallback

def estimate_discocirc_complexity(sentence, d=300):
    """Full-rank cost: operator rank-r  →  d**r FLOPs."""
    doc = nlp(sentence)
    total_flops, breakdown = 0, []
    for tok in doc:
        r = get_token_rank(tok)
        cost = d ** r
        breakdown.append((tok.text, tok.pos_, r, cost))
        total_flops += cost
    return total_flops, breakdown

# --------------------------------------------------------------------------- #
# >>> NEW: CP-DisCoCirc complexity                                            #
# --------------------------------------------------------------------------- #
def estimate_cp_discocirc_complexity(sentence, d=300, R=50):
    """
    CP-decomposed cost:   FLOPs ≈ R * (r+2) * d     (see derivation in notes)

    • r   = multilinear order of the operator
    • d   = embedding dimension
    • R   = chosen CP rank   (default 50 is typical for d≈300)
    """
    doc = nlp(sentence)
    total_flops, breakdown = 0, []
    for tok in doc:
        r = get_token_rank(tok)
        cost = R * (r + 2) * d        # linear in d instead of d**r
        breakdown.append((tok.text, tok.pos_, r, cost))
        total_flops += cost
    return total_flops, breakdown

###############################################################################
# 3) BERT Complexity Estimation (vanilla vs. optimised)
###############################################################################
MAC_FLOP = 2          # multiply+add  counted as 2 FLOPs

def flops_transformer_layer_vanilla(L: int, H: int) -> int:
    """Classical BERT encoder-layer FLOPs (no fusions)."""
    proj_qkv      = 3 * MAC_FLOP * L * H * H
    proj_out      =     MAC_FLOP * L * H * H
    attn_matmul   = 2 * MAC_FLOP * L * L * H          # QKᵀ   + αV
    scale_softmax = L * L + 6 * L * L                 # scaling & soft-max
    feedforward   = 2 * MAC_FLOP * L * H * 4 * H      # two dense layers
    gelu          = 8 * L * 4 * H
    layernorm     = 2 * 8 * L * H
    residual_add  = 2 * L * H
    return (proj_qkv + proj_out + attn_matmul +
            scale_softmax + feedforward + gelu +
            layernorm + residual_add)

def flops_transformer_layer_optimised(L: int, H: int) -> int:
    """FLOPs after FlashAttention-style & fused kernels (accuracy-neutral)."""
    proj_qkv      = 3 * MAC_FLOP * L * H * H          # fused QKV GEMM – FLOPs same
    proj_out      =     MAC_FLOP * L * H * H
    attn_matmul   =     MAC_FLOP * L * L * H          # FlashAttn: QKᵀ+αV in one pass
    # soft-max and scaling fused inside FlashAttention (ignored)
    feedforward   = 2 * MAC_FLOP * L * H * 4 * H
    # fused Bias-GELU – polynomial eval disappears
    layernorm     = 2 * 8 * L * H                    # kept (can’t avoid mean/var)
    residual_add  = 2 * L * H
    return (proj_qkv + proj_out + attn_matmul +
            feedforward + layernorm + residual_add)

def bert_forward_flops(sentence: str,
                       model_name="bert-base-uncased",
                       layers=12, hidden=768,
                       *, optimised=False):
    tok = BertTokenizer.from_pretrained(model_name)
    L = len(tok.encode(sentence, add_special_tokens=True))
    per_layer = (flops_transformer_layer_optimised if optimised
                 else flops_transformer_layer_vanilla)(L, hidden)
    return per_layer * layers, L

###############################################################################
# 4) Example comparison
###############################################################################
if __name__ == "__main__":
    sentence = "Jack loves Diane."

    H = 384        # Unified dimensionality for all systems
    R = 50         # CP rank
    layers = 12    # Standard for BERT-base

    # Compute FLOPs
    full_total, _  = estimate_discocirc_complexity(sentence, d=H)
    cp_total,   _  = estimate_cp_discocirc_complexity(sentence, d=H, R=R)
    bert_total, L  = bert_forward_flops(sentence, hidden=H, layers=layers)
    bert_opt,   _  = bert_forward_flops(sentence, hidden=H, layers=layers, optimised=True)

    # Print each block
    print("\n==== FLOP Comparison (H = "+str(H)+", CP-rank = "+str(R)+") ====\n")

    print(f"Sentence: {sentence}")
    print(f"Tokenized length (w/ specials): {L}")
    print()

    print(f"[1] Full-rank DisCoCirc         : {full_total:,.0f} FLOPs")
    print(f"[2] CP-rank-{R} DisCoCirc        : {cp_total:,.0f} FLOPs")
    print(f"[3] BERT-base (vanilla)         : {bert_total:,.0f} FLOPs")
    print(f"[4] BERT-base (optimised)       : {bert_opt:,.0f} FLOPs\n")

    # Compute relative comparisons
    methods = {
        "Full-rank DisCoCirc": full_total,
        f"CP-rank-{R} DisCoCirc": cp_total,
        "BERT-base (vanilla)": bert_total,
        "BERT-base (optimised)": bert_opt,
    }

    names = list(methods.keys())
    print("==== Relative FLOP Ratios ====\n")
    print(f"{'Method':<28}", end="")
    for name in names:
        print(f"{name:<28}", end="")
    print()

    for i, name_i in enumerate(names):
        print(f"{name_i:<28}", end="")
        for j, name_j in enumerate(names):
            ratio = methods[name_i] / methods[name_j]
            print(f"{ratio:<28.2f}", end="")
        print()