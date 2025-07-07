import spacy
from transformers import BertTokenizer

###############################################################################
# 0) MoE Parser Complexity Models (from parser_complexity.py)
###############################################################################
class DenseModel:
    """spaCy-sm dense parser model"""
    def __init__(self, tagger_params=750_000, parser_params=1_300_000, flops_per_weight=2):
        self.tagger_params = tagger_params
        self.parser_params = parser_params
        self.flops_per_weight = flops_per_weight

    def flops(self, n_tokens: int) -> int:
        total_params = self.tagger_params + self.parser_params
        return n_tokens * total_params * self.flops_per_weight

class MoeBaseline:
    """Baseline MoE parser with quadratic scorer"""
    def __init__(self,
                 expert_dim=32,
                 n_layers=2,
                 scorer_flops_per_pair=2048,     # biaffine O(N²)
                 router_params=50_000,
                 router_flops_per_weight=2):
        self.expert_dim = expert_dim
        self.n_layers = n_layers
        self.scorer_flops_per_pair = scorer_flops_per_pair
        self.router_params = router_params
        self.router_flops_per_weight = router_flops_per_weight

    def flops(self, n_tokens: int, k_active: int) -> int:
        router = n_tokens * self.router_params * self.router_flops_per_weight
        expert_per_tok = k_active * self.n_layers * (self.expert_dim ** 2) * 2
        experts_total = n_tokens * expert_per_tok
        scorer = (n_tokens ** 2) * self.scorer_flops_per_pair
        return router + experts_total + scorer

class MoeFast:
    """Fast MoE parser with linear scorer"""
    def __init__(self,
                 expert_dim=32,
                 n_layers=2,
                 scorer_flops_per_token=512,     # linear alternative scorer
                 router_params=4_096,            # tiny CNN router
                 router_flops_per_weight=2):
        self.expert_dim = expert_dim
        self.n_layers = n_layers
        self.scorer_flops_per_token = scorer_flops_per_token
        self.router_params = router_params
        self.router_flops_per_weight = router_flops_per_weight

    def flops(self, n_tokens: int, k_active: int) -> int:
        router = n_tokens * self.router_params * self.router_flops_per_weight
        expert_per_tok = k_active * self.n_layers * (self.expert_dim ** 2) * 2
        experts_total = n_tokens * expert_per_tok
        scorer = n_tokens * self.scorer_flops_per_token      # linear cost
        return router + experts_total + scorer

###############################################################################
# 1) Load a spaCy model (download "en_core_web_sm" if not installed)
###############################################################################
try:
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except OSError:
    print("Warning: spaCy model not available. Using simple token counting.")
    SPACY_AVAILABLE = False
    nlp = None

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

def estimate_discocirc_complexity(sentence, d=300, include_parser=True):
    """Full-rank cost: operator rank-r  →  d**r FLOPs + parser cost."""
    if not SPACY_AVAILABLE:
        return 0, []
    
    doc = nlp(sentence)
    discocirc_flops, breakdown = 0, []
    for tok in doc:
        r = get_token_rank(tok)
        cost = d ** r
        breakdown.append((tok.text, tok.pos_, r, cost))
        discocirc_flops += cost
    
    if include_parser:
        # Add spaCy parser cost
        n_tokens = len(sentence.split())
        parser = DenseModel()
        parser_flops = parser.flops(n_tokens)
        total_flops = discocirc_flops + parser_flops
        return total_flops, breakdown
    else:
        return discocirc_flops, breakdown

# --------------------------------------------------------------------------- #
# >>> NEW: CP-DisCoCirc complexity                                            #
# --------------------------------------------------------------------------- #
def estimate_cp_discocirc_complexity(sentence, d=300, R=50, include_parser=True):
    """
    CP-decomposed cost:   FLOPs ≈ R * (r+2) * d     (see derivation in notes)

    • r   = multilinear order of the operator
    • d   = embedding dimension
    • R   = chosen CP rank   (default 50 is typical for d≈300)
    """
    if not SPACY_AVAILABLE:
        return 0, []
    
    doc = nlp(sentence)
    discocirc_flops, breakdown = 0, []
    for tok in doc:
        r = get_token_rank(tok)
        cost = R * (r + 2) * d        # linear in d instead of d**r
        breakdown.append((tok.text, tok.pos_, r, cost))
        discocirc_flops += cost
    
    if include_parser:
        # Add spaCy parser cost
        n_tokens = len(sentence.split())
        parser = DenseModel()
        parser_flops = parser.flops(n_tokens)
        total_flops = discocirc_flops + parser_flops
        return total_flops, breakdown
    else:
        return discocirc_flops, breakdown

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
    sentence = "Once upon a midnight dreary, while I pondered, weak and weary, over many a quaint and curious volume of forgotten lore, while I nodded, nearly napping, suddenly there came a tapping, as of some one gently rapping, rapping at my chamber door."

    H = 384*4        # Unified dimensionality for all systems
    R = 50         # CP rank
    layers = 12    # Standard for BERT-base
    k = 1          # MoE active experts

    # Get token count for parser models
    L_parser = len(sentence.split())  # Simple whitespace tokenization for parser models

    # Compute FLOPs
    if SPACY_AVAILABLE:
        full_total, _  = estimate_discocirc_complexity(sentence, d=H)
        cp_total,   _  = estimate_cp_discocirc_complexity(sentence, d=H, R=R)
    else:
        full_total, cp_total = None, None
        print("DisCoCirc complexity skipped (spaCy not available)")

    bert_total, L  = bert_forward_flops(sentence, hidden=H, layers=layers)
    bert_opt,   _  = bert_forward_flops(sentence, hidden=H, layers=layers, optimised=True)

    # MoE Parser models
    dense_parser = DenseModel()
    moe_baseline = MoeBaseline()
    moe_fast = MoeFast()
    
    dense_total = dense_parser.flops(L_parser)
    moe_baseline_total = moe_baseline.flops(L_parser, k)
    moe_fast_total = moe_fast.flops(L_parser, k)

    # Print each block
    print("\n==== FLOP Comparison (H = "+str(H)+", CP-rank = "+str(R)+", MoE k = "+str(k)+") ====\n")

    print(f"Sentence: {sentence}")
    print(f"BERT tokenized length (w/ specials): {L}")
    print(f"Parser tokenized length (whitespace): {L_parser}")
    print()

    if SPACY_AVAILABLE:
        print(f"[1] Full-rank DisCoCirc+Parser  : {full_total:,.0f} FLOPs")
        print(f"[2] CP-rank-{R} DisCoCirc+Parser : {cp_total:,.0f} FLOPs")
    print(f"[3] BERT-base (vanilla)         : {bert_total:,.0f} FLOPs")
    print(f"[4] BERT-base (optimised)       : {bert_opt:,.0f} FLOPs")
    print(f"[5] spaCy-sm Parser only        : {dense_total:,.0f} FLOPs")
    print(f"[6] MoE Baseline Parser (k={k})   : {moe_baseline_total:,.0f} FLOPs")
    print(f"[7] MoE Fast Parser (k={k})       : {moe_fast_total:,.0f} FLOPs\n")

    # Compute relative comparisons
    methods = {
        "BERT-base (vanilla)": bert_total,
        "BERT-base (optimised)": bert_opt,
        "spaCy-sm Parser only": dense_total,
        f"MoE Baseline Parser (k={k})": moe_baseline_total,
        f"MoE Fast Parser (k={k})": moe_fast_total,
    }
    
    if SPACY_AVAILABLE:
        methods["Full-rank DisCoCirc+Parser"] = full_total
        methods[f"CP-rank-{R} DisCoCirc+Parser"] = cp_total

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