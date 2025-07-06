import pandas as pd

# Example sentence (user can modify this string)
example_text = "The quick brown fox jumps over the lazy dog."
tokens = example_text.strip().split()
n_tokens = len(tokens)

# ----------- Constants (rough–order-of-magnitude numbers) -----------
# spaCy-sm parameter sizes (tagger+parser)  ≈ 2.6 M weights
SPACY_TAGGER_PARAMS = 300_000
SPACY_PARSER_PARAMS = 2_300_000
SPACY_FLOPS_PER_WEIGHT = 2  # multiply + add

# Hyper-parameters for our MoE designs
EXPERT_DIM = 32
EXPERT_N_LAYERS = 2
SCORER_FLOPS_PER_PAIR = 2_048  # biaffine ~ 32×32 mult-adds *2
TAGGER_PARAMS_MOE = 50_000     # tiny CNN router
TAGGER_FLOPS_PER_WEIGHT = 2

def spacy_flops(n_tokens: int) -> int:
    total_params = SPACY_TAGGER_PARAMS + SPACY_PARSER_PARAMS
    return n_tokens * total_params * SPACY_FLOPS_PER_WEIGHT

def moe_flops(n_tokens: int, k_active: int) -> int:
    # (1) POS router cost
    router = n_tokens * TAGGER_PARAMS_MOE * TAGGER_FLOPS_PER_WEIGHT
    # (2) Active experts cost
    expert_per_token = k_active * EXPERT_N_LAYERS * (EXPERT_DIM**2) * 2  # mult+add
    experts_total = n_tokens * expert_per_token
    # (3) Pairwise biaffine scorer
    scorer = (n_tokens ** 2) * SCORER_FLOPS_PER_PAIR
    return router + experts_total + scorer

results = []
results.append({"Approach": "spaCy-sm", "Active-experts/token": "N/A", "FLOPs": spacy_flops(n_tokens)})
for k in (1, 2):
    results.append({"Approach": f"MoE (hard, K={k})" if k == 1 else f"MoE (soft, K={k})",
                    "Active-experts/token": k,
                    "FLOPs": moe_flops(n_tokens, k)})

df = pd.DataFrame(results)
# Pretty-print with thousands separator
df["FLOPs"] = df["FLOPs"].apply(lambda x: f"{x:,}")

print("FLOP estimate per approach:")
print("=" * 50)
print(df.to_string(index=False))
print("=" * 50)
print(f"Example sentence: '{example_text}'")
print(f"Number of tokens: {n_tokens}")
