import pandas as pd

# ---------------- Dense spaCy‑sm (more accurate params) ----------------
class DenseModel:
    def __init__(self, tagger_params=750_000, parser_params=1_300_000, flops_per_weight=2):
        self.tagger_params = tagger_params
        self.parser_params = parser_params
        self.flops_per_weight = flops_per_weight

    def flops(self, n_tokens: int) -> int:
        total_params = self.tagger_params + self.parser_params
        return n_tokens * total_params * self.flops_per_weight


# ---------------- Baseline MoE (original toy config) -------------------
class MoeBaseline:
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


# ---------------- Faster MoE (smaller router + linear scorer) ----------
class MoeFast:
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


# ---------------- Comparison helpers -----------------------------------
def compare_models(n_tokens: int, ks=(1, 2)):
    dense = DenseModel()
    baseline = MoeBaseline()
    fast = MoeFast()

    rows = []
    rows.append({"Model": "spaCy‑sm", "K": None, "Tokens": n_tokens,
                 "FLOPs": dense.flops(n_tokens)})
    for k in ks:
        rows.append({"Model": "MoE‑baseline", "K": k, "Tokens": n_tokens,
                     "FLOPs": baseline.flops(n_tokens, k)})
        rows.append({"Model": "MoE‑fast", "K": k, "Tokens": n_tokens,
                     "FLOPs": fast.flops(n_tokens, k)})
    return pd.DataFrame(rows)


def sweep_lengths(lengths=(5, 10, 20, 40, 80), ks=(1, 2)):
    frames = [compare_models(n, ks) for n in lengths]
    df = pd.concat(frames, ignore_index=True)
    return df


# Example sentence
example_text = "The quick brown fox jumps over the lazy dog."
n_tokens = len(example_text.strip().split())

df_single = compare_models(n_tokens)

# Calculate percentage savings and speedup compared to spaCy
spacy_flops = df_single[df_single["Model"] == "spaCy‑sm"]["FLOPs"].iloc[0]
df_single["Savings %"] = df_single["FLOPs"].apply(
    lambda x: f"{((spacy_flops - x) / spacy_flops * 100):.1f}%" if x != spacy_flops else "baseline"
)
df_single["Speedup"] = df_single["FLOPs"].apply(
    lambda x: f"{(spacy_flops / x):.2f}x" if x != spacy_flops else "1.00x"
)

# Add thousands separator for FLOPs
df_single["FLOPs"] = df_single["FLOPs"].map("{:,}".format)

print("FLOP Analysis for Single Sentence:")
print("=" * 60)
print(df_single.to_string(index=False))
print("=" * 60)
print(f"Example sentence: '{example_text}'")
print(f"Number of tokens: {n_tokens}")
print()
print("Note: Negative savings % means the model uses MORE FLOPs than spaCy")
