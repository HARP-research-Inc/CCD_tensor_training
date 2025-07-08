# High‑Speed POS‑MoE Dependency Parser – Implementation & Testing Roadmap

*Last updated: 6 Jul 2025*

---

## Critical Constants

| Symbol           | Value                        | Purpose                                                  |
| ---------------- | ---------------------------- | -------------------------------------------------------- |
| `EXPERT_DIM`     | **32**                       | Hidden size per expert (hyperbolic Möbius linear layers) |
| `K`              | **2**                        | Top‑*K* POS tags (soft routing)                          |
| `N_EXPERTS`      | **17**                       | Distinct experts (hash to 12 if memory‑constrained)      |
| `ROUTER_PARAMS`  | **≈ 4 k**                    | Tiny CNN router parameters                               |
| `LINEAR_SCORER`  | **512 FLOPs / tok**          | Replaces quadratic biaffine scorer                       |
| `FLASH_BIAFFINE` | **INT4, 2 048 FLOPs / pair** | Optional Phase 5 kernel                                  |

---

## Phase 0 – Baseline Metric Harness

| Goal | Build a repeatable FLOP & latency benchmark for spaCy‑sm on target hardware. |
| ---- | ---------------------------------------------------------------------------- |

**Key Tasks**

* Tokenise & sentence‑split sample corpora (UD EN train/dev).
* Implement FLOP estimator skeleton (matches spaCy‑sm param counts).
* Record baseline wall‑clock on CPU & GPU (batch sizes 1/8/32).

**Deliverable**: Jupyter notebook + JSON log of baseline numbers.

**Proof‑of‑Concept Checklist**

* [ ] Notebook runs end‑to‑end in < 30 s.
* [ ] Unit test asserts FLOP count ≈ 82 M for 20‑token sentence.

---

## Phase 1 – Minimal Hard‑Routing MoE Skeleton

| Goal | Forward pass of tiny CNN router + 17 hyperbolic experts + quadratic scorer + MST on CPU & GPU. |
| ---- | ---------------------------------------------------------------------------------------------- |

**Key Tasks**

1. Implement 4 k‑param 1‑D CNN POS router (top‑1 tag per token).
2. Build 32 × 32 Möbius linear + hGLU experts (torch‑geoopt).
3. Assemble quadratic biaffine scorer (`torch.einsum`).
4. Decode with Chu–Liu/Edmonds (torch‑struct).

**Deliverable**: `forward()` returning heads per token.

**Proof‑of‑Concept Checklist**

* [ ] Forward latency < 1 ms / 32 tok on CPU (AVX2).
* [ ] FLOP counter ≤ 1.3 M / 20 tok (≈ 65 × spaCy‑sm).

---

## Phase 2 – Soft Routing & k‑Best Enumeration

| Goal | Probabilistic router with top‑K = 2 tags and stacked‑MST k‑best queue. |
| ---- | ---------------------------------------------------------------------- |

**Key Tasks**

* Add softmax over POS logits; retain top‑2 tags/token.
* Extend scorer to `S[k,ℓ,h,d]` (head, dep, POS₁, POS₂).
* Implement Camerini k‑best MST enumerator.

**Deliverable**: Forward pass producing ranked parse list.

**Proof‑of‑Concept Checklist**

* [ ] ΔFLOPs router cost amortised (< 5 %).
* [ ] UAS drop ≤ 1 pp vs Phase 1 on UD dev.

---

## Phase 3 – Fused Expert + Linear‑Scorer Triton Kernel

| Goal | Single Triton kernel that performs expert mat‑mul & linear scorer in INT8. |
| ---- | -------------------------------------------------------------------------- |

**Key Tasks**

* Prototype kernel; autotune block sizes.
* Row‑wise INT8 weight quantisation with scale per output row.
* Integrate into `torch.autograd.Function` for back‑prop.

**Deliverable**: Triton `.py` + CI test comparing outputs within 1e‑3 RMSE to FP32.

**Proof‑of‑Concept Checklist**

* [ ] Kernel < 200 µs / 32 tok on RTX 4090.
* [ ] Memory bandwidth < 50 MB / batch.

---

## Phase 4 – Accuracy Pass

| Goal | Achieve ≥ 93 % UAS on UD English dev. |
| ---- | ------------------------------------- |

**Key Tasks**

* Training script (TorchLightning) with list‑oracle loss.
* Riemannian Adam, cosine LR schedule.
* Early stopping, Weights & Biases logging.

**Deliverable**: Checkpoint + scalar metrics JSON.

**Proof‑of‑Concept Checklist**

* [ ] Dev UAS ≥ 93 % within 6 h single‑GPU training.
* [ ] No dead experts (usage entropy > 0.5).

---

## Phase 5 – Micro‑Optimisation

| Goal | Hit > 1 M words/s on RTX 4090. |
| ---- | ------------------------------ |

**Key Tasks**

* Flash‑biaffine tiling with CUDA shared memory.
* CUDA graphs for static batch shapes.
* Lexical window pruning for n > 80 token docs.

**Deliverable**: Bench script logging tokens/s across batch sizes.

**Proof‑of‑Concept Checklist**

* [ ] 1 M words/s (batch 512 × 20 tok) on GPU.
* [ ] 100 k words/s on 32‑core CPU.

---

## Phase 6 – Quantisation & Packaging

| Goal | INT8/FP8 runtime wheel; ≤ 550 k FLOPs / 20 tok. |
| ---- | ----------------------------------------------- |

**Key Tasks**

* Quant stubs for hyperbolic ops (GeoOpt fork).
* On‑load weight casting; graph capture with `torch.compile`.
* Build wheels with poetry & GitHub Actions.

**Deliverable**: `pip install pos‑moe‑parser‑0.x.y.whl`.

**Proof‑of‑Concept Checklist**

* [ ] Wheel installs on Linux/macOS/Win.
* [ ] CLI parses 1 k sentences in < 1 s on CPU.

---

## Phase 7 – Benchmarking & Release

| Goal | Public demo (CLI/REST) + through‑put/memory docs. |
| ---- | ------------------------------------------------- |

**Key Tasks**

* spaCy wrapper (`nlp.add_pipe`).
* FastAPI REST endpoint.
* README table (FLOPs, latency, memory) vs spaCy‑sm, Stanza, UDPipe.

**Deliverable**: GitHub release v1.0.

**Proof‑of‑Concept Checklist**

* [ ] README badge shows 180 × speed‑up headline.
* [ ] Twitter/X demo tweet approved.

---

### Appendix A – Testing Matrix

| Phase | Unit Tests         | Integration      | Perf/Regression | Hardware  |
| ----- | ------------------ | ---------------- | --------------- | --------- |
|  0    | ✔️ FLOP count      | n/a              | –               | CPU       |
|  1    | ✔️ Forward tensors | ✔️ small batch   | –               | CPU + GPU |
|  2    | ✔️ Routing entropy | ✔️ parse list    | –               | GPU       |
|  3    | ✔️ Kernel numerics | ✔️ back‑prop     | ✔️ tokens/s     | GPU       |
|  4    | ✔️ Loss grad flow  | ✔️ train loop    | ✔️ Dev UAS      | GPU       |
|  5    | –                  | ✔️ full pipeline | ✔️ throughput   | CPU + GPU |
|  6    | ✔️ INT8 cast       | ✔️ wheel import  | ✔️ FLOPs        | CPU + GPU |
|  7    | –                  | ✔️ REST 200s     | ✔️ latency plot | CPU + GPU |

---

*Feel free to mark boxes, tweak constants, or drop extra notes during execution. Each phase is scoped to be a runnable proof‑of‑concept – ship early, measure often.*
