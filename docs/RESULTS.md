# Experiment Results

## Phase 3: EBM-Guided Masked Diffusion (LLaDA-8B) — Autoresearch Sweep

**Date**: July 4, 2026
**Hardware**: RTX 3090 (24GB VRAM), CUDA 12.4, 8GB RAM
**Model**: LLaDA-8B-Instruct (16GB VRAM), MDLM sampler (128 steps, block_size=32)
**Script**: `tests/autoresearch.py`
**Methodology**: Karpathy autoresearch — hypothesis → 5-min experiment → measure → keep/revert

### Experimental Setup

- **Prompt**: "Write a short story about something interesting." (open-ended, no topic hint)
- **Guidance targets**: 4 topics (space, ocean, horror, cooking)
- **Metrics**: target_sim (cosine sim to target via MiniLM-L6-v2), coherence (half-text cosine sim), diversity, non_rep (1 - most_common_word_ratio)
- **Quality threshold**: target_sim > 0.15 AND coherence > 0.3

### Full Sweep Results (13 experiments)

| # | Config | sim_mean | good% | Space | Ocean | Horror | Cooking | Verdict |
|---|--------|----------|-------|-------|-------|--------|---------|---------|
| e00 | α=5, const, abs_max, mean_emb | 0.1845 | 62% | 0.187 | 0.341 | 0.172 | 0.037 | baseline |
| e01 | α=5, linear_up, z_score | 0.0916 | 12% | 0.120 | 0.021 | 0.193 | 0.033 | ❌ REVERT |
| e02 | α=5, const, prob_additive | 0.0964 | 12% | 0.097 | 0.046 | 0.187 | 0.055 | ❌ REVERT |
| e03 | α=10, cosine, abs_max | 0.2370 | 62% | 0.159 | 0.198 | **0.461** | 0.130 | ✅ KEEP |
| e04 | α=15, cosine, abs_max | 0.2724 | 33% | 0.222 | 0.344 | 0.369 | 0.132 | ❌ REVERT (degenerate) |
| e05 | α=10, cosine, min_max | 0.2927 | 100%* | — | 0.226 | 0.411 | 0.190 | ❌ REVERT (3/8 degenerate) |
| **e06** | **α=10, cosine, abs_max, cosine_all** | **0.2574** | **75%** | 0.179 | 0.339 | **0.440** | 0.072 | **✅ BEST** |
| e07 | e06 + suppress baseline | 0.2574 | 75% | 0.179 | 0.339 | 0.440 | 0.072 | ❌ REVERT (no effect) |
| e08 | e06 + temp=0.3 | 0.2224 | 50% | 0.121 | 0.395 | 0.362 | 0.012 | ❌ REVERT |
| e09 | e06 + steps=64 | 0.1838 | 50% | 0.131 | 0.202 | 0.342 | 0.060 | ❌ REVERT |
| e10 | e06 + linear_down | 0.2437 | 75%* | 0.287 | 0.311 | 0.288 | 0.090 | ❌ REVERT (degenerate) |
| e11 | e06, 3 trials | 0.2446 | 58% | 0.158 | 0.268 | 0.405 | 0.148 | ✅ confirms e06 |
| e12 | e06 + temp=0.9 | 0.1545 | 25% | 0.158 | 0.045 | 0.387 | 0.029 | ❌ REVERT |
| e13 | e06 + logit_blended | 0.1943 | 62% | 0.158 | 0.287 | 0.277 | 0.055 | ❌ REVERT |

\* good% on valid outputs only; degenerate outputs (non_rep < 0.5) excluded

### Best Configuration

**e06: `logit_additive + cosine_all scoring + cosine alpha schedule + abs_max norm + α=10`**

- sim_mean: **0.2574** (+39% vs baseline 0.1845)
- good%: **75%** (6/8 outputs meet quality threshold)
- Strongest topics: **horror (0.44)**, **ocean (0.34)**
- Weakest topics: cooking (0.07), space (0.18)
- Zero degenerate outputs

### Key Findings

1. **Cosine alpha schedule is critical.** Energy guidance ramping 0→α_max→0 across denoising steps consistently outperforms constant or linear schedules. The model needs freedom early (to establish structure) and late (to refine coherence), with maximum guidance in the middle.

2. **α=10 is the sweet spot.** α=5 is too weak (sim barely above baseline). α=15 causes degenerate outputs (repetition collapse). The cosine schedule's peak α is ~5 (half of nominal), so effective peak injection is moderate.

3. **Logit-space additive guidance is superior to probability-space.** Adding energy scores directly to logits (pre-softmax) preserves the model's distribution shape better than blending probabilities.

4. **cosine_all scoring captures topic better than mean_embedding.** Computing max cosine similarity between each target token and all vocab tokens gives sharper per-token energy than projecting a mean direction.

5. **Temperature is bimodal.** temp=0.3 makes the model too deterministic (resists guidance). temp=0.9 adds noise that drowns the guidance signal. temp=0.6 (default) is optimal.

6. **Topic-dependent effectiveness.** Horror and ocean have distinctive vocabulary that competes well with the model's generic story template. Space and cooking are "common" concepts deeply embedded in the model's priors — the energy guidance cannot overcome the model's narrative inertia for these topics.

### Qualitative Analysis

All guided outputs begin with "Once upon a time, there was a girl..." — LLaDA-8B has a strong narrative template for open-ended story prompts. The energy guidance modifies **individual tokens** (cave→darkness, crystal→blood, water→ocean) but cannot redirect the **narrative structure**.

This reveals the fundamental limitation: **logit-level energy injection steers vocabulary selection but not narrative planning.** The masked diffusion model's iterative denoising creates a "commitment cascade" — early tokens lock in a story arc that later tokens must follow.

### Statistical Power

With 2 trials per topic (8 guided samples), the 75% good rate has a wide confidence interval. The 3-trial replication (e11) showed 58% (7/12), suggesting the true good rate is ~55-70%.

### Limitations of This Approach

1. **Prompt-locked narratives**: The open prompt "write a story" activates a dominant template. More specific prompts might show different guidance effectiveness.

2. **Token-level vs sentence-level**: The energy function operates per-token. Sentence-level semantic steering would require a different mechanism (e.g., classifier-free guidance at the sequence level).

3. **No training**: These experiments use inference-time guidance only. Fine-tuning the model with energy-aware objectives could improve steering.

---

## Earlier Experiments (Historical Reference)

### PGLF Sentence Embedding Experiment (April 2026)

| Model | STS-B Spearman | Δ |
|---|---|---|
| MiniLM-L6-v2 (baseline) | **0.8672** | — |
| PGLF + MiniLM-L6-v2 | 0.8264 | **-4.7%** |

PGLF projection layer degraded embedding quality. Pre-trained geometry is near-optimal.

### EBM Language Model Training (March 2026)

Training ran but did not converge. Best val_loss 0.0448 did not translate to coherent generation.

### Salvaged Components

- **Embedding service** → integrated into M2M-Rust
- **Autoresearch protocol** → adapted into this sweep framework
- **NaN debugging** for contrastive losses
- **Score matching pipeline** patterns
