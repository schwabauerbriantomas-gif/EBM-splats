# EBM-Splats

[![CI](https://github.com/schwabauerbriantomas-gif/EBM-splats/actions/workflows/ci.yml/badge.svg)](https://github.com/schwabauerbriantomas-gif/EBM-splats/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/ebm-splats)](https://pypi.org/project/ebm-splats/)

> **Status: Research complete.** Phase 1-3 empirical tests finished. Energy-guided generation continued in [m2m-energy-fields](https://github.com/schwabauerbriantomas-gif/m2m-energy-fields).

Energy-Based Model with Gaussian Splats on a 640D hypersphere. Explores distributional representations for latent spaces, sampling via Langevin dynamics, Rectified Flow, and energy-guided masked diffusion.

## Project Phases

### Phase 1: EBM + PGLF (April 2026) — Discarded

EBM with Gaussian splats as attractors on S^639 + PGLF (projection over MiniLM with contrastive loss).

**Result:** PGLF degraded MiniLM on STS-B (-4.7%). Projection over pre-trained embeddings always destroys geometry.

### Phase 1 Empirical: Discard Tests (July 2026)

3 empirical tests to discard or confirm alternatives. RTX 3090, real data.

| Test | Hypothesis | Result | Verdict |
|------|-----------|-----------|-----------|
| PGLF Grid (14 configs) | Can any config beat MiniLM? | 0/14 beat baseline (0.8672) | **DISCARDED** |
| OOD Detection | Does EBM energy detect OOD? | AUROC=1.0 (EBM) vs 0.999 (random baseline) | **NO ADVANTAGE** |
| RF vs Langevin | Does RF solve the speed bottleneck? | 29.4x faster (RF 5 vs Langevin 200 steps), better quality | **CONFIRMED** |

**Key finding:** The argument that "200 Langevin steps per token" is prohibitive no longer applies. Rectified Flow with 1-2 steps produces better samples than Langevin with 200 steps, 29.4-38.7x faster (`tests/t11_rf_results.jsonl`: RF 5 steps 0.0079s and RF 1 step 0.006s vs Langevin 200 steps 0.2322s).

### Phase 2: Energy-Guided Generation (July 2026)

EBM as a generator that learns its own latent space (not as a layer over pre-trained models), with sampling via Rectified Flow.

| Test | Hypothesis | Result | Verdict |
|------|-----------|-----------|-----------|
| Energy-Guided Generation | Can energy manipulation steer generation? | 100% topic control at gs=1.0-2.0 | **CONFIRMED** |
| Concept Composition | Can multiple concepts be combined? | 4/4 mechanisms work | **CONFIRMED** |

### Phase 3: EBM-Guided Masked Diffusion on LLaDA-8B (July 2026)

Can energy fields steer a large masked diffusion model (LLaDA-8B) via logit injection at each denoising step? **13-experiment autoresearch sweep** using Karpathy methodology.

| Config | sim_mean | good% | Verdict |
|--------|----------|-------|---------|
| Baseline (α=5, constant) | 0.1845 | 62% | reference |
| **Best (α=10, cosine schedule, cosine_all)** | **0.2574** | **75%** | **✅ +39% sim** |
| Worst (z_score + linear_up) | 0.0916 | 12% | ❌ |

**Best config:** `logit_additive + cosine_all scoring + cosine alpha schedule + abs_max norm + α=10`

**What works:** Topics with distinctive vocabulary (horror sim=0.44, ocean sim=0.34) are steered effectively. Energy injection modifies token selection while maintaining coherence.

**What doesn't work:** "Common" topics (cooking sim=0.07, space sim=0.18) resist steering — the model's narrative priors dominate. All outputs begin "Once upon a time, there was a girl..." regardless of guidance target.

**Fundamental limitation:** Logit-level energy injection steers **vocabulary selection** but not **narrative planning**. The masked diffusion model's iterative denoising creates a commitment cascade that energy cannot redirect.

### Security Implications

The Phase 3 experiments inadvertently revealed a **novel attack surface** in masked diffusion language models. Unlike autoregressive models (GPT, Llama) which have 1 logit injection point per token, MDLMs expose **N injection points** (one per denoising step, typically 128). Our experiments show that pre-computed energy vectors can steer generation toward arbitrary semantic targets while maintaining output coherence.

See [`docs/SECURITY_ANALYSIS.md`](docs/SECURITY_ANALYSIS.md) for the full threat model, empirical evidence, and mitigation recommendations.

See [`docs/RESULTS.md`](docs/RESULTS.md) for the full 13-experiment sweep with per-topic breakdowns.

## Repository Structure

```
├── src/ebm/               # Core EBM modules (geometry, splats, energy, model, etc.)
├── pglf/                  # PGLF (archived — discarded empirically)
├── scripts/               # Training and generation scripts
├── tests/
│   ├── phase1_t11_rf_vs_langevin.py    # RF vs Langevin benchmark
│   ├── phase1_t12_pglf_grid.py         # PGLF grid search
│   ├── phase1_t13_ood_energy.py        # OOD detection test
│   ├── phase2_energy_guided.py         # Energy-guided generation test
│   ├── phase2_composition.py           # Concept composition test
│   └── t*_results.jsonl                # Raw results
└── docs/
    ├── PHASE1_RESULTS.md  # Full Phase 1 report
    ├── PHASE2_RESULTS.md  # Phase 2 energy-guided generation report
    └── ...
```

## Detailed Results

See [`docs/PHASE1_RESULTS.md`](docs/PHASE1_RESULTS.md) and [`docs/PHASE2_RESULTS.md`](docs/PHASE2_RESULTS.md) for full empirical test reports.

## Tech Stack

- Python, PyTorch (CUDA 12.4, RTX 3090)
- sentence-transformers, HuggingFace datasets
- Rust (M2M integration via HTTP)

## Experimental GPU kernels

Optional Triton/CUDA-graph kernels for the hot paths, with automatic fallback
to the reference implementation (no CUDA, no Triton, or outside the winning
regime). Install with `pip install -e ".[triton]"` (Linux + CUDA GPU).

Official head-to-head (RTX 3090, 4000 splats × dim 640, B=64, seed 42,
correctness asserted before timing — `benches/bench_headtohead.py`):

| workload | reference | kernel | speedup | correctness |
|---|---|---|---|---|
| Langevin sampling (100 steps, B=8) | 656 ms | **46.7 ms** | **14.1x** | graph ≡ eager bit-exact; \|ΔE\| vs reference 0.021 (sampler noise, 0.2%) |
| score forward | 1.06 ms | 0.25 ms | **4.25x** | maxdiff 8.2e-8 |
| splat energy | 0.80 ms | 0.23 ms | **3.47x** | maxdiff 9.5e-7 |
| find_neighbors (warm, memoized) | 0.02 ms | 0.02 ms | 0.82x | bit-exact |

Honest notes:

- The Langevin win is **launch-overhead elimination** (CUDA graph capture of the
  whole 100-step loop, ~176 kernels/step in eager mode), not math speedups.
  Capture costs ~0.7–1.2 s once per (shape, splats) configuration and is cached.
- The KNN kernel only wins on cold lookups at small batch; warm lookups are
  dominated by the LRU-1 memoization of the reference (0.02 ms) — the wrapper
  routes automatically and falls back outside its winning regime.
- The fused KNN variant (distance+top-k in one launch) was built, measured and
  **rejected**: 15–40x slower than the split path (bitonic merge dominates).
- Reference quirk left untouched (documented, not fixed — it would invalidate
  published results): `find_neighbors` selects the LARGEST d2 values (farthest
  splats) on its torch path, and its legacy faiss path disagreed with it.

## License

Apache-2.0
