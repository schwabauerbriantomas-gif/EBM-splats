# EBM-Splats

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
| OOD Detection | Does EBM energy detect OOD? | AUROC=1.0 but NN=0.999 | **NO ADVANTAGE** |
| RF vs Langevin | Does RF solve the speed bottleneck? | 24-29x faster, better quality | **CONFIRMED** |

**Key finding:** The argument that "200 Langevin steps per token" is prohibitive no longer applies. Rectified Flow with 1-2 steps produces better samples than Langevin with 200 steps, 24x faster.

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
├── docs/
│   ├── PHASE1_RESULTS.md  # Full Phase 1 report
│   ├── PHASE2_RESULTS.md  # Phase 2 energy-guided generation report
│   └── ...
└── benchmark_results/     # Previous benchmarks
```

## Detailed Results

See [`docs/PHASE1_RESULTS.md`](docs/PHASE1_RESULTS.md) and [`docs/PHASE2_RESULTS.md`](docs/PHASE2_RESULTS.md) for full empirical test reports.

## Tech Stack

- Python, PyTorch (CUDA 12.4, RTX 3090)
- sentence-transformers, HuggingFace datasets
- Rust (M2M integration via HTTP)

## License

Apache-2.0
