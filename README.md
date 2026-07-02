# EBM-Splats

> **Status: ACTIVE RESEARCH** — Phase 1 empirical tests complete (July 2026). PGLF embedding approach discarded. EBM generator with Rectified Flow under exploration.

Energy-Based Model with Gaussian Splats on a 640D hypersphere. Explores distributional representations for latent spaces, sampling via Langevin dynamics and Rectified Flow.

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

**Composition results:**
- Equal blend (A+B): balanced similarity to both topics
- Weighted (70/30): asymmetric control confirmed
- Suppression (A−B): sim_B dropped from 0.44 to −0.29
- Triple (A+B+C): all three topics active (sim > 0.48)

**EBM + RF enables semantic arithmetic on the hypersphere.**

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
