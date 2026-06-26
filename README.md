# EBM-Splats

> **Status: ARCHIVED** — Research concluded April 2026. Negative result: the PGLF approach degraded sentence embedding quality by 4.7% vs baseline. Repository preserved for methodological reference and code salvage.

## Summary

An exploration of **Energy-Based Models** combined with **Gaussian Splats** on a Riemannian hypersphere (S^639) for continuous language representation. The project evolved through two phases:

1. **Phase 1 (EBM)**: Language generation via Langevin dynamics sampling on a splat-based energy landscape
2. **Phase 2 (PGLF)**: Pareto-Guided Langevin Flow — sentence embeddings using a custom projection + contrastive loss on top of MiniLM-L6-v2

The central hypothesis was that representing text as **Gaussian distributions** (splats) rather than point vectors could capture semantic uncertainty and improve embedding quality. The experiment disproved this hypothesis for unimodal text similarity.

## Key Result

| Model | STS-B Spearman | Δ |
|---|---|---|
| MiniLM-L6-v2 (baseline) | **0.8672** | — |
| PGLF + MiniLM-L6-v2 | 0.8264 | **-4.7%** |

PGLF degraded performance. Adding a projection layer and training on 50K SNLI pairs disrupted the carefully optimized embedding geometry of the pre-trained backbone.

## What Was Salvaged

- **Embedding service** (`pglf/embedding_service.py`) — integrated into [M2M-Rust](https://github.com/brian-corrientes/m2m-rust) as production embedding provider
- **NaN debugging techniques** for contrastive losses (Gaussian kernel + clamping)
- **Training pipeline patterns** (score matching, Langevin dynamics, flow matching)
- **Autoresearch protocol** (5-minute experiment loop adapted from Karpathy)

## Repository Structure

```
ebm-splats/
├── src/ebm/              Core EBM modules
│   ├── config.py         EBMConfig dataclass (110 parameters)
│   ├── geometry.py       Riemannian ops (exp_map, log_map, geodesic_distance)
│   ├── splats.py         SplatStorage — KNN-indexed Gaussian attractors
│   ├── energy.py         EnergyFunction — splat + geometric + compositional + context
│   ├── score_network.py  Direct parametric score model (DSM)
│   ├── langevin.py       Underdamped Langevin sampler on S^639
│   ├── soc.py            Self-Organized Criticality (adaptive splat addition)
│   ├── decoder.py        MoE decoder (4 experts, 2 active, vectorized)
│   ├── context.py        3-level hierarchical context (local/medium/global)
│   ├── data.py           Dataset utilities (WikiText-103, TinyStories)
│   ├── data_loader.py    Streaming dataset loader
│   ├── evaluation.py     Perplexity and coverage metrics
│   ├── vulkan.py         Vulkan compute engine (note: simulated CPU fallback)
│   └── cuda/energy.py    CUDA-native energy computation
├── pglf/                 PGLF extension (sentence embeddings)
│   ├── encoders.py       TextEncoder (6-layer Transformer, 640D)
│   ├── contrastive_head.py  InfoNCE + alignment/uniformity losses
│   ├── embedding_service.py HTTP embedding service (salvaged into M2M)
│   ├── flow_matching.py  OT-CFM conditional flow matching
│   ├── pareto_filter.py  Multi-objective non-dominated sorting
│   ├── service.py        PGLF inference service
│   └── trainer.py        3-phase training orchestration
├── scripts/              Training and evaluation scripts
├── experiments/          One-off experiments and diagnostics
├── tests/                Unit and integration tests
├── docs/                 Architecture, methodology, specifications
└── shaders/              Vulkan compute shaders (SPIR-V)
```

## Tech Stack

- **Python 3.10+**, **PyTorch** (CUDA 12.4, RTX 3090)
- **sentence-transformers**, **HuggingFace datasets**
- **FAISS** (KNN for splat neighbor search)
- **Flask** (embedding service HTTP API)
- **Rust** (M2M integration via HTTP)

## Installation

```bash
# Archived project — for reference only
pip install -e .[dev]
```

## Lessons Learned

- **Negative results are results.** Knowing what doesn't work saves future investment.
- **Don't compete with scale.** A 30M param model trained on 50K examples won't beat a 22M param model trained on 1B+ examples.
- **Validate early.** 35 seconds of training answered the question that weeks of architecture design couldn't.

## License

Apache-2.0
