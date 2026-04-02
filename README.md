# EBM-Splats → PGLF (Archived)

> **Status: ARCHIVED** — Project concluded April 2026 after empirical evaluation showed the approach did not outperform existing solutions. Preserved for reference.

## What This Was

An exploration of using **Gaussian Splats** to represent energy-based distributions in embedding space, combined with a custom embedding model (PGLF — Persistent Gaussian Latent Flow).

The core idea: instead of representing text as a single point vector, use splats (Gaussian distributions) to capture semantic uncertainty and multi-modal meaning.

## Architecture

- **EBM**: Energy-Based Model with Langevin dynamics sampling
- **PGLF**: 6-layer Transformer encoder + contrastive learning + flow matching
  - InfoNCE contrastive loss with alignment
  - 3-phase training: Langevin exploration → Pareto filtering → Flow Matching distillation
- **Embedding Service**: HTTP service wrapping `sentence-transformers/all-MiniLM-L6-v2` with hypersphere projection

## Key Files

- `pglf/` — PGLF model code (encoders, trainer, contrastive head, embedding service)
- `pglf_checkpoints/` — Trained model weights (303MB)
- `pglf_quick_test.py` — The experiment that answered the core question

## The Experiment & Results

**Question:** Do PGLF layers (projection + contrastive loss) improve over plain MiniLM-L6-v2?

**Method:**
- Baseline: `all-MiniLM-L6-v2` plain on STS-B benchmark
- PGLF: Frozen MiniLM backbone + trainable projection (384→384) + InfoNCE loss
- Training: 50K SNLI pairs, 1 epoch, batch 64
- Evaluation: STS-B Spearman correlation

**Results:**

| Model | STS-B Spearman | Δ |
|---|---|---|
| MiniLM-L6-v2 (baseline) | **0.8672** | — |
| PGLF + MiniLM-L6-v2 | 0.8264 | **-4.7%** |

**PGLF degraded performance by 4.7%.**

## Why It Didn't Work

1. **MiniLM is already optimized** — trained on 1B+ sentence pairs for semantic similarity. Its embedding geometry is already near-optimal for cosine similarity tasks.

2. **Projection destroys geometry** — Adding a randomly-initialized projection layer and training on 50K pairs disrupts the carefully learned embedding space. Even with near-identity initialization, the contrastive loss on limited data introduces noise.

3. **Insufficient data** — PGLF's 50K SNLI pairs vs MiniLM's billion-scale pretraining. Even training on 1M pairs would likely only close the gap to zero, not surpass the baseline.

4. **Unimodal limitation** — The splat-based uncertainty representation might have value for cross-modal alignment (text+image, text+audio) or OOD detection, but for unimodal text similarity, point embeddings are sufficient.

## What Was Salvaged

- The **embedding service** (`pglf/embedding_service.py`) was integrated into [M2M-Rust](https://github.com/brian-corrientes/m2m-rust) as the production embedding provider via HTTP
- NaN debugging techniques in contrastive losses (Gaussian kernel + clamping for `UniformityAlignmentLoss`)
- Experience with flow matching, Langevin dynamics, and EBM training pipelines

## Lessons Learned

- **Negative results are results.** Knowing what doesn't work saves future investment.
- **Don't compete with scale.** A 30M param model trained on 50K examples won't beat a 22M param model trained on 1B+ examples, regardless of architectural novelty.
- **Validate early.** The quick test (35 seconds of training) answered the question that weeks of architecture design couldn't.

## Tech Stack

- Python, PyTorch (CUDA 12.4, RTX 3090)
- sentence-transformers, HuggingFace datasets
- Rust (M2M integration via HTTP)
