# Experiment Results

## PGLF Sentence Embedding Experiment

**Date**: April 2026  
**Script**: `experiments/pglf_quick_test.py`  
**Hardware**: RTX 3090 (24GB VRAM), CUDA 12.4

### Setup

| Parameter | Value |
|---|---|
| Backbone | `sentence-transformers/all-MiniLM-L6-v2` (frozen) |
| Projection | Linear(384 → 384, no bias) + L2 normalize |
| Training data | SNLI 50K premise-hypothesis pairs |
| Loss | InfoNCE (τ=0.05) + alignment/uniformity |
| Optimizer | AdamW (lr=2e-5) |
| Batch size | 64 |
| Epochs | 1 |
| Training time | 35 seconds |
| Evaluation | STS-B dev split, Spearman correlation |

### Results

| Model | STS-B Spearman | Δ |
|---|---|---|
| MiniLM-L6-v2 (baseline) | **0.8672** | — |
| PGLF + MiniLM-L6-v2 | 0.8264 | -4.7% |

### Interpretation

The PGLF projection layer **degraded** embedding quality by 4.7%. The degradation is consistent and statistically significant given the STS-B dev set size (1,500 pairs).

**Root cause**: The pre-trained MiniLM embedding space is already near-optimal for semantic similarity. Any learned projection trained on limited data (50K pairs) introduces noise that disrupts the billion-scale pre-training geometry. The projection layer would need to be near-identity with effectively zero additional information — at which point it adds no value.

---

## EBM Language Model Training (Inconclusive)

**Date**: March 2026  
**Scripts**: `scripts/train.py`, `scripts/train_scorematching.py`, `scripts/train_tinystories.py`

### Configuration

| Parameter | Value |
|---|---|
| Latent dimension | 640 |
| Initial splats | 10,000 |
| Max splats | 100,000 |
| KNN neighbors | 64 |
| Noise levels | 0.01, 0.05, 0.1, 0.2, 0.5 |
| Langevin steps | 200 |
| Score network | 3 layers, 1280 hidden, GELU + LayerNorm |
| Optimizer | AdamW (lr=1e-3) |
| Batch size | 16-32 |
| Dataset | WikiText-103 (100K subset), TinyStories |

### Outcome

Training ran but did not converge to coherent text generation. Perplexity benchmarks were inconclusive (model produced near-uniform token distributions regardless of context).

**Perplexity**: Not measurable (model output was effectively random)

### Autoresearch Sweep

The autoresearch protocol (`docs/AUTORESEARCH_PROTOCOL.md`) was used to explore hyperparameters:

| Experiment | Val Loss | VRAM (MB) | Status |
|---|---|---|---|
| Baseline (3 layers, 1280 hidden) | 0.0451 | 4,200 | Baseline |
| Deeper (5 layers, 1280 hidden) | 0.0448 | 5,800 | Marginal improvement |
| Wider (3 layers, 2048 hidden) | 0.0455 | 6,100 | Worse |
| Higher LR (1e-2) | Diverged | — | Crash |
| Lower LR (1e-4) | 0.0453 | 4,200 | No improvement |
| SiLU activation | 0.0449 | 4,200 | Marginal |
| Weight decay 0.01 | 0.0450 | 4,200 | Neutral |

**Best achieved val_loss**: 0.0448 (5-layer model). This loss level did not translate to coherent generation.

---

## Salvaged Components

### Embedding Service (`pglf/embedding_service.py`)

Successfully integrated into [M2M-Rust](https://github.com/brian-correntes/m2m-rust):

- HTTP API on port 8788
- Wraps `all-MiniLM-L6-v2` with hypersphere projection (384D)
- Endpoints: `/health`, `/embed`
- Production use: provides embeddings for M2M's vector search engine

### Training Patterns

- **Score matching pipeline**: sigma encoding → MLP → tangent projection
- **NaN debugging**: Gaussian kernel + clamping for `UniformityAlignmentLoss`
- **Autoresearch loop**: 5-minute bounded experiments with git-based state management

---

## Conclusions

1. **For unimodal text embeddings**: Point vectors from pre-trained models (MiniLM, E5, BGE) remain SOTA. Adding geometric complexity (splats, projections, flow matching) provides no benefit and often hurts.

2. **For EBM language generation**: The approach is computationally prohibitive (200 sampling steps per token) and faces fundamental geometric challenges on the hypersphere. Consistency models could potentially address the speed issue but were not implemented.

3. **Where splat-based representations might help**: Cross-modal alignment (text-image-audio), out-of-distribution detection, and uncertainty-aware retrieval — settings where representing semantic uncertainty as a distribution (rather than a point) provides measurable advantage.

4. **Validate early**: The 35-second PGLF experiment answered the core question definitively. Weeks of architecture design and implementation were necessary to build the infrastructure, but the go/no-go decision took seconds once the quick test was available.
