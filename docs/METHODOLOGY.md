# Methodology

## Research Question

Can Gaussian Splats on a Riemannian hypersphere improve text representation quality over standard point-vector embeddings?

## Phase 1: EBM Language Model

### Objective
Build a continuous language model using energy-based sampling on S^639, where token embeddings are points on the hypersphere and Gaussian Splats define an energy landscape.

### Approach
1. **Initialization**: Random splats on the hypersphere, optionally initialized from pre-trained word embeddings
2. **Training**: Denoising score matching (DSM) — perturb clean embeddings with Gaussian noise at multiple scales, train ScoreNetwork to predict the denoising direction
3. **Sampling**: Langevin dynamics — start from noise, follow the score field to reach low-energy (high-probability) regions
4. **Generation**: Sample latent via Langevin → decode to vocabulary logits via MoE decoder
5. **Adaptation**: SOC mechanism adds splats at high-energy regions, refining the energy landscape

### Datasets
- **WikiText-103** (100K subset) — primary training data
- **TinyStories** — simplified language for faster convergence experiments

### Configuration
- 640D latent space, 10K initial splats (max 100K)
- 64 nearest neighbors per query (KNN via FAISS)
- 5 noise levels: σ ∈ {0.01, 0.05, 0.1, 0.2, 0.5}
- 200 Langevin steps, dt=0.001
- AdamW, lr=1e-3, gradient clipping at 1.0

### Outcome
Training ran but **no convergence was achieved**. The model could not generate coherent text. Key issues:
- Energy landscape was too flat (insufficient splat differentiation)
- Langevin sampling was computationally expensive (200 steps per token)
- Score matching on the hypersphere is numerically challenging (tangent space projections introduce instabilities near poles)

---

## Phase 2: PGLF Sentence Embeddings

### Objective
Repurpose the EBM infrastructure for sentence embeddings. Test whether PGLF layers (projection + contrastive loss + flow matching) improve over a strong baseline.

### Hypothesis
Adding a learned projection layer trained with InfoNCE contrastive loss on top of a pre-trained sentence encoder would improve semantic similarity by adapting the embedding geometry to a specific objective.

### Experimental Design

**Independent variable**: Model architecture
- Control: `all-MiniLM-L6-v2` (22M params, pre-trained on 1B+ pairs)
- Treatment: Frozen MiniLM backbone + trainable projection (384→384) + InfoNCE loss

**Controlled variables**:
- Evaluation dataset: STS-B benchmark (Spearman correlation)
- Training data: 50K SNLI premise-hypothesis pairs
- Training duration: 1 epoch, batch size 64
- Backbone: identical pre-trained weights

**Measured variable**: STS-B Spearman correlation

### Implementation Details

**Contrastive loss** (InfoNCE with Gaussian kernel):
```
L = -log(exp(sim(z_i, z_j)/τ) / Σ_k exp(sim(z_i, z_k)/τ))
```

**Alignment/Uniformity** (Wang & Isola, 2020):
- Alignment: mean ||z_i - z_j||² for positive pairs
- Uniformity: log mean exp(-2 ||z_i - z_j||²)

NaN prevention: Gaussian kernel smoothing + value clamping in the `UniformityAlignmentLoss`.

**Flow Matching** (OT-CFM): Trained a conditional vector field v(x, t, c) to deterministically map noise → embeddings, distilling the stochastic Langevin trajectories.

### Results

| Model | STS-B Spearman | Δ |
|---|---|---|
| MiniLM-L6-v2 (baseline) | 0.8672 | — |
| PGLF + MiniLM-L6-v2 | 0.8264 | -4.7% |

Training time: 35 seconds. The experiment was run via `experiments/pglf_quick_test.py`.

---

## Autoresearch Protocol

Adapted from Karpathy's autoresearch methodology for automated hyperparameter search:

1. Modify only `autoresearch_train.py` (model/optimizer/training loop)
2. Run for 5 minutes wall clock on single GPU
3. Extract `val_loss` and `peak_vram_mb`
4. If improved → keep commit; if worse → revert
5. Loop indefinitely

Sweep space explored: depth (3-8 layers), hidden dim (256-2048), learning rate (1e-5 to 1e-2), noise levels, activation functions, normalization types, batch sizes, dropout rates, EMA configurations.

See `docs/AUTORESEARCH_PROTOCOL.md` for the full protocol.

---

## Code Quality Improvements Applied

During the archival restructuring, the following fixes were applied to the codebase:

1. **MoE decoder vectorized** — replaced O(B·K) Python loop with batched `torch.einsum` operations
2. **Context energy shape fix** — `compute_context_energy` now returns per-batch-element tensor instead of scalar
3. **Import structure** — converted bare module imports to proper package-relative imports
4. **Junk removed** — 24+ log files, empty files, compiled bytecode, duplicate experiment outputs deleted

---

## Why It Didn't Work

### For Language Generation (Phase 1)

The energy-based approach to language generation on a hypersphere faces fundamental challenges:

1. **Sampling cost**: 200 Langevin steps per token is 100-1000× slower than autoregressive decoding
2. **Landscape flatness**: With limited training data, splats cluster in a small region, leaving most of S^639 unexplored
3. **Gradient instabilities**: Riemannian tangent projections near poles cause numerical issues
4. **Decoder bottleneck**: The MoE decoder must map from continuous S^639 back to discrete vocabulary — a fundamentally harder problem than standard classification

### For Sentence Embeddings (Phase 2)

1. **MiniLM is already optimized** — trained on 1B+ sentence pairs. Its geometry is near-optimal for cosine similarity
2. **Projection destroys geometry** — a randomly-initialized projection layer, even when trained, disrupts the carefully learned embedding space
3. **Insufficient data** — 50K SNLI pairs vs 1B+ pre-training pairs. Even 1M pairs would likely only close the gap to zero, not surpass baseline
4. **Unimodal limitation** — Splat-based uncertainty may have value for cross-modal alignment (text+image) or OOD detection, but for unimodal text similarity, point embeddings are sufficient

### What Would Need to Change

For the EBM approach to be viable:
- **Consistency models** to reduce Langevin steps from 200 to 1-2 (100× speedup)
- **Large-scale training** (10M+ pairs, not 50K)
- **Cross-modal setting** where uncertainty representation provides genuine advantage
- **Pre-trained initialization** of splats from a strong encoder (e.g., Phi-3-mini embeddings)
