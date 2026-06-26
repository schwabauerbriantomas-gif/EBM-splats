# Architecture

## Overview

The EBM-Splats system represents text as points on a 640-dimensional unit hypersphere (S^639). Instead of a single vector per token, the system uses **Gaussian Splats** — localized Gaussian distributions — as energy attractors that define a probability landscape over the embedding space.

## Core Components

### 1. Hypersphere Geometry (`geometry.py`)

All operations happen on S^639, the unit sphere in R^640:

- `exp_map(p, v)` — Riemannian exponential map (tangent vector → sphere)
- `log_map(p, x)` — Inverse map (sphere → tangent vector)
- `project_to_tangent(x, v)` — Orthogonal projection to tangent space
- `geodesic_distance(x, y)` — Angular distance = arccos(x·y)
- `normalize_sphere(x)` — L2 normalization

### 2. Splat Storage (`splats.py`)

The SplatStore holds up to 100K Gaussian splats, each defined by:
- **μ** (mu) — center on the hypersphere (learnable)
- **α** (alpha) — sharpness/intensity (learnable, positive)
- **κ** (kappa) — concentration parameter (learnable, positive)
- **importance** — weight for energy computation (learnable)

Neighbor finding uses FAISS Inner Product search (falls back to brute-force `torch.cdist` if FAISS unavailable).

### 3. Energy Function (`energy.py`)

The total energy combines four terms:

```
E(x) = E_splats(x) + λ_geom · E_geom(x) + λ_comp · E_comp(x) + E_context(x)
```

**Splat energy** (V2 sign convention):
```
E_splats(x) = -log Σ_k exp(α_k · (x·μ_k - 1) / τ + log w_k)
```
- `(x·μ_k - 1) ∈ [-2, 0]`, minimum energy at splat centers
- Importance-weighted via normalized kappa

**Geometric energy** — collapse regularization enforcing batch-level diversity.

**Compositional energy** — pairwise splat interaction via learned bilinear form.

**Context energy** (V2 §4.4) — 3-level hierarchical attraction:
```
E_trans(x) = -Σ_{l∈{local,medium,global}} λ_l · (x · c_l)
```

### 4. Score Network (`score_network.py`)

Direct parametric score model for denoising score matching (DSM):
- Input: x (640D) + sigma encoding (random Fourier features)
- Architecture: 3-layer MLP with LayerNorm + GELU
- Output: score vector projected to tangent space at x

Replaces the original broken approach using `torch.autograd.grad` on the energy function.

### 5. Langevin Dynamics (`langevin.py`)

Underdamped Langevin sampling on S^639:
- Velocity Verlet integrator with exponential map
- Gradient clipping for stability
- Adaptive noise injection on stagnation detection
- Typical: 200 steps, dt=0.001, γ=0.1, T=1.0

### 6. Self-Organized Criticality (`soc.py`)

Adaptive splat management:
- HistoryBuffer tracks recent states and energies
- Order parameter φ = mean(α · ρ/ρ_avg) measures tension
- When φ > threshold, new splat added at highest-energy location
- Minimum distance constraint prevents duplicates

### 7. MoE Decoder (`decoder.py`)

Mixture-of-Experts decoder (4 experts, 2 active):
- Router: linear → softmax → top-K
- Expert FFN: 2-layer MLP per expert
- **Vectorized** via `torch.einsum` (replaces O(B·K) Python loop)
- Input: [latent; context] → output: vocab logits

### 8. Hierarchical Context (`context.py`)

Three-level context for long-range dependencies:
- **Local** (β=0.5, 8-16 tokens) — fast adaptation
- **Medium** (β=0.8, 64-128 tokens) — moderate
- **Global** (β=0.95, 512+ tokens) — slow, stable

Each level maintains an EMA of recent token embeddings on the hypersphere.

## PGLF Extension

PGLF (Pareto-Guided Langevin Flow) extends the EBM for sentence embeddings:

1. **Text Encoder** — 6-layer Transformer (640D) with token + positional embeddings
2. **Contrastive Head** — InfoNCE loss with alignment and uniformity terms
3. **Pareto Filter** — multi-objective non-dominated sorting (quality × diversity × fidelity)
4. **Flow Matching** — OT-CFM learns deterministic vector field from noise → Pareto-optimal embeddings
5. **Embedding Service** — HTTP API wrapping sentence-transformers backbone

The 3-phase training pipeline:
```
Phase 1: Langevin Exploration → maps energy landscape
Phase 2: Pareto Filtering → selects golden trajectories  
Phase 3: Flow Matching → distills into fast inference model
```

## Known Limitations

1. **Vulkan engine is simulated** — falls back to CPU tensor ops, not real GPU compute
2. **FAISS index recreated per call** — SplatStorage.find_neighbors creates a new IndexFlatIP each time (perf bottleneck for large splat stores)
3. **Context hierarchy stores on CPU** — buffers move between CPU/GPU on each update
4. **No convergence achieved** — training started but perplexity benchmarks were inconclusive
5. **PGLF degraded baseline** — the core experiment showed -4.7% vs MiniLM-L6-v2

## Data Flow

```
Input tokens
    │
    ▼
nn.Embedding(50257, 640)
    │
    ▼
normalize_sphere ──→ x ∈ S^639
    │
    ├──→ EnergyFunction.compute_energy(x, context)  → E(x) ∈ R
    ├──→ EnergyFunction.compute_score(x)            → s(x) ∈ T_x S^639
    │         │
    │         ▼
    │    Langevin sampling (200 steps)
    │         │
    │         ▼
    │    x_sampled ∈ S^639 (low energy)
    │
    ├──→ HierarchicalContext.update(x)
    │
    ▼
EBMDecoder([x_sampled; context])
    │
    ▼
vocab logits (50257)
```
