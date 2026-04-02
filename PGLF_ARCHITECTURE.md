# PGLF-EBM: Pareto-Guided Langevin Flow Embedding Model

## Architecture Overview

Omnimodal embedding model that maps text, images, and audio to a shared 640D hypersphere (S^639),
using the PGLF methodology built on top of EBM-splats.

## Three-Phase Pipeline

### Phase 1: Langevin Exploration (Score Field Mapping)
- Reuse EBM-splats ScoreNetwork + Langevin sampler
- Energy function: E(x) = E_splats + λ_align*E_contrastive + λ_div*E_diversity
- Langevin maps the density landscape of the embedding space
- Output: Score field (gradient of log-probability at every point)

### Phase 2: Pareto Filtering (Multi-Objective Selection)
- For each sampled trajectory, compute:
  - Objective A: Embedding quality (alignment score between modalities)
  - Objective B: Information diversity (spread across the sphere)
  - Objective C: Reconstruction fidelity (can we decode back?)
- Non-dominated sorting → keep only Pareto frontier
- Output: "Golden trajectories" — optimal paths from noise to embeddings

### Phase 3: Flow Matching (Deterministic Distillation)
- Train a conditional Flow Matching model (OT-CFM)
- Learns the vector field: v(x,t,c) that maps noise → Pareto-optimal embeddings
- Conditioned on modality (text/image/audio) + content
- At inference: single ODE step → high-quality embedding
- Output: Fast, deterministic embedding generator

## Model Components

### Encoders (project to 640D sphere)
- Text: nn.Embedding(50257, 640) + TransformerEncoder (6 layers) → normalize to S^639
- Image: CLIP ViT-B/32 encoder → Linear(512, 640) → normalize to S^639
- Audio: Whisper base encoder → Linear(768, 640) → normalize to S^639

### Shared Core (from EBM-splats)
- geometry.py: Riemannian ops (exp_map, log_map, tangent projection)
- splats.py: SplatStore (KNN attractors, cross-modal concepts)
- score_network.py: Score field predictor
- langevin.py: Underdamped sampler on S^639

### New Components
- pareto_filter.py: Multi-objective ranking + non-dominated sorting
- flow_matching.py: OT-CFM with conditional vector field
- contrastive_head.py: Cross-modal alignment loss (InfoNCE on hypersphere)
- pglf_trainer.py: Orchestrates the 3-phase training loop

## Integration with M2M-Rust
- Export trained Flow Matching model to ONNX
- Python embedding service (HTTP/gRPC) that M2M MCP calls
- Or: embed in Hermes as a pre-computation step
- Dimension: 640 (same as M2M's current dim)

## Training Strategy
1. Phase 0: Pre-train text encoder with DSM (reuse autoresearch checkpoints)
2. Phase 1: Add contrastive loss, run Langevin exploration
3. Phase 2: Filter with Pareto, collect golden trajectories
4. Phase 3: Train Flow Matching on golden trajectories
5. Iterative: Re-run Phase 1-3 with improved energy landscape

## Hardware Budget (RTX 3090 24GB)
- Text encoder (Transformer 6L 640D): ~40M params, ~2GB VRAM
- CLIP ViT-B/32 (frozen): ~90M params, ~1.5GB VRAM
- Whisper base (frozen): ~75M params, ~1.5GB VRAM
- ScoreNetwork (12L 4096H): ~50M params, ~3GB VRAM
- Flow Matching (6L 640D): ~20M params, ~1GB VRAM
- SplatStore (100K splats): ~250K params, ~0.1GB VRAM
- Training overhead (gradients, optimizer, batches): ~10GB
- Total: ~19GB → fits in 24GB with AMP
