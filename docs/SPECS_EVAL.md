# SPECS_EVAL_EBM.md — Document Evaluation for EBM-splats

**Generated:** 2026-03-20  
**Target:** [EBM-splats](https://github.com/schwabauerbriantomas-gif/EBM-splats)  
**Repo commit:** Latest (cloned 2026-03-20)  
**Local workspace:** `C:\Users\Brian\.openclaw\workspace\projects\ebm\`  
**Documents analyzed:** 21 PDFs from `D:\a evaluar y considerar\`

---

## Repo vs Local Workspace Comparison

The GitHub repo contains **27 files** (Python + shaders). The local workspace contains **38 Python files** — **11 files NOT in the repo**:

| File | Status | Description |
|------|--------|-------------|
| `score_network.py` | **Local only** | Direct parametric score model replacing broken autograd approach |
| `train_ebm_optimized.py` | **Local only** | 4-phase training (DSM → Rectified Flow → Reflow → Benchmark) with EMA, β-annealing |
| `energy_cuda.py` | **Local only** | CUDA-native energy function replacing Vulkan dependency |
| `train_cuda.py` | **Local only** | CUDA training variant |
| `train_scorematching.py` | **Local only** | Score matching training |
| `train_rectified_flow.py` | **Local only** | Rectified flow training |
| `train_tinystories.py` | **Local only** | Training on TinyStories dataset |
| `train_tinystories_fast.py` | **Local only** | Fast TinyStories training |
| `train_tinystories_streaming.py` | **Local only** | Streaming TinyStories training |
| `generate_samples.py` | **Local only** | Sample generation |
| `dataset_loader.py` | **Local only** | Alternative dataset loader |

**Key finding:** The local workspace has significant advancements (ScoreNetwork, CUDA energy, rectified flow, optimized training) that are NOT pushed to the GitHub repo. The repo is **outdated** relative to the workspace.

---

## Current Code State Summary (Repo)

- **Architecture:** EBM on S^639 hypersphere, Gaussian Splats as attractors, Langevin underdamped sampling
- **Training:** Denoising score matching on WikiText-103, batch size 16-32, AdamW
- **Energy:** E(x) = E_splats + λ_geom·E_geom + λ_comp·E_comp (logsumexp over KNN neighbors)
- **Splats:** FAISS-CPU KNN, max 100K, learnable μ/α/κ
- **Vulkan:** Simulated (falls back to CPU tensor ops — not real GPU compute)
- **SOC:** HistoryBuffer + order parameter, adds splats at criticality
- **Decoder:** MoE (4 experts, 2 active)
- **Score computation:** Uses `torch.autograd.grad` on energy (identified as broken in local workspace)

**Critical issues in repo code:**
1. Vulkan engine is simulated, not real GPU compute
2. Score matching uses broken autograd approach (fixed in local `score_network.py`)
3. `config.py` has duplicate field `init_alpha` (defined twice)
4. `splats.py` `add_splat` has swapped alpha/kappa logic
5. `dataset_utils.py` returns tuple `(dataloader, tokenizer)` but `train.py` expects just `dataloader`
6. No actual convergence achieved (training started but no perplexity results)

---

## Document-by-Document Evaluation

---

### 1. EBM_Composicionalidad_Analisis.pdf

**Summary:** Deep analysis of compositionality in the EBM system. Evaluates lexical, syntactic, and semantic compositionality. Finds the current model has limited, mostly implicit compositional capabilities with clear architectural improvement opportunities.

**Applicability: HIGH**  
The model's `CompositionalEngine` in `model.py` and `compute_comp_energy` in `energy.py` are rudimentary. This document provides the theoretical framework for making compositionality explicit and robust.

**Actionable Items:**
- Redesign `CompositionalEngine` from simple MLP to tensor-product decomposition (TPD) or HELM architecture
- Add explicit syntactic composition operators (merge, select, shift) per Section 3.2
- Implement compositional scoring metrics for evaluation
- Add hierarchical composition for multi-token sequences

**Priority:** P1  
**Dependencies:** ScoreNetwork must be working first; basic training convergence needed  
**Estimated effort:** Large  
**Overlap:** `model.py` (CompositionalEngine), `energy.py` (compute_comp_energy), `decoder.py` (MoE context)

---

### 2. EBM_Costo_Beneficio_Composicionalidad.pdf

**Summary:** Compares training from scratch vs compositional-assisted training. Quantifies FLOPs, GPU time, energy, and memory. Concludes compositional strategy has significantly better ROI for consumer hardware.

**Applicability: MEDIUM**  
Useful for planning but assumes hardware that doesn't match reality (doc references RTX 3070, actual GPU is AMD RX 6650XT with no CUDA). Numbers need adjustment for AMD Vulkan compute.

**Actionable Items:**
- Recalculate FLOPs estimates for AMD RX 6650XT (no Tensor Cores, Vulkan not CUDA)
- Use as budget justification framework when planning training phases
- Consider the compositional training strategy as phase ordering guide

**Priority:** P2  
**Dependencies:** None (planning document)  
**Estimated effort:** Small  
**Overlap:** None directly — planning document

---

### 3. EBM_Implementacion_Especificacion_Completa.pdf

**Summary:** Complete 20-section technical specification for the EBM system. Covers latent space, energy function, splats, Riemannian ops, score matching, Langevin, SOC, context hierarchy, MoE decoder, compositional extensions, APIs, hyperparameters, training/inference pipelines, dependencies, implementation phases, and evaluation metrics.

**Applicability: HIGH**  
This is the **canonical spec** for the entire project. The repo's `spec.txt` is an abbreviated version. This PDF contains the full mathematical derivations and implementation details that are partially missing from the codebase.

**Actionable Items:**
- Compare spec Sections 2-8 against actual code — several spec requirements are not implemented:
  - Sec 8: Context hierarchy (3 levels) — not implemented
  - Sec 9: Hierarchical context manager — not implemented  
  - Sec 14: Many hyperparameters in spec are not in `config.py`
  - Sec 18: Phase implementation plan — only Phase 1 partially done
- Implement missing contextual attention mechanism (Sec 9)
- Add proper evaluation metrics pipeline (Sec 19)
- Spec references CUDA 11.8+ but actual code uses Vulkan — resolve discrepancy

**Priority:** P0  
**Dependencies:** None (this IS the spec)  
**Estimated effort:** Large (many missing components)  
**Overlap:** ALL files — this is the reference spec for the entire codebase

---

### 4. EBM_Sistema_V2_Especificacion_Tecnica.pdf

**Summary:** V2 revised technical specification with corrections to the original. Fixes sign conventions in energy function, corrects score matching formulations, improves decoder with context handling, adds hierarchical context system, complete evaluation metrics, error recovery, and explicit decision rules for SOC consolidation.

**Applicability: HIGH**  
Critical corrections to the V1 spec. The sign convention fix (E(x) related to p(x) via exp(-E(x))/Z) is essential — the repo code has inconsistent sign handling between `compute_score` (returns -grad_R) and `forward` (returns energy directly).

**Actionable Items:**
- Fix sign convention throughout: ensure p(x) ∝ exp(-E(x)) consistently
- Implement the corrected score matching derivation (Sec 2.3)
- Add context-capable decoder (Sec 4)
- Implement hierarchical context system (Sec 5)
- Add error recovery and gradient safety mechanisms (Sec 7)

**Priority:** P0  
**Dependencies:** None (supersedes V1 spec)  
**Estimated effort:** Large  
**Overlap:** `energy.py`, `model.py`, `decoder.py`, `config.py`, `soc.py`

---

### 5. EBM_V2_Analisis_Compatibilidad_Final.pdf

**Summary:** Categorizes 15 proposed methods (8 ML, 7 physics/math) and analyzes their pairwise compatibility. Identifies optimal method combinations: EDLM+Consistency Models (quality), HOMER+SOC (context), Hamiltonian+Underdamped Langevin (sampling).

**Applicability: MEDIUM**  
Good roadmap for which improvements to implement together. The compatibility matrix prevents implementing conflicting methods.

**Actionable Items:**
- Follow the recommended combination priorities from the compatibility matrix
- EDLM + Consistency Models should be the first ML improvement batch
- Hamiltonian integrator should replace current Störmer-Verlet when implementing
- Avoid implementing conflicting pairs (e.g., Cyclical Noise + Consistency Models)

**Priority:** P2  
**Dependencies:** Basic convergence first  
**Estimated effort:** Small (planning document)  
**Overlap:** `langevin.py` (integrator choice), `energy.py` (EDLM energy)

---

### 6. EBM_V2_Datasets_Especificacion.pdf

**Summary:** Specifies datasets for initialization, training, evaluation, and auxiliary components. Defines 4 categories: embeddings base (~1M samples), training (10M-1B), evaluation (100K-1M), auxiliary (variable). Includes specific dataset recommendations per phase.

**Applicability: HIGH**  
The repo currently only uses WikiText-103 with 100K subset. The spec calls for much richer data strategy including semantic initialization from pre-trained models.

**Actionable Items:**
- Implement Dataset of Base Embeddings (DEB) — use Phi-3-mini or GPT-2 embeddings to initialize SplatStore (already partially in config `init_from_vocab_embeddings`)
- Add evaluation datasets beyond WikiText (perplexity benchmarks)
- Implement domain-specific training datasets for geometry specialization
- Add auxiliary datasets for decoder pre-training

**Priority:** P1  
**Dependencies:** `splats.py` (initialization), `dataset_utils.py` (loader)  
**Estimated effort:** Medium  
**Overlap:** `dataset_utils.py`, `splats.py`, `config.py`

---

### 7. EBM_V2_Estimacion_Tiempos_Hardware.pdf

**Summary:** Training time estimates for RTX 3070 + Ryzen 5 3400G + SSD. Calculates FLOPs per operation, GPU utilization, and total time estimates.

**Applicability: LOW**  
References RTX 3070 (CUDA) but actual hardware is AMD RX 6650XT (Vulkan, no CUDA). All performance estimates are invalid for the actual hardware. Vulkan compute has different throughput characteristics.

**Actionable Items:**
- Recalculate all estimates for AMD RX 6650XT (22 TFLOPS FP32, 8GB VRAM, no Tensor Cores)
- Vulkan compute shaders need separate benchmarking (not CUDA cores)
- Consider that the Vulkan engine in the repo is simulated (CPU fallback) — actual GPU time unknown

**Priority:** P3  
**Dependencies:** None  
**Estimated effort:** Small  
**Overlap:** None

---

### 8. EBM_V2_Fisica_Matematicas_Aplicadas.pdf

**Summary:** Deep dive into physics/math concepts applicable to EBM: Free Energy Principle (Friston), Hamiltonian Neural Networks, Underdamped Langevin (5-10x acceleration), Self-Organized Criticality, Ricci Flow for manifold evolution, Natural Gradient/Fisher Information optimization.

**Applicability: MEDIUM**  
Theoretical foundation document. Some concepts are already implemented (Langevin underdamped, SOC), others are aspirational (Ricci Flow, Hamiltonian NN). Good for understanding the theoretical framework.

**Actionable Items:**
- Implement GAUL correction for Langevin sampling (Sec 4) — corrects for manifold curvature
- Consider Ricci Flow for adaptive manifold geometry (Sec 5.3) — advanced
- Natural gradient via Fisher Information for optimizer (Sec 6) — medium effort
- Free Energy Principle formulation for unified training objective (Sec 2)

**Priority:** P2  
**Dependencies:** Basic convergence, stable Langevin sampling  
**Estimated effort:** Medium  
**Overlap:** `langevin.py` (GAUL correction), `energy.py` (FEP formulation)

---

### 9. EBM_V2_Mejoras_Optimizaciones.pdf

**Summary:** Survey of 2024-2025 papers for improvements: Consistency Models (1-2 steps, 10-100x speedup), EDLM (EBM+Diffusion integration), B+ANN for billion-scale search, Riemannian optimization, HOMER for hierarchical context, Hierarchical WSD for polysemy.

**Applicability: HIGH**  
Consistency Models are the single highest-impact improvement identified — could reduce Langevin steps from 200 to 1-2. EDLM integration could fundamentally improve generation quality. These are concrete, implementable improvements from peer-reviewed papers.

**Actionable Items:**
- **Consistency Models**: Implement consistency training + distilled sampler for 10-100x inference speedup
- **EDLM**: Integrate diffusion bridge into energy function for improved generation
- **B+ANN/HARMONY**: Replace FAISS-CPU with ANN for O(log N) search (also shared with M2M)
- **Riemannian optimization**: Replace standard Adam with Riemannian Adam on S^639
- Add paper references to code documentation

**Priority:** P1  
**Dependencies:** Stable base training, ScoreNetwork working  
**Estimated effort:** Large (Consistency Models), Medium (others)  
**Overlap:** `langevin.py` (Consistency sampler), `energy.py` (EDLM), `splats.py` (B+ANN)

---

### 10. inferencia_activa_ebm_splats.pdf

**Summary:** Integrates Friston's Free Energy Principle with EBM-splats architecture. Maps perception-action cycle to energy minimization and generation. Proposes active inference agents that can act proactively, not just react to queries.

**Applicability: MEDIUM**  
Conceptually important but operationally the mapping is already implicit (Langevin = variational inference, energy = expected energy). The active inference agent concept is forward-looking.

**Actionable Items:**
- Add explicit FEP formulation to training objective (combine expected energy + KL divergence)
- Implement perception-action cycle for interactive generation
- Add "surprise" metric (negative log probability) as monitoring signal
- Design agent architecture where EBM state encodes beliefs about the world

**Priority:** P3  
**Dependencies:** Stable base system, basic convergence  
**Estimated effort:** Large  
**Overlap:** `energy.py` (FEP energy), `model.py` (agent architecture)

---

### 11. Analisis_Sistema_EBM_Criticidad_Geometrica.pdf

**Summary:** Technical verification analysis (in Chinese) identifying critical issues: sign convention problems in energy function, boundary condition ambiguity, numerical instability in normalization, gradient direction instability near splat centers, missing implementation details. Rates theoretical consistency as "Medium" and engineering completeness as "Insufficient".

**Applicability: HIGH**  
This document identifies **real bugs** in the current code:
1. Energy function sign convention inconsistency
2. Score matching formula needs sign reversal clarification  
3. Gradient instability near splat centers → need gradient clipping/regularization
4. Numerical instability in normalization after each Langevin step

**Actionable Items:**
- **Fix energy sign convention** — document and enforce p(x) ∝ exp(-E(x)) everywhere
- **Add gradient clipping** in `langevin.py` (already `grad_clip: float = 1.0` in config but not used in langevin.py)
- **Add adaptive noise levels** based on local density (high density → small σ, sparse → large σ)
- **Implement numerical stability** for normalization (clamp before normalize)
- Add proper evaluation metrics (perplexity, coverage, etc.)

**Priority:** P0  
**Dependencies:** None (bug fixes)  
**Estimated effort:** Medium  
**Overlap:** `energy.py`, `langevin.py`, `config.py`, `model.py`

---

### 12. integraciones_teoricas_ebm_splats.pdf

**Summary:** Synthesis of 10 analysis documents, identifying three theoretical pillars: Aragon's geometric paradigm, Friston's Free Energy Principle, and Gaussian Splats on Riemannian hyperspheres. Argues these are different manifestations of the same underlying principle.

**Applicability: MEDIUM**  
Provides high-level theoretical coherence. The unified manifold concept (M) as simultaneously FEP state space, Aragon's knowledge geometry, and EBM's S^639 is the key insight.

**Actionable Items:**
- Use the unified manifold perspective in code documentation
- Implement MACSL (continuous learning mechanism) from Aragon's framework
- Design experiments that validate the geometric knowledge hypothesis
- Consider the manifold unification when designing transfer learning

**Priority:** P3  
**Dependencies:** None (conceptual framework)  
**Estimated effort:** Small (documentation), Large (MACSL implementation)  
**Overlap:** Conceptual — informs architecture decisions across all files

---

### 13. requisitos_hardware_ebm_splats.pdf

**Summary:** Hardware requirements analysis ranking components by priority: GPU VRAM (critical), RAM (high), CPU (moderate), SSD (moderate), HDD (low). Specifies minimum 6GB VRAM, recommended 8-12GB.

**Applicability: MEDIUM**  
Validates that the current AMD RX 6650XT (8GB) meets recommended requirements. However, the analysis assumes CUDA GPU, not AMD Vulkan.

**Actionable Items:**
- Verify Vulkan compute performance matches CUDA estimates for RX 6650XT
- Profile actual VRAM usage per component (splats, gradients, activations)
- Consider memory optimization (gradient checkpointing, mixed precision)
- Note: actual Vulkan engine is simulated — real GPU memory usage unknown

**Priority:** P2  
**Dependencies:** Real Vulkan implementation  
**Estimated effort:** Small  
**Overlap:** `config.py`, `vulkan_engine.py`

---

### 14. analisis_geometria_ebm_splats.pdf

**Summary:** Applies Richard Aragon's "Geometry Beneath the Weights" paradigm to EBM-splats. Knowledge = geometric organization, learning = spatial transformation. Proposes treating the latent space as an explorable universe rather than an optimization artifact.

**Applicability: MEDIUM**  
Philosophical/architectural direction document. The key actionable insight is that splat organization itself IS the knowledge representation, and interventions should target geometry, not just weights.

**Actionable Items:**
- Implement geometric diagnostics: measure manifold curvature, spectral properties
- Add splat distribution visualization as training monitor
- Design geometric interventions (splat merging, splitting, warping) as explicit operations
- Consider VIVERE-style spectral compression for knowledge transfer

**Priority:** P2  
**Dependencies:** Basic system working  
**Estimated effort:** Medium  
**Overlap:** `splats.py`, `soc.py`, `geometry.py`

---

### 15. arquitectura_transferencia_geometrica_unificada.pdf

**Summary:** Proposes replacing the "separate geometries" paradigm (.geom files) with a unified shared manifold where all knowledge domains coexist as regions. Transfer learning becomes a native operation (region mapping) rather than file-based transfer.

**Applicability: MEDIUM**  
Important architectural vision for the future. Currently the system uses a single SplatStore, which already approximates this. The key advancement would be explicit region-based management and cross-domain transfer.

**Actionable Items:**
- Add region metadata to splats (domain label, importance score)
- Implement region-based attention in energy computation
- Design domain transfer as geometric transformation (rotation/warping in tangent space)
- Add transfer learning API: `transfer_geometry(source_domain, target_domain)`

**Priority:** P2  
**Dependencies:** Basic training convergence, domain-specific training  
**Estimated effort:** Large  
**Overlap:** `splats.py` (region metadata), `energy.py` (region-aware energy), `model.py` (transfer API)

---

### 16. geometrias_compatibilidad_integracion.pdf

**Summary:** Technical specifications for knowledge geometry interoperability: .geom packaging format, 3-level validation (structural/semantic/functional), spectral conflict detection, compatibility scoring.

**Applicability: LOW**  
The unified manifold architecture (doc #15) supersedes the .geom file approach. This document's validation concepts (spectral analysis, structural checks) are still useful.

**Actionable Items:**
- Extract spectral validation concepts for use in unified manifold
- Implement eigenvalue-based compatibility checking when merging splat regions
- Keep manifest.json concept for versioning knowledge regions within unified manifold

**Priority:** P3  
**Dependencies:** Unified manifold architecture  
**Estimated effort:** Small  
**Overlap:** `splats.py` (validation)

---

### 17. estrategia_geometrias_conocimiento.pdf

**Summary:** Monetization strategy for "knowledge geometries" — compressed geometric structures as products. Compares traditional model-as-API vs geometry-as-product business models. Proposes selling compressed splat stores as transferable knowledge packages.

**Applicability: LOW**  
Business strategy document, not technical. Interesting vision but premature — needs working product first.

**Actionable Items:**
- Design splat compression/export format for future productization
- Consider VIVERE-style PNG spectral compression as ultra-compact transfer
- Ensure SplatStore serialization is clean and portable

**Priority:** P3  
**Dependencies:** Working product  
**Estimated effort:** Small  
**Overlap:** `splats.py` (serialization)

---

### 18. transfer_learning_geometrias.pdf

**Summary:** Transfer learning between knowledge geometries. Proposes 4 methods: initialization by projection, fine-tuning with geometric constraints, splat merging, and adaptive interpolation. Goal: learn Go from Python geometry in hours not weeks.

**Applicability: MEDIUM**  
Concrete technical proposal for a key feature. The 4 methods are implementable but require the base system to work first.

**Actionable Items:**
- Implement Method 1 (initialization by projection) — map equivalent tokens between languages
- Add geometric constraint loss for fine-tuning (Method 2)
- Implement splat merging algorithm with distance thresholds (Method 3)
- Design adaptive interpolation based on domain similarity (Method 4)

**Priority:** P2

**Dependencies:** Basic training convergence, domain-specific geometries  
**Estimated effort:** Medium (Method 1), Large (all methods)  
**Overlap:** \splats.py\ (merging/projection), \	rain.py\ (fine-tuning), \energy.py\ (constraint loss)

---

### 19. analisis_tiempo_entrenamiento_mvp.pdf

**Summary:** Training time estimates for MVP on RTX 3090 (24GB). 5 phases: Setup+Refactor, Transfer Learning, Domain Specific, Optimization, API+Deploy. Total: 8-12 weeks, 200-330h GPU.

**Applicability: MEDIUM**  
Good project planning but references RTX 3090 (24GB) vs actual AMD RX 6650XT (8GB). Batch sizes and VRAM assumptions optimistic.

**Actionable Items:**
- Use 5-phase framework as project structure
- Reduce batch size for 8GB VRAM
- Account for Vulkan vs CUDA overhead

**Priority:** P2 | **Effort:** Small | **Overlap:** None

---

### 20. guia_implementacion_tecnica.pdf

**Summary:** Practical guide recommending hybrid approach: SOTA embeddings (Phi-3-mini/Qwen) → normalize to S^639 → initialize SplatStore → EBM fine-tune on domain data. Already partially implemented.

**Applicability: HIGH**  
Most actionable document. The hybrid initialization strategy is the clear path forward.

**Actionable Items:**
- Implement Phi-3-mini or Qwen embedding initialization
- Follow 5-step hybrid process from Table 2
- Implement recommended evaluation pipeline

**Priority:** P1 | **Effort:** Medium | **Overlap:** config.py, splats.py, model.py

---

### 21. consejo_agentes_iso9001_2025.pdf

**Summary:** Multi-agent ISO 9001 quality management system on EBM-splats with active inference. Specialized agents with individual knowledge geometries on shared manifold.

**Applicability: LOW**  
Domain-specific application. Multi-agent pattern interesting but not core EBM work.

**Priority:** P3 | **Effort:** Small | **Overlap:** None

---

## Priority Matrix

### P0 — Critical
1. **#11** Criticidad Geometrica — Fix energy sign, gradient clipping → energy.py, langevin.py
2. **#3** Implementacion Completa — Missing spec components → Multiple
3. **#4** V2 Especificacion — Apply V2 corrections → energy.py, model.py

### P1 — High
4. **#1** Composicionalidad — Redesign compositional engine → model.py
5. **#6** Datasets — Rich dataset strategy → dataset_utils.py
6. **#9** Mejoras — Consistency Models, EDLM → langevin.py
7. **#20** Guia Tecnica — Hybrid SOTA initialization → splats.py

### P2 — Medium
8-14. Docs #14, #15, #18, #5, #8, #13, #19 — Geometric diagnostics, unified manifold, transfer learning, method roadmap, GAUL, Vulkan validation, project planning

### P3 — Low
15-21. Docs #2, #7, #10, #12, #16, #17, #21 — Budget planning, invalid estimates, active inference, theory, .geom format, monetization, ISO case study

---

## Key Discrepancies

### Code vs Spec
1. Context hierarchy (3 levels) — NOT implemented
2. Score matching — spec says denoising, repo uses simple energy minimization
3. Vulkan engine — simulated CPU fallback, not real GPU compute
4. Training pipeline — basic loop vs spec multi-phase curriculum
5. Evaluation metrics — spec defines many; repo has none working

### Repo vs Local Workspace
**Local workspace is significantly ahead:** score_network.py (fixes broken autograd), train_ebm_optimized.py (4-phase with EMA/reflow), energy_cuda.py (real CUDA), multiple training variants.
**Recommendation: Push to repo immediately.**

### Hardware Mismatch
Docs reference RTX 3070/3090 (CUDA, Tensor Cores). Actual: AMD RX 6650XT (Vulkan, no CUDA). Invalidates all FLOPs, batch size, memory, and time estimates.

---

## Recommended Action Plan

**Phase 1: Fix Foundation (Week 1-2)**
1. Push local workspace code to repo
2. Fix energy sign convention (#4, #11)
3. Add gradient clipping to Langevin
4. Fix config duplicates and splats.py bugs

**Phase 2: Converge (Week 2-4)**
1. Switch to ScoreNetwork for proper DSM
2. Hybrid SOTA initialization (Phi-3-mini)
3. Expand datasets beyond WikiText-103
4. Implement evaluation metrics

**Phase 3: Accelerate (Week 4-6)**
1. Consistency Models for 10-100x speedup
2. B+ANN replacing FAISS-CPU
3. EDLM integration
4. Real GPU compute (Vulkan or CUDA)

**Phase 4: Advance (Week 6+)**
1. Compositional engine redesign
2. Unified manifold architecture
3. Transfer learning between geometries
4. Context hierarchy implementation

---
*21 documents evaluated. Repo cloned and compared against local workspace.*
