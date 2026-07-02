# Phase 2: Energy-Guided Generation with Rectified Flow

**Date:** July 2, 2026
**Hardware:** RTX 3090 24GB, PyTorch 2.6, CUDA 12.4

---

## Hypothesis

Manipulating the energy of specific splats allows controlling what type of content is generated. Boosting energy toward topic A + suppressing energy of topic B should produce samples that retrieve texts from A.

## Setup

1. **Data:** 10K TinyStories embedded with MiniLM-L6-v2 (384D, normalized to S^383)
2. **Clusters:** 50 topics via KMeans, selected 2 well-separated ones (sim=0.27)
   - Topic A (cluster 25): stories about parks, birds, children playing (126 stories)
   - Topic B (cluster 27): stories with explicit morals (57 stories)
3. **Splats:** 50 cluster centers as splat centers
4. **RF:** VelocityNet (3-layer MLP, 512 hidden), trained 1000 steps (5.9s)
5. **Sampling:** 500 samples per condition, 2 RF steps, decode via nearest-neighbor retrieval
6. **Guidance:** boost/suppress direction toward/away from cluster centers in tangent space

## Results

| Guidance Scale | Boost A (% in A) | Boost B (% in B) | Baseline (% in A) |
|---|---|---|---|
| 0.0 (baseline) | — | — | 0.6% |
| 0.5 | 62.8% | 99.2% | — |
| **1.0** | **99.6%** | **100.0%** | — |
| **2.0** | **100.0%** | **100.0%** | — |
| 5.0 | 4.4% | 8.2% | — |

## Analysis

### Works
- **gs=1.0 to 2.0: perfect control** (99.6-100% of samples in the target topic)
- From baseline 0.6% to 100% with a single boost direction
- Control works in both directions (boost A and boost B)
- RF with 2 steps + guidance = instant sampling

### Limitations
- **gs=5.0 collapses**: too much guidance pushes samples off the data manifold
- **Decode is retrieval**: does not generate new text, retrieves existing texts from corpus
- **2 topics tested**: needs validation with more pairs and subtler topics
- **Small corpus**: 10K stories limits retrieval diversity

### Required next steps
1. **Neural decoder**: train a decoder that maps samples -> text (not retrieval)
2. **Composition**: boost A + boost B simultaneously -> does it generate content at the intersection?
3. **Continuous control**: guidance_scale as a continuous variable -> interpolation between topics
4. **More topics**: validate with 10+ clusters, not just 2
5. **Larger corpus**: 100K+ stories for better coverage

## Verdict

**Energy-guided generation via EBM + RF works.** This is the first positive result of the project. The combination of energy landscape + Rectified Flow enables controlling which region of the semantic space is sampled, in real time, with 100% accuracy.

The next critical step is replacing retrieval-based decoding with a real neural decoder, so the system generates new text instead of retrieving existing text.
