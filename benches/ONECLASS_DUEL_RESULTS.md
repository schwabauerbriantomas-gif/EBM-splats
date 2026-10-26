# One-class scoring duel — EBM-splats vs classical alternatives

Reproducible with `benches/oneclass_duel.py` (seed 42).

**Setup:** all-MiniLM-L6-v2 (384d, normalized) · 20 Newsgroups, 18,846 docs ·
20 one-class tasks (fit on 70% of class C only — no negative labels) ·
test: positives (30% of C) vs 2x sample of the rest · AUROC per class,
mean ± std over 20 classes · per-document latency, batch=1, CPU-only ·
EBM fit: k-means (k<=256) over the positive embeddings, alpha=5.0 const,
kappa=cluster size — no score-network training.

## Results

| method | AUROC (mean±std) | min | max | best/class | p50 ms/doc | p95 ms/doc | model size/task |
|---|---|---|---|---|---|---|---|
| **ebm-splats** | **0.9585 ± 0.0181** | 0.920 | 0.987 | **10/20** | 0.45 | 0.60 | 386 KB |
| knn-density | 0.9584 ± 0.0200 | 0.916 | 0.987 | 9/20 | 1.75 | 2.00 | 802 KB |
| mahalanobis (LW) | 0.9523 ± 0.0201 | 0.911 | 0.982 | 1/20 | 5.26 | 7.15 | 578 KB |
| gmm (diag, 64) | 0.9439 ± 0.0228 | 0.888 | 0.981 | 0/20 | 0.38 | 0.45 | 192 KB |
| centroid | 0.9324 ± 0.0339 | 0.842 | 0.976 | 0/20 | <0.01 | <0.01 | 2 KB |
| iforest | 0.8828 ± 0.0511 | 0.786 | 0.963 | 0/20 | 18.71 | 21.35 | 200 KB |

## Verdict

1. **vs kNN-density: statistical tie on quality** (Δ=0.0001), won on operation:
   **3.9x faster on CPU** and constant O(k·d) memory vs kNN's O(n·d) that grows
   with the corpus (802 KB at ~600 docs — the gap widens linearly).
2. **vs GMM** (its closest classical equivalent): clear AUROC win (+0.0146) at
   comparable latency.
3. **vs Mahalanobis**: +0.006 AUROC and **11.7x faster** (per-doc 384x384
   inversion dominates). **vs IsolationForest**: +0.076 AUROC, 42x faster.
4. Unique to the splat model (not measured here, by construction): generates
   samples from the fitted corpus (RF / Langevin), composes corpora by summing
   energy fields, and grows incrementally by adding splats — no retraining.
5. Supervised classifiers with labels beat every one-class method where labels
   exist; this duel only ranks the label-free league. Generic-text results may
   not transfer to heavily imbalanced domains.

## Calibration finding (applies to any zero-training splat fit)

The first run scored **AUROC 0.06 — systematically inverted** — with
alpha = ln(cluster size), the "natural" calibration: small clusters got
alpha ~ 0, the distance term vanished, and the score degenerated to ranking
by splat weight. Positive docs land near small, specific splats and received
MORE energy than negatives. With **alpha held constant at 5.0** the distance
term dominates and AUROC jumps 0.06 -> 0.96; the kappa importance weighting
of the neighbor window stops mattering once alpha is large enough. If you fit
splats without training, use a constant alpha large enough that
alpha/temperature >> 1 across the (x·mu - 1) range.
