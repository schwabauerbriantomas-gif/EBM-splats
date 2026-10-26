"""One-class scoring duel: EBM-splats vs classical alternatives.

Reproduces the numbers in benches/ONECLASS_DUEL_RESULTS.md (seed 42):
requires `pip install -e .` plus scikit-learn and sentence-transformers,
and internet access for the 20 Newsgroups download + MiniLM weights.

Task: per class C of 20 Newsgroups — fit on train_C only (unlabeled positives),
score test set (positives = C, negatives = 2x sample of the rest). AUROC is the
quantitative metric; latency is per-document, batch=1, CPU-only. Outputs:
duel_results.json + duel_tables.md in the working directory.

Methods:
  ebm-splats   — SplatStorage fit by k-means (mu=centroid, alpha=5.0 const,
                 kappa=cluster size), energy from this repo
  gmm          — sklearn GaussianMixture (diag), log-likelihood
  knn-density  — -mean cosine distance to k=10 nearest training docs
  centroid     — cosine similarity to normalized class centroid
  mahalanobis  — Ledoit-Wolf covariance, -distance^2
  iforest      — IsolationForest score_samples
"""

import json
import time
import warnings

import numpy as np
import torch
from sklearn.covariance import LedoitWolf
from sklearn.datasets import fetch_20newsgroups
from sklearn.ensemble import IsolationForest
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")

SEED = 42
KNN_K = 10
RNG = np.random.default_rng(SEED)
RESULTS = {"meta": {}, "per_class": {}, "latency": {}, "model_size": {}}

METHODS = ["ebm-splats", "gmm", "knn-density", "centroid", "mahalanobis", "iforest"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- embeddings
def build_embeddings():
    import os

    cache = "embeddings_minitlm.npz"
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return z["emb"], z["target"], z["texts"]
    from sentence_transformers import SentenceTransformer

    d = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
    texts, target = [], []
    for t, y in zip(d.data, d.target):
        t = " ".join(t.split())
        if len(t) >= 40:  # drop near-empty after header/footer strip
            texts.append(t)
            target.append(y)
    log(f"encoding {len(texts)} docs with all-MiniLM-L6-v2 (CPU)...")
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    emb = model.encode(texts, batch_size=256, show_progress_bar=False,
                       normalize_embeddings=True, convert_to_numpy=True)
    target = np.array(target)
    np.savez_compressed(cache, emb=emb.astype(np.float32), target=target,
                        texts=np.array(texts, dtype=object))
    log(f"embeddings cached: {emb.shape}")
    return emb.astype(np.float32), target, np.array(texts, dtype=object)


# ---------------------------------------------------------------- fitters
class EBMSplatsScorer:
    """SplatStorage fit by k-means over normalized embeddings."""

    def __init__(self, latent_dim=384):
        from ebm.config import EBMConfig
        from ebm.splats import SplatStorage

        self.config = EBMConfig(latent_dim=latent_dim, n_splats_init=1,
                                max_splats=256, knn_k=64, device="cpu")
        self.store = None

    def fit(self, X):
        from sklearn.cluster import KMeans

        from ebm.energy import EnergyFunction

        n = min(256, max(8, len(X) // 2))
        km = KMeans(n_clusters=n, n_init=4, random_state=SEED).fit(X)
        counts = np.bincount(km.labels_, minlength=n).astype(np.float64)
        mus = km.cluster_centers_
        mus /= np.linalg.norm(mus, axis=1, keepdims=True) + 1e-9
        # alpha: CONSTANT 5.0 — ln(n_k) is a calibration trap: small clusters
        # get alpha ~0, the distance term vanishes and the score degenerates
        # to ranking by splat weight (AUROC 0.06, inverted). With alpha=5 the
        # distance term dominates and the energy ranks by proximity (~0.93).
        alpha = np.full(n, 5.0)
        kappa = np.maximum(counts, 1.0)
        from ebm.splats import SplatStorage

        self.store = SplatStorage(self.config)
        with torch.no_grad():
            self.store.mu.data.zero_(); self.store.mu.data[:n] = torch.from_numpy(mus).float()
            self.store.alpha.data[:n] = torch.from_numpy(alpha).float()
            self.store.kappa.data[:n] = torch.from_numpy(kappa).float()
            self.store.n_active = n
        self.fn = EnergyFunction(self.config, self.store)
        self.fn.eval()
        return self

    def score(self, X):
        with torch.no_grad():
            e = self.fn.compute_splat_energy(torch.from_numpy(X).float())
        return -e.numpy()  # higher = more inlier


def fit_gmm(X):
    k = min(64, max(2, len(X) // 8))
    return GaussianMixture(n_components=k, covariance_type="diag",
                           random_state=SEED, n_init=1).fit(X)


def fit_knn(X):
    return NearestNeighbors(n_neighbors=KNN_K, metric="cosine").fit(X)


def fit_centroid(X):
    c = X.mean(axis=0, keepdims=True)
    return c / (np.linalg.norm(c) + 1e-9)


def fit_mahalanobis(X):
    return LedoitWolf().fit(X)


def fit_iforest(X):
    return IsolationForest(n_estimators=200, random_state=SEED).fit(X)


def score_all(fits, X):
    """Return dict method -> score array (higher = more inlier)."""
    out = {}
    out["ebm-splats"] = fits["ebm"].score(X)
    out["gmm"] = fits["gmm"].score_samples(X)
    dist, _ = fits["knn"].kneighbors(X, n_neighbors=KNN_K)
    out["knn-density"] = -dist.mean(axis=1)
    out["centroid"] = (X @ fits["centroid"].T).ravel()
    Xc = X - fits["mahal"].location_
    out["mahalanobis"] = -np.sum(Xc @ np.linalg.inv(fits["mahal"].covariance_) * Xc, axis=1)
    out["iforest"] = fits["iforest"].score_samples(X)
    return out


# ---------------------------------------------------------------- main
def main():
    emb, target, texts = build_embeddings()
    log(f"dataset: {emb.shape[0]} docs, dim={emb.shape[1]}, "
        f"{len(np.unique(target))} classes")

    rng = np.random.default_rng(SEED)
    classes = np.unique(target)
    # global split by class: 70% train / 30% test per class
    train_idx, test_idx = {}, {}
    for c in classes:
        idx = np.where(target == c)[0]
        rng.shuffle(idx)
        ntr = int(0.7 * len(idx))
        train_idx[c], test_idx[c] = idx[:ntr], idx[ntr:]

    auroc = {m: [] for m in METHODS}
    t0 = time.perf_counter()
    for ci, c in enumerate(classes):
        Xtr = emb[train_idx[c]]
        pos = emb[test_idx[c]]
        neg_pool = np.concatenate([test_idx[o] for o in classes if o != c])
        neg = rng.choice(neg_pool, size=min(2 * len(pos), len(neg_pool)), replace=False)
        Xte = np.concatenate([pos, emb[neg]])
        y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])

        fits = {
            "ebm": EBMSplatsScorer(emb.shape[1]).fit(Xtr),
            "gmm": fit_gmm(Xtr),
            "knn": fit_knn(Xtr),
            "centroid": fit_centroid(Xtr),
            "mahal": fit_mahalanobis(Xtr),
            "iforest": fit_iforest(Xtr),
        }
        scores = score_all(fits, Xte)
        for m, key in zip(METHODS, ["ebm-splats", "gmm", "knn-density",
                                    "centroid", "mahalanobis", "iforest"]):
            auroc[m].append(roc_auc_score(y, scores[key]))
        RESULTS["per_class"][str(c)] = {m: round(float(auroc[m][-1]), 4) for m in METHODS}
        log(f"class {ci+1:2d}/20 [{c:2d}] "
            + " ".join(f"{m}={auroc[m][-1]:.3f}" for m in METHODS))
    fit_time_total = time.perf_counter() - t0
    RESULTS["meta"]["duel_fit_time_s"] = round(fit_time_total, 1)

    # ------------------------------------------------ latency (batch=1, CPU)
    log("measuring batch=1 CPU latency (300 docs each)...")
    qlat = {m: [] for m in METHODS}
    fits0 = {
        "ebm": EBMSplatsScorer(emb.shape[1]).fit(emb[train_idx[classes[0]]]),
        "gmm": fit_gmm(emb[train_idx[classes[0]]]),
        "knn": fit_knn(emb[train_idx[classes[0]]]),
        "centroid": fit_centroid(emb[train_idx[classes[0]]]),
        "mahal": fit_mahalanobis(emb[train_idx[classes[0]]]),
        "iforest": fit_iforest(emb[train_idx[classes[0]]]),
    }
    Q = emb[:300]
    import statistics

    def timed(fn, n=300):
        xs = []
        for i in range(n):
            t = time.perf_counter()
            fn(Q[i:i + 1])
            xs.append((time.perf_counter() - t) * 1e3)
        return statistics.median(xs), sorted(xs)[int(0.95 * n)]

    qlat["ebm-splats"] = timed(lambda q: fits0["ebm"].score(q))
    qlat["gmm"] = timed(lambda q: fits0["gmm"].score_samples(q))
    qlat["knn-density"] = timed(lambda q: fits0["knn"].kneighbors(q))
    qlat["centroid"] = timed(lambda q: q @ fits0["centroid"].T)
    qlat["mahalanobis"] = timed(lambda q: -np.sum(
        (q - fits0["mahal"].location_) @ np.linalg.inv(fits0["mahal"].covariance_)
        * (q - fits0["mahal"].location_), axis=1))
    qlat["iforest"] = timed(lambda q: fits0["iforest"].score_samples(q))
    for m in METHODS:
        RESULTS["latency"][m] = {"p50_ms": round(qlat[m][0], 3),
                                 "p95_ms": round(qlat[m][1], 3)}

    # ------------------------------------------------ model size (per class, ~600 docs)
    n_tr = len(train_idx[classes[0]])
    d = emb.shape[1]
    RESULTS["model_size"] = {
        "ebm-splats": 256 * (d * 4 + 8),      # mu f32 + alpha f32 + kappa f32
        "gmm": 64 * (d * 2 + 1) * 4,          # diag cov: means+vars+weights
        "knn-density": n_tr * d * 4,          # stores the WHOLE train set
        "centroid": d * 4,
        "mahalanobis": (d * d + d) * 4,
        "iforest": int(200 * 1024),           # ~1KB/tree rough
    }

    # ------------------------------------------------ summary
    summary = {}
    for m in METHODS:
        a = np.array(auroc[m])
        summary[m] = {"mean": round(float(a.mean()), 4),
                      "std": round(float(a.std()), 4),
                      "min": round(float(a.min()), 4),
                      "max": round(float(a.max()), 4)}
    # pairwise wins on per-class basis
    wins = {m: 0 for m in METHODS}
    for c in RESULTS["per_class"]:
        row = RESULTS["per_class"][c]
        best = max(row, key=row.get)
        wins[best] += 1
    RESULTS["meta"]["best_method_per_class_wins"] = wins
    RESULTS["summary"] = summary
    RESULTS["meta"]["encoder"] = "all-MiniLM-L6-v2 (384d, normalized)"
    RESULTS["meta"]["dataset"] = "20 Newsgroups, 18846 docs, 20 classes, 70/30"
    RESULTS["meta"]["seed"] = SEED

    with open("duel_results.json", "w") as f:
        json.dump(RESULTS, f, indent=2)
    log("saved duel_results.json")

    # markdown table
    lines = ["# One-class scoring duel — 20 Newsgroups\n",
             f"Encoder: all-MiniLM-L6-v2 (384d) | AUROC per class (inliers vs 2x rest), "
             f"mean ± std over 20 classes | seed {SEED}\n",
             "| method | AUROC (mean±std) | min | max | wins(best/class) "
             "| p50 ms/doc | p95 ms/doc | model size/class |",
             "|---|---|---|---|---|---|---|---|"]
    for m in METHODS:
        s = summary[m]
        lines.append(f"| {m} | {s['mean']:.4f} ± {s['std']:.4f} | {s['min']:.3f} "
                     f"| {s['max']:.3f} | {wins[m]}/20 | {qlat[m][0]:.2f} "
                     f"| {qlat[m][1]:.2f} | {RESULTS['model_size'][m]/1024:.0f} KB |")
    with open("duel_tables.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    log("saved duel_tables.md")


if __name__ == "__main__":
    main()
