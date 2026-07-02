#!/usr/bin/env python3
"""
PHASE 2 — TEST: Concept Composition via EBM Energy

Hypothesis: Adding energy from multiple topics simultaneously produces
samples in the semantic intersection. Like word2vec arithmetic but
with continuous control of the energy landscape.

Tests:
  1. Boost A alone → samples near A
  2. Boost B alone → samples near B
  3. Boost A+B together → samples at the midpoint/intersection
  4. Boost A with weight 0.7 + B with weight 0.3 → asymmetric blend
  5. Boost A, suppress B → pure A, far from B
  6. 3 topics: Boost A+B+C → triple intersection

Metric: average cosine similarity of samples to each topic center.
If composition works, A+B should have high similarity to BOTH centers.
"""

import time, json, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans

device = "cuda"
torch.cuda.set_per_process_memory_fraction(0.85)
RESULTS = "/root/EBM-splats/tests/phase2_composition_results.jsonl"

def normalize_sphere(x):
    return F.normalize(x, dim=-1)

def project_to_tangent(x, v):
    dot = (x * v).sum(dim=-1, keepdim=True)
    return v - dot * x

# ── 1. Data ──
print("=== LOAD DATA ===", flush=True)
from sentence_transformers import SentenceTransformer
st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
emb_dim = st_model.get_embedding_dimension()

data_path = "/mnt/d/datasets/ebm/tinystories_train.txt"
print("Loading TinyStories...", flush=True)
with open(data_path, "r", encoding="utf-8", errors="ignore") as f:
    all_texts = [line.strip() for line in f.readlines() if len(line.strip()) > 80]
texts = all_texts[:10000]
print(f"Stories: {len(texts)}", flush=True)

t0 = time.time()
embeddings = st_model.encode(texts, batch_size=256, show_progress_bar=False,
                             convert_to_tensor=True, normalize_embeddings=True)
print(f"Embedded in {time.time()-t0:.1f}s", flush=True)

# ── 2. Clusters ──
print("\n=== KMEANS ===", flush=True)
N_CLUSTERS = 50
emb_np = embeddings.cpu().numpy()
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(emb_np)
cluster_centers = normalize_sphere(torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=device))
cluster_counts = np.bincount(cluster_labels, minlength=N_CLUSTERS)

# Pick 4 well-separated topics with enough stories
center_sims = (cluster_centers @ cluster_centers.T).cpu().numpy()
candidates = [i for i in range(N_CLUSTERS) if cluster_counts[i] >= 80]

# Find 4 most mutually-dissimilar topics
from itertools import combinations
best_quad = None
best_score = 999
for quad in combinations(candidates, 4):
    avg_sim = np.mean([center_sims[a, b] for a, b in combinations(quad, 2)])
    if avg_sim < best_score:
        best_score = avg_sim
        best_quad = quad

tA, tB, tC, tD = best_quad
print(f"Topics: A={tA}({cluster_counts[tA]}), B={tB}({cluster_counts[tB]}), C={tC}({cluster_counts[tC]}), D={tD}({cluster_counts[tD]})", flush=True)
print(f"Average pairwise similarity: {best_score:.4f}", flush=True)

for name, ci in [("A", tA), ("B", tB), ("C", tC), ("D", tD)]:
    idxs = np.where(cluster_labels == ci)[0][:2]
    print(f"\n  Topic {name} (cluster {ci}):", flush=True)
    for idx in idxs:
        print(f"    {texts[idx][:120]}...", flush=True)

splats_mu = cluster_centers.clone()

# ── 3. Train RF ──
print("\n=== TRAIN RF ===", flush=True)

class VelocityNet(nn.Module):
    def __init__(self, dim, hidden=512):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.Linear(1, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.net = nn.Sequential(
            nn.Linear(dim + hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim))
    def forward(self, x_t, t):
        t_emb = self.time_mlp(t)
        v = self.net(torch.cat([x_t, t_emb], dim=-1))
        return project_to_tangent(x_t, v)

def geodesic_interpolate(p, q, t):
    cos_theta = (p * q).sum(dim=-1, keepdim=True).clamp(-1 + 1e-7, 1 - 1e-7)
    theta = torch.acos(cos_theta)
    sin_theta = torch.sin(theta).clamp(min=1e-7)
    return normalize_sphere(
        torch.sin((1 - t) * theta) / sin_theta * p +
        torch.sin(t * theta) / sin_theta * q)

vel_net = VelocityNet(emb_dim).to(device)
optimizer = torch.optim.AdamW(vel_net.parameters(), lr=1e-3, weight_decay=0.01)

N_STORIES = len(texts)
t0 = time.time()
for step in range(1000):
    idx = torch.randint(0, N_STORIES, (256,), device=device)
    x_1 = embeddings[idx]
    x_0 = normalize_sphere(torch.randn(256, emb_dim, device=device))
    t = torch.rand(256, 1, device=device)
    x_t = geodesic_interpolate(x_0, x_1, t)
    target_v = project_to_tangent(x_t, x_1 - x_t)
    pred_v = vel_net(x_t, t)
    loss = F.mse_loss(pred_v, target_v)
    optimizer.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(vel_net.parameters(), 1.0)
    optimizer.step()
print(f"RF trained in {time.time()-t0:.1f}s", flush=True)

# ── 4. Compositional sampling ──
print("\n=== COMPOSITION TESTS ===", flush=True)

def sample_composed(n_samples, guidance_vec, n_steps=2):
    """
    guidance_vec: [N_topics, 2] where [:, 0] = cluster index, [:, 1] = weight
    Positive weight = boost (move toward), negative = suppress (move away)
    """
    x = normalize_sphere(torch.randn(n_samples, emb_dim, device=device))
    dt = 1.0 / n_steps
    for step in range(n_steps):
        t = torch.full((n_samples, 1), step * dt, device=device)
        v = vel_net(x, t)

        for ci, weight in guidance_vec:
            center = splats_mu[int(ci)]
            direction = project_to_tangent(x, center.unsqueeze(0) - x)
            direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
            v = v + weight * direction

        v_norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        x = torch.cos(v_norm * dt) * x + torch.sin(v_norm * dt) * (v / v_norm)
        x = normalize_sphere(x)
    return x

def measure_alignment(samples, centers_to_check):
    """Measure avg cosine similarity to each center."""
    results = {}
    with torch.no_grad():
        for name, ci in centers_to_check.items():
            center = splats_mu[ci]
            sims = (samples.detach() @ center).mean().item()
            results[name] = round(sims, 4)
    return results

def measure_nn_cluster(samples, labels, centers_info):
    """What cluster do nearest-neighbors belong to?"""
    with torch.no_grad():
        sims = samples.detach() @ embeddings.detach().T
        nn_idx = sims.argmax(dim=-1)
        nn_clusters = labels[nn_idx.cpu().numpy()]
    dist = {}
    for name, ci in centers_info.items():
        dist[name] = round(np.mean(nn_clusters == ci), 4)
    return dist

N_SAMPLES = 500
GS = 1.5  # Sweet spot from previous test

topics = {"A": tA, "B": tB, "C": tC, "D": tD}

experiments = [
    ("baseline", []),
    ("boost_A", [(tA, GS)]),
    ("boost_B", [(tB, GS)]),
    ("boost_C", [(tC, GS)]),
    # Composition: A + B
    ("boost_AB_equal", [(tA, GS), (tB, GS)]),
    # Weighted: 70% A, 30% B
    ("boost_AB_70_30", [(tA, GS * 0.7 / 0.5), (tB, GS * 0.3 / 0.5)]),
    # Weighted: 30% A, 70% B
    ("boost_AB_30_70", [(tA, GS * 0.3 / 0.5), (tB, GS * 0.7 / 0.5)]),
    # Boost A, suppress B
    ("boost_A_minus_B", [(tA, GS), (tB, -GS)]),
    # Triple: A + B + C
    ("boost_ABC", [(tA, GS), (tB, GS), (tC, GS)]),
    # Boost A + D (most dissimilar pair)
    ("boost_AD", [(tA, GS), (tD, GS)]),
]

print(f"\n{'Experiment':<25} {'sim_A':>7} {'sim_B':>7} {'sim_C':>7} {'sim_D':>7} | {'nn_A%':>6} {'nn_B%':>6} {'nn_C%':>6} {'nn_D%':>6}", flush=True)
print("-" * 100)

all_results = []

for name, guidance in experiments:
    samples = sample_composed(N_SAMPLES, guidance)
    sims = measure_alignment(samples, topics)
    nn_pct = measure_nn_cluster(samples, cluster_labels, topics)

    result = {"experiment": name, "guidance": str(guidance), "sim_to": sims, "nn_pct": nn_pct}
    all_results.append(result)
    with open(RESULTS, "a") as f:
        f.write(json.dumps(result) + "\n")

    print(f"{name:<25} {sims['A']:>7.4f} {sims['B']:>7.4f} {sims['C']:>7.4f} {sims['D']:>7.4f} | "
          f"{nn_pct['A']*100:>5.1f}% {nn_pct['B']*100:>5.1f}% {nn_pct['C']*100:>5.1f}% {nn_pct['D']*100:>5.1f}%", flush=True)

# ── 5. Key comparisons ──
print("\n" + "=" * 70)
print("COMPOSITION ANALYSIS")
print("=" * 70)

base = {r["experiment"]: r for r in all_results}

# Does A+B have high sim to BOTH A and B?
ab = base["boost_AB_equal"]
a_only = base["boost_A"]
b_only = base["boost_B"]

print(f"\n1. EQUAL COMPOSITION (A+B)")
print(f"   boost_A only:    sim_A={a_only['sim_to']['A']:.4f}, sim_B={a_only['sim_to']['B']:.4f}")
print(f"   boost_B only:    sim_A={b_only['sim_to']['A']:.4f}, sim_B={b_only['sim_to']['B']:.4f}")
print(f"   boost_AB equal:  sim_A={ab['sim_to']['A']:.4f}, sim_B={ab['sim_to']['B']:.4f}")
ab_balanced = ab['sim_to']['A'] > 0.5 * a_only['sim_to']['A'] and ab['sim_to']['B'] > 0.5 * b_only['sim_to']['B']
print(f"   → Balanced? {'YES' if ab_balanced else 'NO'} (both sims > 50% of single-boost)")

# Does 70/30 produce asymmetric blend?
ab_73 = base["boost_AB_70_30"]
ab_37 = base["boost_AB_30_70"]
print(f"\n2. WEIGHTED BLEND")
print(f"   70%A+30%B:  sim_A={ab_73['sim_to']['A']:.4f}, sim_B={ab_73['sim_to']['B']:.4f} → A{'>' if ab_73['sim_to']['A'] > ab_73['sim_to']['B'] else '<'}B")
print(f"   30%A+70%B:  sim_A={ab_37['sim_to']['A']:.4f}, sim_B={ab_37['sim_to']['B']:.4f} → A{'>' if ab_37['sim_to']['A'] > ab_37['sim_to']['B'] else '<'}B")
weighted_works = (ab_73['sim_to']['A'] > ab_73['sim_to']['B']) and (ab_37['sim_to']['B'] > ab_37['sim_to']['A'])
print(f"   → Asymmetric? {'YES' if weighted_works else 'NO'}")

# Does A-B suppress B?
amb = base["boost_A_minus_B"]
print(f"\n3. SUPPRESSION (A - B)")
print(f"   boost_A only:      sim_A={a_only['sim_to']['A']:.4f}, sim_B={a_only['sim_to']['B']:.4f}, nn_B={a_only['nn_pct']['B']*100:.1f}%")
print(f"   boost_A minus_B:   sim_A={amb['sim_to']['A']:.4f}, sim_B={amb['sim_to']['B']:.4f}, nn_B={amb['nn_pct']['B']*100:.1f}%")
suppression_works = amb['sim_to']['B'] < a_only['sim_to']['B']
print(f"   → B suppressed? {'YES' if suppression_works else 'NO'} (sim_B decreased)")

# Triple composition
abc = base["boost_ABC"]
print(f"\n4. TRIPLE COMPOSITION (A+B+C)")
print(f"   boost_ABC:  sim_A={abc['sim_to']['A']:.4f}, sim_B={abc['sim_to']['B']:.4f}, sim_C={abc['sim_to']['C']:.4f}")
all_three = abc['sim_to']['A'] > 0.4 and abc['sim_to']['B'] > 0.4 and abc['sim_to']['C'] > 0.4
print(f"   → All three active? {'YES' if all_three else 'NO'}")

# Overall verdict
n_works = sum([ab_balanced, weighted_works, suppression_works, all_three])
print(f"\n{'='*70}")
print(f"VERDICT: {n_works}/4 composition mechanisms work")
if n_works >= 3:
    print("  → Composición de conceptos FUNCIONA. EBM + RF enables semantic arithmetic.", flush=True)
elif n_works >= 2:
    print("  → Composición parcial. Some mechanisms work, others need tuning.", flush=True)
else:
    print("  → Composición NO funciona reliably.", flush=True)

# ── 6. Qualitative: show retrieved stories for key conditions ──
print("\n=== QUALITATIVE EXAMPLES ===", flush=True)

for exp_name in ["boost_A", "boost_AB_equal", "boost_ABC"]:
    exp = base[exp_name]
    print(f"\n--- {exp_name} ---", flush=True)
    guidance = eval(exp["guidance"])
    samples = sample_composed(3, guidance)
    with torch.no_grad():
        sims = samples.detach() @ embeddings.detach().T
        topk_sims, topk_idx = sims.topk(1, dim=-1)
    for i in range(3):
        idx = topk_idx[i, 0].item()
        print(f"  [{topk_sims[i,0].item():.3f}] {texts[idx][:130]}...", flush=True)

print("\nDone!", flush=True)
