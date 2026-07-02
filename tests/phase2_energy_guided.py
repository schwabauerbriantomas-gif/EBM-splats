#!/usr/bin/env python3
"""
PHASE 2 — TEST: Energy-Guided Generation with Rectified Flow

Hypothesis: Manipulating the energy of specific splats allows controlling
what type of content is generated. Adding energy to topic A splats
and subtracting energy from topic B splats should produce samples
semantically closer to A and farther from B.

Setup:
  1. Embed 10K TinyStories with MiniLM → distribution on S^383
  2. Cluster into ~50 topics via KMeans
  3. Initialize splats from cluster centers
  4. Train RF velocity network on the full distribution
  5. Baseline sampling (no guidance) vs guided sampling (boost/suppress)
  6. Decoding via nearest-neighbor retrieval (no neural decoder needed)
  7. Measure: do guided samples retrieve texts from the correct topic?

Metric: for each sample, measure which topic the nearest-neighbor
in the corpus belongs to. If guidance works, samples guided toward
topic A should retrieve more texts from A than baseline samples.
"""

import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}", flush=True)
if device == "cuda":
    torch.cuda.set_per_process_memory_fraction(0.85)

RESULTS = "/root/EBM-splats/tests/phase2_energy_guided_results.jsonl"
import sys
sys.path.insert(0, "/root/EBM-splats/src/ebm")
sys.path.insert(0, "/root/EBM-splats")

# Inline geometry functions (avoid import path issues)
def normalize_sphere(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=-1)

def project_to_tangent(x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    dot = (x * v).sum(dim=-1, keepdim=True)
    return v - dot * x

# ── 1. Load data ──
print("\n=== STEP 1: Load TinyStories + embed ===", flush=True)
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
emb_dim = st_model.get_embedding_dimension()
print(f"MiniLM dim: {emb_dim}", flush=True)

# Load TinyStories from local file
data_path = "/mnt/d/datasets/ebm/tinystories_train.txt"
print(f"Loading TinyStories from {data_path}...", flush=True)
with open(data_path, "r", encoding="utf-8", errors="ignore") as f:
    all_texts = [line.strip() for line in f.readlines() if len(line.strip()) > 80]

N_STORIES = 10000
texts = all_texts[:N_STORIES]
print(f"Using {len(texts)} stories", flush=True)

t0 = time.time()
embeddings = st_model.encode(texts, batch_size=256, show_progress_bar=False,
                             convert_to_tensor=True, normalize_embeddings=True)
print(f"Embedded in {time.time()-t0:.1f}s. Shape: {embeddings.shape}", flush=True)

# ── 2. Cluster into topics ──
print("\n=== STEP 2: KMeans clustering ===", flush=True)
N_CLUSTERS = 50
emb_np = embeddings.cpu().numpy()
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(emb_np)
cluster_centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=device)
cluster_centers = normalize_sphere(cluster_centers)

# Count stories per cluster
cluster_counts = np.bincount(cluster_labels, minlength=N_CLUSTERS)
print(f"Clusters: {N_CLUSTERS}", flush=True)
print(f"Stories per cluster: min={cluster_counts.min()}, max={cluster_counts.max()}, mean={cluster_counts.mean():.0f}", flush=True)

# Find 3 well-separated topics for testing
print("\nFinding well-separated topic pairs for testing...", flush=True)
center_sims = cluster_centers @ cluster_centers.T
np_sims = center_sims.cpu().numpy()
# Find pairs with lowest similarity (most different topics)
min_sim = 1.0
best_pair = (0, 1)
for i in range(N_CLUSTERS):
    for j in range(i+1, N_CLUSTERS):
        if cluster_counts[i] >= 50 and cluster_counts[j] >= 50:
            sim = np_sims[i, j]
            if sim < min_sim:
                min_sim = sim
                best_pair = (i, j)

topic_a, topic_b = best_pair
print(f"Topic A: cluster {topic_a} ({cluster_counts[topic_a]} stories)", flush=True)
print(f"Topic B: cluster {topic_b} ({cluster_counts[topic_b]} stories)", flush=True)
print(f"Similarity between topics: {min_sim:.4f}", flush=True)

# Show example texts from each topic
print(f"\nTopic A examples:", flush=True)
a_indices = np.where(cluster_labels == topic_a)[0][:3]
for idx in a_indices:
    print(f"  [{idx}] {texts[idx][:120]}...", flush=True)

print(f"\nTopic B examples:", flush=True)
b_indices = np.where(cluster_labels == topic_b)[0][:3]
for idx in b_indices:
    print(f"  [{idx}] {texts[idx][:120]}...", flush=True)

# ── 3. Setup splats from all cluster centers ──
print("\n=== STEP 3: Initialize splats ===", flush=True)
splats_mu = cluster_centers.clone()  # [50, 384]
splats_alpha = torch.tensor(cluster_counts, dtype=torch.float32, device=device)
splats_alpha = splats_alpha / splats_alpha.max()  # Normalize to [0, 1]
print(f"Splats: {splats_mu.shape[0]}, alpha range: [{splats_alpha.min():.3f}, {splats_alpha.max():.3f}]", flush=True)

# ── 4. Train Rectified Flow velocity network ──
print("\n=== STEP 4: Train RF velocity network ===", flush=True)

class VelocityNet(nn.Module):
    def __init__(self, dim, hidden=512):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.net = nn.Sequential(
            nn.Linear(dim + hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim)
        )
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
        torch.sin(t * theta) / sin_theta * q
    )

vel_net = VelocityNet(emb_dim).to(device)
optimizer = torch.optim.AdamW(vel_net.parameters(), lr=1e-3, weight_decay=0.01)

N_STEPS = 1000
BATCH = 256
t0 = time.time()
for step in range(N_STEPS):
    idx = torch.randint(0, N_STORIES, (BATCH,), device=device)
    x_1 = embeddings[idx]
    x_0 = normalize_sphere(torch.randn(BATCH, emb_dim, device=device))
    t = torch.rand(BATCH, 1, device=device)

    x_t = geodesic_interpolate(x_0, x_1, t)
    target_v = project_to_tangent(x_t, x_1 - x_t)
    pred_v = vel_net(x_t, t)
    loss = F.mse_loss(pred_v, target_v)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(vel_net.parameters(), 1.0)
    optimizer.step()

    if (step + 1) % 200 == 0:
        print(f"  Step {step+1}/{N_STEPS} | Loss: {loss.item():.6f} | {(time.time()-t0)/(step+1):.3f}s/step", flush=True)

print(f"RF trained in {time.time()-t0:.1f}s", flush=True)

# ── 5. Energy-guided sampling ──
print("\n=== STEP 5: Energy-guided sampling ===", flush=True)

def sample_rf_guided(vel_net, n_samples, dim, n_steps=2,
                      splats_mu=None, splats_alpha=None,
                      boost_clusters=None, suppress_clusters=None,
                      guidance_scale=1.0):
    """
    Sample with energy guidance.

    boost_clusters: list of cluster indices to boost (increase probability)
    suppress_clusters: list of cluster indices to suppress (decrease probability)
    guidance_scale: strength of guidance
    """
    x = normalize_sphere(torch.randn(n_samples, dim, device=device))
    dt = 1.0 / n_steps

    for step in range(n_steps):
        t_val = step * dt
        t = torch.full((n_samples, 1), t_val, device=device)

        # Base velocity from RF
        v = vel_net(x, t)

        # Energy guidance: add gradient toward boosted clusters
        if boost_clusters is not None and splats_mu is not None:
            for ci in boost_clusters:
                # Direction toward cluster center
                center = splats_mu[ci]
                # Project to tangent at x
                direction = project_to_tangent(x, center.unsqueeze(0) - x)
                direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
                v = v + guidance_scale * direction

        # Suppress: move away from suppressed clusters
        if suppress_clusters is not None and splats_mu is not None:
            for ci in suppress_clusters:
                center = splats_mu[ci]
                direction = project_to_tangent(x, center.unsqueeze(0) - x)
                direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
                v = v - guidance_scale * direction

        # Move on sphere
        v_norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        x = torch.cos(v_norm * dt) * x + torch.sin(v_norm * dt) * (v / v_norm)
        x = normalize_sphere(x)

    return x

def decode_by_retrieval(samples, corpus_embeddings, corpus_texts, k=3):
    """Decode samples by finding nearest neighbors in corpus."""
    with torch.no_grad():
        sims = samples.detach() @ corpus_embeddings.detach().T
        topk_sims, topk_idx = sims.topk(k, dim=-1)
    results = []
    for i in range(samples.shape[0]):
        neighbors = [(corpus_texts[topk_idx[i, j].item()], topk_sims[i, j].item())
                     for j in range(k)]
        results.append(neighbors)
    return results

def evaluate_topic_alignment(samples, corpus_embeddings, cluster_labels, target_cluster):
    """
    Measure what fraction of nearest-neighbors belong to target_cluster.
    """
    with torch.no_grad():
        sims = samples.detach() @ corpus_embeddings.detach().T
        topk_idx = sims.argmax(dim=-1)
        nn_clusters = cluster_labels[topk_idx.cpu().numpy()]
    return np.mean(nn_clusters == target_cluster)

# Sampling parameters
N_SAMPLES = 500
GUIDANCE_SCALES = [0.0, 0.5, 1.0, 2.0, 5.0]

print(f"\nSampling {N_SAMPLES} points per condition...", flush=True)

all_results = []

for gs in GUIDANCE_SCALES:
    print(f"\n--- Guidance scale: {gs} ---", flush=True)

    # Baseline (no guidance)
    if gs == 0.0:
        samples_base = sample_rf_guided(vel_net, N_SAMPLES, emb_dim, n_steps=2,
                                         splats_mu=None, splats_alpha=None,
                                         guidance_scale=0.0)
        base_in_a = evaluate_topic_alignment(samples_base, embeddings, cluster_labels, topic_a)
        base_in_b = evaluate_topic_alignment(samples_base, embeddings, cluster_labels, topic_b)
        print(f"  Baseline: {base_in_a*100:.1f}% in topic A, {base_in_b*100:.1f}% in topic B", flush=True)

        result = {
            "condition": "baseline",
            "guidance_scale": 0.0,
            "pct_topic_a": round(base_in_a, 4),
            "pct_topic_b": round(base_in_b, 4),
        }
        all_results.append(result)
        with open(RESULTS, "a") as f:
            f.write(json.dumps(result) + "\n")
        continue

    # Boost A, suppress B
    samples_ab = sample_rf_guided(vel_net, N_SAMPLES, emb_dim, n_steps=2,
                                   splats_mu=splats_mu, splats_alpha=splats_alpha,
                                   boost_clusters=[topic_a],
                                   suppress_clusters=[topic_b],
                                   guidance_scale=gs)
    ab_in_a = evaluate_topic_alignment(samples_ab, embeddings, cluster_labels, topic_a)
    ab_in_b = evaluate_topic_alignment(samples_ab, embeddings, cluster_labels, topic_b)
    print(f"  Boost A / Suppress B: {ab_in_a*100:.1f}% in A, {ab_in_b*100:.1f}% in B", flush=True)

    # Boost B, suppress A
    samples_ba = sample_rf_guided(vel_net, N_SAMPLES, emb_dim, n_steps=2,
                                   splats_mu=splats_mu, splats_alpha=splats_alpha,
                                   boost_clusters=[topic_b],
                                   suppress_clusters=[topic_a],
                                   guidance_scale=gs)
    ba_in_a = evaluate_topic_alignment(samples_ba, embeddings, cluster_labels, topic_a)
    ba_in_b = evaluate_topic_alignment(samples_ba, embeddings, cluster_labels, topic_b)
    print(f"  Boost B / Suppress A: {ba_in_a*100:.1f}% in A, {ba_in_b*100:.1f}% in B", flush=True)

    result = {
        "condition": f"boost_a_suppress_b_gs{gs}",
        "guidance_scale": gs,
        "pct_topic_a": round(ab_in_a, 4),
        "pct_topic_b": round(ab_in_b, 4),
    }
    all_results.append(result)
    with open(RESULTS, "a") as f:
        f.write(json.dumps(result) + "\n")

    result = {
        "condition": f"boost_b_suppress_a_gs{gs}",
        "guidance_scale": gs,
        "pct_topic_a": round(ba_in_a, 4),
        "pct_topic_b": round(ba_in_b, 4),
    }
    all_results.append(result)
    with open(RESULTS, "a") as f:
        f.write(json.dumps(result) + "\n")

# ── 6. Qualitative: show retrieved texts ──
print("\n=== STEP 6: Qualitative examples ===", flush=True)

# Baseline
print("\n--- Baseline (no guidance) ---", flush=True)
samples_base = sample_rf_guided(vel_net, 5, emb_dim, n_steps=2, guidance_scale=0.0)
retrieved = decode_by_retrieval(samples_base, embeddings, texts, k=1)
for i, neighbors in enumerate(retrieved):
    print(f"  Sample {i}: [{neighbors[0][1]:.3f}] {neighbors[0][0][:100]}...", flush=True)

# Boost A strongly
print(f"\n--- Boost Topic A (gs=5.0) ---", flush=True)
samples_a = sample_rf_guided(vel_net, 5, emb_dim, n_steps=2,
                              splats_mu=splats_mu, splats_alpha=splats_alpha,
                              boost_clusters=[topic_a],
                              suppress_clusters=[topic_b],
                              guidance_scale=5.0)
retrieved_a = decode_by_retrieval(samples_a, embeddings, texts, k=1)
for i, neighbors in enumerate(retrieved_a):
    print(f"  Sample {i}: [{neighbors[0][1]:.3f}] {neighbors[0][0][:100]}...", flush=True)

# ── Summary ──
print("\n" + "=" * 70)
print("ENERGY-GUIDED GENERATION — RESULTS")
print("=" * 70)

print(f"\nTopics: A=cluster {topic_a} ({cluster_counts[topic_a]} stories), B=cluster {topic_b} ({cluster_counts[topic_b]} stories)")
print(f"Topic similarity: {min_sim:.4f}\n")

print(f"{'Condition':<35} {'% in A':>8} {'% in B':>8}")
print("-" * 55)
for r in all_results:
    print(f"{r['condition']:<35} {r['pct_topic_a']*100:>7.1f}% {r['pct_topic_b']*100:>7.1f}%")

# Check if guidance works
baseline_a = all_results[0]["pct_topic_a"]
baseline_b = all_results[0]["pct_topic_b"]

best_boost_a = max(all_results, key=lambda r: r["pct_topic_a"] if "boost_a" in r["condition"] else 0)
best_boost_b = max(all_results, key=lambda r: r["pct_topic_b"] if "boost_b" in r["condition"] else 0)

print(f"\n--- Analysis ---")
print(f"Baseline: {baseline_a*100:.1f}% A, {baseline_b*100:.1f}% B")
print(f"Best boost A: {best_boost_a['pct_topic_a']*100:.1f}% A (was {baseline_a*100:.1f}%) → +{(best_boost_a['pct_topic_a']-baseline_a)*100:.1f}pp")
print(f"Best boost B: {best_boost_b['pct_topic_b']*100:.1f}% B (was {baseline_b*100:.1f}%) → +{(best_boost_b['pct_topic_b']-baseline_b)*100:.1f}pp")

a_improvement = best_boost_a["pct_topic_a"] - baseline_a
b_improvement = best_boost_b["pct_topic_b"] - baseline_b

if a_improvement > 0.1 and b_improvement > 0.1:
    print(f"\nVERDICT: Energy guidance WORKS. Both topics boosted significantly.", flush=True)
    print(f"  → EBM with RF enables controlled generation via energy manipulation.", flush=True)
elif a_improvement > 0.05 or b_improvement > 0.05:
    print(f"\nVERDICT: Energy guidance shows PARTIAL effect.", flush=True)
    print(f"  → Some control but not strong enough for practical use.", flush=True)
else:
    print(f"\nVERDICT: Energy guidance DOES NOT work.", flush=True)
    print(f"  → Manipulating splat energy doesn't control generation direction.", flush=True)

print("\nDone!", flush=True)
