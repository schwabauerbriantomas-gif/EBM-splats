#!/usr/bin/env python3
"""
TEST 1.1: Rectified Flow vs Langevin — Speed Benchmark

Hypothesis: Rectified Flow reduces sampling from 200 steps to 5 steps
without losing quality.

Since we don't have a trained EBM, this test measures:
1. Pure speed: how long does RF (5 steps) take vs Langevin (200 steps)?
2. Interpolation quality: does RF produce samples that look like real data?

We use real MiniLM embeddings as the "data distribution" and compare:
- Langevin: starts from noise, follows score field for 200 steps
- Rectified Flow: starts from noise, follows velocity field for 5 steps
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}", flush=True)

sys_path = "/root/EBM-splats"
import sys
sys.path.insert(0, sys_path)

RESULTS_FILE = "/root/EBM-splats/tests/t11_rf_results.jsonl"

from geometry import normalize_sphere, exp_map, project_to_tangent

# ── Load real embeddings as "data distribution" ──
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

print("Loading model + data...", flush=True)
st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
emb_dim = st_model.get_sentence_embedding_dimension()

wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
texts = [t.strip() for t in wiki["text"] if len(t.strip()) > 50][:5000]

print(f"Encoding {len(texts)} texts...", flush=True)
data_embs = st_model.encode(texts, batch_size=256, show_progress_bar=False, convert_to_tensor=True, normalize_embeddings=True)
print(f"Data shape: {data_embs.shape}", flush=True)

# ── Simple velocity network for Rectified Flow ──
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

# ── Geodesic interpolation ──
def geodesic_interpolate(p, q, t):
    cos_theta = (p * q).sum(dim=-1, keepdim=True).clamp(-1 + 1e-7, 1 - 1e-7)
    theta = torch.acos(cos_theta)
    sin_theta = torch.sin(theta).clamp(min=1e-7)
    result = torch.sin((1 - t) * theta) / sin_theta * p + torch.sin(t * theta) / sin_theta * q
    return normalize_sphere(result)

# ── Train Rectified Flow ──
print("\n=== Training Rectified Flow ===", flush=True)
vel_net = VelocityNet(emb_dim).to(device)
optimizer = torch.optim.AdamW(vel_net.parameters(), lr=1e-3, weight_decay=0.01)

N_DATA = data_embs.shape[0]
BATCH = 256
N_STEPS = 500  # 2 epochs over 5K data

t0 = time.time()
for step in range(N_STEPS):
    idx = torch.randint(0, N_DATA, (BATCH,), device=device)
    x_1 = data_embs[idx]  # [B, D]
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

    if (step + 1) % 100 == 0:
        print(f"  Step {step+1}/{N_STEPS} | Loss: {loss.item():.6f} | {(time.time()-t0)/(step+1):.3f}s/step", flush=True)

rf_train_time = time.time() - t0
print(f"RF training done: {rf_train_time:.1f}s", flush=True)

# ── RF Sampling ──
def sample_rf(vel_net, n_samples, dim, n_steps=5):
    x = normalize_sphere(torch.randn(n_samples, dim, device=device))
    dt = 1.0 / n_steps
    for step in range(n_steps):
        t = torch.full((n_samples, 1), step * dt, device=device)
        v = vel_net(x, t)
        v_norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        x = torch.cos(v_norm * dt) * x + torch.sin(v_norm * dt) * (v / v_norm)
        x = normalize_sphere(x)
    return x

# ── Simple Langevin sampler (using kNN-based score) ──
def sample_langevin_simple(n_samples, dim, data, n_steps=200, dt=0.001, k=32):
    """Simple Langevin: move toward nearest data points."""
    x = normalize_sphere(torch.randn(n_samples, dim, device=device))
    for step in range(n_steps):
        # Compute score: direction toward nearest data points
        with torch.no_grad():
            sims = x @ data.T  # [n_samples, N]
            topk_sims, topk_idx = sims.topk(k, dim=-1)
            topk_data = data[topk_idx]  # [n_samples, k, D]
            # Score = mean direction toward neighbors
            weights = F.softmax(topk_sims * 10, dim=-1).unsqueeze(-1)  # [n_samples, k, 1]
            target = (topk_data * weights).sum(dim=1)  # [n_samples, D]
            score = project_to_tangent(x, target - x)
            # Normalize score for stable steps
            score = score / (score.norm(dim=-1, keepdim=True) + 1e-8)

        # Langevin step on sphere
        v = score * dt * 10
        v_norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        x = torch.cos(v_norm) * x + torch.sin(v_norm) * (v / v_norm)
        x = normalize_sphere(x)

    return x

# ── Quality metric: MMD (Maximum Mean Discrepancy) ──
def compute_mmd(samples, data, kernel_bandwidth=None):
    """Compute MMD^2 between samples and data using RBF kernel."""
    if kernel_bandwidth is None:
        # Median heuristic
        with torch.no_grad():
            dists = torch.cdist(samples[:100], data[:100])
            kernel_bandwidth = dists.median().item()

    def rbf(x, y, h):
        d = torch.cdist(x.detach(), y.detach()) ** 2
        return torch.exp(-d / (2 * h ** 2))

    n, m = samples.shape[0], data.shape[0]
    # Subsample for memory — clone to avoid inference tensor issues
    max_n = min(n, 500)
    max_m = min(m, 500)
    s = samples[:max_n].clone().detach()
    d = data[:max_m].clone().detach()

    with torch.no_grad():
        K_ss = rbf(s, s, kernel_bandwidth)
        K_dd = rbf(d, d, kernel_bandwidth)
        K_sd = rbf(s, d, kernel_bandwidth)

    mmd = K_ss.mean() + K_dd.mean() - 2 * K_sd.mean()
    return mmd.item()

# ── Benchmark ──
print("\n=== BENCHMARK: Speed + Quality ===", flush=True)

N_SAMPLES = 1024
data_subset = data_embs[:2000]  # Use 2K for scoring + MMD

results = []

# Test different RF step counts
for rf_steps in [1, 2, 5, 10, 20]:
    t0 = time.time()
    samples = sample_rf(vel_net, N_SAMPLES, emb_dim, n_steps=rf_steps)
    rf_time = time.time() - t0

    mmd = compute_mmd(samples, data_subset)

    result = {
        "method": f"rectified_flow_{rf_steps}_steps",
        "n_steps": rf_steps,
        "time_s": round(rf_time, 4),
        "mmd": round(mmd, 6),
    }
    results.append(result)
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(result) + "\n")
    print(f"  RF ({rf_steps:>2d} steps): {rf_time:.3f}s | MMD={mmd:.6f}", flush=True)

# Langevin (different step counts)
for lang_steps in [10, 50, 100, 200]:
    t0 = time.time()
    samples = sample_langevin_simple(N_SAMPLES, emb_dim, data_subset, n_steps=lang_steps)
    lang_time = time.time() - t0

    mmd = compute_mmd(samples, data_subset)

    result = {
        "method": f"langevin_{lang_steps}_steps",
        "n_steps": lang_steps,
        "time_s": round(lang_time, 4),
        "mmd": round(mmd, 6),
    }
    results.append(result)
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(result) + "\n")
    print(f"  Langevin ({lang_steps:>3d} steps): {lang_time:.3f}s | MMD={mmd:.6f}", flush=True)

# Random baseline (noise, no sampling)
t0 = time.time()
random_samples = normalize_sphere(torch.randn(N_SAMPLES, emb_dim, device=device))
random_mmd = compute_mmd(random_samples, data_subset)
random_time = time.time() - t0
result = {
    "method": "random_noise_baseline",
    "n_steps": 0,
    "time_s": round(random_time, 4),
    "mmd": round(random_mmd, 6),
}
results.append(result)
with open(RESULTS_FILE, "a") as f:
    f.write(json.dumps(result) + "\n")
print(f"  Random noise: {random_time:.3f}s | MMD={random_mmd:.6f}", flush=True)

# ── Summary ──
print("\n" + "=" * 70)
print("RECTIFIED FLOW vs LANGEVIN — SUMMARY")
print("=" * 70)

print(f"\n{'Method':<30} {'Steps':>6} {'Time':>8} {'MMD':>10}")
print("-" * 58)
for r in results:
    print(f"{r['method']:<30} {r['n_steps']:>6} {r['time_s']:>7.3f}s {r['mmd']:>10.6f}")

# Find best RF and Langevin
rf_results = [r for r in results if "rectified" in r["method"]]
lang_results = [r for r in results if "langevin" in r["method"]]

best_rf = min(rf_results, key=lambda r: r["mmd"])
best_lang = min(lang_results, key=lambda r: r["mmd"])

rf_5 = next((r for r in rf_results if r["n_steps"] == 5), best_rf)
lang_200 = next((r for r in lang_results if r["n_steps"] == 200), best_lang)

print(f"\n--- Comparison ---")
print(f"  Best RF ({best_rf['n_steps']} steps): {best_rf['time_s']:.3f}s, MMD={best_rf['mmd']:.6f}")
print(f"  Best Langevin ({best_lang['n_steps']} steps): {best_lang['time_s']:.3f}s, MMD={best_lang['mmd']:.6f}")
print(f"  Random baseline: {random_mmd:.6f}")

if lang_200["time_s"] > 0:
    speedup = lang_200["time_s"] / rf_5["time_s"]
    print(f"\n  Speedup RF(5) vs Langevin(200): {speedup:.1f}x")

if rf_5["mmd"] < lang_200["mmd"]:
    print(f"  RF(5) MMD is BETTER than Langevin(200): {rf_5['mmd']:.6f} < {lang_200['mmd']:.6f}")
elif rf_5["mmd"] < lang_200["mmd"] * 1.1:
    print(f"  RF(5) MMD ≈ Langevin(200): {rf_5['mmd']:.6f} ≈ {lang_200['mmd']:.6f}")
else:
    print(f"  RF(5) MMD is WORSE than Langevin(200): {rf_5['mmd']:.6f} > {lang_200['mmd']:.6f}")

# Verdict
if speedup > 10 and rf_5["mmd"] <= lang_200["mmd"] * 1.1:
    print(f"\nVERDICT: Rectified Flow WINS — {speedup:.0f}x faster, similar quality", flush=True)
    print("  → Sampling bottleneck is SOLVABLE. RF makes EBM sampling viable.", flush=True)
elif speedup > 5 and rf_5["mmd"] < random_mmd * 0.8:
    print(f"\nVERDICT: Rectified Flow is PROMISING — {speedup:.0f}x faster, decent quality", flush=True)
    print("  → Worth more training. Speed problem is addressed.", flush=True)
elif speedup > 5:
    print(f"\nVERDICT: Rectified Flow is FASTER but quality needs work", flush=True)
    print("  → Speed bottleneck solved but quality gap remains.", flush=True)
else:
    print(f"\nVERDICT: No significant speedup or quality improvement", flush=True)

print(f"\nTraining time: {rf_train_time:.1f}s (500 steps on 5K embeddings)", flush=True)
print("Done!", flush=True)
