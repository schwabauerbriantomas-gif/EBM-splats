#!/usr/bin/env python3
"""
TEST 1.3: OOD Detection with EBM Energy

Hypothesis: The EBM energy (Gaussian splat function) discriminates
in-distribution (ID) vs out-of-distribution (OOD) samples.

If AUROC > 0.75 → the energy has practical value for quantifying uncertainty.
If AUROC ~ 0.5 → the energy does not discriminate, discard for OOD.

Setup:
  1. Generate 10K ID embeddings with MiniLM from normal texts (WikiText)
  2. Generate OOD embeddings from:
     - Python code (very different distribution)
     - Random text (random tokens)
     - Other languages (Chinese, Arabic)
  3. Create SplatStorage with ID embeddings
  4. For each embedding (ID + OOD), compute energy
  5. Calculate AUROC: does the energy distinguish ID from OOD?
"""

import time
import sys
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}", flush=True)

sys.path.insert(0, "/root/EBM-splats")
RESULTS_FILE = "/root/EBM-splats/tests/t13_ood_results.jsonl"

# ── Load model ──
from sentence_transformers import SentenceTransformer
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
st_model = SentenceTransformer(MODEL_NAME, device=device)
emb_dim = st_model.get_sentence_embedding_dimension()
print(f"Model: {MODEL_NAME}, dim: {emb_dim}", flush=True)

# ── Load ID data: WikiText-103 ──
from datasets import load_dataset

print("Loading WikiText-103 (ID distribution)...", flush=True)
wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
wiki_texts = [t.strip() for t in wiki["text"] if len(t.strip()) > 50][:10000]
print(f"  ID texts: {len(wiki_texts)}", flush=True)

# ── Load OOD data ──
print("Loading OOD data...", flush=True)

# OOD Type 1: Python code
python_code = [
    "def fibonacci(n):\n    if n <= 1: return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    "import numpy as np\narr = np.random.randn(100, 100)\nprint(arr.mean())",
    "class NeuralNetwork:\n    def __init__(self, layers):\n        self.weights = []\n        for i in range(len(layers)-1):\n            self.weights.append(np.random.randn(layers[i], layers[i+1]))",
    "for i in range(100):\n    x = np.sin(i * 0.1)\n    print(f'Value: {x:.4f}')",
    "async def fetch_data(url):\n    response = await aiohttp.get(url)\n    return await response.json()",
    "torch.manual_seed(42)\nmodel = nn.Sequential(nn.Linear(768, 256), nn.ReLU(), nn.Linear(256, 10))",
    "df = pd.read_csv('data.csv')\ndf.groupby('category').agg({'price': 'mean'})",
    "from transformers import AutoModel\nmodel = AutoModel.from_pretrained('bert-base-uncased')",
    "@app.route('/api/users', methods=['POST'])\ndef create_user():\n    data = request.json\n    db.insert(data)",
    "git rebase -i HEAD~3\n# Squash the last 3 commits",
    "sudo systemctl restart nginx\nsudo ufw allow 80/tcp",
    "const express = require('express');\nconst app = express();",
    "SELECT u.name, COUNT(o.id) FROM users u LEFT JOIN orders o ON u.id = o.user_id GROUP BY u.name;",
    "docker run -d --name db -p 5432:5432 -e POSTGRES_PASSWORD=secret postgres:16",
    "kubectl apply -f deployment.yaml\nkubectl get pods -n production",
] * 100  # 1500 code samples

# OOD Type 2: Random tokens (extreme OOD)
import string
random_texts = []
for _ in range(1000):
    words = [''.join(np.random.choice(list(string.ascii_lowercase), np.random.randint(3, 10))) for _ in range(np.random.randint(10, 30))]
    random_texts.append(' '.join(words))

# OOD Type 3: Non-English text
non_english = [
    "今天天气很好，我们去公园散步吧。阳光明媚，鸟语花香。",
    "اللغة العربية هي لغة القرآن الكريم وهي لغة رسمية في العديد من الدول",
    "학교에 가고 싶지 않아요. 오늘은 너무 피곤해요.",
    "Я не понимаю, что происходит. Это очень сложно для меня.",
    "日本人は礼儀正しいです。彼らは常に他人を尊重します。",
    "¿Dónde está la biblioteca? Necesito estudiar para mi examen de mañana.",
    "Le chat est sur la table. Il dort tranquillement au soleil.",
    "Die Katze schläft auf dem Sofa. Es ist sehr gemütlich.",
    "Il mio gatto mangia il pesce. Gli piace molto il tonno.",
    "O gato está dormindo no sofá. Ele está muito confortável.",
] * 150  # 1500 samples

print(f"  OOD code: {len(python_code)}", flush=True)
print(f"  OOD random: {len(random_texts)}", flush=True)
print(f"  OOD non-English: {len(non_english)}", flush=True)

# ── Encode everything ──
print("\nEncoding embeddings...", flush=True)
t0 = time.time()

id_embs = st_model.encode(wiki_texts, batch_size=256, show_progress_bar=False, convert_to_tensor=True, normalize_embeddings=True)
code_embs = st_model.encode(python_code[:1000], batch_size=256, show_progress_bar=False, convert_to_tensor=True, normalize_embeddings=True)
random_embs = st_model.encode(random_texts, batch_size=256, show_progress_bar=False, convert_to_tensor=True, normalize_embeddings=True)
foreign_embs = st_model.encode(non_english[:1000], batch_size=256, show_progress_bar=False, convert_to_tensor=True, normalize_embeddings=True)

print(f"  Encoding done in {time.time()-t0:.1f}s", flush=True)
print(f"  ID shape: {id_embs.shape}, Code: {code_embs.shape}, Random: {random_embs.shape}, Foreign: {foreign_embs.shape}", flush=True)

# ── Energy function: Gaussian splat energy on hypersphere ──
# E(x) = -log Σ_k exp(α_k * (x·μ_k - 1) / τ)
# Lower energy = closer to data manifold (ID)
# Higher energy = far from data (OOD)

def compute_energy_simple(x, splats_mu, splats_alpha, temperature=0.1, k_neighbors=64):
    """
    Compute energy of point x given splat centers.
    Uses KNN to find nearest k splats.

    Args:
        x: [B, D] query points (normalized)
        splats_mu: [N, D] splat centers (normalized)
        splats_alpha: [N] splat weights
        temperature: softmax temperature
        k_neighbors: number of nearest neighbors
    Returns:
        energy: [B] energy values
    """
    # Compute similarities [B, N]
    sims = x @ splats_mu.T

    # For each query, get top-k nearest splats
    topk_sims, topk_idx = sims.topk(min(k_neighbors, splats_mu.shape[0]), dim=-1)

    # Gather corresponding alphas
    topk_alpha = splats_alpha[topk_idx]  # [B, k]

    # Energy: E = -logsumexp(alpha * (sim - 1) / tau)
    # (sim - 1) ∈ [-2, 0], max=0 when x=mu
    exponent = topk_alpha * (topk_sims - 1.0) / temperature  # [B, k]
    energy = -torch.logsumexp(exponent, dim=-1)  # [B]

    return energy

# ── Setup splats from ID data ──
print("\nSetting up splats from ID embeddings...", flush=True)

# Use all 10K ID embeddings as splat centers
splats_mu = id_embs.clone()  # [10000, 384]
splats_alpha = torch.ones(splats_mu.shape[0], device=device)

# ── Compute energy for ID and OOD ──
print("\nComputing energies...", flush=True)

# Test multiple temperature settings
temperatures = [0.01, 0.05, 0.1, 0.5, 1.0]
k_values = [16, 32, 64, 128]

all_results = []

for tau in temperatures:
    for k in k_values:
        # Compute energy for each distribution
        with torch.no_grad():
            id_energy = compute_energy_simple(id_embs[:1000], splats_mu, splats_alpha, tau, k)
            code_energy = compute_energy_simple(code_embs, splats_mu, splats_alpha, tau, k)
            random_energy = compute_energy_simple(random_embs, splats_mu, splats_alpha, tau, k)
            foreign_energy = compute_energy_simple(foreign_embs, splats_mu, splats_alpha, tau, k)

        # ID vs Code OOD
        labels_code = np.concatenate([np.zeros(1000), np.ones(1000)])
        scores_code = torch.cat([id_energy, code_energy]).cpu().numpy()
        auroc_code = roc_auc_score(labels_code, scores_code)

        # ID vs Random OOD
        labels_random = np.concatenate([np.zeros(1000), np.ones(1000)])
        scores_random = torch.cat([id_energy, random_energy]).cpu().numpy()
        auroc_random = roc_auc_score(labels_random, scores_random)

        # ID vs Foreign OOD
        labels_foreign = np.concatenate([np.zeros(1000), np.ones(1000)])
        scores_foreign = torch.cat([id_energy, foreign_energy]).cpu().numpy()
        auroc_foreign = roc_auc_score(labels_foreign, scores_foreign)

        # Combined OOD (all types vs ID)
        all_ood_energy = torch.cat([code_energy, random_energy, foreign_energy])
        labels_all = np.concatenate([np.zeros(1000), np.ones(len(all_ood_energy))])
        scores_all = torch.cat([id_energy, all_ood_energy]).cpu().numpy()
        auroc_all = roc_auc_score(labels_all, scores_all)

        # Stats
        id_mean = id_energy.mean().item()
        id_std = id_energy.std().item()
        code_mean = code_energy.mean().item()
        random_mean = random_energy.mean().item()
        foreign_mean = foreign_energy.mean().item()

        result = {
            "tau": tau,
            "k": k,
            "auroc_code": round(auroc_code, 4),
            "auroc_random": round(auroc_random, 4),
            "auroc_foreign": round(auroc_foreign, 4),
            "auroc_all": round(auroc_all, 4),
            "id_energy_mean": round(id_mean, 4),
            "code_energy_mean": round(code_mean, 4),
            "random_energy_mean": round(random_mean, 4),
            "foreign_energy_mean": round(foreign_mean, 4),
        }
        all_results.append(result)

        with open(RESULTS_FILE, "a") as f:
            f.write(json.dumps(result) + "\n")

        print(f"  τ={tau:.2f} k={k:>3d} | AUROC: code={auroc_code:.3f} random={auroc_random:.3f} foreign={auroc_foreign:.3f} all={auroc_all:.3f} | "
              f"E[ID]={id_mean:.3f} E[code]={code_mean:.3f} E[rand]={random_mean:.3f} E[foreign]={foreign_mean:.3f}", flush=True)

# ── Also test cosine distance to nearest neighbor as baseline comparison ──
print("\n--- Baseline comparison: cosine distance to nearest ID neighbor ---", flush=True)

# For each OOD point, find max similarity to any ID point
with torch.no_grad():
    # ID: sample 1000, find max sim to all 10K ID (excluding self)
    sim_id_to_id = id_embs[:1000] @ id_embs.T  # [1000, 10000]
    # Zero out diagonal (self-similarity)
    for i in range(1000):
        sim_id_to_id[i, i] = -2.0
    max_sim_id = sim_id_to_id.max(dim=-1).values  # [1000]
    # Convert to "distance" (higher = more anomalous)
    nn_dist_id = 1.0 - max_sim_id

    sim_code_to_id = code_embs @ id_embs.T
    nn_dist_code = 1.0 - sim_code_to_id.max(dim=-1).values

    sim_random_to_id = random_embs @ id_embs.T
    nn_dist_random = 1.0 - sim_random_to_id.max(dim=-1).values

    sim_foreign_to_id = foreign_embs @ id_embs.T
    nn_dist_foreign = 1.0 - sim_foreign_to_id.max(dim=-1).values

# AUROC for nearest-neighbor baseline
labels = np.concatenate([np.zeros(1000), np.ones(1000)])

auroc_nn_code = roc_auc_score(labels, torch.cat([nn_dist_id, nn_dist_code]).cpu().numpy())
auroc_nn_random = roc_auc_score(labels, torch.cat([nn_dist_id, nn_dist_random]).cpu().numpy())
auroc_nn_foreign = roc_auc_score(labels, torch.cat([nn_dist_id, nn_dist_foreign]).cpu().numpy())

all_ood_nn = torch.cat([nn_dist_code, nn_dist_random, nn_dist_foreign])
labels_all = np.concatenate([np.zeros(1000), np.ones(len(all_ood_nn))])
auroc_nn_all = roc_auc_score(labels_all, torch.cat([nn_dist_id, all_ood_nn]).cpu().numpy())

print(f"  Nearest-Neighbor baseline: code={auroc_nn_code:.3f} random={auroc_nn_random:.3f} foreign={auroc_nn_foreign:.3f} all={auroc_nn_all:.3f}", flush=True)

baseline_result = {
    "method": "nearest_neighbor_cosine",
    "auroc_code": round(auroc_nn_code, 4),
    "auroc_random": round(auroc_nn_random, 4),
    "auroc_foreign": round(auroc_nn_foreign, 4),
    "auroc_all": round(auroc_nn_all, 4),
}
all_results.append(baseline_result)
with open(RESULTS_FILE, "a") as f:
    f.write(json.dumps(baseline_result) + "\n")

# ── Summary ──
print("\n" + "=" * 70)
print("OOD DETECTION SUMMARY")
print("=" * 70)

print(f"\nBest splat energy config per OOD type:", flush=True)

# Find best configs
best_all = max(all_results, key=lambda r: r.get("auroc_all", 0))
best_code = max([r for r in all_results if "tau" in r], key=lambda r: r["auroc_code"])
best_random = max([r for r in all_results if "tau" in r], key=lambda r: r["auroc_random"])
best_foreign = max([r for r in all_results if "tau" in r], key=lambda r: r["auroc_foreign"])

_best_method = best_all.get('method', 'tau=%s, k=%s' % (best_all.get('tau'), best_all.get('k')))
print("\n  Best overall AUROC:     %.4f (%s)" % (best_all.get('auroc_all', 0), _best_method))
print(f"  Best for code OOD:      {best_code['auroc_code']:.4f} (tau={best_code['tau']}, k={best_code['k']})")
print(f"  Best for random OOD:    {best_random['auroc_random']:.4f} (tau={best_random['tau']}, k={best_random['k']})")
print(f"  Best for foreign OOD:   {best_foreign['auroc_foreign']:.4f} (tau={best_foreign['tau']}, k={best_foreign['k']})")
print(f"  Nearest-neighbor baseline: all={auroc_nn_all:.4f}", flush=True)

# Verdict
best_splat_auroc = max(r.get("auroc_all", 0) for r in all_results if "tau" in r)
if best_splat_auroc > 0.85:
    print(f"\nVERDICT: Energy function is EXCELLENT for OOD detection (AUROC={best_splat_auroc:.3f})", flush=True)
    print("  → Splat energy has genuine practical value for uncertainty quantification", flush=True)
elif best_splat_auroc > 0.75:
    print(f"\nVERDICT: Energy function is GOOD for OOD detection (AUROC={best_splat_auroc:.3f})", flush=True)
    print("  → Worth investigating further. But nearest-neighbor may be just as good.", flush=True)
elif best_splat_auroc > 0.65:
    print(f"\nVERDICT: Energy function is WEAK for OOD detection (AUROC={best_splat_auroc:.3f})", flush=True)
    print("  → Some signal but not strong enough to be useful in practice", flush=True)
else:
    print(f"\nVERDICT: Energy function FAILS for OOD detection (AUROC={best_splat_auroc:.3f})", flush=True)
    print("  → Splat energy does not discriminate ID vs OOD. DESCARTAR.", flush=True)

# Comparison
if best_splat_auroc > auroc_nn_all + 0.02:
    print(f"\n  Splat energy ({best_splat_auroc:.3f}) BEATS nearest-neighbor ({auroc_nn_all:.3f})!", flush=True)
elif best_splat_auroc > auroc_nn_all - 0.02:
    print(f"\n  Splat energy ({best_splat_auroc:.3f}) ≈ nearest-neighbor ({auroc_nn_all:.3f}). No advantage.", flush=True)
else:
    print(f"\n  Splat energy ({best_splat_auroc:.3f}) is WORSE than nearest-neighbor ({auroc_nn_all:.3f}).", flush=True)

print("\nDone!", flush=True)
