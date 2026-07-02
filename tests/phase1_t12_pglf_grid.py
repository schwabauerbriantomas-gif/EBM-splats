#!/usr/bin/env python3
"""
TEST 1.2: PGLF Grid Search — Can any config beat the MiniLM baseline?

Hypothesis: the -4.7% was due to conservative init (gain=0.1), 1 epoch, 50K pairs.
If any config beats 0.8672 → PGLF has potential.
If none beats → discard PGLF for unimodal embeddings.

Grid:
  - data_size: 50K, 200K, 500K
  - epochs: 1, 3, 5
  - init_gain: 0.1, 0.5, 1.0
  - lr: 1e-4, 5e-4, 1e-3
  - temperature: 0.05, 0.07, 0.1

To avoid exploding the grid (3^5=243 configs), we use a strategy:
  Phase A: Vary ONE parameter at a time from baseline (15 configs)
  Phase B: If any improves, explore around it (10 more configs)
"""

import time
import sys
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import spearmanr
from itertools import product

# ── Setup ──
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}", flush=True)
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    torch.cuda.set_per_process_memory_fraction(0.85)

RESULTS_FILE = "/root/EBM-splats/tests/t12_results.jsonl"
BASELINE_TARGET = 0.8672  # MiniLM-L6-v2 known STS-B Spearman

# ── Load data ONCE ──
from datasets import load_dataset

print("Loading STS-B...", flush=True)
stsb = load_dataset("glue", "stsb", split="validation")
sts_s1 = [str(x) for x in stsb["sentence1"]]
sts_s2 = [str(x) for x in stsb["sentence2"]]
sts_scores = np.array(stsb["label"], dtype=np.float32) / 5.0

print("Loading SNLI full...", flush=True)
snli_full = load_dataset("snli", split="train")
snli_clean = snli_full.filter(lambda x: x["label"] != -1)
print(f"SNLI clean: {len(snli_clean)} pairs", flush=True)

# ── Load model ONCE ──
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
st_model = SentenceTransformer(MODEL_NAME, device=device)
backbone_dim = st_model.get_sentence_embedding_dimension()
backbone = st_model[0].auto_model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Freeze backbone
for param in backbone.parameters():
    param.requires_grad = False
backbone.eval()

# ── Evaluation function ──
def evaluate_sts(encode_fn, name=""):
    emb1 = encode_fn(sts_s1)
    emb2 = encode_fn(sts_s2)
    cos = np.sum(emb1 * emb2, axis=1) / (
        np.linalg.norm(emb1, axis=1) * np.linalg.norm(emb2, axis=1) + 1e-8
    )
    corr, _ = spearmanr(cos, sts_scores)
    print(f"  [{name}] STS-B Spearman: {corr:.4f}", flush=True)
    return corr

# ── Baseline ──
print("\n=== BASELINE ===", flush=True)
baseline_spearman = evaluate_sts(
    lambda texts: st_model.encode(texts, batch_size=128, show_progress_bar=False, convert_to_numpy=True),
    "MiniLM-baseline"
)
print(f"Baseline: {baseline_spearman:.4f} (target to beat: {BASELINE_TARGET})", flush=True)

# ── Pre-encode SNLI through backbone (ONCE) ──
def pre_encode_snli(n_samples):
    """Pre-encode SNLI premises + hypotheses through frozen backbone."""
    indices = np.random.RandomState(42).choice(len(snli_clean), n_samples, replace=False)
    premises = [str(snli_clean[int(i)]["premise"]) for i in indices]
    hypotheses = [str(snli_clean[int(i)]["hypothesis"]) for i in indices]
    labels = np.array([snli_clean[int(i)]["label"] for i in indices])

    print(f"  Pre-encoding {n_samples} pairs through backbone...", flush=True)
    t0 = time.time()
    with torch.no_grad():
        # Batch encode
        all_p, all_h = [], []
        bs = 256
        for start in range(0, len(premises), bs):
            batch_p = premises[start:start+bs]
            batch_h = hypotheses[start:start+bs]
            enc_p = tokenizer(batch_p, truncation=True, max_length=128, padding=True, return_tensors="pt")
            enc_h = tokenizer(batch_h, truncation=True, max_length=128, padding=True, return_tensors="pt")
            enc_p = {k: v.to(device) for k, v in enc_p.items()}
            enc_h = {k: v.to(device) for k, v in enc_h.items()}

            out_p = backbone(**enc_p).last_hidden_state
            mask_p = enc_p["attention_mask"].unsqueeze(-1).float()
            pooled_p = (out_p * mask_p).sum(dim=1) / mask_p.sum(dim=1).clamp(min=1)

            out_h = backbone(**enc_h).last_hidden_state
            mask_h = enc_h["attention_mask"].unsqueeze(-1).float()
            pooled_h = (out_h * mask_h).sum(dim=1) / mask_h.sum(dim=1).clamp(min=1)

            all_p.append(pooled_p.cpu())
            all_h.append(pooled_h.cpu())

    p_embs = torch.cat(all_p)  # [N, 384]
    h_embs = torch.cat(all_h)  # [N, 384]
    labels_t = torch.tensor(labels, dtype=torch.long)
    print(f"  Pre-encode done in {time.time()-t0:.1f}s. Shape: {p_embs.shape}", flush=True)
    return p_embs, h_embs, labels_t

# Pre-encode all sizes we'll need
print("\n=== PRE-ENCODING DATA ===", flush=True)
cached = {}
for n in [50000, 200000, 500000]:
    if n <= len(snli_clean):
        cached[n] = pre_encode_snli(n)

# ── PGLF Projection Head ──
class PGLFHead(nn.Module):
    def __init__(self, dim, hidden_dim, gain):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, dim),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=gain)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

# ── Training function ──
def train_pglf(p_embs, h_embs, labels, epochs, lr, temperature, gain, batch_size=128):
    """Train PGLF head on pre-encoded embeddings."""
    n = len(p_embs)
    head = PGLFHead(backbone_dim, backbone_dim, gain).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.01)

    p_embs_gpu = p_embs.to(device)
    h_embs_gpu = h_embs.to(device)
    labels_gpu = labels.to(device)

    head.train()
    global_step = 0

    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = perm[start:start+batch_size]
            p = head(p_embs_gpu[idx])
            h = head(h_embs_gpu[idx])
            lbl = labels_gpu[idx]
            B = p.shape[0]

            p_norm = F.normalize(p, dim=-1)
            h_norm = F.normalize(h, dim=-1)

            sim = (p_norm @ h_norm.T) / temperature
            sim = sim.clamp(-20.0, 20.0)
            diag = torch.arange(B, device=device)

            loss_p = F.cross_entropy(sim, diag, reduction='none')
            loss_h = F.cross_entropy(sim.T, diag, reduction='none')
            weights = torch.where(lbl == 0, 1.0, torch.where(lbl == 1, 0.3, 0.05))
            loss = (loss_p * weights + loss_h * weights).mean() / 2

            # Alignment for entailment
            is_ent = (lbl == 0)
            if is_ent.any():
                align = ((p_norm[is_ent] - h_norm[is_ent]) ** 2).sum(dim=-1).mean()
                loss = loss + 0.5 * align

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            global_step += 1

    return head

# ── Evaluate trained head ──
def eval_with_head(head):
    """Evaluate head on STS-B."""
    head.eval()
    def encode_fn(texts):
        all_embs = []
        bs = 128
        with torch.no_grad():
            for start in range(0, len(texts), bs):
                batch = texts[start:start+bs]
                enc = tokenizer(batch, truncation=True, max_length=128, padding=True, return_tensors="pt")
                enc = {k: v.to(device) for k, v in enc.items()}
                out = backbone(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).float()
                pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                projected = head(pooled)
                normalized = F.normalize(projected, dim=-1)
                all_embs.append(normalized.cpu().numpy())
        return np.concatenate(all_embs)

    return evaluate_sts(encode_fn, "PGLF")

# ── Grid Search Phase A: One-at-a-time variation ──
print("\n" + "=" * 70)
print("PHASE A: Grid Search — One-at-a-time variation from baseline")
print("=" * 70)

# Baseline config
base_config = {"data": 50000, "epochs": 1, "lr": 1e-4, "temp": 0.07, "gain": 0.1}

# Variations to test
configs = [
    # (name, overrides_from_base)
    ("baseline", {}),
    # Data scaling
    ("data=200K", {"data": 200000}),
    ("data=500K", {"data": 500000}),
    # More epochs
    ("epochs=3", {"epochs": 3}),
    ("epochs=5", {"epochs": 5}),
    # Higher gain (less conservative init)
    ("gain=0.5", {"gain": 0.5}),
    ("gain=1.0", {"gain": 1.0}),
    # Higher LR
    ("lr=5e-4", {"lr": 5e-4}),
    ("lr=1e-3", {"lr": 1e-3}),
    # Temperature variations
    ("temp=0.05", {"temp": 0.05}),
    ("temp=0.1", {"temp": 0.1}),
    # Combined: more data + more epochs + higher gain
    ("combo1", {"data": 200000, "epochs": 3, "gain": 0.5}),
    ("combo2", {"data": 500000, "epochs": 3, "gain": 0.5, "lr": 5e-4}),
    ("combo3", {"data": 200000, "epochs": 5, "gain": 1.0, "lr": 5e-4, "temp": 0.05}),
]

results = []
best_spearman = baseline_spearman
best_config = "baseline"

for name, overrides in configs:
    cfg = {**base_config, **overrides}

    print(f"\n--- Config: {name} ---", flush=True)
    print(f"  data={cfg['data']}, epochs={cfg['epochs']}, lr={cfg['lr']}, temp={cfg['temp']}, gain={cfg['gain']}", flush=True)

    if cfg["data"] not in cached:
        print(f"  SKIP: data size {cfg['data']} not pre-encoded", flush=True)
        continue

    p_embs, h_embs, labels = cached[cfg["data"]]

    t0 = time.time()
    head = train_pglf(p_embs, h_embs, labels, cfg["epochs"], cfg["lr"], cfg["temp"], cfg["gain"])
    train_time = time.time() - t0

    spearman = eval_with_head(head)

    delta = spearman - baseline_spearman
    verdict = "IMPROVES" if delta > 0.002 else ("NEUTRAL" if delta > -0.002 else "HURTS")

    result = {
        "config": name,
        "data": cfg["data"],
        "epochs": cfg["epochs"],
        "lr": cfg["lr"],
        "temp": cfg["temp"],
        "gain": cfg["gain"],
        "spearman": round(spearman, 4),
        "delta": round(delta, 4),
        "verdict": verdict,
        "train_time_s": round(train_time, 1),
    }
    results.append(result)

    # Save incrementally
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(result) + "\n")

    print(f"  Spearman: {spearman:.4f} (Δ={delta:+.4f}) [{verdict}] ({train_time:.1f}s)", flush=True)

    if spearman > best_spearman:
        best_spearman = spearman
        best_config = name
        print(f"  *** NEW BEST! ***", flush=True)

    # Cleanup GPU
    del head
    torch.cuda.empty_cache()

# ── Final Summary ──
print("\n" + "=" * 70)
print("FINAL RESULTS SUMMARY")
print("=" * 70)
print(f"\nBaseline (MiniLM-L6-v2): {baseline_spearman:.4f}")
print(f"\n{'Config':<25} {'Spearman':>10} {'Delta':>8} {'Verdict':>10} {'Time':>8}")
print("-" * 65)
for r in results:
    print(f"{r['config']:<25} {r['spearman']:>10.4f} {r['delta']:>+8.4f} {r['verdict']:>10} {r['train_time_s']:>7.1f}s")

print(f"\nBest: {best_config} → {best_spearman:.4f}", flush=True)

if best_spearman > BASELINE_TARGET + 0.002:
    print(f"\nVERDICT: PGLF IMPROVES over baseline! ({best_spearman:.4f} > {BASELINE_TARGET})", flush=True)
    print("  → The approach has potential. Worth deeper investigation.", flush=True)
elif best_spearman > BASELINE_TARGET - 0.002:
    print(f"\nVERDICT: PGLF is NEUTRAL (within noise of baseline).", flush=True)
    print("  → No benefit but no harm either. Not worth pursuing.", flush=True)
else:
    print(f"\nVERDICT: PGLF consistently HURTS. All configs worse than baseline.", flush=True)
    print("  → DESCARTADO: The projection layer approach is fundamentally flawed for unimodal text.", flush=True)

print("\nDone!", flush=True)
