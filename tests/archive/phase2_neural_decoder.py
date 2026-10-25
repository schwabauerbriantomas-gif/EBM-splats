#!/usr/bin/env python3
"""
Phase 2: Neural Decoder for Energy-Guided Generation

Instead of retrieving existing texts, generate NEW text conditioned on
a sampled point in the hypersphere. The sampled point serves as a
"semantic seed" that a small GPT-2 decoder uses as its initial prefix.

Architecture:
  1. Sample point z in S^383 via RF + energy guidance
  2. Project z -> GPT-2 embedding space (384D -> 768D) via MLP
  3. Feed z_projected as a "virtual token" prefix to GPT-2
  4. Generate text autoregressively from GPT-2 conditioned on the prefix

Training:
  - For each TinyStory: embed with MiniLM -> get z
  - Project z -> 768D -> feed as prefix to GPT-2
  - Train GPT-2 to generate the original story given the prefix
  - Loss: standard cross-entropy on the story tokens

This tests whether the energy-guided samples produce coherent novel text.
"""

import time, json, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
from transformers import GPT2LMHeadModel, GPT2Tokenizer

device = "cuda"
torch.cuda.set_per_process_memory_fraction(0.80)
RESULTS = "/root/EBM-splats/tests/phase2_decoder_results.jsonl"

def normalize_sphere(x):
    return F.normalize(x, dim=-1)

def project_to_tangent(x, v):
    dot = (x * v).sum(dim=-1, keepdim=True)
    return v - dot * x

# ── 1. Load models ──
print("=== STEP 1: Load models ===", flush=True)
from sentence_transformers import SentenceTransformer

st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
emb_dim = st_model.get_embedding_dimension()
print(f"MiniLM dim: {emb_dim}", flush=True)

gpt2_name = "gpt2"
gpt2 = GPT2LMHeadModel.from_pretrained(gpt2_name).to(device)
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_name)
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
gpt2_dim = gpt2.config.n_embd  # 768
print(f"GPT-2 dim: {gpt2_dim}", flush=True)

# Freeze GPT-2 for prefix-tuning approach
for p in gpt2.parameters():
    p.requires_grad = False
gpt2.eval()

# ── 2. Load data ──
print("\n=== STEP 2: Load TinyStories ===", flush=True)
data_path = "/mnt/d/datasets/ebm/tinystories_train.txt"
with open(data_path, "r", encoding="utf-8", errors="ignore") as f:
    all_texts = [line.strip() for line in f.readlines() if len(line.strip()) > 80]

N_STORIES = 8000  # Keep smaller for faster training
texts = all_texts[:N_STORIES]
print(f"Stories: {len(texts)}", flush=True)

# ── 3. Embed all stories ──
print("Embedding stories...", flush=True)
t0 = time.time()
embeddings = st_model.encode(texts, batch_size=256, show_progress_bar=False,
                             convert_to_tensor=True, normalize_embeddings=True)
print(f"Embedded in {time.time()-t0:.1f}s. Shape: {embeddings.shape}", flush=True)

# ── 4. Train RF velocity network (same as before) ──
print("\n=== STEP 3: Train RF ===", flush=True)

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
optimizer_rf = torch.optim.AdamW(vel_net.parameters(), lr=1e-3, weight_decay=0.01)

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
    optimizer_rf.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(vel_net.parameters(), 1.0)
    optimizer_rf.step()
print(f"RF trained in {time.time()-t0:.1f}s", flush=True)

# ── 5. Clusters for guidance ──
print("\n=== STEP 4: KMeans ===", flush=True)
N_CLUSTERS = 30
emb_np = embeddings.cpu().numpy()
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(emb_np)
cluster_centers = normalize_sphere(torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=device))
cluster_counts = np.bincount(cluster_labels, minlength=N_CLUSTERS)
print(f"Clusters: {N_CLUSTERS}, sizes: min={cluster_counts.min()}, max={cluster_counts.max()}", flush=True)

# Pick 2 well-separated topics
center_sims = (cluster_centers @ cluster_centers.T).cpu().numpy()
candidates = [i for i in range(N_CLUSTERS) if cluster_counts[i] >= 100]
best_pair = None
best_sim = 1.0
for i in candidates:
    for j in candidates:
        if i < j and center_sims[i, j] < best_sim:
            best_sim = center_sims[i, j]
            best_pair = (i, j)
topic_a, topic_b = best_pair
print(f"Topics: A={topic_a} ({cluster_counts[topic_a]}), B={topic_b} ({cluster_counts[topic_b]}), sim={best_sim:.4f}", flush=True)

# ── 6. Prefix projection layer ──
print("\n=== STEP 5: Train prefix projection ===", flush=True)

class PrefixProjector(nn.Module):
    """Project MiniLM embedding (384D) -> GPT-2 prefix tokens (n_prefix x 768D)."""
    def __init__(self, in_dim, gpt2_dim, n_prefix=4):
        super().__init__()
        self.n_prefix = n_prefix
        self.net = nn.Sequential(
            nn.Linear(in_dim, gpt2_dim * 2),
            nn.GELU(),
            nn.Linear(gpt2_dim * 2, gpt2_dim * n_prefix),
        )
    def forward(self, z):
        """z: [B, in_dim] -> [B, n_prefix, gpt2_dim]"""
        out = self.net(z)
        return out.view(z.shape[0], self.n_prefix, -1)

N_PREFIX = 4
prefix_proj = PrefixProjector(emb_dim, gpt2_dim, N_PREFIX).to(device)
optimizer_pp = torch.optim.AdamW(prefix_proj.parameters(), lr=5e-4, weight_decay=0.01)

# Training: for each story, embed -> project -> use as GPT-2 prefix -> predict story tokens
MAX_LEN = 64
BATCH = 16

print(f"Training prefix projector: {N_PREFIX} prefix tokens, max_len={MAX_LEN}, batch={BATCH}", flush=True)
t0 = time.time()
train_losses = []

for epoch in range(3):
    perm = torch.randperm(N_STORIES)
    epoch_loss = 0
    n_batches = 0

    for start in range(0, N_STORIES, BATCH):
        batch_idx = perm[start:start+BATCH]
        if len(batch_idx) < BATCH:
            continue

        batch_texts = [texts[i] for i in batch_idx.cpu().numpy()]
        batch_embs = embeddings[batch_idx]  # [B, 384]

        # Tokenize stories
        enc = gpt2_tokenizer(batch_texts, truncation=True, max_length=MAX_LEN,
                             padding="max_length", return_tensors="pt")
        input_ids = enc["input_ids"].to(device)  # [B, MAX_LEN]
        attention_mask = enc["attention_mask"].to(device)

        # Project embeddings -> prefix tokens
        prefix = prefix_proj(batch_embs)  # [B, n_prefix, 768]

        # Get GPT-2 word embeddings for the story
        with torch.no_grad():
            word_embs = gpt2.transformer.wte(input_ids)  # [B, MAX_LEN, 768]

        # Concatenate prefix + story embeddings
        inputs_embeds = torch.cat([prefix, word_embs[:, :-1]], dim=1)  # [B, n_prefix + MAX_LEN - 1, 768]

        # Build attention mask for prefix + story
        prefix_mask = torch.ones(BATCH, N_PREFIX, device=device)
        full_mask = torch.cat([prefix_mask, attention_mask[:, :-1]], dim=1)

        # Forward through GPT-2 — gradient flows through inputs_embeds to prefix
        outputs = gpt2(inputs_embeds=inputs_embeds, attention_mask=full_mask)

        # GPT-2 logits: [B, n_prefix + MAX_LEN - 1, vocab]
        # We want to predict tokens at positions [n_prefix:] (the story tokens)
        logits = outputs.logits[:, N_PREFIX:, :]  # [B, MAX_LEN - 1, vocab]

        # Target: story tokens shifted by 1
        targets = input_ids[:, 1:]  # [B, MAX_LEN - 1]

        # Cross-entropy loss
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction="none"
        )
        # Mask out padding
        target_mask = (targets != gpt2_tokenizer.pad_token_id).float().reshape(-1)
        loss = (loss * target_mask).sum() / (target_mask.sum() + 1e-8)

        optimizer_pp.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(prefix_proj.parameters(), 1.0)
        optimizer_pp.step()

        epoch_loss += loss.item()
        n_batches += 1

    avg_loss = epoch_loss / max(n_batches, 1)
    train_losses.append(avg_loss)
    print(f"  Epoch {epoch+1}/3 | Loss: {avg_loss:.4f} | {time.time()-t0:.1f}s elapsed", flush=True)

print(f"Prefix projector trained in {time.time()-t0:.1f}s", flush=True)

# ── 7. Generation function ──
print("\n=== STEP 6: Generation test ===", flush=True)

def sample_rf_guided(n_samples, guidance_vec, n_steps=2):
    x = normalize_sphere(torch.randn(n_samples, emb_dim, device=device))
    dt = 1.0 / n_steps
    for step in range(n_steps):
        t = torch.full((n_samples, 1), step * dt, device=device)
        v = vel_net(x, t)
        for ci, weight in guidance_vec:
            center = cluster_centers[int(ci)]
            direction = project_to_tangent(x, center.unsqueeze(0) - x)
            direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
            v = v + weight * direction
        v_norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        x = torch.cos(v_norm * dt) * x + torch.sin(v_norm * dt) * (v / v_norm)
        x = normalize_sphere(x)
    return x

def generate_story(z_sample, prefix_proj, gpt2, tokenizer, max_new=50):
    """Generate a story from a sampled point z."""
    prefix = prefix_proj(z_sample.unsqueeze(0))  # [1, n_prefix, 768]

    # Start generation from the prefix
    inputs_embeds = prefix
    generated = []

    with torch.no_grad():
        for _ in range(max_new):
            outputs = gpt2(inputs_embeds=inputs_embeds)
            next_logits = outputs.logits[:, -1, :]
            next_token = next_logits.argmax(dim=-1)
            generated.append(next_token.item())

            if next_token.item() == tokenizer.eos_token_id:
                break

            next_emb = gpt2.transformer.wte(next_token).unsqueeze(1)  # [1, 1, 768]
            inputs_embeds = torch.cat([inputs_embeds, next_emb], dim=1)

    return tokenizer.decode(generated, skip_special_tokens=True)

def generate_batch(z_samples, prefix_proj, gpt2, tokenizer, max_new=50):
    """Generate stories from multiple sampled points."""
    B = z_samples.shape[0]
    prefix = prefix_proj(z_samples)  # [B, n_prefix, 768]

    inputs_embeds = prefix
    generated = [[] for _ in range(B)]
    done = [False] * B

    with torch.no_grad():
        for _ in range(max_new):
            outputs = gpt2(inputs_embeds=inputs_embeds)
            next_logits = outputs.logits[:, -1, :]
            next_tokens = next_logits.argmax(dim=-1)  # [B]

            for i in range(B):
                if not done[i]:
                    tid = next_tokens[i].item()
                    if tid == tokenizer.eos_token_id:
                        done[i] = True
                    else:
                        generated[i].append(tid)

            if all(done):
                break

            next_emb = gpt2.transformer.wte(next_tokens).unsqueeze(1)
            inputs_embeds = torch.cat([inputs_embeds, next_emb], dim=1)

    return [tokenizer.decode(g, skip_special_tokens=True) for g in generated]

# ── 8. Generate stories under different conditions ──
GS = 1.5
N_GEN = 8

conditions = [
    ("baseline", []),
    ("boost_A", [(topic_a, GS)]),
    ("boost_B", [(topic_b, GS)]),
    ("boost_AB", [(topic_a, GS * 0.7 / 0.5), (topic_b, GS * 0.3 / 0.5)]),
]

print(f"\nGenerating {N_GEN} stories per condition...\n", flush=True)

all_generated = {}

for cond_name, guidance in conditions:
    print(f"--- Condition: {cond_name} ---", flush=True)
    z_samples = sample_rf_guided(N_GEN, guidance)
    stories = generate_batch(z_samples, prefix_proj, gpt2, gpt2_tokenizer, max_new=40)

    all_generated[cond_name] = stories

    for i, s in enumerate(stories):
        print(f"  [{i}] {s[:150]}", flush=True)

    # Compute similarity to cluster centers for verification
    with torch.no_grad():
        gen_embs = st_model.encode(stories, batch_size=32, show_progress_bar=False,
                                   convert_to_tensor=True, normalize_embeddings=True)
        sim_a = (gen_embs @ cluster_centers[topic_a]).mean().item()
        sim_b = (gen_embs @ cluster_centers[topic_b]).mean().item()

    result = {
        "condition": cond_name,
        "sim_to_A": round(sim_a, 4),
        "sim_to_B": round(sim_b, 4),
        "n_stories": len(stories),
    }
    with open(RESULTS, "a") as f:
        f.write(json.dumps(result) + "\n")

    print(f"  Avg sim to topic A: {sim_a:.4f}, topic B: {sim_b:.4f}\n", flush=True)

# ── 9. Summary ──
print("=" * 70)
print("NEURAL DECODER — RESULTS")
print("=" * 70)

print(f"\nPrefix projector loss: {' -> '.join(f'{l:.4f}' for l in train_losses)}")
print(f"\nGeneration quality check — similarity to target topics:")
print(f"{'Condition':<20} {'Sim to A':>10} {'Sim to B':>10}")
print("-" * 44)
for cond_name, _ in conditions:
    r = [json.loads(line) for line in open(RESULTS)]
    for rr in r:
        if rr["condition"] == cond_name:
            print(f"{cond_name:<20} {rr['sim_to_A']:>10.4f} {rr['sim_to_B']:>10.4f}")
            break

# Check if guidance affects generated text
base_sim_a = [rr for rr in [json.loads(l) for l in open(RESULTS)] if rr["condition"] == "baseline"][0]["sim_to_A"]
boost_a_sim_a = [rr for rr in [json.loads(l) for l in open(RESULTS)] if rr["condition"] == "boost_A"][0]["sim_to_A"]

delta = boost_a_sim_a - base_sim_a
print(f"\nGuidance effect: sim_A went from {base_sim_a:.4f} (baseline) to {boost_a_sim_a:.4f} (boost A) → delta={delta:+.4f}")

if delta > 0.02:
    print("\nVERDICT: Neural decoder RESPONDS to energy guidance.", flush=True)
    print("  Generated text shifts toward the guided topic.", flush=True)
    print("  → EBM + RF + neural decoder = controllable text generation pipeline.", flush=True)
elif delta > 0.005:
    print("\nVERDICT: Weak response. Some topic shift but not strong.", flush=True)
else:
    print("\nVERDICT: Neural decoder does NOT respond to energy guidance.", flush=True)
    print("  The prefix projector may need more training or a different architecture.", flush=True)

print("\nDone!", flush=True)
