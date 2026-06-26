"""
PGLF Quick Test: Do PGLF-style projection layers add value over plain MiniLM-L6-v2?

Compares:
  1. BASELINE: plain sentence-transformers/all-MiniLM-L6-v2 on STS-B
  2. PGLF: frozen MiniLM + trained projection (384->384) + contrastive loss on SNLI

Metrics: Spearman rank correlation on STS-B (standard sentence embedding benchmark)
"""

import time
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import spearmanr

print("=" * 70)
print("PGLF Quick Test: MiniLM-L6-v2 vs MiniLM + PGLF Projection")
print("=" * 70)

# ---------------------------------------------------------------------------
# 0. Device
# ---------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n[Setup] Device: {device}")
if device == "cuda":
    print(f"[Setup] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[Setup] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ---------------------------------------------------------------------------
# 1. Load STS-B for evaluation
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 1: Loading STS Benchmark (GLUE STS-B)")
print("=" * 70)

from datasets import load_dataset

print("[Data] Downloading STS-B validation split...")
stsb = load_dataset("glue", "stsb", split="validation")
print(f"[Data] STS-B loaded: {len(stsb)} pairs")

# Extract sentences and scores
sts_sentences1 = [str(x) for x in stsb["sentence1"]]
sts_sentences2 = [str(x) for x in stsb["sentence2"]]
sts_scores = np.array(stsb["label"], dtype=np.float32)
# Normalize scores to [0, 1] range (STS-B uses 0-5)
sts_scores = sts_scores / 5.0
print(f"[Data] Score range: {sts_scores.min():.2f} - {sts_scores.max():.2f}")
print(f"[Data] Sample pair: '{sts_sentences1[0][:60]}...' / '{sts_sentences2[0][:60]}...'")
print(f"[Data] Sample score: {sts_scores[0]:.2f}")

# ---------------------------------------------------------------------------
# 2. Load SNLI for training the PGLF adapter
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 2: Loading SNLI train split (first 50K)")
print("=" * 70)

print("[Data] Downloading SNLI train split...")
snli_train = load_dataset("snli", split="train[:50000]")
print(f"[Data] SNLI loaded: {len(snli_train)} pairs")

# Filter out examples with label -1 (no consensus)
snli_filtered = snli_train.filter(lambda x: x["label"] != -1)
print(f"[Data] After filtering invalid labels: {len(snli_filtered)} pairs")

# Label mapping: 0=entailment (positive), 2=contradiction (negative), 1=neutral
snli_premises = [str(x) for x in snli_filtered["premise"]]
snli_hypotheses = [str(x) for x in snli_filtered["hypothesis"]]
snli_labels = np.array(snli_filtered["label"], dtype=np.int64)
print(f"[Data] Label distribution: entailment={np.sum(snli_labels==0)}, "
      f"neutral={np.sum(snli_labels==1)}, contradiction={np.sum(snli_labels==2)}")

# ---------------------------------------------------------------------------
# 3. Load sentence-transformers model
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 3: Loading all-MiniLM-L6-v2")
print("=" * 70)

from sentence_transformers import SentenceTransformer

print("[Model] Loading sentence-transformers/all-MiniLM-L6-v2...")
model_name = "sentence-transformers/all-MiniLM-L6-v2"
st_model = SentenceTransformer(model_name, device=device)
backbone_dim = st_model.get_sentence_embedding_dimension()
print(f"[Model] Loaded. Embedding dim: {backbone_dim}")

# ---------------------------------------------------------------------------
# 4. Define evaluation function
# ---------------------------------------------------------------------------
def evaluate_sts(model_or_encoder, name, batch_size=128):
    """Evaluate on STS-B using Spearman correlation of cosine similarities."""
    print(f"\n[Eval] Evaluating {name} on STS-B ({len(sts_sentences1)} pairs)...")
    
    if isinstance(model_or_encoder, SentenceTransformer):
        # Direct sentence-transformers encoding
        emb1 = model_or_encoder.encode(sts_sentences1, batch_size=batch_size, 
                                        show_progress_bar=True, convert_to_numpy=True)
        emb2 = model_or_encoder.encode(sts_sentences2, batch_size=batch_size,
                                        show_progress_bar=True, convert_to_numpy=True)
    else:
        # Custom encode function (for PGLF model)
        emb1 = model_or_encoder(sts_sentences1, batch_size=batch_size)
        emb2 = model_or_encoder(sts_sentences2, batch_size=batch_size)
    
    # Cosine similarity
    cos_sims = np.sum(emb1 * emb2, axis=1) / (
        np.linalg.norm(emb1, axis=1) * np.linalg.norm(emb2, axis=1) + 1e-8
    )
    
    # Spearman correlation
    corr, pval = spearmanr(cos_sims, sts_scores)
    print(f"[Eval] {name} Spearman: {corr:.4f} (p={pval:.2e})")
    return corr

# ---------------------------------------------------------------------------
# 5. BASELINE: Evaluate plain MiniLM
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 4: BASELINE - Plain MiniLM-L6-v2 on STS-B")
print("=" * 70)

t0 = time.time()
baseline_spearman = evaluate_sts(st_model, "MiniLM-L6-v2 (baseline)")
baseline_time = time.time() - t0
print(f"[Eval] Baseline time: {baseline_time:.1f}s")

# ---------------------------------------------------------------------------
# 6. Build PGLF adapter: frozen MiniLM + trainable projection
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 5: Building PGLF adapter (frozen MiniLM + trainable projection)")
print("=" * 70)

# Get the underlying transformer model to freeze it
backbone = st_model[0].auto_model  # The actual BERT/MiniLM model
print(f"[PGLF] Backbone: {type(backbone).__name__}")

# Freeze backbone
for param in backbone.parameters():
    param.requires_grad = False
frozen_params = sum(p.numel() for p in backbone.parameters())
trainable_in_backbone = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
print(f"[PGLF] Backbone params: {frozen_params:,} (trainable: {trainable_in_backbone:,})")

# PGLF projection head - inspired by the architecture in contrastive_head.py
# Linear -> GELU -> Linear (same as encoders.py proj head)
class PGLFProjectionHead(nn.Module):
    """Projection head: maps backbone embeddings through a trainable adapter."""
    def __init__(self, in_dim, hidden_dim=None, out_dim=None, dropout=0.1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = in_dim
        if out_dim is None:
            out_dim = in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        # Initialize with near-identity for stable start
        nn.init.xavier_uniform_(self.net[0].weight, gain=0.1)
        nn.init.zeros_(self.net[0].bias)
        nn.init.xavier_uniform_(self.net[3].weight, gain=0.1)
        nn.init.zeros_(self.net[3].bias)
    
    def forward(self, x):
        return self.net(x)

pglf_head = PGLFProjectionHead(backbone_dim, backbone_dim, backbone_dim).to(device)
pglf_params = sum(p.numel() for p in pglf_head.parameters())
print(f"[PGLF] Projection head params: {pglf_params:,}")
print(f"[PGLF] Total trainable params: {pglf_params:,} (backbone is frozen)")

# Tokenizer for SNLI
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# ---------------------------------------------------------------------------
# 7. Train PGLF adapter with contrastive loss on SNLI
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 6: Training PGLF adapter on SNLI (contrastive)")
print("=" * 70)

# Training hyperparameters
BATCH_SIZE = 64
LR = 1e-4
N_EPOCHS = 1
TEMPERATURE = 0.07
MAX_LEN = 128

# Prepare dataset - we need (premise, hypothesis, label) triplets
# Strategy: for each batch, encode premises and hypotheses through frozen MiniLM + PGLF head
# Use entailment pairs as positives, contradiction as hard negatives
# InfoNCE loss (same as HypersphereContrastiveLoss in contrastive_head.py)

optimizer = torch.optim.AdamW(pglf_head.parameters(), lr=LR, weight_decay=0.01)

# DataLoader
from torch.utils.data import Dataset, DataLoader

class SNLIDataset(Dataset):
    def __init__(self, premises, hypotheses, labels, tokenizer, max_len=128):
        self.premises = premises
        self.hypotheses = hypotheses
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.premises)
    
    def __getitem__(self, idx):
        p = self.tokenizer(
            self.premises[idx], truncation=True, max_length=self.max_len,
            padding="max_length", return_tensors="pt"
        )
        h = self.tokenizer(
            self.hypotheses[idx], truncation=True, max_length=self.max_len,
            padding="max_length", return_tensors="pt"
        )
        return {
            "premise_ids": p["input_ids"].squeeze(0),
            "premise_mask": p["attention_mask"].squeeze(0),
            "hypo_ids": h["input_ids"].squeeze(0),
            "hypo_mask": h["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }

print(f"[Train] Preparing SNLI DataLoader (batch_size={BATCH_SIZE})...")
snli_dataset = SNLIDataset(snli_premises, snli_hypotheses, snli_labels, tokenizer, MAX_LEN)
snli_loader = DataLoader(
    snli_dataset, batch_size=BATCH_SIZE, shuffle=True, 
    num_workers=2, pin_memory=True, drop_last=True
)
n_steps = len(snli_loader) * N_EPOCHS
print(f"[Train] Steps per epoch: {len(snli_loader)}, total: {n_steps}")

# Move backbone to eval mode
backbone.eval()
pglf_head.train()

def encode_with_backbone(input_ids, attention_mask):
    """Encode through frozen backbone (MiniLM)."""
    with torch.no_grad():
        output = backbone(input_ids=input_ids, attention_mask=attention_mask)
        # Mean pooling
        token_embs = output.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).float()
        pooled = (token_embs * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
    return pooled

# Training loop
print(f"\n[Train] Starting training...")
global_step = 0
running_loss = 0.0
running_acc = 0.0
t_train_start = time.time()

for epoch in range(N_EPOCHS):
    for batch_idx, batch in enumerate(snli_loader):
        # Move to device
        p_ids = batch["premise_ids"].to(device)
        p_mask = batch["premise_mask"].to(device)
        h_ids = batch["hypo_ids"].to(device)
        h_mask = batch["hypo_mask"].to(device)
        labels = batch["label"].to(device)
        
        B = p_ids.shape[0]
        
        # Encode through frozen backbone
        p_embs = encode_with_backbone(p_ids, p_mask)  # [B, 384]
        h_embs = encode_with_backbone(h_ids, h_mask)  # [B, 384]
        
        # Through trainable PGLF projection head
        p_proj = pglf_head(p_embs)  # [B, 384]
        h_proj = pglf_head(h_embs)  # [B, 384]
        
        # Normalize on hypersphere (like PGLF)
        p_norm = F.normalize(p_proj, dim=-1)
        h_norm = F.normalize(h_proj, dim=-1)
        
        # InfoNCE contrastive loss (same as HypersphereContrastiveLoss)
        # For entailment pairs (label=0): positive pairs
        # Use all pairs in batch with symmetric InfoNCE
        sim_matrix = p_norm @ h_norm.T / TEMPERATURE  # [B, B]
        sim_matrix = sim_matrix.clamp(-20.0, 20.0)  # prevent fp16 overflow
        
        # Weight the loss based on labels:
        # - entailment (0): strong positive
        # - neutral (1): weak positive (lower weight)
        # - contradiction (2): negative (skip or zero weight)
        
        # Strategy: only use entailment pairs as positives for InfoNCE
        # Create a mask where (i,j) is positive if labels[i]==0 AND i==j
        is_entailment = (labels == 0)
        
        # Simple approach: standard InfoNCE where diagonal is positive
        # But mask out non-entailment anchors
        diag_labels = torch.arange(B, device=device)
        
        loss_p = F.cross_entropy(sim_matrix, diag_labels, reduction='none')
        loss_h = F.cross_entropy(sim_matrix.T, diag_labels, reduction='none')
        
        # Weight: full weight for entailment, 0.3 for neutral, 0.0 for contradiction
        weights = torch.where(labels == 0, 1.0, 
                     torch.where(labels == 1, 0.3, 0.05))
        
        loss = (loss_p * weights + loss_h * weights).mean() / 2
        
        # Also add alignment loss for entailment pairs (like UniformityAlignmentLoss)
        if is_entailment.any():
            align_loss = ((p_norm[is_entailment] - h_norm[is_entailment]) ** 2).sum(dim=-1).mean()
            loss = loss + 0.5 * align_loss
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pglf_head.parameters(), 1.0)
        optimizer.step()
        
        # Track accuracy (how often diagonal is argmax)
        with torch.no_grad():
            acc = ((sim_matrix.argmax(dim=1) == diag_labels).float().mean() +
                   (sim_matrix.T.argmax(dim=1) == diag_labels).float().mean()) / 2
        
        global_step += 1
        running_loss += loss.item()
        running_acc += acc.item()
        
        if global_step % 50 == 0:
            avg_loss = running_loss / 50
            avg_acc = running_acc / 50
            elapsed = time.time() - t_train_start
            steps_per_sec = global_step / elapsed
            eta = (n_steps - global_step) / steps_per_sec if steps_per_sec > 0 else 0
            print(f"  Step {global_step:>5d}/{n_steps} | Loss: {avg_loss:.4f} | "
                  f"Acc: {avg_acc:.3f} | Speed: {steps_per_sec:.1f} steps/s | "
                  f"ETA: {eta/60:.1f}min")
            running_loss = 0.0
            running_acc = 0.0

train_time = time.time() - t_train_start
print(f"\n[Train] Training complete! Time: {train_time/60:.1f} min ({train_time:.0f}s)")
print(f"[Train] Final head weights norm: {pglf_head.net[0].weight.norm().item():.4f}")

# ---------------------------------------------------------------------------
# 8. Evaluate PGLF + MiniLM
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 7: Evaluating PGLF+MiniLM on STS-B")
print("=" * 70)

pglf_head.eval()

def encode_pglf(texts, batch_size=128):
    """Encode texts through frozen MiniLM + trained PGLF projection."""
    all_embs = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]
        encoded = tokenizer(
            batch_texts, truncation=True, max_length=MAX_LEN,
            padding=True, return_tensors="pt"
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        
        with torch.no_grad():
            backbone_embs = encode_with_backbone(encoded["input_ids"], encoded["attention_mask"])
            projected = pglf_head(backbone_embs)
            normalized = F.normalize(projected, dim=-1)
        
        all_embs.append(normalized.cpu().numpy())
    
    return np.concatenate(all_embs, axis=0)

t0 = time.time()
pglf_spearman = evaluate_sts(encode_pglf, "PGLF+MiniLM-L6-v2")
pglf_time = time.time() - t0
print(f"[Eval] PGLF eval time: {pglf_time:.1f}s")

# ---------------------------------------------------------------------------
# 9. Results
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

diff = pglf_spearman - baseline_spearman
pct = (diff / abs(baseline_spearman)) * 100 if baseline_spearman != 0 else 0

print(f"""
  Model                          Spearman (STS-B)    Time
  ------------------------------ ------------------  ----------
  MiniLM-L6-v2 (baseline)        {baseline_spearman:.4f}           {baseline_time:.1f}s
  PGLF + MiniLM-L6-v2            {pglf_spearman:.4f}           {pglf_time:.1f}s
  ------------------------------ ------------------  ----------
  Difference:                    {diff:+.4f} ({pct:+.2f}%)
  Training time:                 {train_time/60:.1f} min
""")

if diff > 0.005:
    print("  VERDICT: PGLF projection IMPROVES over plain MiniLM! (+{:.4f})".format(diff))
    print("  The contrastive training on SNLI teaches useful structure.")
elif diff > -0.005:
    print("  VERDICT: PGLF projection is NEUTRAL (~same as baseline).")
    print("  The adapter doesn't hurt but doesn't help either.")
else:
    print("  VERDICT: PGLF projection HURTS ({:.4f}).".format(diff))
    print("  Possible: insufficient training, wrong hyperparams, or task mismatch.")

print("\n" + "=" * 70)
print("Done!")
print("=" * 70)
