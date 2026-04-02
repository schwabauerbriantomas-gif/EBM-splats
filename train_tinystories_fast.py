#!/usr/bin/env python3
"""EBM Training - Fast and optimized version."""

import os
import sys
import time
import json
import math
import copy
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import EBMConfig
from geometry import normalize_sphere, project_to_tangent, exp_map
from score_network import ScoreNetwork

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
log = logging.getLogger('ebm_train')

device = torch.device('cuda')
config = EBMConfig()
log.info(f'GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

# Load TinyStories from local file
log.info('Loading TinyStories train (100MB subset)...')
token_texts = []
with open('D:/datasets/ebm/tinystories_train.txt', 'r', encoding='utf-8') as f:
    for line in f:
        text = line.strip()
        if text:
            token_texts.append(text)

log.info(f'Read {len(token_texts):,} story texts')

# Tokenize
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('gpt2')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

token_ids = tokenizer.encode('\n'.join(token_texts), add_special_tokens=False)
log.info(f'  {len(token_ids):,} tokens')

# Create dataset chunks
seq_len = 64
batch_size = 64
chunks = []
for i in range(0, len(token_ids), seq_len):
    chunk = token_ids[i:i+seq_len]
    if len(chunk) < seq_len:
        # Pad short chunks
        chunk = chunk + [tokenizer.pad_token_id] * (seq_len - len(chunk))
    chunks.append(chunk)

log.info(f'Created {len(chunks)} chunks of [{batch_size} x {seq_len}]')

# DataLoader
import torch
from torch.utils.data import TensorDataset, DataLoader

class TokenDataset(TensorDataset):
    def __init__(self, tokens):
        self.tokens = tokens
    
    def __len__(self):
        return len(self.tokens)
    
    def __getitem__(self, idx):
        return self.tokens[idx]

dataset = TokenDataset(torch.tensor(chunks, dtype=torch.long))
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                        num_workers=0, pin_memory=True)

log.info(f'Dataloader: {len(dataloader)} batches')

# Models
class Embedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(50257, 640)
    
    def forward(self, tokens):
        return normalize_sphere(self.emb(tokens))

class MLP(nn.Module):
    def __init__(self, dim=640, hidden=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 64, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
    
    def forward(self, x):
        return self.net(x)

# Score Network
score_net = ScoreNetwork(dim=config.latent_dim, hidden_dim=2048, sigma_emb_dim=64).to(device)
embedder = Embedder().to(device)

# EMA
ema_decay = 0.999
ema_shadow = {k: v.clone().detach() for k, v in score_net.state_dict().items()}

# Optimizer
optimizer = torch.optim.AdamW(score_net.parameters(), lr=1e-4, weight_decay=0.01)

# Scaler
scaler = torch.amp.GradScaler('cuda')

# Training loop
log.info('Starting training...')
history = []

N_EPOCHS = 100
noise_levels = config.noise_levels

for epoch in range(N_EPOCHS):
    score_net.train()
    t0 = time.time()
    total_loss = 0.0
    nan_count = 0
    
    for bi, batch in enumerate(dataloader):
        tokens = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        
        with torch.amp.autocast('cuda'):
            # Noise
            x_0 = embedder(tokens)
            B, S, D = x_0.shape
            
            # Beta annealing
            beta = min(1.0, 0.1 + 0.9 * epoch / max(N_EPOCHS * 0.5, 1))
            
            # Multi-noise DSM
            for sigma in noise_levels:
                sigma_eff = sigma * beta
                
                with torch.no_grad():
                    noise = torch.randn_like(x_0)
                    noise = project_to_tangent(x_0, noise)
                    x_t = normalize_sphere(x_0 + sigma_eff * noise)
                    x_t_flat = x_t.reshape(B * S, D)
                    residual = x_0.reshape(B * S, D) - x_t_flat
                    score_target = project_to_tangent(x_t_flat, residual) / (sigma_eff ** 2)
                
                # Input perturbation (10% chance)
                if torch.rand(1).item() < 0.1:
                    with torch.no_grad():
                        noise2 = torch.randn_like(x_t_flat) * 0.01
                        noise2 = project_to_tangent(x_t_flat, noise2)
                        x_t_flat = x_t_flat + noise2
                
                sigma_t = torch.full((x_t_flat.size(0),), sigma_eff, device=device)
                score_pred = score_net(x_t_flat, sigma_t)
                loss = F.mse_loss(score_pred, score_target)
                total_loss += loss.item()
        
        if total_loss != 0.0 and not torch.isnan(total_loss) and not torch.isinf(total_loss):
            scaler.scale(total_loss).backward()
            torch.nn.utils.clip_grad_norm_(score_net.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        
        # EMA update
        with torch.no_grad():
            for k, v in score_net.state_dict().items():
                if k in ema_shadow:
                    ema_shadow[k].mul_(ema_decay).add_(v.data, alpha=1 - ema_decay)
    
    avg_loss = total_loss / len(dataloader)
    dur = time.time() - t0
    mem = torch.cuda.max_memory_allocated() / 1e9
    
    log.info(f'E{epoch+1:3d}/{N_EPOCHS} | Loss: {avg_loss:.4f} | {dur:.1f}s | {mem:.1f}GB')
    history.append({'epoch': epoch + 1, 'loss': avg_loss, 'duration': dur, 'gpu_gb': mem})
    
    # Save every 10 epochs
    if (epoch + 1) % 10 == 0:
        torch.save(score_net.state_dict(), f'checkpoints_tinystories/score_net_epoch_{epoch+1}.pt')
        ema_state = {k: v.cpu() for k, v in ema_shadow.items()}
        torch.save(ema_state, f'checkpoints_tinystories/score_net_ema_epoch_{epoch+1}.pt')
        log.info(f'Checkpoint saved: {epoch+1}')
    
    # Save final
torch.save(score_net.state_dict(), f'checkpoints_tinystories/score_net_final.pt')
torch.save({k: v.cpu() for k, v in ema_shadow.items()}, f'checkpoints_tinystories/score_net_ema_final.pt')
log.info(f'Training complete. Final loss: {history[-1]["loss"]:.4f}')
