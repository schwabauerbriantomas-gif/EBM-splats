#!/usr/bin/env python3
"""Train EBM ScoreNetwork on TinyStories dataset."""

import sys, os, time, json, copy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger('ebm_train')

from config import EBMConfig
from score_network import ScoreNetwork
from geometry import normalize_sphere, project_to_tangent
from dataset_loader import get_dataloader

device = torch.device('cuda')
config = EBMConfig()
log.info(f'GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

# Load TinyStories
log.info('Loading TinyStories train (200MB subset)...')
dataloader, tokenizer = get_dataloader('tinystories', seq_len=64, batch_size=64, split='train', max_chars=200_000_000)
log.info(f'Dataloader: {len(dataloader)} batches')

# Models
class Embedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(50257, 640)
    def forward(self, tokens):
        return normalize_sphere(self.emb(tokens))

score_net = ScoreNetwork(dim=config.latent_dim).to(device)
embedder = Embedder().to(device)

# EMA
ema_decay = 0.999
ema_shadow = {k: v.clone().detach() for k, v in score_net.state_dict().items()}

optimizer = torch.optim.AdamW(score_net.parameters(), lr=1e-4, weight_decay=0.01)
scaler = torch.amp.GradScaler('cuda')

ckpt_dir = 'checkpoints_tinystories'
os.makedirs(ckpt_dir, exist_ok=True)

# Resume from checkpoint if exists
start_epoch = 0
resume_ckpt = os.path.join(ckpt_dir, 'score_net_epoch_1.pt')
if os.path.exists(resume_ckpt):
    log.info(f'Resuming from {resume_ckpt}...')
    score_net.load_state_dict(torch.load(resume_ckpt, map_location=device, weights_only=True))
    # Find latest epoch
    for e in range(100, 0, -1):
        if os.path.exists(os.path.join(ckpt_dir, f'score_net_epoch_{e}.pt')):
            start_epoch = e
            break
    log.info(f'Resuming from epoch {start_epoch}')
    if os.path.exists(os.path.join(ckpt_dir, 'history.json')):
        with open(os.path.join(ckpt_dir, 'history.json')) as f:
            history = json.load(f)
        log.info(f'Loaded {len(history)} history entries')

noise_levels = config.noise_levels
N_EPOCHS = 100


def dsm_loss_fn(score_net, embed_fn, tokens, epoch, n_epochs):
    x_0 = embed_fn(tokens)  # [B, S, D]
    B, S, D = x_0.shape
    beta = min(1.0, 0.1 + 0.9 * epoch / max(n_epochs * 0.5, 1))

    total_loss = torch.tensor(0.0, device=device)
    for sigma in noise_levels:
        sigma_eff = sigma * beta
        with torch.no_grad():
            noise = torch.randn_like(x_0)
            noise = project_to_tangent(x_0, noise)
            x_t = normalize_sphere(x_0 + sigma_eff * noise)
            x_t_flat = x_t.reshape(-1, D)
            residual = x_0.reshape(-1, D) - x_t_flat
            score_target = project_to_tangent(x_t_flat, residual) / (sigma_eff ** 2)

        # Input perturbation (10%)
        if torch.rand(1).item() < 0.1:
            with torch.no_grad():
                x_t_flat = x_t_flat + torch.randn_like(x_t_flat) * 0.01
                x_t_flat = normalize_sphere(x_t_flat)

        sigma_t = torch.full((x_t_flat.size(0),), sigma_eff, device=device)
        score_pred = score_net(x_t_flat, sigma_t)
        loss = (1.0 / len(noise_levels)) * F.mse_loss(score_pred, score_target)
        total_loss = total_loss + loss
    return total_loss


log.info('Starting training...')
history = []

for epoch in range(start_epoch, N_EPOCHS):
    score_net.train()
    t0 = time.time()
    total_loss = 0.0
    n_batches = len(dataloader)
    nan_count = 0

    for bi, batch in enumerate(dataloader):
        tokens = batch.to(device) if not isinstance(batch, dict) else batch['tokens'].to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda'):
            loss = dsm_loss_fn(score_net, embedder, tokens, epoch, N_EPOCHS)

        if torch.isnan(loss) or torch.isinf(loss):
            nan_count += 1
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(score_net.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()

        # Log every 500 batches
        if (bi + 1) % 500 == 0:
            batch_avg = total_loss / (bi + 1 - nan_count)
            log.info(f'  E{epoch+1} batch {bi+1}/{n_batches} | running_loss: {batch_avg:.4f} | NaN: {nan_count}')

        # EMA
        for k, v in score_net.state_dict().items():
            if k in ema_shadow:
                ema_shadow[k].mul_(ema_decay).add_(v.data, alpha=1 - ema_decay)

    avg_loss = total_loss / max(n_batches - nan_count, 1)
    dur = time.time() - t0
    mem = torch.cuda.max_memory_allocated() / 1e9

    log.info(f'E{epoch+1:3d}/{N_EPOCHS} | Loss: {avg_loss:.4f} | NaN: {nan_count} | {dur:.1f}s | {mem:.1f}GB')
    history.append({'epoch': epoch + 1, 'loss': avg_loss, 'duration': dur, 'gpu_gb': mem, 'nan': nan_count})

    # Save every 10 epochs
    if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == N_EPOCHS - 1:
        torch.save(score_net.state_dict(), f'{ckpt_dir}/score_net_epoch_{epoch+1}.pt')
        torch.save({k: v.cpu() for k, v in ema_shadow.items()}, f'{ckpt_dir}/score_net_ema_epoch_{epoch+1}.pt')
        with open(f'{ckpt_dir}/history.json', 'w') as f:
            json.dump(history, f)
        log.info(f'  Checkpoint saved')

    torch.cuda.reset_peak_memory_stats()

# Final
torch.save(score_net.state_dict(), f'{ckpt_dir}/score_net_final.pt')
torch.save({k: v.cpu() for k, v in ema_shadow.items()}, f'{ckpt_dir}/score_net_ema_final.pt')
with open(f'{ckpt_dir}/history.json', 'w') as f:
    json.dump(history, f)

final_loss = history[-1]['loss']
log.info(f'Training complete. Final loss: {final_loss:.4f}')
