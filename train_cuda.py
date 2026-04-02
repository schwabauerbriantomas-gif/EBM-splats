#!/usr/bin/env python3
"""
SPEC 2 + SPEC 1 combined: CUDA training with AMP + Denoising Score Matching.

Replaces vulkan_device="vulkan" with pure CUDA:
  - torch.cuda.amp.GradScaler for mixed precision
  - CUDA-native energy function (no Vulkan)
  - FAISS GPU index when available
  - Denoising score matching training loop
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from config import EBMConfig
from geometry import normalize_sphere, project_to_tangent
from score_network import ScoreNetwork
from energy_cuda import EnergyFunctionCUDA
from splats import SplatStorage
from decoder import EBMDecoder
from soc import HistoryBuffer, maybe_consolidate
from dataset_utils import get_dataloader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EBMModelCUDA(nn.Module):
    """EBM model configured for CUDA — no Vulkan dependency."""
    
    def __init__(self, config: EBMConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.latent_dim)
        self.splats = SplatStorage(config)
        self.energy_fn = EnergyFunctionCUDA(config, self.splats)
        self.decoder = EBMDecoder(config)
    
    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(token_ids)
        return normalize_sphere(x)
    
    def compute_energy(self, x: torch.Tensor) -> torch.Tensor:
        return self.energy_fn(x)


def dsm_loss_cuda(score_net, embed_fn, tokens, config):
    """
    Denoising score matching with CUDA AMP using ScoreNetwork.
    Gradient flows properly through score_net — no autograd.grad needed.
    """
    device = tokens.device
    noise_levels = config.noise_levels

    with torch.no_grad():
        x_0 = embed_fn(tokens)  # [B, S, D]
    B, S, D = x_0.shape

    total_loss = torch.tensor(0.0, device=device)

    for sigma in noise_levels:
        with torch.no_grad():
            noise = torch.randn_like(x_0)
            noise = project_to_tangent(x_0, noise)
            x_t = normalize_sphere(x_0 + sigma * noise)
            x_t_flat = x_t.reshape(-1, D)

            residual = x_0.reshape(-1, D) - x_t_flat
            score_target = project_to_tangent(x_t_flat, residual) / (sigma ** 2)

        sigma_t = torch.full((x_t_flat.size(0),), sigma, device=device)
        score_pred = score_net(x_t_flat, sigma_t)

        weight = 1.0 / len(noise_levels)
        loss = weight * F.mse_loss(score_pred, score_target)
        total_loss = total_loss + loss

    return total_loss


def train_epoch_cuda(score_net, embed_fn, dataloader, optimizer, scaler, config, epoch, soc_buffer=None):
    """Train one epoch with CUDA AMP using ScoreNetwork."""
    score_net.train()
    total_loss = 0.0
    num_batches = len(dataloader)

    start = time.time()

    for batch_idx, batch in enumerate(dataloader):
        tokens = batch['tokens'] if isinstance(batch, dict) else batch
        tokens = tokens.cuda()

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda'):
            loss = dsm_loss_cuda(score_net, embed_fn, tokens, config)

        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(f"NaN/Inf loss at batch {batch_idx}, skipping")
            continue

        scaler.scale(loss).backward()

        if config.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(score_net.parameters(), config.grad_clip)

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        if batch_idx % 100 == 0:
            avg = total_loss / (batch_idx + 1)
            mem = torch.cuda.memory_allocated() / 1e9
            logger.info(
                f"Epoch {epoch} | Batch {batch_idx}/{num_batches} | "
                f"DSM Loss: {avg:.6f} | GPU Mem: {mem:.1f}GB"
            )

    duration = time.time() - start
    avg_loss = total_loss / num_batches

    mem_peak = torch.cuda.max_memory_allocated() / 1e9
    logger.info(
        f"Epoch {epoch} done | Avg DSM Loss: {avg_loss:.6f} | "
        f"Duration: {duration:.1f}s | Peak GPU: {mem_peak:.1f}GB"
    )

    return avg_loss


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="EBM CUDA Training with DSM + AMP")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=100000)
    parser.add_argument("--checkpoint-dir", default="checkpoints_cuda")
    args = parser.parse_args()
    
    assert torch.cuda.is_available(), "CUDA not available"
    device = torch.cuda.current_device()
    logger.info(f"CUDA device: {torch.cuda.get_device_name(device)}")
    logger.info(f"CUDA memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
    
    config = EBMConfig(device="cuda")
    
    dataloader, tokenizer = get_dataloader(
        tokenizer_name="gpt2",
        dataset_name="wikitext",
        config_name="wikitext-103-raw-v1",
        split="train",
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        max_samples=args.max_samples
    )
    
    model = EBMModelCUDA(config).cuda()

    # Score network — the trainable model
    score_net = ScoreNetwork(dim=config.latent_dim).cuda()
    params = sum(p.numel() for p in score_net.parameters() if p.requires_grad)
    logger.info(f"ScoreNetwork params: {params:,}")

    optimizer = torch.optim.AdamW(score_net.parameters(), lr=args.lr, weight_decay=config.reg_weight)
    scaler = torch.amp.GradScaler('cuda')
    soc_buffer = None
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    for epoch in range(args.epochs):
        avg_loss = train_epoch_cuda(score_net, model.embed, dataloader, optimizer, scaler, config, epoch, soc_buffer)
        ckpt_path = os.path.join(args.checkpoint_dir, f"score_net_epoch_{epoch+1}.pt")
        torch.save(score_net.state_dict(), ckpt_path)
        logger.info(f"Saved: {ckpt_path}")
    
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
