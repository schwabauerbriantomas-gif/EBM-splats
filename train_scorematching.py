#!/usr/bin/env python3
"""
SPEC 1: Denoising Score Matching Training Loop

Fixes the bug where training minimized energy.mean() directly instead of
performing denoising score matching with multiple noise levels.

Denoising Score Matching (Vincent, 2011):
  - Sample x_0 from real data
  - Corrupt: x_t = normalize(x_0 + sigma * epsilon), epsilon ~ N(0, I)
  - Score model predicts s_theta(x_t, sigma) ≈ (x_0 - x_t) / sigma^2
  - Loss: ||s_theta(x_t, sigma) - (x_0 - x_t) / sigma^2||^2

On the hypersphere S^{d-1}, we use tangent-space projections:
  - Noise added in tangent space: epsilon_tangent ~ N(0, I), then project
  - Target score: score_target = project_to_tangent(x_t, x_0 - x_t) / sigma^2
  - Predicted score: s_theta(x_t) from energy gradient
"""

import os
import time
import torch
import torch.nn.functional as F
import logging

from config import EBMConfig
from model import EBMModel
from score_network import ScoreNetwork
from geometry import normalize_sphere, project_to_tangent
from soc import HistoryBuffer, maybe_consolidate
from dataset_utils import get_dataloader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def denoising_score_matching_loss(score_net, embed_fn, tokens, config):
    """
    Compute denoising score matching loss over multiple noise levels.

    Uses a direct ScoreNetwork instead of autograd on energy function,
    ensuring proper gradient flow for training.

    Args:
        score_net: ScoreNetwork model
        embed_fn: callable to embed tokens to sphere (model.embed)
        tokens: [B, S] integer token IDs from real data
        config: EBMConfig with noise_levels

    Returns:
        scalar loss
    """
    device = next(score_net.parameters()).device
    noise_levels = config.noise_levels

    # Embed tokens -> points on hypersphere [B, S, D]
    with torch.no_grad():
        x_0 = embed_fn(tokens)  # [B, S, D]
    B, S, D = x_0.shape

    total_loss = torch.tensor(0.0, device=device)
    n_noise = len(noise_levels)

    for sigma in noise_levels:
        # Sample noise in tangent space of x_0
        with torch.no_grad():
            noise = torch.randn_like(x_0)
            noise = project_to_tangent(x_0, noise)
            x_t = normalize_sphere(x_0 + sigma * noise)
            x_t_flat = x_t.reshape(-1, D)

            # Target score
            residual = x_0.reshape(-1, D) - x_t_flat
            score_target = project_to_tangent(x_t_flat, residual) / (sigma ** 2)

        # Predicted score from ScoreNetwork (gradient flows through this!)
        sigma_t = torch.full((x_t_flat.size(0),), sigma, device=device)
        score_pred = score_net(x_t_flat, sigma_t)

        weight = 1.0 / n_noise
        loss = weight * F.mse_loss(score_pred, score_target)
        total_loss = total_loss + loss

    return total_loss


def train_epoch_sm(score_net, embed_fn, dataloader, optimizer, config, epoch, soc_buffer=None):
    """
    Train one epoch with denoising score matching using ScoreNetwork.
    """
    score_net.train()
    total_loss = 0.0
    num_batches = len(dataloader)

    start_time = time.time()

    for batch_idx, batch in enumerate(dataloader):
        tokens = batch['tokens'] if isinstance(batch, dict) else batch
        tokens = tokens.to(next(score_net.parameters()).device)

        optimizer.zero_grad()

        loss = denoising_score_matching_loss(score_net, embed_fn, tokens, config)

        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(f"NaN/Inf loss at batch {batch_idx}, skipping")
            continue

        loss.backward()

        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(score_net.parameters(), config.grad_clip)

        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 100 == 0:
            avg = total_loss / (batch_idx + 1)
            logger.info(
                f"Epoch {epoch} | Batch {batch_idx}/{num_batches} | "
                f"DSM Loss: {avg:.6f}"
            )

    duration = time.time() - start_time
    avg_loss = total_loss / num_batches

    logger.info(
        f"Epoch {epoch} done | Avg DSM Loss: {avg_loss:.6f} | Duration: {duration:.1f}s"
    )

    return avg_loss


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="EBM Denoising Score Matching Training")
    parser.add_argument("--device", default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=100000)
    parser.add_argument("--checkpoint-dir", default="checkpoints_sm")
    args = parser.parse_args()
    
    config = EBMConfig(device=args.device)
    
    logger.info(f"Device: {args.device}")
    logger.info(f"Noise levels: {config.noise_levels}")
    logger.info(f"Latent dim: {config.latent_dim}")
    
    # Dataset
    dataloader, tokenizer = get_dataloader(
        tokenizer_name="gpt2",
        dataset_name="wikitext",
        config_name="wikitext-103-raw-v1",
        split="train",
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        max_samples=args.max_samples
    )
    
    # Model (for embedding only — not trained by score matching)
    ebm_model = EBMModel(config).to(args.device)

    # Score network (the actual trainable model for score matching)
    score_net = ScoreNetwork(dim=config.latent_dim).to(args.device)
    param_count = sum(p.numel() for p in score_net.parameters() if p.requires_grad)
    logger.info(f"ScoreNetwork parameters: {param_count:,}")

    # Optimizer — only score network parameters
    optimizer = torch.optim.AdamW(score_net.parameters(), lr=args.lr, weight_decay=config.reg_weight)
    
    # SOC buffer
    soc_buffer = None  # Score network doesn't use SOC

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for epoch in range(args.epochs):
        avg_loss = train_epoch_sm(score_net, ebm_model.embed, dataloader, optimizer, config, epoch, soc_buffer)

        ckpt_path = os.path.join(args.checkpoint_dir, f"score_net_epoch_{epoch+1}.pt")
        torch.save(score_net.state_dict(), ckpt_path)
        logger.info(f"Saved: {ckpt_path}")
    
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
