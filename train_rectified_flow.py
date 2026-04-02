#!/usr/bin/env python3
"""
SPEC 3: Rectified Flow Sampler for EBM

Replaces 200-step Langevin dynamics with 1-10 step ODE solver.

Rectified Flow (Liu et al., 2022):
  Training: v_theta(x_t, t) learns velocity field
    x_t = (1-t)*x_0 + t*x_1  where x_0=noise, x_1=data
    Loss: ||v_theta(x_t, t) - (x_1 - x_0)||^2
    
  Sampling: Euler ODE from t=1 to t=0
    x_{t-dt} = x_t - dt * v_theta(x_t, t)
    ~5-10 steps vs 200 Langevin steps

On the hypersphere, we interpolate geodesically:
  x_t = geodesic_interpolate(x_0, x_1, t)
  x_0 = random point on sphere (noise)
  x_1 = data point on sphere

The velocity network reuses the energy function's score as a base,
but trains a separate lightweight velocity head.
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from config import EBMConfig
from model import EBMModel
from geometry import normalize_sphere, exp_map, project_to_tangent
from dataset_utils import get_dataloader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RectifiedFlowVelocityNet(nn.Module):
    """
    Lightweight velocity network for rectified flow on the hypersphere.
    Maps (x_t, t) -> velocity vector in tangent space at x_t.
    """
    def __init__(self, latent_dim: int, hidden_dim: int = 1024):
        super().__init__()
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # Main network: x_t + time_embed -> velocity
        self.net = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
    
    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_t: [B, D] point on hypersphere
            t: [B, 1] time in [0, 1]
        Returns:
            velocity: [B, D] in tangent space at x_t
        """
        t_emb = self.time_mlp(t)  # [B, H]
        combined = torch.cat([x_t, t_emb], dim=-1)  # [B, D+H]
        v = self.net(combined)  # [B, D]
        # Project to tangent space of x_t
        v = project_to_tangent(x_t, v)
        return v


def geodesic_interpolate(p: torch.Tensor, q: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Geodesic interpolation on S^{d-1}.
    gamma(t) = sin((1-t)*theta)/sin(theta) * p + sin(t*theta)/sin(theta) * q
    where theta = arccos(p·q).
    
    Args:
        p: [B, D] start point
        q: [B, D] end point  
        t: [B, 1] interpolation parameter in [0, 1]
    Returns:
        [B, D] interpolated point on sphere
    """
    cos_theta = (p * q).sum(dim=-1, keepdim=True).clamp(-1 + 1e-7, 1 - 1e-7)
    theta = torch.acos(cos_theta)  # [B, 1]
    
    sin_theta = torch.sin(theta).clamp(min=1e-7)
    
    result = torch.sin((1 - t) * theta) / sin_theta * p + \
             torch.sin(t * theta) / sin_theta * q
    
    return normalize_sphere(result)


def rectified_flow_loss(vel_net, x_0, x_1):
    """
    Compute rectified flow training loss.
    
    x_t = geodesic_interpolate(x_0, x_1, t)
    target_velocity = geodesic direction from x_t toward x_1
    loss = ||v_theta(x_t, t) - target_velocity||^2
    
    Args:
        vel_net: RectifiedFlowVelocityNet
        x_0: [B, D] noise points on sphere
        x_1: [B, D] data points on sphere
    Returns:
        scalar loss
    """
    B = x_0.size(0)
    device = x_0.device
    
    # Sample random times uniformly in [0, 1]
    t = torch.rand(B, 1, device=device)
    
    # Geodesic interpolation
    x_t = geodesic_interpolate(x_0, x_1, t)  # [B, D]
    
    # Target velocity: direction from x_t toward x_1 in tangent space
    # For small steps, this approximates (x_1 - x_t), but on sphere we project
    target_v = project_to_tangent(x_t, x_1 - x_t)  # [B, D]
    
    # Predicted velocity
    pred_v = vel_net(x_t, t)  # [B, D]
    
    return F.mse_loss(pred_v, target_v)


def sample_rectified_flow(vel_net, n_samples, latent_dim, device, n_steps=5):
    """
    Generate samples using Euler ODE solver for rectified flow.
    
    Starts from random noise (t=1) and integrates to data (t=0).
    
    Args:
        vel_net: trained velocity network
        n_samples: number of samples
        latent_dim: dimension
        device: torch device
        n_steps: number of Euler steps (1-10 typical)
    Returns:
        [n_samples, latent_dim] on hypersphere
    """
    # x_0 = random noise on sphere (t=1)
    x = normalize_sphere(torch.randn(n_samples, latent_dim, device=device))
    
    dt = 1.0 / n_steps
    
    for step in range(n_steps):
        t_val = 1.0 - step * dt  # t goes from 1 to 0
        t = torch.full((n_samples, 1), t_val, device=device)
        
        v = vel_net(x, t)  # velocity pointing toward data
        
        # Euler step: move x toward data
        # On sphere, we use exponential map for the position update
        x = exp_map(x, -dt * v)  # negative because we go from t=1 to t=0
        x = normalize_sphere(x)
    
    return x


def train_rectified_flow(model, vel_net, dataloader, optimizer_v, config, epochs):
    """
    Train the rectified flow velocity network.
    """
    device = next(model.parameters()).device
    
    for epoch in range(epochs):
        vel_net.train()
        total_loss = 0.0
        n_batches = len(dataloader)
        
        for batch_idx, batch in enumerate(dataloader):
            tokens = batch['tokens'] if isinstance(batch, dict) else batch
            tokens = tokens.to(device)
            
            # Get data points on sphere
            with torch.no_grad():
                x_1 = model.embed(tokens)  # [B, S, D]
                B, S, D = x_1.shape
                x_1 = x_1.reshape(B * S, D)
            
            # Random noise on sphere
            x_0 = normalize_sphere(torch.randn_like(x_1))
            
            # Train velocity network
            optimizer_v.zero_grad()
            loss = rectified_flow_loss(vel_net, x_0, x_1)
            
            if torch.isnan(loss):
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vel_net.parameters(), config.grad_clip)
            optimizer_v.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                avg = total_loss / (batch_idx + 1)
                logger.info(
                    f"[RF] Epoch {epoch} | Batch {batch_idx}/{n_batches} | "
                    f"Loss: {avg:.6f}"
                )
        
        avg_loss = total_loss / n_batches
        logger.info(f"[RF] Epoch {epoch} done | Avg Loss: {avg_loss:.6f}")


def benchmark_comparison(model, vel_net, config, device):
    """
    Compare Langevin (200 steps) vs Rectified Flow (5 steps) sampling speed.
    """
    from langevin import sample_langevin
    
    n_samples = 8
    
    # Benchmark Langevin
    x_init = normalize_sphere(torch.randn(n_samples, config.latent_dim, device=device))
    
    start = time.time()
    with torch.no_grad():
        _ = sample_langevin(x_init, model.energy_fn, config)
    langevin_time = time.time() - start
    
    # Benchmark Rectified Flow (5 steps)
    start = time.time()
    with torch.no_grad():
        _ = sample_rectified_flow(vel_net, n_samples, config.latent_dim, device, n_steps=5)
    rf_time_5 = time.time() - start
    
    # Benchmark Rectified Flow (1 step)
    start = time.time()
    with torch.no_grad():
        _ = sample_rectified_flow(vel_net, n_samples, config.latent_dim, device, n_steps=1)
    rf_time_1 = time.time() - start
    
    logger.info("=" * 50)
    logger.info("SAMPLING SPEED BENCHMARK")
    logger.info(f"  Langevin (200 steps): {langevin_time:.3f}s")
    logger.info(f"  Rectified Flow (5 steps): {rf_time_5:.3f}s")
    logger.info(f"  Rectified Flow (1 step): {rf_time_1:.3f}s")
    logger.info(f"  Speedup (5 steps): {langevin_time / rf_time_5:.1f}x")
    logger.info(f"  Speedup (1 step): {langevin_time / rf_time_1:.1f}x")
    logger.info("=" * 50)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="EBM Rectified Flow Training")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr-v", type=float, default=1e-4, help="Velocity net LR")
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=100000)
    parser.add_argument("--rf-steps", type=int, default=5, help="Sampling steps for RF")
    args = parser.parse_args()
    
    config = EBMConfig(device=args.device)
    
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
    
    # Base EBM model
    model = EBMModel(config).to(args.device)
    
    # Rectified Flow velocity network
    vel_net = RectifiedFlowVelocityNet(config.latent_dim).to(args.device)
    logger.info(f"Velocity net params: {sum(p.numel() for p in vel_net.parameters()):,}")
    
    # Optimizer for velocity network only
    optimizer_v = torch.optim.AdamW(vel_net.parameters(), lr=args.lr_v, weight_decay=config.reg_weight)
    
    # Train
    logger.info("Training Rectified Flow velocity network...")
    train_rectified_flow(model, vel_net, dataloader, optimizer_v, config, args.epochs)
    
    # Save
    os.makedirs("checkpoints_rf", exist_ok=True)
    torch.save(vel_net.state_dict(), "checkpoints_rf/vel_net.pt")
    logger.info("Saved velocity network to checkpoints_rf/vel_net.pt")
    
    # Benchmark
    logger.info("Running speed benchmark...")
    benchmark_comparison(model, vel_net, config, args.device)
    
    # Quick generation test
    logger.info("Quick generation test...")
    with torch.no_grad():
        samples = sample_rectified_flow(vel_net, 4, config.latent_dim, args.device, n_steps=args.rf_steps)
        # Decode
        context = normalize_sphere(samples.mean(dim=0, keepdim=True).expand(4, -1))
        logits = model.decode(samples, context)
        generated_tokens = torch.argmax(logits, dim=-1)
        logger.info(f"Generated token IDs (first sample): {generated_tokens[0, :20].tolist()}")
        try:
            text = tokenizer.decode(generated_tokens[0].tolist())
            logger.info(f"Decoded text (first sample): {text[:200]}")
        except Exception:
            pass


if __name__ == "__main__":
    main()
