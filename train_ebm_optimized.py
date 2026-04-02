#!/usr/bin/env python3
"""
EBM Optimized Training Pipeline — Unified 4-phase training.

Phase 1: DSM training with β-annealing + EMA + input perturbation
Phase 2: Rectified Flow training
Phase 3: Reflow (1-2 rounds)
Phase 4: Benchmark sampling (Langevin vs RF vs RF+reflow)

Optimizations from papers:
  A. β-Temperature Annealing     (2603.06875)
  B. Adaptive Step Size          (2603.11319)
  C. EMA Weights                 (2601.10679)
  D. Input Perturbation          (2601.10679)
  E. Reflow                      (2209.03003)
  F. Fractional Langevin Sampler (NeurIPS 2025)
"""

import os
import sys
import time
import math
import copy
import logging
import argparse
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import EBMConfig
from geometry import normalize_sphere, project_to_tangent, exp_map, geodesic_distance
from score_network import ScoreNetwork

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# A. EMA (Exponential Moving Average) — Paper 4
# ──────────────────────────────────────────────────────────────

class EMAHelper:
    """Maintains exponential moving average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}
        self.backup = {}

    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v, alpha=1.0 - self.decay)

    def apply(self, model: nn.Module):
        """Apply EMA weights to model (for inference)."""
        self.backup = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow)

    def restore(self, model: nn.Module):
        """Restore original weights after inference."""
        model.load_state_dict(self.backup)
        self.backup = {}


# ──────────────────────────────────────────────────────────────
# A. β-Temperature Annealing — Paper 1
# ──────────────────────────────────────────────────────────────

class BetaScheduler:
    """Linear annealing of inverse temperature β from beta_start to beta_end."""

    def __init__(self, beta_start: float = 0.1, beta_end: float = 1.0, total_steps: int = 1000):
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.total_steps = total_steps

    def get_beta(self, step: int) -> float:
        progress = min(step / self.total_steps, 1.0)
        return self.beta_start + progress * (self.beta_end - self.beta_start)


# ──────────────────────────────────────────────────────────────
# F. Fractional Langevin Sampler — Paper 2
# ──────────────────────────────────────────────────────────────

class FractionalLangevinSampler:
    """
    Fractional Langevin Dynamics with Hurst exponent H.
    Uses precomputed kernel for long-range temporal correlations.
    
    ⟨v(t)v(s)⟩ ∝ |t-s|^(2H-1) with H=0.75
    """

    def __init__(self, H: float = 0.75, max_steps: int = 500):
        self.H = H
        self.kernel_cache = {}
        self.max_steps = max_steps
        # Precompute kernel
        self._build_kernel(max_steps)

    def _build_kernel(self, n_steps: int):
        """Precompute correlation kernel matrix."""
        K = torch.zeros(n_steps, n_steps)
        for i in range(n_steps):
            for j in range(n_steps):
                diff = abs(i - j)
                K[i, j] = (diff + 1e-8) ** (2 * self.H - 1)
        # Cholesky for correlated noise generation: noise = L @ z, z~N(0,I)
        # Clamp for numerical stability
        K = K + 1e-6 * torch.eye(n_steps)
        try:
            self.L = torch.linalg.cholesky(K)
        except RuntimeError:
            # Fallback: use diagonal (standard Langevin) if not PSD
            self.L = torch.sqrt(torch.diag(K)).diag()

    def sample(self, x_init, score_fn, n_steps: int = 200, dt: float = 0.001,
               gamma: float = 0.1, device='cpu'):
        """
        Fractional Langevin sampling via Euler-Maruyama.
        
        Args:
            x_init: [B, D] starting points
            score_fn: function x -> score tensor
            n_steps: number of steps
            dt: base step size
            gamma: friction coefficient
        """
        B, D = x_init.shape
        x = x_init.clone()
        L = self.L[:n_steps, :n_steps].to(device)

        # Generate all correlated noise at once
        z = torch.randn(n_steps, B, D, device=device)
        correlated_noise = torch.einsum('ij,jbd->ibd', L, z)  # [n_steps, B, D]
        # Scale by dt^H for fractional scaling
        scale = (dt ** self.H) * math.sqrt(2 * gamma * dt)
        correlated_noise = correlated_noise * scale

        for step in range(n_steps):
            with torch.no_grad():
                score = score_fn(x)

            # Update: x = x + dt * score + fractional_noise
            x = x + dt * score + correlated_noise[step]
            x = normalize_sphere(x)

        return x


# ──────────────────────────────────────────────────────────────
# B. Adaptive Step Size Langevin — Paper 3
# ──────────────────────────────────────────────────────────────

class AdaptiveLangevinSampler:
    """
    Langevin sampler with adaptive step size.
    dt = dt_base / (1 + lambda * ||score||^2), clamped to [dt_min, dt_max].
    """

    def __init__(self, dt_base: float = 0.001, dt_min: float = 1e-5,
                 dt_max: float = 0.01, lambda_reg: float = 0.1):
        self.dt_base = dt_base
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.lambda_reg = lambda_reg

    def sample(self, x_init, score_fn, n_steps: int = 200, device='cpu'):
        B, D = x_init.shape
        x = x_init.clone()

        for step in range(n_steps):
            with torch.no_grad():
                score = score_fn(x)
                score_norm_sq = (score ** 2).sum(dim=-1, keepdim=True)  # [B, 1]

            # Adaptive step size
            dt = self.dt_base / (1.0 + self.lambda_reg * score_norm_sq)
            dt = dt.clamp(self.dt_min, self.dt_max)

            # Langevin update
            noise = torch.randn_like(x)
            noise = project_to_tangent(x, noise)
            x = x + dt * score + torch.sqrt(2 * dt) * noise
            x = normalize_sphere(x)

        return x


# ──────────────────────────────────────────────────────────────
# Rectified Flow Velocity Net (reuse from train_rectified_flow.py)
# ──────────────────────────────────────────────────────────────

class RectifiedFlowVelocityNet(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int = 1024):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.net = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x_t, t):
        t_emb = self.time_mlp(t)
        combined = torch.cat([x_t, t_emb], dim=-1)
        v = self.net(combined)
        v = project_to_tangent(x_t, v)
        return v


def geodesic_interpolate(p, q, t):
    cos_theta = (p * q).sum(dim=-1, keepdim=True).clamp(-1 + 1e-7, 1 - 1e-7)
    theta = torch.acos(cos_theta)
    sin_theta = torch.sin(theta).clamp(min=1e-7)
    result = torch.sin((1 - t) * theta) / sin_theta * p + \
             torch.sin(t * theta) / sin_theta * q
    return normalize_sphere(result)


# ──────────────────────────────────────────────────────────────
# E. Reflow — Paper 5
# ──────────────────────────────────────────────────────────────

def reflow_round(vel_net, embed_fn, dataloader, optimizer, config,
                 device, n_batches=None):
    """
    One round of reflow:
    1. Sample x_0 (real data), x_1 = noise on sphere
    2. Use current vel_net to compute x_t and predicted x_1'
    3. Re-train vel_net with (x_t, x_1') pairs
    """
    vel_net.train()
    total_loss = 0.0
    batches_done = 0

    for batch_idx, batch in enumerate(dataloader):
        if n_batches is not None and batch_idx >= n_batches:
            break
        tokens = batch['tokens'] if isinstance(batch, dict) else batch
        tokens = tokens.to(device)

        with torch.no_grad():
            x_1 = embed_fn(tokens)
            B, S, D = x_1.shape
            x_1 = x_1.reshape(B * S, D)
            x_0 = normalize_sphere(torch.randn_like(x_1))

        t = torch.rand(x_0.size(0), 1, device=device)
        x_t = geodesic_interpolate(x_0, x_1, t)

        # Predict target using current model
        with torch.no_grad():
            v_pred = vel_net(x_t, t)
            # Compute x_1' = x_t + (1-t) * v_pred, then normalize
            x_1_prime = x_t + (1 - t) * v_pred
            x_1_prime = normalize_sphere(x_1_prime)

        # Re-train with (x_t, x_1_prime)
        optimizer.zero_grad()
        target_v = project_to_tangent(x_t, x_1_prime - x_t)
        v_new = vel_net(x_t, t)
        loss = F.mse_loss(v_new, target_v)

        if torch.isnan(loss):
            continue

        loss.backward()
        nn.utils.clip_grad_norm_(vel_net.parameters(), config.grad_clip)
        optimizer.step()

        total_loss += loss.item()
        batches_done += 1

    return total_loss / max(batches_done, 1)


# ──────────────────────────────────────────────────────────────
# Sampling utilities
# ──────────────────────────────────────────────────────────────

def sample_rf_euler(vel_net, n_samples, latent_dim, device, n_steps=5):
    """Standard RF Euler sampler."""
    x = normalize_sphere(torch.randn(n_samples, latent_dim, device=device))
    dt = 1.0 / n_steps
    with torch.no_grad():
        for step in range(n_steps):
            t_val = 1.0 - step * dt
            t = torch.full((n_samples, 1), t_val, device=device)
            v = vel_net(x, t)
            x = exp_map(x, -dt * v)
            x = normalize_sphere(x)
    return x


def compute_energy_stats(x, embed_fn, dataloader, config, device, n_batches=5):
    """Compute average cosine similarity to real data (proxy for energy)."""
    # Compute mean cosine sim to nearest real data
    all_data = []
    count = 0
    with torch.no_grad():
        for batch in dataloader:
            if count >= n_batches:
                break
            tokens = batch['tokens'] if isinstance(batch, dict) else batch
            tokens = tokens.to(device)
            x_1 = embed_fn(tokens)
            B, S, D = x_1.shape
            all_data.append(x_1.reshape(B * S, D))
            count += 1
    all_data = torch.cat(all_data, dim=0)  # [N, D]

    # For each sample, find max cosine sim to data
    cos_sims = torch.mm(x, all_data.T)  # [n_samples, N]
    max_sims, _ = cos_sims.max(dim=1)
    return max_sims.mean().item()


def compute_diversity(x):
    """Compute average pairwise cosine distance (diversity metric)."""
    B = x.size(0)
    if B < 2:
        return 0.0
    # Sample pairs for efficiency
    n_pairs = min(B * 10, B * (B - 1) // 2)
    idx_i = torch.randint(0, B, (n_pairs,))
    idx_j = torch.randint(0, B, (n_pairs,))
    cos_sims = (x[idx_i] * x[idx_j]).sum(dim=-1)
    return (1.0 - cos_sims).mean().item()


# ──────────────────────────────────────────────────────────────
# Phase 1: DSM Training with β-annealing + EMA + Input Perturbation
# ──────────────────────────────────────────────────────────────

def phase1_dsm_training(score_net, embed_fn, dataloader, optimizer, scaler,
                         config, args, device):
    """
    DSM training with:
      A. β-Temperature Annealing — modulates noise schedule
      C. EMA weights
      D. Input Perturbation — extra noise with prob 0.1
    """
    logger.info("=" * 60)
    logger.info("PHASE 1: DSM Training with β-annealing + EMA + Input Perturbation")
    logger.info("=" * 60)

    n_epochs = args.dsm_epochs
    ema = EMAHelper(score_net, decay=0.999)
    total_steps = n_epochs * len(dataloader)
    global_step = 0
    beta_scheduler = BetaScheduler(beta_start=0.1, beta_end=1.0, total_steps=total_steps)

    losses = []

    for epoch in range(n_epochs):
        score_net.train()
        epoch_loss = 0.0
        n_batches = len(dataloader)
        epoch_start = time.time()

        for batch_idx, batch in enumerate(dataloader):
            tokens = batch['tokens'] if isinstance(batch, dict) else batch
            tokens = tokens.to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                x_0 = embed_fn(tokens)
            B, S, D = x_0.shape

            beta = beta_scheduler.get_beta(global_step)
            sigma_eff = [s * beta for s in config.noise_levels]

            batch_loss = torch.tensor(0.0, device=device)

            for sigma in sigma_eff:
                with torch.no_grad():
                    noise = torch.randn_like(x_0)
                    noise = project_to_tangent(x_0, noise)
                    x_t = normalize_sphere(x_0 + sigma * noise)
                    x_t_flat = x_t.reshape(-1, D)

                    # D. Input Perturbation — extra noise with prob 0.1
                    if torch.rand(1).item() < 0.1:
                        extra_noise = torch.randn_like(x_t_flat) * 0.01
                        extra_noise = project_to_tangent(x_t_flat, extra_noise)
                        x_t_flat = normalize_sphere(x_t_flat + extra_noise)

                    residual = x_0.reshape(-1, D) - x_t_flat
                    score_target = project_to_tangent(x_t_flat, residual) / (sigma ** 2)

                sigma_t = torch.full((x_t_flat.size(0),), sigma, device=device)
                score_pred = score_net(x_t_flat, sigma_t)

                weight = 1.0 / len(sigma_eff)
                loss = weight * F.mse_loss(score_pred, score_target)
                batch_loss = batch_loss + loss

            if torch.isnan(batch_loss) or torch.isinf(batch_loss):
                logger.warning(f"NaN/Inf loss at step {global_step}, skipping")
                global_step += 1
                continue

            with torch.amp.autocast('cuda'):
                batch_loss = batch_loss.float()

            scaler.scale(batch_loss).backward()

            if config.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(score_net.parameters(), config.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            # C. Update EMA
            ema.update(score_net)

            epoch_loss += batch_loss.item()
            global_step += 1

            if batch_idx % 50 == 0:
                avg = epoch_loss / (batch_idx + 1)
                mem = torch.cuda.memory_allocated() / 1e9
                logger.info(
                    f"[DSM] Epoch {epoch+1}/{n_epochs} | Batch {batch_idx}/{n_batches} | "
                    f"Loss: {avg:.6f} | β: {beta:.3f} | GPU: {mem:.1f}GB"
                )

        duration = time.time() - epoch_start
        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        mem_peak = torch.cuda.max_memory_allocated() / 1e9
        logger.info(
            f"[DSM] Epoch {epoch+1} done | Loss: {avg_loss:.6f} | "
            f"Time: {duration:.1f}s | Peak GPU: {mem_peak:.1f}GB"
        )

        # Save both regular and EMA checkpoints
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        torch.save(score_net.state_dict(),
                    os.path.join(args.checkpoint_dir, f"score_net_epoch_{epoch+1}.pt"))
        ema.apply(score_net)
        torch.save(score_net.state_dict(),
                    os.path.join(args.checkpoint_dir, f"score_net_ema_epoch_{epoch+1}.pt"))
        ema.restore(score_net)
        logger.info(f"Saved checkpoints (regular + EMA) for epoch {epoch+1}")

    return score_net, ema, losses


# ──────────────────────────────────────────────────────────────
# Phase 2: Rectified Flow Training
# ──────────────────────────────────────────────────────────────

def phase2_rf_training(vel_net, embed_fn, dataloader, optimizer, config,
                        args, device):
    """Train Rectified Flow velocity network."""
    logger.info("=" * 60)
    logger.info("PHASE 2: Rectified Flow Training")
    logger.info("=" * 60)

    n_epochs = args.rf_epochs
    losses = []

    for epoch in range(n_epochs):
        vel_net.train()
        total_loss = 0.0
        n_batches = len(dataloader)
        epoch_start = time.time()

        for batch_idx, batch in enumerate(dataloader):
            tokens = batch['tokens'] if isinstance(batch, dict) else batch
            tokens = tokens.to(device)

            with torch.no_grad():
                x_1 = embed_fn(tokens)
                B, S, D = x_1.shape
                x_1 = x_1.reshape(B * S, D)

            x_0 = normalize_sphere(torch.randn_like(x_1))
            B_flat = x_0.size(0)

            t = torch.rand(B_flat, 1, device=device)
            x_t = geodesic_interpolate(x_0, x_1, t)
            target_v = project_to_tangent(x_t, x_1 - x_t)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                pred_v = vel_net(x_t, t)
                loss = F.mse_loss(pred_v, target_v)

            if torch.isnan(loss):
                continue

            loss.backward()
            nn.utils.clip_grad_norm_(vel_net.parameters(), config.grad_clip)
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 50 == 0:
                avg = total_loss / (batch_idx + 1)
                logger.info(
                    f"[RF] Epoch {epoch+1}/{n_epochs} | Batch {batch_idx}/{n_batches} | "
                    f"Loss: {avg:.6f}"
                )

        duration = time.time() - epoch_start
        avg_loss = total_loss / n_batches
        losses.append(avg_loss)
        logger.info(f"[RF] Epoch {epoch+1} done | Loss: {avg_loss:.6f} | Time: {duration:.1f}s")

        os.makedirs(args.checkpoint_dir, exist_ok=True)
        torch.save(vel_net.state_dict(),
                    os.path.join(args.checkpoint_dir, f"vel_net_epoch_{epoch+1}.pt"))

    return vel_net, losses


# ──────────────────────────────────────────────────────────────
# Phase 4: Benchmark
# ──────────────────────────────────────────────────────────────

def phase4_benchmark(score_net, ema, vel_net, embed_fn, dataloader, config, args, device):
    """Benchmark different sampling strategies."""
    logger.info("=" * 60)
    logger.info("PHASE 4: SAMPLING BENCHMARK")
    logger.info("=" * 60)

    n_samples = 64
    latent_dim = config.latent_dim
    results = {}

    # Apply EMA weights for sampling
    ema.apply(score_net)

    def make_score_fn(sigma_val):
        def score_fn(x):
            sigma_t = torch.full((x.size(0),), sigma_val, device=device)
            return score_net(x, sigma_t)
        return score_fn

    # ── Standard Langevin (200 steps) ──
    logger.info("Benchmarking: Standard Langevin (200 steps)...")
    x_init = normalize_sphere(torch.randn(n_samples, latent_dim, device=device))
    score_fn = make_score_fn(0.1)
    langevin_sampler = AdaptiveLangevinSampler(dt_base=0.001)
    t0 = time.time()
    with torch.no_grad():
        samples_langevin = langevin_sampler.sample(x_init, score_fn, n_steps=200, device=device)
    t_langevin = time.time() - t0
    energy_langevin = compute_energy_stats(samples_langevin, embed_fn, dataloader, config, device, n_batches=3)
    diversity_langevin = compute_diversity(samples_langevin)
    results['Langevin 200 steps'] = {
        'time': t_langevin,
        'energy_proxy': energy_langevin,
        'diversity': diversity_langevin
    }
    logger.info(f"  Time: {t_langevin:.3f}s | Energy proxy (max cos sim): {energy_langevin:.4f} | Diversity: {diversity_langevin:.4f}")

    # ── RF Euler (5 steps) ──
    logger.info("Benchmarking: RF Euler (5 steps)...")
    t0 = time.time()
    with torch.no_grad():
        samples_rf5 = sample_rf_euler(vel_net, n_samples, latent_dim, device, n_steps=5)
    t_rf5 = time.time() - t0
    energy_rf5 = compute_energy_stats(samples_rf5, embed_fn, dataloader, config, device, n_batches=3)
    diversity_rf5 = compute_diversity(samples_rf5)
    results['RF 5 steps'] = {
        'time': t_rf5,
        'energy_proxy': energy_rf5,
        'diversity': diversity_rf5
    }
    logger.info(f"  Time: {t_rf5:.3f}s | Energy proxy: {energy_rf5:.4f} | Diversity: {diversity_rf5:.4f}")

    # ── RF Euler (10 steps) ──
    logger.info("Benchmarking: RF Euler (10 steps)...")
    t0 = time.time()
    with torch.no_grad():
        samples_rf10 = sample_rf_euler(vel_net, n_samples, latent_dim, device, n_steps=10)
    t_rf10 = time.time() - t0
    energy_rf10 = compute_energy_stats(samples_rf10, embed_fn, dataloader, config, device, n_batches=3)
    diversity_rf10 = compute_diversity(samples_rf10)
    results['RF 10 steps'] = {
        'time': t_rf10,
        'energy_proxy': energy_rf10,
        'diversity': diversity_rf10
    }
    logger.info(f"  Time: {t_rf10:.3f}s | Energy proxy: {energy_rf10:.4f} | Diversity: {diversity_rf10:.4f}")

    # ── Fractional Langevin (200 steps) ──
    logger.info("Benchmarking: Fractional Langevin (200 steps, H=0.75)...")
    try:
        x_init = normalize_sphere(torch.randn(n_samples, latent_dim, device=device))
        frac_sampler = FractionalLangevinSampler(H=0.75, max_steps=200)
        t0 = time.time()
        with torch.no_grad():
            samples_frac = frac_sampler.sample(x_init, score_fn, n_steps=200, dt=0.001,
                                               device=device)
        t_frac = time.time() - t0
        energy_frac = compute_energy_stats(samples_frac, embed_fn, dataloader, config, device, n_batches=3)
        diversity_frac = compute_diversity(samples_frac)
        results['Fractional Langevin 200'] = {
            'time': t_frac,
            'energy_proxy': energy_frac,
            'diversity': diversity_frac
        }
        logger.info(f"  Time: {t_frac:.3f}s | Energy proxy: {energy_frac:.4f} | Diversity: {diversity_frac:.4f}")
    except Exception as e:
        logger.warning(f"Fractional Langevin failed: {e}")
        results['Fractional Langevin 200'] = {'time': None, 'energy_proxy': None, 'diversity': None}

    # Restore original weights
    ema.restore(score_net)

    # ── Summary ──
    logger.info("=" * 60)
    logger.info("BENCHMARK SUMMARY")
    logger.info(f"{'Method':<30} {'Time (s)':<10} {'Energy (cos)':<14} {'Diversity':<10}")
    logger.info("-" * 60)
    for name, r in results.items():
        t_str = f"{r['time']:.3f}" if r['time'] is not None else "N/A"
        e_str = f"{r['energy_proxy']:.4f}" if r['energy_proxy'] is not None else "N/A"
        d_str = f"{r['diversity']:.4f}" if r['diversity'] is not None else "N/A"
        logger.info(f"{name:<30} {t_str:<10} {e_str:<14} {d_str:<10}")

    # Speedups
    if results['Langevin 200 steps']['time'] and results['RF 5 steps']['time']:
        speedup = results['Langevin 200 steps']['time'] / results['RF 5 steps']['time']
        logger.info(f"RF 5-step speedup vs Langevin: {speedup:.1f}x")
    if results['Langevin 200 steps']['time'] and results['RF 10 steps']['time']:
        speedup = results['Langevin 200 steps']['time'] / results['RF 10 steps']['time']
        logger.info(f"RF 10-step speedup vs Langevin: {speedup:.1f}x")

    logger.info("=" * 60)

    return results


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="EBM Optimized Training Pipeline")
    parser.add_argument("--dsm-epochs", type=int, default=5)
    parser.add_argument("--rf-epochs", type=int, default=3)
    parser.add_argument("--reflow-rounds", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-v", type=float, default=1e-4)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--checkpoint-dir", default="checkpoints_optimized")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA not available"
    device = torch.cuda.current_device()
    logger.info(f"CUDA device: {torch.cuda.get_device_name(device)}")

    config = EBMConfig(device="cuda")

    # Dataset
    from dataset_utils import get_dataloader
    dataloader, tokenizer = get_dataloader(
        tokenizer_name="gpt2",
        dataset_name="wikitext",
        config_name="wikitext-103-raw-v1",
        split="train",
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        max_samples=args.max_samples
    )
    logger.info(f"Dataset: {len(dataloader)} batches")

    # Embedding function (no trainable params — uses pretrained GPT-2 embeddings)
    from model import EBMModel
    base_model = EBMModel(config).to(device)
    base_model.eval()
    embed_fn = base_model.embed

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ── Phase 1: DSM Training ──
    score_net = ScoreNetwork(dim=config.latent_dim).to(device)
    params = sum(p.numel() for p in score_net.parameters() if p.requires_grad)
    logger.info(f"ScoreNetwork params: {params:,}")

    optimizer = torch.optim.AdamW(score_net.parameters(), lr=args.lr,
                                   weight_decay=config.reg_weight)
    scaler = torch.amp.GradScaler('cuda')

    score_net, ema, dsm_losses = phase1_dsm_training(
        score_net, embed_fn, dataloader, optimizer, scaler, config, args, device
    )

    # ── Phase 2: Rectified Flow Training ──
    vel_net = RectifiedFlowVelocityNet(config.latent_dim).to(device)
    params_v = sum(p.numel() for p in vel_net.parameters() if p.requires_grad)
    logger.info(f"Velocity net params: {params_v:,}")

    optimizer_v = torch.optim.AdamW(vel_net.parameters(), lr=args.lr_v,
                                     weight_decay=config.reg_weight)

    vel_net, rf_losses = phase2_rf_training(
        vel_net, embed_fn, dataloader, optimizer_v, config, args, device
    )

    # ── Phase 3: Reflow ──
    logger.info("=" * 60)
    logger.info(f"PHASE 3: Reflow ({args.reflow_rounds} rounds)")
    logger.info("=" * 60)

    reflow_losses = []
    for r in range(args.reflow_rounds):
        logger.info(f"Reflow round {r+1}/{args.reflow_rounds}...")
        optimizer_reflow = torch.optim.AdamW(vel_net.parameters(), lr=args.lr_v * 0.5,
                                              weight_decay=config.reg_weight)
        reflow_loss = reflow_round(vel_net, embed_fn, dataloader, optimizer_reflow,
                                    config, device)
        reflow_losses.append(reflow_loss)
        logger.info(f"Reflow round {r+1} done | Loss: {reflow_loss:.6f}")
        torch.save(vel_net.state_dict(),
                    os.path.join(args.checkpoint_dir, f"vel_net_reflow_{r+1}.pt"))

    # ── Phase 4: Benchmark ──
    benchmark_results = phase4_benchmark(
        score_net, ema, vel_net, embed_fn, dataloader, config, args, device
    )

    # Save summary
    summary = {
        'dsm_losses': dsm_losses,
        'rf_losses': rf_losses,
        'reflow_losses': reflow_losses,
        'benchmark': {k: v for k, v in benchmark_results.items()},
    }
    import json
    with open(os.path.join(args.checkpoint_dir, 'training_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved training summary to {args.checkpoint_dir}/training_summary.json")

    logger.info("All phases complete.")


if __name__ == "__main__":
    main()