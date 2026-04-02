"""
Hierarchical Context System — V2 §4.

Three-level context for long-range dependencies:
  - Local:  last 8-16 tokens  (β≈0.5, fast adaptation)
  - Medium: last 64-128 tokens (β≈0.8, moderate)
  - Global: full sequence       (β≈0.95, slow, stable)

Integration:
  E_trans(x_t) = -Σ_{l∈{local,medium,global}} λ_l · (x_t · c_l)
"""

import torch
import torch.nn as nn
from typing import Dict

from config import EBMConfig
from geometry import normalize_sphere


class HierarchicalContext(nn.Module):
    """Three-level context system for EBM language model."""

    def __init__(self, config: EBMConfig):
        super().__init__()
        self.config = config
        self.latent_dim = config.latent_dim

        # Persistent context vectors (on hypersphere)
        self.register_buffer('c_local', torch.zeros(1, config.latent_dim))
        self.register_buffer('c_medium', torch.zeros(1, config.latent_dim))
        self.register_buffer('c_global', torch.zeros(1, config.latent_dim))

        self.beta_local = config.beta_local
        self.beta_medium = config.beta_medium
        self.beta_global = config.beta_global

        # Update counters
        self.register_buffer('_step', torch.tensor(0, dtype=torch.long))
        self.register_buffer('_local_count', torch.tensor(0, dtype=torch.long))
        self.register_buffer('_medium_count', torch.tensor(0, dtype=torch.long))

        # Buffers stored as lists to avoid device/shape issues
        self._local_buffer = []
        self._medium_buffer = []

    def reset(self, batch_size: int, device: torch.device):
        """Reset context for a new sequence."""
        self.c_local = normalize_sphere(torch.randn(batch_size, self.latent_dim, device=device))
        self.c_medium = normalize_sphere(torch.randn(batch_size, self.latent_dim, device=device))
        self.c_global = normalize_sphere(torch.randn(batch_size, self.latent_dim, device=device))
        self._step.zero_()
        self._local_count.zero_()
        self._medium_count.zero_()
        self._local_buffer = []
        self._medium_buffer = []

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        """Update context with new token embedding x: [B, D]."""
        B = x.size(0)
        device = x.device
        step = self._step.item()

        # Ensure batch/device consistency
        if self.c_local.size(0) != B or self.c_local.device != device:
            self.c_local = normalize_sphere(torch.randn(B, self.latent_dim, device=device))
            self.c_medium = normalize_sphere(torch.randn(B, self.latent_dim, device=device))
            self.c_global = normalize_sphere(torch.randn(B, self.latent_dim, device=device))

        # ── Local: every step ──
        self.c_local = normalize_sphere(
            self.beta_local * self.c_local.to(device) + (1 - self.beta_local) * x
        )

        # Store batch-averaged embedding for medium context
        self._local_buffer.append(x.mean(dim=0).detach().cpu())
        if len(self._local_buffer) > self.config.context_local_window:
            self._local_buffer.pop(0)
        self._local_count.add_(1)

        # ── Medium: every 4 steps ──
        if step > 0 and step % 4 == 0 and len(self._local_buffer) > 0:
            local_avg = torch.stack(self._local_buffer).mean(dim=0).to(device)
            self.c_medium = normalize_sphere(
                self.beta_medium * self.c_medium.to(device) + (1 - self.beta_medium) * local_avg.unsqueeze(0).expand(B, -1)
            )
            self._medium_buffer.append(self.c_medium.mean(dim=0).detach().cpu())
            if len(self._medium_buffer) > self.config.context_medium_window:
                self._medium_buffer.pop(0)
            self._medium_count.add_(1)

        # ── Global: every 16 steps ──
        if step > 0 and step % 16 == 0 and len(self._medium_buffer) > 0:
            medium_avg = torch.stack(self._medium_buffer).mean(dim=0).to(device)
            self.c_global = normalize_sphere(
                self.beta_global * self.c_global.to(device) + (1 - self.beta_global) * medium_avg.unsqueeze(0).expand(B, -1)
            )

        self._step.add_(1)

    def get_context(self) -> Dict[str, torch.Tensor]:
        """Return dict of context vectors."""
        return {
            'local': self.c_local,
            'medium': self.c_medium,
            'global': self.c_global,
        }

    def get_combined_context(self) -> torch.Tensor:
        """Weighted average context."""
        c = (
            self.config.lambda_context_local * self.c_local +
            self.config.lambda_context_medium * self.c_medium +
            self.config.lambda_context_global * self.c_global
        )
        return normalize_sphere(c)
