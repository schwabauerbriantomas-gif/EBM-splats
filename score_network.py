#!/usr/bin/env python3
"""
Score Network — direct parametric score model for denoising score matching.

Replaces the broken approach of using torch.autograd.grad on the energy
function (which had create_graph=False + .detach(), breaking gradient flow).

Architecture:
  - Input: x (batch, dim), sigma (scalar noise level)
  - sigma → MLP embedding (64-dim), concatenated with x
  - 3 hidden layers (2×dim) with GELU + LayerNorm
  - Output: score vector in tangent space (batch, dim), projected via project_to_tangent
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from geometry import project_to_tangent


class SigmaEncoder(nn.Module):
    """Encode noise level sigma into a fixed-size embedding vector."""

    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.out_dim = out_dim
        # Random Fourier features for sigma (sinusoidal encoding)
        self.register_buffer('freqs', torch.randn(out_dim // 2) * math.pi)

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sigma: [...] tensor of noise levels
        Returns:
            [...] × out_dim embedding
        """
        # sigma is shape [B] or scalar → project to [..., 1]
        sigma = sigma.unsqueeze(-1)  # [..., 1]
        proj = sigma * self.freqs.unsqueeze(0)  # [..., out_dim//2]
        emb = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)  # [..., out_dim]
        return emb


class ScoreNetwork(nn.Module):
    """
    Direct score model: s_theta(x, sigma) → tangent-space vector.

    s_theta(x, sigma) ≈ (x_0 - x) / sigma^2  (the denoising score).
    """

    def __init__(self, dim: int = 640, hidden_dim: int = None, sigma_emb_dim: int = 64, n_layers: int = 3):
        super().__init__()
        self.dim = dim
        hidden_dim = hidden_dim or (2 * dim)

        self.sigma_encoder = SigmaEncoder(sigma_emb_dim)

        # Input: x (dim) + sigma_emb (sigma_emb_dim)
        self.input_proj = nn.Linear(dim + sigma_emb_dim, hidden_dim)

        # Hidden layers
        layers = []
        for _ in range(n_layers - 1):
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.hidden = nn.Sequential(*layers)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, D] points on hypersphere
            sigma: [B] or scalar noise levels
        Returns:
            [B, D] score vector, projected to tangent space at x
        """
        B, D = x.shape

        # Expand sigma to match batch
        if sigma.dim() == 0:
            sigma = sigma.expand(B)
        sigma_emb = self.sigma_encoder(sigma)  # [B, sigma_emb_dim]

        # Concatenate and feed through network
        h = torch.cat([x, sigma_emb], dim=-1)  # [B, D + sigma_emb_dim]
        h = self.input_proj(h)
        h = self.hidden(h)
        h = self.output_proj(h)  # [B, D]

        # Project to tangent space at x
        score = project_to_tangent(x, h)
        return score
