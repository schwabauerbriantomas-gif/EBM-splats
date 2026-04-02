"""
CUDA-native Energy Function — V2 Compliant.

Same as energy.py but uses CUDA AMP-safe autograd instead of manual gradients.
The V2 manual gradient approach in energy.py is preferred for training stability.
This module serves as CUDA fallback when V2 manual gradients aren't needed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import EBMConfig
from geometry import project_to_tangent


class EnergyFunctionCUDA(nn.Module):
    """CUDA-native energy with V2 sign conventions."""

    def __init__(self, config: EBMConfig, splat_store):
        super().__init__()
        self.config = config
        self.splats = splat_store
        self.W_comp = nn.Linear(3, 1)

    def compute_splat_energy(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape[:-1]
        if x.dim() > 2:
            x = x.reshape(-1, x.size(-1))

        neighbors_mu, neighbors_alpha, neighbors_kappa = self.splats.find_neighbors(x, self.config.knn_k)

        dot_products = torch.bmm(neighbors_mu, x.unsqueeze(-1)).squeeze(-1)

        # Importance weighting (V2 §2.2)
        importance = neighbors_kappa.clamp(min=1e-4)
        weights = importance / importance.sum(dim=-1, keepdim=True)

        exponent = neighbors_alpha * (dot_products - 1.0) / self.config.temperature
        weighted_exponent = exponent + torch.log(weights.clamp(min=1e-8))

        energy = -torch.logsumexp(weighted_exponent, dim=-1)
        return energy.view(original_shape)

    def compute_geom_energy(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape[:-1]
        if x.dim() > 2:
            x = x.reshape(-1, x.size(-1))

        if x.size(0) < 2:
            return torch.tensor(0.0, device=x.device)

        batch_sims = torch.mm(x, x.T)
        mask = ~torch.eye(x.size(0), dtype=torch.bool, device=x.device)
        off_diag = batch_sims[mask]
        spread_energy = -torch.log(1.0 - off_diag.clamp(max=1.0 - 1e-4) + 1e-4).mean()

        return spread_energy

    def compute_comp_energy(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape[:-1]
        if x.dim() > 2:
            x = x.reshape(-1, x.size(-1))

        neighbors_mu, _, _ = self.splats.find_neighbors(x, 2)
        u = torch.bmm(neighbors_mu[:, 0:1, :], x.unsqueeze(-1)).squeeze(-1)
        v = torch.bmm(neighbors_mu[:, 1:2, :], x.unsqueeze(-1)).squeeze(-1)
        uv_concat = torch.cat([u, v, u * v], dim=-1)
        comp_energy = torch.sigmoid(self.W_comp(uv_concat)).squeeze(-1)
        return comp_energy.view(original_shape)

    def compute_context_energy(self, x: torch.Tensor, context_vecs: dict) -> torch.Tensor:
        energy = torch.tensor(0.0, device=x.device)
        for level, vec in context_vecs.items():
            lam = getattr(self.config, f'lambda_context_{level}', 0.0)
            energy = energy - lam * (x * vec).sum(dim=-1)
        return energy

    def forward(self, x: torch.Tensor, context_vecs: dict = None) -> torch.Tensor:
        e_splat = self.compute_splat_energy(x)
        e_geom = self.compute_geom_energy(x) * self.config.lambda_geom
        e_comp = self.compute_comp_energy(x) * self.config.lambda_comp
        e_total = e_splat + e_geom + e_comp

        if context_vecs is not None:
            e_trans = self.compute_context_energy(x, context_vecs)
            e_total = e_total + e_trans

        return e_total

    def compute_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Riemannian score via AMP-safe autograd.
        For training stability, prefer energy.py manual gradient approach.
        """
        with torch.amp.autocast('cuda', enabled=False):
            x_f32 = x.float().detach().requires_grad_(True)
            energy = self.forward(x_f32.float())
            grad_e = torch.autograd.grad(energy.sum(), x_f32, create_graph=False)[0]
            grad_riemann = project_to_tangent(x_f32, grad_e)

        return -grad_riemann
