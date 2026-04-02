"""
Energy Function — V2 Specification Compliant.

V2 fixes applied:
  §2.1  Sign convention: p(x) = exp(-E(x))/Z, low energy near splats
  §2.2  Importance-weighted logsumexp with (x·μ_k - 1) ∈ [-2, 0]
  §2.3  Gradient stability: adaptive normalization, Riemannian projection
  §2.4  Collapse regularization between neighboring splats
  §4.4  Context transition energy E_trans(x_t) = -Σ λ_l · (x_t · c_l)

Gradient computation (V2 §2.3):
  ∇E_local(x) = -Σ p_k · (α_k/τ) · proj_tangent(μ_k) / clip(||proj_tangent(μ_k)||, ε, ∞)

Riemannian gradient for score:
  ∇_R E(x) = ∇E(x) - (x·∇E(x))·x   (project to tangent plane)
  score(x) = -∇_R E(x)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import EBMConfig
from geometry import project_to_tangent


class EnergyFunction(nn.Module):
    def __init__(self, config: EBMConfig, splat_store):
        super().__init__()
        self.config = config
        self.splats = splat_store
        self.W_comp = nn.Linear(3, 1)

    def compute_splat_energy(self, x: torch.Tensor) -> torch.Tensor:
        """
        E_splats(x) = -log Σ_{k∈N(x)} exp(α_k · (x·μ_k - 1) / τ)
        
        Sign convention (V2 §2.1):
          (x·μ_k - 1) ∈ [-2, 0], max=0 when x=μ_k
          → logsumexp over non-positive exponents
          → -logsumexp is NON-NEGATIVE, MINIMUM=0 at splat centers
          → Low energy near splats ✓
        """
        original_shape = x.shape[:-1]
        if x.dim() > 2:
            x = x.reshape(-1, x.size(-1))

        neighbors_mu, neighbors_alpha, neighbors_kappa = self.splats.find_neighbors(x, self.config.knn_k)

        # Inner products: [B*S, K]
        dot_products = torch.bmm(neighbors_mu, x.unsqueeze(-1)).squeeze(-1)

        # Importance-weighted (V2 §2.2): importance = density proxy via kappa
        # w_k = kappa_k / Σ kappa_j (normalized weights)
        importance = neighbors_kappa.clamp(min=1e-4)
        weights = importance / importance.sum(dim=-1, keepdim=True)

        # Weighted exponent: α_k · (x·μ_k - 1) / τ
        exponent = neighbors_alpha * (dot_products - 1.0) / self.config.temperature

        # Weighted logsumexp (add log(w_k) to exponents)
        weighted_exponent = exponent + torch.log(weights.clamp(min=1e-8))

        energy = -torch.logsumexp(weighted_exponent, dim=-1)
        return energy.view(original_shape)

    def compute_splat_energy_grad(self, x: torch.Tensor) -> torch.Tensor:
        """
        Manual gradient of E_splats for V2 §2.3 stability.
        
        ∇E_local(x) = -Σ_k p_k · (α_k/τ) · proj_tangent(μ_k) / clip(||proj_tangent(μ_k)||, ε, ∞)
        
        Returns raw Euclidean gradient [B, D] (not yet projected to tangent).
        """
        original_shape = x.shape[:-1]
        if x.dim() > 2:
            x = x.reshape(-1, x.size(-1))

        neighbors_mu, neighbors_alpha, neighbors_kappa = self.splats.find_neighbors(x, self.config.knn_k)

        dot_products = torch.bmm(neighbors_mu, x.unsqueeze(-1)).squeeze(-1)  # [B, K]

        importance = neighbors_kappa.clamp(min=1e-4)
        weights = importance / importance.sum(dim=-1, keepdim=True)

        # Compute attention weights p_k = softmax(α_k · (x·μ_k - 1) / τ + log w_k)
        exponents = neighbors_alpha * (dot_products - 1.0) / self.config.temperature
        weighted_logits = exponents + torch.log(weights.clamp(min=1e-8))
        p_k = torch.softmax(weighted_logits, dim=-1)  # [B, K]

        # Project mu_k to tangent plane at x
        # proj_tangent(μ_k) = μ_k - (x·μ_k)·x
        x_dot_mu = dot_products.unsqueeze(-1)  # [B, K, 1]
        mu_tangent = neighbors_mu - x_dot_mu * x.unsqueeze(1)  # [B, K, D]

        # Norm and clip for stability
        mu_tangent_norm = mu_tangent.norm(dim=-1, keepdim=True).clamp(min=1e-6)  # [B, K, 1]
        mu_tangent_stable = mu_tangent / mu_tangent_norm  # Normalized tangent vectors

        # Weighted sum: -Σ p_k · (α_k/τ) · direction
        coeff = p_k * (neighbors_alpha / self.config.temperature)  # [B, K]
        grad = -(coeff.unsqueeze(-1) * mu_tangent_stable).sum(dim=1)  # [B, D]

        return grad.view(original_shape + (self.config.latent_dim,))

    def compute_geom_energy(self, x: torch.Tensor) -> torch.Tensor:
        """
        E_geom(x) = -λ_spread · mean(log(1 - x_i·x_j))
        V2 §2.4: Collapse regularization for batch-level diversity.
        Requires batch_size >= 2; returns 0 for batch_size < 2.
        """
        if x.dim() > 2:
            x = x.reshape(-1, x.size(-1))

        if x.size(0) < 2:
            return torch.tensor(0.0, device=x.device)

        batch_sims = torch.mm(x, x.T)
        mask = ~torch.eye(x.size(0), dtype=torch.bool, device=x.device)
        off_diag = batch_sims[mask]
        # Clamp to avoid log(0) or log(negative)
        spread_energy = -torch.log(1.0 - off_diag.clamp(max=1.0 - 1e-4) + 1e-4).mean()

        return spread_energy

    def compute_comp_energy(self, x: torch.Tensor) -> torch.Tensor:
        """Compositional interaction energy."""
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
        """
        V2 §4.4: E_trans(x_t) = -Σ_{l∈{local,medium,global}} λ_l · (x_t · c_l)
        
        Args:
            x: [B, D] current latent
            context_vecs: dict with keys 'local', 'medium', 'global', each [B, D]
        """
        energy = torch.tensor(0.0, device=x.device)
        for level, vec in context_vecs.items():
            lam = getattr(self.config, f'lambda_context_{level}', 0.0)
            energy = energy - lam * (x * vec).sum(dim=-1)
        return energy

    def forward(self, x: torch.Tensor, context_vecs: dict = None) -> torch.Tensor:
        """
        Total energy E(x) = E_splats + λ_geom·E_geom + λ_comp·E_comp + E_trans
        """
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
        Score: s(x) = -∇_R E(x) = -(∇E(x) - (x·∇E(x))·x)
        
        Uses manual gradient computation from V2 §2.3 for stability,
        then projects to Riemannian tangent plane.
        """
        x_flat = x.reshape(-1, self.config.latent_dim)

        # Manual splat energy gradient (stable, no autograd needed for core term)
        grad_e = self.compute_splat_energy_grad(x_flat)  # Already Euclidean

        # Riemannian projection: ∇_R E = ∇E - (x·∇E)·x
        x_dot_grad = (x_flat * grad_e).sum(dim=-1, keepdim=True)
        grad_riemann = grad_e - x_dot_grad * x_flat

        return -grad_riemann
