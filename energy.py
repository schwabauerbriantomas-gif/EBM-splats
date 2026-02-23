import torch
import torch.nn as nn
import torch.nn.functional as F

from config import EBMConfig
from geometry import project_to_tangent

try:
    from vulkan_engine import VulkanEBMRunner
    HAS_VULKAN = True
except ImportError:
    HAS_VULKAN = False

class EnergyFunction(nn.Module):
    def __init__(self, config: EBMConfig, splat_store):
        super().__init__()
        self.config = config
        self.splats = splat_store
        
        self.vulkan_runner = None
        if self.config.device == "vulkan" and HAS_VULKAN:
            print("INFO: Initializing Vulkan hardware bindings for Energy Compute...")
            self.vulkan_runner = VulkanEBMRunner(latent_dim=config.latent_dim)
        
        # Compositional Interaction G-function weights
        self.W_comp = nn.Linear(3, 1)

    def compute_splat_energy(self, x: torch.Tensor) -> torch.Tensor:
        """
        E_splats(x) = -log Σ_{k=1}^K exp(α_k * (x·μ_k - 1) / τ)
        Handles both [B, D] and [B, S, D] inputs.
        """
        original_shape = x.shape[:-1]
        if x.dim() > 2:
            x = x.view(-1, x.size(-1)) # Flatten to [B*S, D]
            
        neighbors_mu, neighbors_alpha, _ = self.splats.find_neighbors(x, self.config.knn_k)
        
        # x is [B*S, D]. neighbors_mu is [B*S, K, D]
        # Calculate inner dot products broadcasted: x * mu_k
        dot_products = torch.bmm(neighbors_mu, x.unsqueeze(-1)).squeeze(-1) # -> [B*S, K]
        
        # Calculate inner exponent
        exponent = neighbors_alpha * (dot_products - 1.0) / self.config.temperature
        
        # Output is [B*S]
        energy = -torch.logsumexp(exponent, dim=-1)
        
        # Reshape back to original batch/seq layout
        return energy.view(original_shape)

    def compute_geom_energy(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates simple geometric regularization terms explicitly avoiding global collapse.
        For exactness based on specs: E_geom(x) = -λ_spread * Σ_{i<j} log(1 - x_i·x_j) 
        We approximate this via random batch-level spreading during train time.
        """
        if x.dim() > 2:
            x = x.view(-1, x.size(-1)) # Flatten to [B*S, D]
            
        # Batch-level pairwise dot products
        batch_sims = torch.mm(x, x.T)
        
        # Exclude self-similarity diagonal
        mask = ~torch.eye(x.size(0), dtype=torch.bool, device=x.device)
        spread_energy = -torch.log(1.0 - batch_sims[mask] + 1e-4).mean()
        
        # Coverage regularization implicitly mapped to global splats distance 
        # (simplified for batch level approximation)
        coverage_energy = torch.tensor(0.0, device=x.device) 
        
        return spread_energy + coverage_energy

    def compute_comp_energy(self, x: torch.Tensor) -> torch.Tensor:
        """
        g(u, v) = σ(W·[u, v, u·v] + b)
        Approximate interaction for top-2 nearest neighbors for compositionality.
        """
        original_shape = x.shape[:-1]
        if x.dim() > 2:
            x = x.view(-1, x.size(-1))
            
        neighbors_mu, _, _ = self.splats.find_neighbors(x, 2)
        
        u = torch.bmm(neighbors_mu[:, 0:1, :], x.unsqueeze(-1)).squeeze(-1) # [B*S, 1]
        v = torch.bmm(neighbors_mu[:, 1:2, :], x.unsqueeze(-1)).squeeze(-1) # [B*S, 1]
        
        uv_concat = torch.cat([u, v, u * v], dim=-1)
        comp_energy = torch.sigmoid(self.W_comp(uv_concat)).squeeze(-1)
        
        return comp_energy.view(original_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns total energy [B]."""
        e_splat = self.compute_splat_energy(x)
        e_geom = self.compute_geom_energy(x) * 0.01  # lambda_geom
        e_comp = self.compute_comp_energy(x) * 0.05  # lambda_comp
        
        return e_splat + e_geom + e_comp
        
    def compute_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        s_theta(x) = -Grad_R E(x)
        Returns the exact Riemannian Score representation.
        """
        # --- Vulkan GPU Acceleration Path ---
        if self.vulkan_runner is not None:
            # The Vulkan compiled schema naturally calculates the Riemannian projections
            # and gradient inversions inside the `energy.comp` GLSL directly.
            mu_active = self.splats.mu[:self.splats.n_active]
            alpha_active = self.splats.alpha[:self.splats.n_active]
            
            vk_score = self.vulkan_runner.run_compute(x, mu_active, alpha_active)
            return vk_score
            
        # --- CPU PyTorch Fallback Path ---
        with torch.enable_grad():
            x.requires_grad_(True)
            energy: torch.Tensor = self.forward(x)
            
            # Sum for batch
            loss = energy.sum()
            grad_e = torch.autograd.grad(loss, x)[0]
            
            # Riemannian Gradient Projection
            grad_r = project_to_tangent(x, grad_e)
            
        return -grad_r
