import torch
import torch.nn as nn
from config import EBMConfig
from splats import SplatStorage
from energy import EnergyFunction
from langevin import sample_langevin
from decoder import EBMDecoder
from geometry import normalize_sphere

class CompositionalEngine(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.comp_net = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim)
        )
    
    def forward(self, v_a: torch.Tensor, v_b: torch.Tensor) -> torch.Tensor:
        """Compose two tangent vectors"""
        combined = torch.cat([v_a, v_b], dim=-1)
        return self.comp_net(combined)


class EBMModel(nn.Module):
    def __init__(self, config: EBMConfig):
        super().__init__()
        self.config = config
        
        # Core embedding layer
        self.embedding = nn.Embedding(config.vocab_size, config.latent_dim)
        self.splats = SplatStorage(config)
        self.energy_fn = EnergyFunction(config, self.splats)
        self.decoder = EBMDecoder(config)
        self.comp_engine = CompositionalEngine(config.latent_dim)

    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Embeds tokens and maps exactly to the hypersphere."""
        x = self.embedding(token_ids)
        x = normalize_sphere(x)
        return x

    def compute_energy(self, x: torch.Tensor) -> torch.Tensor:
        """Energy for batch x."""
        return self.energy_fn(x)

    def compute_score(self, x: torch.Tensor) -> torch.Tensor:
        """Riemannian score for denoising matching."""
        return self.energy_fn.compute_score(x)

    def sample(self, n_samples: int, context: torch.Tensor = None) -> torch.Tensor:
        """Generates minimal energy states via underdamped Langevin dynamics."""
        active_device = "cpu" if self.config.device == "vulkan" else self.config.device
        # Initialize uniformly on hypersphere
        x_init = torch.randn(n_samples, self.config.latent_dim, device=active_device)
        x_init = normalize_sphere(x_init)
        
        # If context is given, start near the context
        if context is not None:
            # We add slight noise to the context and normalize
            x_init = normalize_sphere(context + 0.1 * x_init)

        sampled_x = sample_langevin(x_init, self.energy_fn, self.config)
        return sampled_x

    def decode(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Converts low-energy spatial states back to discrete token spaces."""
        return self.decoder(x, context)
        
    def to_device(self):
        """Moves entire model implicitly preserving faiss structures."""
        # We manually keep some tensors specifically constrained.
        self.to(self.config.device)
        return self
