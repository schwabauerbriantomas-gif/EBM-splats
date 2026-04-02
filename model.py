"""
EBM Model — V2 Integrated.

Integrates:
  - HierarchicalContext for long-range dependencies
  - Fixed energy function with V2 sign conventions
  - ScoreNetwork-compatible scoring
  - Context-aware decoding
  - Gradient clipping in compute methods
"""

import torch
import torch.nn as nn

from config import EBMConfig
from splats import SplatStorage
from energy import EnergyFunction
from langevin import sample_langevin
from decoder import EBMDecoder
from geometry import normalize_sphere
from context_hierarchy import HierarchicalContext


class CompositionalEngine(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.comp_net = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, v_a: torch.Tensor, v_b: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([v_a, v_b], dim=-1)
        return self.comp_net(combined)


class EBMModel(nn.Module):
    def __init__(self, config: EBMConfig):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.latent_dim)
        self.splats = SplatStorage(config)
        self.energy_fn = EnergyFunction(config, self.splats)
        self.decoder = EBMDecoder(config)
        self.comp_engine = CompositionalEngine(config.latent_dim)
        self.context = HierarchicalContext(config)

    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Embed tokens and project onto hypersphere."""
        x = self.embedding(token_ids)
        return normalize_sphere(x)

    def compute_energy(self, x: torch.Tensor, context_vecs=None) -> torch.Tensor:
        return self.energy_fn(x, context_vecs=context_vecs)

    def compute_score(self, x: torch.Tensor) -> torch.Tensor:
        """Riemannian score using V2 stable manual gradient."""
        return self.energy_fn.compute_score(x)

    def generate(self, prompt_tokens: torch.Tensor, max_new_tokens: int = 32,
                 device: torch.device = torch.device('cpu')) -> torch.Tensor:
        """
        Autoregressive generation using Langevin sampling + decoding.
        
        Args:
            prompt_tokens: [B, S] token IDs
            max_new_tokens: number of tokens to generate
            device: torch device
        Returns:
            [B, S + max_new_tokens] generated token IDs
        """
        B = prompt_tokens.size(0)
        active_device = "cpu" if self.config.device == "vulkan" else self.config.device

        # Initialize context
        self.context.reset(B, device)

        # Embed prompt and update context
        x_prompt = self.embed(prompt_tokens.to(device))
        for t in range(x_prompt.size(1)):
            self.context.update(x_prompt[:, t])

        generated = [prompt_tokens]
        last_logits = None

        for step in range(max_new_tokens):
            # Sample latent via Langevin
            context_combined = self.context.get_combined_context()
            x_sampled = self.sample(n_samples=B, context=context_combined)

            # Decode to logits
            logits = self.decoder(x_sampled, context_combined)  # [B, vocab_size]

            # Sample token (top-k, k=50)
            probs = torch.softmax(logits, dim=-1)
            topk_probs, topk_ids = torch.topk(probs, k=50, dim=-1)
            topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

            # Sample from top-k
            sampled_flat = torch.multinomial(topk_probs.view(-1, 50), 1)
            next_tokens = topk_ids.view(-1, 50).gather(
                1, sampled_flat.view(-1, 1)
            ).squeeze(-1)  # [B]

            generated.append(next_tokens.unsqueeze(1))

            # Update context with new token embedding
            x_new = self.embed(next_tokens.to(device))
            self.context.update(x_new)

            last_logits = logits

        return torch.cat(generated, dim=1)

    def sample(self, n_samples: int, context: torch.Tensor = None) -> torch.Tensor:
        """Generate low-energy states via Langevin dynamics."""
        active_device = "cpu" if self.config.device == "vulkan" else self.config.device
        x_init = torch.randn(n_samples, self.config.latent_dim, device=active_device)
        x_init = normalize_sphere(x_init)

        if context is not None:
            x_init = normalize_sphere(context + 0.1 * x_init)

        return sample_langevin(x_init, self.energy_fn, self.config)
