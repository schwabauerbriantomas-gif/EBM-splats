"""
Evaluation Metrics — V2 §19.

Metrics:
  - Energy stats (mean, std, NaN check)
  - Cosine similarity to data (energy proxy)
  - Diversity (pairwise angular distance)
  - Perplexity estimation
  - Coherence score
  - Gradient health (magnitude, NaN check)
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional
from collections import deque


class EBMEvaluator:
    """Evaluation metrics for the EBM language model."""

    def __init__(self, device: torch.device = torch.device('cpu')):
        self.device = device
        self.energy_history: deque = deque(maxlen=1000)

    def compute_energy_stats(self, energy: torch.Tensor) -> Dict[str, float]:
        """Basic energy statistics with NaN/Inf detection."""
        e = energy.detach().cpu().float()
        return {
            'energy_mean': e.mean().item(),
            'energy_std': e.std().item(),
            'energy_min': e.min().item(),
            'energy_max': e.max().item(),
            'has_nan': torch.isnan(e).any().item(),
            'has_inf': torch.isinf(e).any().item(),
            'has_zero': (e == 0.0).any().item(),
            'all_positive': (e >= 0).all().item(),
            'is_real': not (torch.isnan(e).any() or torch.isinf(e).any()),
        }

    def compute_cosine_similarity(self, samples: torch.Tensor, references: torch.Tensor) -> float:
        """
        Average max cosine similarity from samples to nearest reference.
        Proxy for energy quality: higher = samples closer to real data.
        """
        with torch.no_grad():
            # samples: [N, D], references: [M, D]
            cos_sims = torch.mm(samples, references.T)  # [N, M]
            max_sims, _ = cos_sims.max(dim=1)
            return max_sims.mean().item()

    def compute_diversity(self, x: torch.Tensor) -> float:
        """Average pairwise angular distance (diversity). Higher = more diverse."""
        B = x.size(0)
        if B < 2:
            return 0.0
        n_pairs = min(B * 10, B * (B - 1) // 2)
        idx_i = torch.randint(0, B, (n_pairs,))
        idx_j = torch.randint(0, B, (n_pairs,))
        cos_sims = (x[idx_i] * x[idx_j]).sum(dim=-1)
        return (1.0 - cos_sims).mean().item()

    def estimate_perplexity(self, energy: torch.Tensor) -> float:
        """
        Approximate perplexity from energy: PPL ≈ exp(mean_energy).
        
        Since p(x) ∝ exp(-E(x)), average energy gives a proxy for
        log(1/p(x)), so PPL ≈ exp(E_mean).
        """
        e = energy.detach().cpu().float().mean()
        return torch.exp(e.clamp(max=100)).item()

    def compute_coherence_score(self, token_sequence: torch.Tensor) -> float:
        """
        Bigram coherence: average cosine similarity between consecutive token embeddings.
        Higher = more coherent generation.
        """
        if token_sequence.dim() != 2 or token_sequence.size(1) < 2:
            return 0.0
        # Normalize rows as proxy embeddings
        embeddings = F.normalize(token_sequence.float(), dim=-1)
        # Bigram cosine similarity
        cos_sims = (embeddings[:, :-1] * embeddings[:, 1:]).sum(dim=-1)
        return cos_sims.mean().item()

    def check_gradient_health(self, score: torch.Tensor) -> Dict[str, float]:
        """Check gradient/score tensor health."""
        s = score.detach().cpu().float()
        norms = s.norm(dim=-1)
        return {
            'score_mean_norm': norms.mean().item(),
            'score_std_norm': norms.std().item(),
            'score_has_nan': torch.isnan(s).any().item(),
            'score_has_inf': torch.isinf(s).any().item(),
            'score_is_zero': (norms < 1e-8).all().item(),
        }

    def full_report(self, energy: torch.Tensor, score: torch.Tensor,
                    samples: Optional[torch.Tensor] = None,
                    references: Optional[torch.Tensor] = None) -> Dict:
        """Generate a full evaluation report."""
        report = {}
        report['energy'] = self.compute_energy_stats(energy)
        report['perplexity'] = self.estimate_perplexity(energy)
        report['gradient'] = self.check_gradient_health(score)

        if samples is not None:
            report['diversity'] = self.compute_diversity(samples)
        if samples is not None and references is not None:
            report['cosine_to_data'] = self.compute_cosine_similarity(samples, references)

        return report

    def is_healthy(self, report: Dict) -> bool:
        """Quick health check: no NaN/Inf, non-zero energies, real scores."""
        e = report.get('energy', {})
        g = report.get('gradient', {})
        return (
            not e.get('has_nan', True) and
            not e.get('has_inf', True) and
            not e.get('all_positive', True) and  # Should have both positive and negative
            not g.get('score_has_nan', True) and
            not g.get('score_has_inf', True) and
            not g.get('score_is_zero', True)
        )
