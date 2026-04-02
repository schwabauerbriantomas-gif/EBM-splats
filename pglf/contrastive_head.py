"""
Contrastive learning head for cross-modal alignment on S^639.

Uses InfoNCE loss adapted for the hypersphere — encourages embeddings
from the same concept (different modalities) to be close, and different
concepts to be far apart on the sphere.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class HypersphereContrastiveLoss(nn.Module):
    """
    InfoNCE loss on the 640D hypersphere.
    
    For each anchor, positive pairs are the same concept in different modalities.
    Negatives are different concepts. Uses cosine similarity on the sphere.
    
    Supports multiple modalities and temperature scaling.
    """
    
    def __init__(self, temperature: float = 0.07, hard_negative_weight: float = 0.0):
        super().__init__()
        self.temperature = temperature
        self.hard_negative_weight = hard_negative_weight
    
    def forward(
        self,
        embeddings_a: torch.Tensor,  # [B, D] modality A embeddings (normalized)
        embeddings_b: torch.Tensor,  # [B, D] modality B embeddings (normalized)
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute symmetric InfoNCE loss.
        
        Positive pairs: (a_i, b_i) for each i in batch
        Negative pairs: (a_i, b_j) for j != i
        """
        B, D = embeddings_a.shape
        
        # Ensure unit norm
        a = F.normalize(embeddings_a, dim=-1)
        b = F.normalize(embeddings_b, dim=-1)
        
        # Cosine similarity matrix — scale by temperature
        sim_matrix = a @ b.T / self.temperature  # [B, B]
        
        # Clamp to prevent overflow in float16
        sim_matrix = sim_matrix.clamp(-20.0, 20.0)
        
        # Labels: diagonal is positive
        labels = torch.arange(B, device=a.device)
        
        # Symmetric loss
        loss_a = F.cross_entropy(sim_matrix, labels)
        loss_b = F.cross_entropy(sim_matrix.T, labels)
        loss = (loss_a + loss_b) / 2
        
        # Logging info
        with torch.no_grad():
            # Accuracy: how often is the positive pair the most similar?
            pos_sim = torch.diag(sim_matrix)
            acc_a = (sim_matrix.argmax(dim=1) == labels).float().mean()
            acc_b = (sim_matrix.T.argmax(dim=1) == labels).float().mean()
            
            # Mean similarity stats
            mean_pos = pos_sim.mean().item()
            mean_neg = (sim_matrix.sum(dim=1) - pos_sim).mean().item() / max(B - 1, 1)
        
        metrics = {
            "contrastive_loss": loss.item(),
            "pos_sim": mean_pos,
            "neg_sim": mean_neg,
            "accuracy": ((acc_a + acc_b) / 2).item(),
        }
        
        return loss, metrics


class TripletHypersphereLoss(nn.Module):
    """
    Triplet loss on the hypersphere for fine-grained control.
    
    anchor · positive > anchor · negative + margin
    """
    
    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin
    
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        a = F.normalize(anchor, dim=-1)
        p = F.normalize(positive, dim=-1)
        n = F.normalize(negative, dim=-1)
        
        pos_sim = (a * p).sum(dim=-1)
        neg_sim = (a * n).sum(dim=-1)
        
        loss = F.relu(neg_sim - pos_sim + self.margin).mean()
        return loss


class UniformityAlignmentLoss(nn.Module):
    """
    Encourages embeddings to be uniformly distributed on the hypersphere
    AND aligned across modalities.
    
    Based on: "Understanding Contrastive Representation Learning through
    Alignment and Uniformity on the Hypersphere" (Wang & Isola, 2020)
    """
    
    def __init__(self, alpha: float = 2.0, t_uniform: float = 2.0):
        super().__init__()
        self.alpha = alpha  # Uniformity weight
        self.t_uniform = t_uniform
    
    def alignment_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Minimize distance between aligned pairs."""
        return ((x - y).norm(dim=-1) ** 2).mean()
    
    def uniformity_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Maximize uniformity on the hypersphere (minimize this)."""
        # Gaussian potential kernel — clamp to avoid log(0) or log(neg)
        pw_distances = torch.pdist(x, p=2)
        gaussian = (-2 * pw_distances.pow(2) / self.t_uniform).exp()
        return gaussian.log().clamp(min=-20).mean()
    
    def forward(
        self,
        embeddings_a: torch.Tensor,
        embeddings_b: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        a = F.normalize(embeddings_a, dim=-1)
        b = F.normalize(embeddings_b, dim=-1)
        
        align = self.alignment_loss(a, b)
        uniform = (self.uniformity_loss(a) + self.uniformity_loss(b)) / 2
        loss = align + self.alpha * uniform
        
        return loss, {
            "alignment": align.item(),
            "uniformity": uniform.item(),
        }
