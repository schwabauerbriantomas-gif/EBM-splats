"""
Flow Matching for embedding generation on S^639.

Implements Optimal Transport Conditional Flow Matching (OT-CFM) that learns
to map noise → Pareto-optimal embeddings on the 640D hypersphere.

At inference, a single ODE solve produces high-quality embeddings deterministically.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class HypersphereFlowMatching(nn.Module):
    """
    Flow Matching model that operates on the 640D unit hypersphere.
    
    Learns a time-dependent vector field v(x, t, c) where:
    - x: current point on S^639
    - t: time in [0, 1]
    - c: conditioning (modality + content encoding)
    
    The field transports noise (t=0) to Pareto-optimal embeddings (t=1).
    """
    
    def __init__(
        self,
        dim: int = 640,
        hidden_dim: int = 1280,
        n_layers: int = 6,
        condition_dim: int = 640,
        n_modalities: int = 3,  # text, image, audio
    ):
        super().__init__()
        self.dim = dim
        self.condition_dim = condition_dim
        
        # Time embedding (sinusoidal)
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(dim),
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Modality embedding
        self.modality_embed = nn.Embedding(n_modalities, hidden_dim)
        
        # Condition projection
        self.cond_proj = nn.Linear(condition_dim, hidden_dim)
        
        # Flow network: predicts tangent vector at (x, t, c)
        layers = []
        in_dim = dim + hidden_dim  # x + time_hidden
        for i in range(n_layers):
            out_dim = hidden_dim if i < n_layers - 1 else dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < n_layers - 1:
                layers.append(nn.SiLU())
            in_dim = out_dim
        self.flow_net = nn.Sequential(*layers)
        
        # Initialize last layer near zero for stable training
        last_layer = self.flow_net[-1] if isinstance(self.flow_net[-1], nn.Linear) else self.flow_net[-2]
        nn.init.zeros_(last_layer.weight)
        nn.init.zeros_(last_layer.bias)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            if m not in [self.flow_net[-1], self.flow_net[-2]]:
                nn.init.xavier_uniform_(m.weight, gain=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def project_to_tangent(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Project vector v onto the tangent plane at x on the hypersphere."""
        # v_tangent = v - (v · x) * x
        dot = (v * x).sum(dim=-1, keepdim=True)
        return v - dot * x
    
    def forward(
        self,
        x: torch.Tensor,      # [B, D] current point on sphere
        t: torch.Tensor,       # [B, 1] time
        condition: torch.Tensor,  # [B, cond_dim] content conditioning
        modality: torch.Tensor,   # [B] modality index
    ) -> torch.Tensor:
        """
        Predict tangent vector at (x, t, c).
        Returns: [B, D] tangent vector on the hypersphere at x.
        """
        # Time embedding
        t_emb = self.time_embed(t)  # [B, hidden]
        
        # Modality + condition
        mod_emb = self.modality_embed(modality)  # [B, hidden]
        cond_emb = self.cond_proj(condition)  # [B, hidden]
        
        # Combine conditioning
        cond = t_emb + mod_emb + cond_emb  # [B, hidden]
        
        # Flow network input: [x, cond]
        h = torch.cat([x, cond], dim=-1)  # [B, D + hidden]
        v = self.flow_net(h)  # [B, D]
        
        # Project to tangent plane
        v_tangent = self.project_to_tangent(x, v)
        
        return v_tangent
    
    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        modality: torch.Tensor,
        n_steps: int = 10,
        dt: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Generate embedding via ODE solve (Euler method on sphere).
        
        Args:
            condition: [B, cond_dim] content encoding
            modality: [B] modality indices
            n_steps: Number of ODE steps
        Returns:
            [B, D] embeddings on the hypersphere
        """
        B = condition.shape[0]
        device = condition.device
        
        if dt is None:
            dt = 1.0 / n_steps
        
        # Start from random point on sphere (noise)
        x = torch.randn(B, self.dim, device=device)
        x = F.normalize(x, dim=-1)
        
        # Euler integration on the sphere
        for step in range(n_steps):
            t = torch.full((B, 1), step * dt, device=device)
            v = self.forward(x, t, condition, modality)
            
            # Riemannian exponential map (approximate for small steps)
            # On sphere: exp_x(v) = cos(||v||) * x + sin(||v||) * v/||v||
            v_norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            x = torch.cos(v_norm * dt) * x + torch.sin(v_norm * dt) * (v / v_norm)
            x = F.normalize(x, dim=-1)  # Ensure unit norm
        
        return x


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for continuous time t in [0, 1]."""
    
    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: [B, 1] time values
        Returns:
            [B, dim] sinusoidal embeddings
        """
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half_dim, device=t.device, dtype=torch.float32) / half_dim
        )
        args = t * freqs.unsqueeze(0)  # [B, half_dim]
        embedding = torch.cat([args.cos(), args.sin()], dim=-1)
        
        if self.dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))
        
        return embedding


class FlowMatchingLoss(nn.Module):
    """
    Optimal Transport CFM loss for hypersphere flow matching.
    
    Given pairs (x_0=noise, x_1=target embedding), the OT path is:
        x_t = cos(arccos(x_0·x_1) * t) * x_0 + sin(arccos(x_0·x_1) * t) * v_ot/||v_ot||
    where v_ot is the optimal transport direction.
    
    The model learns to predict the tangent vector at each point along this path.
    """
    
    def __init__(self, sigma_min: float = 1e-4):
        super().__init__()
        self.sigma_min = sigma_min
    
    def geodesic_interpolate(
        self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        Interpolate along geodesic on the hypersphere.
        x_t = slerp(x_0, x_1, t)
        """
        # Cosine of angle between x_0 and x_1
        dot = (x_0 * x_1).sum(dim=-1, keepdim=True).clamp(-0.999, 0.999)
        omega = torch.acos(dot)
        sin_omega = torch.sin(omega).clamp(min=1e-8)
        
        x_t = (torch.sin((1 - t) * omega) / sin_omega) * x_0 + \
              (torch.sin(t * omega) / sin_omega) * x_1
        
        return F.normalize(x_t, dim=-1)
    
    def geodesic_velocity(
        self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the tangent velocity of the geodesic at time t.
        This is the target for the flow matching model.
        """
        dot = (x_0 * x_1).sum(dim=-1, keepdim=True).clamp(-0.999, 0.999)
        omega = torch.acos(dot)
        sin_omega = torch.sin(omega).clamp(min=1e-8)
        
        # d/dt of slerp
        v = (-omega * torch.cos((1 - t) * omega) / sin_omega) * x_0 + \
            (omega * torch.cos(t * omega) / sin_omega) * x_1
        
        return v
    
    def forward(
        self,
        model: nn.Module,
        x_1: torch.Tensor,       # [B, D] target embeddings (from Pareto frontier)
        condition: torch.Tensor,   # [B, cond_dim] content encoding
        modality: torch.Tensor,    # [B] modality indices
    ) -> torch.Tensor:
        """
        Compute flow matching loss.
        
        1. Sample noise x_0 on sphere
        2. Sample random time t ~ U(0, 1)
        3. Interpolate to get x_t
        4. Compute target velocity v_t
        5. Loss = ||model(x_t, t, c) - v_t||^2
        """
        B, D = x_1.shape
        device = x_1.device
        
        # Sample noise on sphere
        x_0 = torch.randn(B, D, device=device)
        x_0 = F.normalize(x_0, dim=-1)
        
        # Random time
        t = torch.rand(B, 1, device=device)  # [B, 1]
        
        # Interpolate
        x_t = self.geodesic_interpolate(x_0, x_1, t)
        
        # Target velocity
        v_target = self.geodesic_velocity(x_0, x_1, t)
        
        # Add small noise for numerical stability
        noise = torch.randn_like(x_t) * self.sigma_min
        noise = noise - (noise * x_t).sum(dim=-1, keepdim=True) * x_t  # Project to tangent
        x_t_noisy = F.normalize(x_t + noise, dim=-1)
        
        # Model prediction
        v_pred = model(x_t_noisy, t, condition, modality)
        
        # MSE loss in tangent space (only direction matters, not magnitude)
        loss = F.mse_loss(v_pred, v_target)
        
        return loss
