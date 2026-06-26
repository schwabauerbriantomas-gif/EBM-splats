import torch
import torch.nn as nn
from .config import EBMConfig

class MoELayer(nn.Module):
    def __init__(self, config: EBMConfig, in_features: int):
        super().__init__()
        self.num_experts = config.moe_experts
        self.k = config.moe_active
        self.hidden_dim = config.hidden_dim
        
        # Router
        self.router = nn.Linear(in_features, self.num_experts)
        
        # Experts (simplified as parallel linear layers for CPU efficiency)
        self.experts_w1 = nn.Parameter(torch.randn(self.num_experts, in_features, self.hidden_dim) / in_features**0.5)
        self.experts_b1 = nn.Parameter(torch.zeros(self.num_experts, self.hidden_dim))
        
        self.experts_w2 = nn.Parameter(torch.randn(self.num_experts, self.hidden_dim, in_features) / self.hidden_dim**0.5)
        self.experts_b2 = nn.Parameter(torch.zeros(self.num_experts, in_features))
        
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is [B, D]
        B, D = x.shape
        router_logits = self.router(x)
        routing_weights = torch.softmax(router_logits, dim=-1)

        # Top-K routing
        top_k_weights, top_k_indices = torch.topk(routing_weights, self.k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        # Vectorized MoE: compute all experts, then gather top-K
        # x: [B, D] → expand to [B, 1, D] → [B, num_experts, D]
        x_expanded = x.unsqueeze(1).expand(-1, self.num_experts, -1)  # [B, E, D]

        # Expert FFN (batched over all experts)
        # experts_w1: [E, D, H], x_expanded: [B, E, D] → h: [B, E, H]
        h = torch.einsum('bed,edh->beh', x_expanded, self.experts_w1) + self.experts_b1.unsqueeze(0)
        h = self.activation(h)
        # experts_w2: [E, H, D], h: [B, E, H] → o: [B, E, D]
        o = torch.einsum('beh,ehd->bed', h, self.experts_w2) + self.experts_b2.unsqueeze(0)

        # Gather top-K expert outputs: [B, E, D] → [B, K, D]
        top_k_idx = top_k_indices.unsqueeze(-1).expand(-1, -1, D)  # [B, K, D]
        selected = torch.gather(o, 1, top_k_idx)  # [B, K, D]

        # Weighted sum: [B, K, 1] * [B, K, D] → [B, D]
        out = (top_k_weights.unsqueeze(-1) * selected).sum(dim=1)
        return out

class EBMDecoder(nn.Module):
    def __init__(self, config: EBMConfig):
        super().__init__()
        
        # Context concat -> [X; c_total] means dim is 2 * latent_dim
        in_dim = config.latent_dim * 2 
        
        self.moe = MoELayer(config, in_dim)
        self.output_layer = nn.Linear(in_dim, config.vocab_size)
        
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        x: Current latent state [B, D]
        context: Context hierarchical vector [B, D]
        """
        combined = torch.cat([x, context], dim=-1)
        moe_out = self.moe(combined)
        logits = self.output_layer(moe_out)
        return logits
