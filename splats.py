import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Dict, Any

from config import EBMConfig
from energy import EnergyFunction
from geometry import normalize_sphere, geodesic_distance
from decoder import EBMDecoder


class SplatStorage(nn.Module):
    def __init__(self, config: EBMConfig):
        super().__init__()
        self.config = config
        self.max_splats = config.max_splats
        self.latent_dim = config.latent_dim
        
        # Learnable parameters with proper initialization
        self.mu = nn.Parameter(torch.randn(self.max_splats, self.latent_dim))
        self.alpha = nn.Parameter(torch.ones(self.max_splats))
        self.kappa = nn.Parameter(torch.ones(self.max_splats) * config.init_kappa)
        
        # Persistent buffers for state tracking
        self.register_buffer("frequency", torch.zeros(self.max_splats))
        self.register_buffer("age", torch.zeros(self.max_splats, dtype=torch.long))
        
        self.n_active = config.n_splats_init
        self.normalize()
    
    def normalize(self):
        with torch.no_grad():
            self.mu.data[:self.n_active] = normalize_sphere(self.mu.data[:self.n_active])
            # Epsilon for numerical stability
            self.alpha.data[:self.n_active] = F.relu(self.alpha.data[:self.n_active]) + 0.01
            self.kappa.data[:self.n_active] = F.relu(self.kappa.data[:self.n_active]) + self.config.init_kappa
        
    def find_neighbors(self, x: torch.Tensor, k: int):
        with torch.no_grad():
            query_vectors = x.detach().cpu().numpy().astype(np.float32)
            db_vectors = self.mu[:self.n_active].detach().cpu().numpy().astype(np.float32)
            
            # Build FAISS inner product index
            try:
                import faiss
                indexer = faiss.IndexFlatIP(self.latent_dim)
                indexer.add(db_vectors)
                
                distances, indices = indexer.search(query_vectors, k)
                
                idx_tensor = torch.from_numpy(indices).to(self.mu.device).long()
                
                neighbors_mu = self.mu[idx_tensor]
                neighbors_alpha = self.alpha[idx_tensor]
                neighbors_kappa = self.kappa[idx_tensor]
                
                return neighbors_mu, neighbors_alpha, neighbors_kappa
            except Exception:
                # Fallback to simple distance calculation if FAISS fails
                distances = torch.cdist(x, self.mu[:self.n_active])
                topk_indices = torch.topk(distances, k=k, dim=-1).indices
                idx_tensor = topk_indices.squeeze(-1)
                
                neighbors_mu = self.mu[idx_tensor]
                neighbors_alpha = self.alpha[idx_tensor]
                neighbors_kappa = self.kappa[idx_tensor]
                
                return neighbors_mu, neighbors_alpha, neighbors_kappa
    
    def add_splat(self, center: torch.Tensor, alpha: float = None, kappa: float = None):
        if self.n_active >= self.max_splats:
            return False
        
        with torch.no_grad():
            self.mu.data[self.n_active] = normalize_sphere(center)
            if alpha is not None:
                self.alpha.data[self.n_active] = self.config.init_alpha
            else:
                self.alpha.data[self.n_active] = alpha
            if kappa is not None:
                self.kappa.data[self.n_active] = self.config.init_kappa
            else:
                self.kappa.data[self.n_active] = self.config.init_kappa
            self.frequency[self.n_active] = 0
            self.age[self.n_active] = 0
            self.n_active += 1
            
        return True
    
    def update_statistics(self, x: torch.Tensor):
        with torch.no_grad():
            unique_idx, counts = torch.unique(x.detach().argmax(dim=-1), return_counts=True)
            self.frequency[unique_idx] += counts.float()
            self.age += 1
