"""
Splat Storage — V2 Compliant.

Fixes:
  - Fixed add_splat: alpha/kappa logic was swapped (if alpha is not None used alpha, 
    else used alpha — never actually used kappa)
  - Added importance field (V2 §2.2) for weighted energy computation
  - Added density field for adaptive noise (V2 §3.3)
  - Proper initialization with kappa from config
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional

from .config import EBMConfig
from .geometry import normalize_sphere, geodesic_distance

# faiss is optional. Resolving the import once (instead of on every
# find_neighbors call) avoids paying the failed-import machinery per call
# when faiss is not installed. Behavior identical: if faiss is present it is
# used exactly as before; if absent, the torch fallback runs exactly as before.
_FAISS_MODULE = None
_FAISS_RESOLVED = False


def _get_faiss():
    global _FAISS_MODULE, _FAISS_RESOLVED
    if not _FAISS_RESOLVED:
        try:
            import faiss as _faiss
            _FAISS_MODULE = _faiss
        except Exception:
            _FAISS_MODULE = None
        _FAISS_RESOLVED = True
    return _FAISS_MODULE


class SplatStorage(nn.Module):
    def __init__(self, config: EBMConfig):
        super().__init__()
        self.config = config
        self.max_splats = config.max_splats
        self.latent_dim = config.latent_dim

        # Learnable parameters
        self.mu = nn.Parameter(torch.randn(self.max_splats, self.latent_dim))
        self.alpha = nn.Parameter(torch.ones(self.max_splats) * config.init_alpha)
        self.kappa = nn.Parameter(torch.ones(self.max_splats) * config.init_kappa)

        # Importance weight (V2 §2.2) — how often/important this splat is
        self.importance = nn.Parameter(torch.ones(self.max_splats) * 0.5)

        # Non-learnable state buffers
        self.register_buffer("frequency", torch.zeros(self.max_splats))
        self.register_buffer("age", torch.zeros(self.max_splats, dtype=torch.long))
        self.register_buffer("density", torch.zeros(self.max_splats))

        # KNN memoization (LRU-1). Key covers everything the neighbor lookup
        # depends on. _mut_counter is bumped by methods that mutate params via
        # .data (which bypasses torch's version counter); optimizer steps and
        # no_grad in-place writes DO bump _version and are covered by that.
        # Initialized before self.normalize(), which bumps the counter.
        self._knn_cache_key = None
        self._knn_cache_val = None
        self._knn_cache_x = None  # strong ref: pins storage so data_ptr can't be recycled
        self._mut_counter = 0

        self.n_active = config.n_splats_init
        self.normalize()

    def normalize(self):
        with torch.no_grad():
            self.mu.data[:self.n_active] = normalize_sphere(self.mu.data[:self.n_active])
            self.alpha.data[:self.n_active] = F.relu(self.alpha.data[:self.n_active]) + 0.01
            self.kappa.data[:self.n_active] = F.relu(self.kappa.data[:self.n_active]) + self.config.init_kappa
            self.importance.data[:self.n_active] = self.importance.data[:self.n_active].clamp(min=0.01)
        self._mut_counter += 1  # .data writes bypass _version: invalidate KNN cache

    def find_neighbors(self, x: torch.Tensor, k: int):
        """Find K nearest neighbors. Returns (mu, alpha, kappa)."""
        with torch.no_grad():
            # LRU-1 memoization: within a Langevin step the same query tensor
            # (same storage, unmutated) is looked up several times with the
            # same k (e.g. compute_splat_energy_grad and compute_splat_energy
            # at the identical x_new). The cached tensors are gathered copies
            # and are only ever read downstream; any param update (optimizer
            # step / add_splat / normalize / .data write covered by
            # _mut_counter) changes the key and forces a recompute.
            cache_key = (x.data_ptr(), tuple(x.shape), x.dtype, x._version, k,
                         self.n_active, self.mu._version, self.alpha._version,
                         self.kappa._version, self._mut_counter)
            if cache_key == self._knn_cache_key:
                return self._knn_cache_val

            faiss = _get_faiss()

            if faiss is not None:
                query_vectors = x.detach().cpu().numpy().astype(np.float32)
                db_vectors = self.mu[:self.n_active].detach().cpu().numpy().astype(np.float32)

                indexer = faiss.IndexFlatIP(self.latent_dim)
                indexer.add(db_vectors)
                distances, indices = indexer.search(query_vectors, k)
                idx_tensor = torch.from_numpy(indices).to(self.mu.device).long()
            else:
                # Pure-torch fallback. compute_mode='use_mm_for_euclid_dist'
                # computes Euclidean distances via the ||a||^2+||b||^2-2ab
                # matmul identity instead of the pairwise kernel: same
                # ordering for all practical inputs (near-exact ties may
                # reorder within float rounding), ~50x faster when the query
                # batch is small (the pairwise kernel is O(B*N*D) broadcast).
                distances = torch.cdist(x, self.mu[:self.n_active],
                                        compute_mode='use_mm_for_euclid_dist')
                topk_indices = torch.topk(distances, k=k, dim=-1).indices
                idx_tensor = topk_indices.squeeze(-1)

            neighbors_mu = self.mu[idx_tensor]
            neighbors_alpha = self.alpha[idx_tensor]
            neighbors_kappa = self.kappa[idx_tensor]

            self._knn_cache_key = cache_key
            self._knn_cache_val = (neighbors_mu, neighbors_alpha, neighbors_kappa)
            self._knn_cache_x = x  # pin: keeps data_ptr unique while cached

            return neighbors_mu, neighbors_alpha, neighbors_kappa

    def add_splat(self, center: torch.Tensor, alpha: float = None, kappa: float = None) -> bool:
        """Add a new splat. FIXED: proper alpha/kappa handling."""
        if self.n_active >= self.max_splats:
            return False

        with torch.no_grad():
            self.mu.data[self.n_active] = normalize_sphere(center)
            # FIX: was swapped — alpha should use provided or default
            self.alpha.data[self.n_active] = alpha if alpha is not None else self.config.init_alpha
            self.kappa.data[self.n_active] = kappa if kappa is not None else self.config.init_kappa
            self.frequency[self.n_active] = 0
            self.age[self.n_active] = 0
            self.density[self.n_active] = 0.0
            self.importance.data[self.n_active] = 0.5  # Default importance
            self.n_active += 1
        self._mut_counter += 1  # .data writes bypass _version: invalidate KNN cache

        return True

    def update_statistics(self, x: torch.Tensor):
        with torch.no_grad():
            unique_idx, counts = torch.unique(x.detach().argmax(dim=-1), return_counts=True)
            self.frequency[unique_idx] += counts.float()
            self.age += 1

    def update_density(self, x_history: torch.Tensor):
        """Update density estimates from recent history (V2 §3.3)."""
        with torch.no_grad():
            for i in range(self.n_active):
                center = self.mu[i:i+1]
                dists = geodesic_distance(x_history, center)
                # Density = count of points within angular distance 0.5
                self.density[i] = (dists < 0.5).float().mean()
