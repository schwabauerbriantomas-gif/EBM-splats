"""
Pareto Filter — Non-dominated sorting for multi-objective embedding optimization.

Takes candidate embeddings from Langevin exploration and filters them
using Pareto frontier analysis across multiple quality objectives.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional


def dominates(obj_a: np.ndarray, obj_b: np.ndarray) -> bool:
    """Check if solution a dominates solution b (all objectives minimized)."""
    return np.all(obj_a <= obj_b) and np.any(obj_a < obj_b)


def fast_non_dominated_sort(objectives: np.ndarray) -> List[List[int]]:
    """
    Fast non-dominated sorting (NSGA-II style).
    
    Args:
        objectives: [N, M] array of M objectives for N solutions
    Returns:
        List of fronts (each front is a list of indices)

    The pairwise domination relation is computed with vectorized broadcasting
    (identical boolean semantics to the previous per-pair `dominates` calls:
    all(a <= b) and any(a < b)). Front construction keeps the exact iteration
    order of the original loops, so the returned fronts (membership AND
    ordering) are identical.
    """
    objectives = np.asarray(objectives)
    n = len(objectives)

    # dom[i, j] = i dominates j, computed in row blocks to bound memory.
    dominated_set = [[] for _ in range(n)]
    domination_count = np.zeros(n, dtype=int)
    block = max(1, min(n, 2 ** 22 // max(1, n * max(1, objectives.shape[1]))))
    for start in range(0, n, block):
        stop = min(start + block, n)
        a = objectives[start:stop]  # [b, M]
        le = (a[:, None, :] <= objectives[None, :, :]).all(axis=2)  # [b, n]
        lt = (a[:, None, :] < objectives[None, :, :]).any(axis=2)   # [b, n]
        dom = le & lt
        domination_count += dom.sum(axis=0)
        for offset in range(stop - start):
            dominated_set[start + offset] = np.flatnonzero(dom[offset]).tolist()

    # First front: solutions with domination_count == 0
    fronts = [[]]
    for i in range(n):
        if domination_count[i] == 0:
            fronts[0].append(i)
    
    # Subsequent fronts
    k = 0
    while fronts[k]:
        next_front = []
        for i in fronts[k]:
            for j in dominated_set[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        k += 1
        fronts.append(next_front)
    
    return fronts[:-1]  # Remove last empty front


def crowding_distance(objectives: np.ndarray, front: List[int]) -> np.ndarray:
    """
    Compute crowding distance for solutions within a front.
    Higher distance = more diverse = preferred.
    """
    if len(front) <= 2:
        return np.full(len(front), np.inf)
    
    n_obj = objectives.shape[1]
    distances = np.zeros(len(front))
    front_objectives = objectives[front]
    
    for m in range(n_obj):
        sorted_idx = np.argsort(front_objectives[:, m])
        distances[sorted_idx[0]] = np.inf
        distances[sorted_idx[-1]] = np.inf
        
        obj_range = front_objectives[sorted_idx[-1], m] - front_objectives[sorted_idx[0], m]
        if obj_range > 1e-10:
            for i in range(1, len(front) - 1):
                distances[sorted_idx[i]] += (
                    front_objectives[sorted_idx[i + 1], m] - front_objectives[sorted_idx[i - 1], m]
                ) / obj_range
    
    return distances


class ParetoFilter:
    """
    Multi-objective Pareto filter for embedding quality.
    
    Objectives (all minimized):
    1. Alignment loss (contrastive cross-modal distance)
    2. Uniformity loss (how well spread on the hypersphere)
    3. Reconstruction loss (can we recover input from embedding?)
    4. Energy (EBM energy — lower = more "natural" embedding)
    """
    
    def __init__(self, n_keep: int = 512, n_objectives: int = 4):
        self.n_keep = n_keep
        self.n_objectives = n_objectives
    
    def compute_objectives(
        self,
        embeddings: torch.Tensor,  # [N, D] on hypersphere
        alignment_scores: Optional[torch.Tensor] = None,  # [N] lower = better
        energy_scores: Optional[torch.Tensor] = None,  # [N] EBM energy
    ) -> np.ndarray:
        """Compute all objectives for each embedding."""
        n = embeddings.shape[0]
        objectives = np.zeros((n, self.n_objectives))
        
        # Obj 0: Alignment loss (if provided, else approximate via energy)
        if alignment_scores is not None:
            objectives[:, 0] = alignment_scores.detach().cpu().numpy()
        else:
            objectives[:, 0] = 0.0  # Placeholder
        
        # Obj 1: Uniformity loss (how clustered are the embeddings?)
        # Ideal: uniform distribution on hypersphere (maximize spread = minimize negative spread)
        with torch.no_grad():
            # Pairwise cosine similarity
            sim_matrix = embeddings @ embeddings.T  # [N, N]
            # Uniformity = log of mean pairwise Gaussian potential
            # Lower = more uniform (better), so we minimize negative uniformity
            uniformity = sim_matrix.pow(2).mean().log().item()
            objectives[:, 1] = abs(uniformity)  # All embeddings get same score
        
        # Obj 2: Reconstruction loss (placeholder — needs decoder)
        objectives[:, 2] = 0.0
        
        # Obj 3: EBM energy
        if energy_scores is not None:
            objectives[:, 3] = energy_scores.detach().cpu().numpy()
        else:
            objectives[:, 3] = 0.0
        
        return objectives
    
    def filter(
        self,
        embeddings: torch.Tensor,
        alignment_scores: Optional[torch.Tensor] = None,
        energy_scores: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, np.ndarray, List[List[int]]]:
        """
        Filter embeddings using Pareto front selection.
        
        Returns:
            selected_embeddings: [n_keep, D]
            selected_objectives: [n_keep, M]
            fronts: All Pareto fronts (for analysis)
        """
        objectives = self.compute_objectives(embeddings, alignment_scores, energy_scores)
        fronts = fast_non_dominated_sort(objectives)
        
        # Select from fronts until we have n_keep
        selected_indices = []
        for front in fronts:
            if len(selected_indices) + len(front) <= self.n_keep:
                selected_indices.extend(front)
            else:
                # Need partial front — use crowding distance
                remaining = self.n_keep - len(selected_indices)
                distances = crowding_distance(objectives, front)
                top_in_front = np.argsort(-distances)[:remaining]  # Highest distance first
                selected_indices.extend([front[i] for i in top_in_front])
                break
        
        selected_idx = np.array(selected_indices[:self.n_keep])
        
        return (
            embeddings[selected_idx],
            objectives[selected_idx],
            fronts,
        )
    
    def get_pareto_front(self, embeddings: torch.Tensor, objectives: np.ndarray) -> Dict:
        """Get analysis of the Pareto front for logging."""
        fronts = fast_non_dominated_sort(objectives)
        
        return {
            "n_fronts": len(fronts),
            "pareto_front_size": len(fronts[0]) if fronts else 0,
            "total_candidates": len(embeddings),
            "front_sizes": [len(f) for f in fronts[:5]],
            "pareto_mean_objectives": objectives[fronts[0]].mean(axis=0).tolist() if fronts else [],
        }
