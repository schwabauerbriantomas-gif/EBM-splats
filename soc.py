import torch
from config import EBMConfig
from splats import SplatStorage
from geometry import geodesic_distance

class HistoryBuffer:
    def __init__(self, capacity: int = 10000, latent_dim: int = 640):
        self.capacity = capacity
        # Stored explicitly onto cpu memory for CPU-only validation stages
        self.states = torch.zeros(capacity, latent_dim)
        self.energies = torch.zeros(capacity)
        self.ptr = 0
        self.full = False
        
    def push(self, x: torch.Tensor, energy: torch.Tensor):
        self.states[self.ptr] = x.detach().cpu()
        self.energies[self.ptr] = energy.detach().cpu()
        self.ptr = (self.ptr + 1) % self.capacity
        self.full = self.full or self.ptr == 0

def compute_order_parameter(splats: SplatStorage, x_history: torch.Tensor) -> float:
    """
    Computes tension over recent history phi = (1/K) sum(alpha * rho / rho_avg)
    using approximation rules per PDF.
    """
    with torch.no_grad():
        if splats.frequency.sum() == 0:
            return 0.0

        # Mean frequency (rho_avg)
        active_counts = splats.frequency[:splats.n_active]
        rho_avg = active_counts.mean().clamp(min=1.0)
        
        # phi summation simplified
        phi = (splats.alpha[:splats.n_active] * active_counts / rho_avg).mean().item()
        return phi

def maybe_consolidate(splats: SplatStorage, config: EBMConfig, buffer: HistoryBuffer) -> bool:
    """SOC evaluation and expansion if necessary."""
    if not buffer.full and buffer.ptr < 100:
        return False
        
    # Get recent chunk
    hist_len = buffer.capacity if buffer.full else buffer.ptr
    recent_x = buffer.states[:hist_len]
    recent_e = buffer.energies[:hist_len]
    
    phi = compute_order_parameter(splats, recent_x)
    
    if phi > config.soc_threshold:
        # High uncertainty region based on maximum energy
        high_energy_idx = torch.argmax(recent_e).item()
        new_center = recent_x[high_energy_idx].to(splats.mu.device)
        
        # Guard against duplicative splats (dist check against nearest mean)
        with torch.no_grad():
            dists = geodesic_distance(new_center, splats.mu[:splats.n_active])
            min_dist = dists.min().item()
            
        if min_dist > config.min_splat_distance:
            # Reached criticality -> form new attractive stable Gaussian
            return splats.add_splat(new_center)
            
    return False
