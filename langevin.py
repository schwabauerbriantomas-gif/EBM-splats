import torch
import numpy as np

from config import EBMConfig
from geometry import exp_map, project_to_tangent
from energy import EnergyFunction

def langevin_step(x: torch.Tensor, v: torch.Tensor, energy_fn: EnergyFunction, config: EBMConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Executes a single step of the underdamped Langevin dynamics sampling loop.
    Uses the Störmer-Verlet symplectic integrator format.
    """
    gamma = config.langevin_gamma
    T = config.langevin_T
    dt = config.langevin_dt
    
    # Half-step momentum
    # Note: compute_score returns -grad_E.
    score = energy_fn.compute_score(x) 
    grad_e = -score  # Recover the gradient
    
    v = v - 0.5 * dt * (gamma * v + grad_e)
    
    # Full-step position (geodesic map exponentiation)
    x = exp_map(x, dt * v)
    
    # Re-evaluate gradient at new position for second half-step momentum
    score_new = energy_fn.compute_score(x)
    grad_e_new = -score_new
    v = v - 0.5 * dt * (gamma * v + grad_e_new)
    
    # White noise injection
    noise = torch.randn_like(v) * np.sqrt(2 * gamma * T * dt)
    v = v + noise
    
    # Constrain velocity back to the tangent space of the new position x
    v = project_to_tangent(x, v)
    
    return x, v

def sample_langevin(x_init: torch.Tensor, energy_fn: EnergyFunction, config: EBMConfig) -> torch.Tensor:
    """
    Runs the full Langevin dynamic loop from x_init.
    Returns the traversed, minimal-energy state.
    """
    x = x_init.clone()
    v = torch.zeros_like(x)
    
    for _ in range(config.langevin_steps):
        x, v = langevin_step(x, v, energy_fn, config)
        
    return x
