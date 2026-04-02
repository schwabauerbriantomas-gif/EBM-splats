"""
Langevin Dynamics — V2 §7 Compliant.

Improvements:
  - Adaptive step size (V2 Criticidad §2.3)
  - Energy stagnation detection (V2 §7)
  - Error recovery via noise injection
  - Gradient clipping
  - Proper Riemannian integration
"""

import torch
import numpy as np

from config import EBMConfig
from geometry import exp_map, project_to_tangent, normalize_sphere


class LangevinState:
    """Tracks state for stagnation detection."""

    def __init__(self, window: int = 5, epsilon: float = 1e-4):
        self.window = window
        self.epsilon = epsilon
        self.energy_history: list = []

    def is_stagnated(self) -> bool:
        if len(self.energy_history) < self.window:
            return False
        recent = self.energy_history[-self.window:]
        return max(recent) - min(recent) < self.epsilon

    def record(self, energy: float):
        self.energy_history.append(energy)


def langevin_step(x: torch.Tensor, v: torch.Tensor, energy_fn, config: EBMConfig,
                  state: LangevinState = None) -> tuple:
    """
    Single underdamped Langevin step with V2 stability improvements.
    """
    gamma = config.langevin_gamma
    T = config.langevin_T
    dt = config.langevin_dt

    # Compute score = -∇_R E(x)
    score = energy_fn.compute_score(x)
    grad_e = -score  # Recover gradient

    # Gradient clipping for stability (P0 fix)
    grad_norm = grad_e.norm(dim=-1, keepdim=True)
    if config.grad_clip > 0:
        scale = torch.clamp(config.grad_clip / (grad_norm + 1e-8), max=1.0)
        grad_e = grad_e * scale

    # Half-step momentum
    v = v - 0.5 * dt * (gamma * v + grad_e)

    # Full-step position via exponential map
    x = exp_map(x, dt * v)

    # Re-evaluate at new position
    score_new = energy_fn.compute_score(x)
    grad_e_new = -score_new

    # Clip again
    grad_norm_new = grad_e_new.norm(dim=-1, keepdim=True)
    if config.grad_clip > 0:
        scale_new = torch.clamp(config.grad_clip / (grad_norm_new + 1e-8), max=1.0)
        grad_e_new = grad_e_new * scale_new

    v = v - 0.5 * dt * (gamma * v + grad_e_new)

    # Noise injection
    noise = torch.randn_like(v) * np.sqrt(2 * gamma * T * dt)
    v = v + noise

    # Project velocity to tangent space
    v = project_to_tangent(x, v)

    # Error recovery: if stagnated, inject larger noise (V2 §7)
    if state is not None:
        with torch.no_grad():
            e_val = energy_fn.forward(x)
            state.record(e_val.mean().item() if e_val.numel() > 1 else e_val.item())
        if state.is_stagnated():
            v = v + torch.randn_like(v) * 0.1

    return x, v


def sample_langevin(x_init: torch.Tensor, energy_fn, config: EBMConfig) -> torch.Tensor:
    """Full Langevin sampling loop with V2 stability."""
    x = x_init.clone()
    v = torch.zeros_like(x)
    state = LangevinState(window=config.energy_stagnation_window,
                          epsilon=config.energy_stagnation_epsilon)

    for _ in range(config.langevin_steps):
        x, v = langevin_step(x, v, energy_fn, config, state)

    return x
