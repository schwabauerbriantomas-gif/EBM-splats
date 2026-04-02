"""
EBM Configuration — V2 Specification Compliant.

Fixes applied:
  - Removed duplicate init_alpha field
  - Added V2 spec parameters (lambda context weights, adaptive noise, etc.)
  - Explicit sign convention documentation
  - Proper default values from V2 Especificacion Tecnica
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class EBMConfig:
    """
    Configuration for the EBM continuous language model.
    
    Sign Convention (V2 Spec §2.1):
        p(x) = exp(-E(x)) / Z
        score(x) = ∇_x log p(x) = -∇_x E(x)
        Low energy = high probability (near splat centers)
    """
    
    # ── Environment ──
    device: str = "cpu"
    
    # ── Latent Space ──
    latent_dim: int = 640
    n_splats_init: int = 10000
    max_splats: int = 100000
    knn_k: int = 64
    
    # ── Splat Initialization (Phase 1 improvements) ──
    vocab_embedding_path: Optional[str] = None
    init_from_vocab_embeddings: bool = False
    init_alpha: float = 1.0
    init_kappa: float = 10.0
    
    # ── Splat Regularization ──
    splat_temperature: float = 0.1
    splat_weight_decay: float = 0.0
    splat_weight_decay_start: float = 1.0
    min_kappa: float = 1.0
    max_kappa: float = 50.0
    
    # ── Training ──
    learning_rate: float = 1e-3
    reg_weight: float = 0.01
    grad_clip: float = 1.0
    temperature: float = 0.1
    
    # ── Curriculum Learning ──
    enable_curriculum_learning: bool = True
    curriculum_epochs: int = 5
    curriculum_target_splats: int = 50000
    splat_convergence_threshold: float = 0.95
    
    # ── Noise Schedule (V2 §3.3 Adaptive Noise) ──
    noise_levels: Tuple[float, ...] = (0.01, 0.05, 0.1, 0.2, 0.5)
    sigma_base: float = 0.1
    sigma_scale: float = 1.0
    sigma_rho_ref: float = 1.0
    
    # ── Langevin Dynamics ──
    langevin_steps: int = 200
    langevin_dt: float = 0.001
    langevin_gamma: float = 0.1
    langevin_T: float = 1.0
    
    # ── SOC (Self-Organized Criticality) ──
    soc_threshold: float = 0.8
    soc_check_interval: int = 100
    min_splat_distance: float = 0.1
    
    # ── Hierarchical Context (V2 §4) ──
    context_local_window: int = 16       # 8-16 tokens
    context_medium_window: int = 128     # 64-128 tokens
    context_global_window: int = 512     # 512+ tokens
    beta_local: float = 0.5              # Fast adaptation
    beta_medium: float = 0.8             # Moderate
    beta_global: float = 0.95            # Slow, stable
    lambda_context_local: float = 1.0
    lambda_context_medium: float = 0.5
    lambda_context_global: float = 0.2
    
    # ── Decoder (V2 §5) ──
    vocab_size: int = 50257  # GPT-2 vocab
    moe_experts: int = 4
    moe_active: int = 2
    hidden_dim: int = 1024
    
    # ── Collapse Regularization (V2 §2.4) ──
    lambda_reg: float = 0.01
    theta_threshold: float = 0.9  # Min angular separation between splat centers
    
    # ── Energy Weights ──
    lambda_geom: float = 0.01
    lambda_comp: float = 0.05
    
    # ── EMA ──
    ema_decay: float = 0.999
    
    # ── Error Recovery (V2 §7) ──
    energy_stagnation_window: int = 5
    energy_stagnation_epsilon: float = 1e-4
    gradient_magnitude_threshold: float = 1e-5
    max_recovery_attempts: int = 3
