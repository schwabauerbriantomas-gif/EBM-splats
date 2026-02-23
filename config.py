from dataclasses import dataclass
from typing import Optional
from typing import List
from typing import Dict
from typing import Any

@dataclass
class EBMConfig:
    # Environment
    device: str = "cpu"
    
    # Latent space
    latent_dim: int = 640
    n_splats_init: int = 10000
    max_splats: int = 100000
    knn_k: int = 64
    
    # FASE 1 MEJORAS: Inicialización Inteligente de Splats
    vocab_embedding_path: Optional[str] = None  # Path to GPT-2 pretrained embeddings
    init_from_vocab_embeddings: bool = False  # Initialize splats from embeddings instead of random
    init_alpha: float = 1.0  # Initial splat weight
    init_kappa: float = 10.0  # Initial splat concentration
    
    # Splat regularization
    splat_temperature: float = 0.1  # Temperature for splat energy exploration
    splat_weight_decay: float = 0.0  # Weight decay per epoch for splats
    splat_weight_decay_start: float = 1.0  # Starting weight decay
    min_kappa: float = 1.0  # Minimum kappa to avoid collapse
    max_kappa: float = 50.0  # Maximum kappa limit
    
    # Training monitoring
    enable_curriculum_learning: bool = True  # Enable curriculum learning (gradual n_splats increase)
    curriculum_epochs: int = 5  # Phases for curriculum (init=5, mid=10, max=15k)
    curriculum_target_splats: int = 50000  # Target splats for end of curriculum
    splat_convergence_threshold: float = 0.95  # Convergence threshold (mean similarity between epochs)
    temperature: float = 0.1
    
    # Langevin Dynamics
    langevin_steps: int = 200
    langevin_dt: float = 0.001
    langevin_gamma: float = 0.1
    langevin_T: float = 1.0
    
    # Self-Organized Criticality (SOC)
    soc_threshold: float = 0.8
    soc_check_interval: int = 100
    min_splat_distance: float = 0.1
    init_alpha: float = 1.0
    init_kappa: float = 10.0
    
    # Hierarchical Context
    context_local: int = 12
    context_medium: int = 64
    context_global: int = 512
    
    # MoE Decoder
    vocab_size: int = 50000
    moe_experts: int = 4
    moe_active: int = 2
    hidden_dim: int = 1024
    
    # Training
    noise_levels: tuple = (0.01, 0.05, 0.1, 0.2, 0.5)
    learning_rate: float = 1e-3
    reg_weight: float = 0.01
    grad_clip: float = 1.0
