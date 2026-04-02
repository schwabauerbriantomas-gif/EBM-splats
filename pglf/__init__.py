"""
PGLF — Pareto-Guided Langevin Flow Embedding Model.

Omnimodal embedding system built on EBM-splats with three-phase training:
1. Langevin exploration (energy landscape mapping)
2. Pareto filtering (multi-objective optimization)
3. Flow Matching (deterministic distillation)
"""

from .encoders import (
    TextEncoder,
    ImageEncoder,
    AudioEncoder,
    OmnimodalEncoder,
)
from .flow_matching import HypersphereFlowMatching, FlowMatchingLoss
from .contrastive_head import HypersphereContrastiveLoss, UniformityAlignmentLoss
from .pareto_filter import ParetoFilter
from .trainer import PGLFTrainer

__version__ = "0.1.0"
__all__ = [
    "TextEncoder",
    "ImageEncoder",
    "AudioEncoder",
    "OmnimodalEncoder",
    "HypersphereFlowMatching",
    "FlowMatchingLoss",
    "HypersphereContrastiveLoss",
    "UniformityAlignmentLoss",
    "ParetoFilter",
    "PGLFTrainer",
]
