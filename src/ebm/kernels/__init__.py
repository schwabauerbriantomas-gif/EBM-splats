"""
Optimized GPU kernels for EBM-splats.

Each module wraps a reference implementation from ``ebm.*`` with a
GPU-optimized version that preserves the reference semantics. Every
entry point keeps a CPU / eager fallback so results are always available
without CUDA or Triton. Imports are lazy (module __getattr__) so that
``import ebm.kernels`` stays cheap and parallel additions don't conflict.

Modules:

    knn       - find_neighbors_triton: Triton d2 GEMM + top-k, conservative
                auto-routing (falls back outside its winning regime)
    energy    - FusedEnergyFunction: drop-in EnergyFunction subclass fusing
                splat-energy / score into a single Triton kernel
    langevin  - sample_langevin_fused: whole 100-step sampler captured as a
                CUDA graph (launch-overhead elimination), seeded determinism
"""

__all__ = [
    "find_neighbors_triton",
    "FusedEnergyFunction",
    "kernel_energy_and_score",
    "kernel_score",
    "kernel_splat_energy",
    "kernel_splat_energy_grad",
    "make_fused_energy_fn",
    "langevin_step_fused",
    "sample_langevin_fused",
]


def __getattr__(name):
    if name == "find_neighbors_triton":
        from .knn import find_neighbors_triton

        return find_neighbors_triton
    if name in ("FusedEnergyFunction", "kernel_energy_and_score", "kernel_score",
                "kernel_splat_energy", "kernel_splat_energy_grad",
                "make_fused_energy_fn"):
        from . import energy

        return getattr(energy, name)
    if name in ("langevin_step_fused", "sample_langevin_fused"):
        from .langevin import langevin_step_fused, sample_langevin_fused

        return sample_langevin_fused if name == "sample_langevin_fused" \
            else langevin_step_fused
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
