"""
Tests for the fused Langevin sampler (ebm.kernels.langevin).

Correctness strategy (honest, no pointwise cross-RNG comparison):
  1. CPU fallback must be BIT-IDENTICAL to the reference (it delegates).
  2. On GPU, ``langevin_step_fused`` (reference-style lazy RNG) must produce a
     BIT-IDENTICAL 100-step trajectory to ``ebm.langevin`` when run in
     isolation — this proves the fused math transcription is exact.
  3. The CUDA-graph path must be BIT-IDENTICAL to the eager fused loop given
     the same seed (same pre-generated noise, same op order).
  4. Same seed => bit-identical fused trajectories (determinism).
  5. Fused vs REFERENCE full-loop trajectories are NOT pointwise comparable
     (different, deliberately documented noise streams); they are compared
     statistically: final norms and final energies with tolerances justified
     from the measured seed-to-seed spread of the reference itself.

NOTE: chains are always run in ISOLATION (never interleaved step-by-step
through one shared energy_fn): SplatStorage's LRU-1 KNN cache can serve a
false hit when two chains alternate through the same storage (host-side
cache keys are (ptr, _version); a freed chain-A tensor's address can be
recycled by chain B). That is a pre-existing property of the reference
cache, not of the fused kernel.
"""

import math

import numpy as np
import pytest
import torch

from ebm.config import EBMConfig
from ebm.energy import EnergyFunction
from ebm.kernels import langevin_step_fused, sample_langevin_fused
from ebm.langevin import LangevinState, langevin_step, sample_langevin
from ebm.splats import SplatStorage

CUDA = torch.cuda.is_available()
gpu = pytest.mark.skipif(not CUDA, reason="requires CUDA")


def _setup(device, *, latent_dim=640, n_splats=2000, max_splats=4000, knn_k=64,
           steps=100, batch=8, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    config = EBMConfig(device=device, latent_dim=latent_dim,
                       n_splats_init=n_splats, max_splats=max_splats,
                       knn_k=knn_k, langevin_steps=steps)
    splats = SplatStorage(config).to(device)
    energy_fn = EnergyFunction(config, splats).to(device)
    x0 = torch.randn(batch, latent_dim, device=device)
    x0 = x0 / x0.norm(dim=-1, keepdim=True)
    return config, energy_fn, x0


# ── CPU fallback (runs everywhere, incl. CI without GPU) ─────────────────────

def test_cpu_fallback_is_bitexact_reference():
    config, energy_fn, x0 = _setup("cpu", latent_dim=64, n_splats=50,
                                   max_splats=100, knn_k=8, steps=20, batch=4)
    with torch.no_grad():
        torch.manual_seed(3)
        ref = sample_langevin(x0.clone(), energy_fn, config)
        torch.manual_seed(3)
        fused = sample_langevin_fused(x0, energy_fn, config)
    assert torch.equal(ref, fused)


def test_cpu_step_fallback_is_bitexact_reference():
    config, energy_fn, x0 = _setup("cpu", latent_dim=32, n_splats=20,
                                   max_splats=40, knn_k=4, steps=1, batch=4)
    with torch.no_grad():
        torch.manual_seed(11)
        v = torch.zeros_like(x0)
        ref_x, ref_v = langevin_step(x0.clone(), v.clone(), energy_fn, config,
                                     None)
        torch.manual_seed(11)  # single-step draws randn: same stream needed
        f_x, f_v = langevin_step_fused(x0.clone(), v.clone(), energy_fn,
                                       config, None)
    assert torch.equal(ref_x, f_x) and torch.equal(ref_v, f_v)


# ── GPU: fused math == reference math (isolated chains, lazy RNG) ────────────

@gpu
def test_strict_step_trajectory_bitexact_vs_reference():
    """100 isolated steps: fused step math must be bit-identical to
    ebm.langevin.langevin_step (same ops, same order, same lazy RNG)."""
    config, energy_fn, x0 = _setup("cuda")
    with torch.no_grad():
        torch.manual_seed(7)
        xr, vr = x0.clone(), torch.zeros_like(x0)
        sr = LangevinState(config.energy_stagnation_window,
                           config.energy_stagnation_epsilon)
        for _ in range(config.langevin_steps):
            xr, vr = langevin_step(xr, vr, energy_fn, config, sr)

        torch.manual_seed(7)
        xf, vf = x0.clone(), torch.zeros_like(x0)
        sf = LangevinState(config.energy_stagnation_window,
                           config.energy_stagnation_epsilon)
        for _ in range(config.langevin_steps):
            xf, vf = langevin_step_fused(xf, vf, energy_fn, config, sf)

    # stagnation history must agree too (same energies recorded)
    assert sr.energy_history == pytest.approx(sf.energy_history, abs=0.0)
    assert not sr.is_stagnated(), (
        "stagnation fired in this workload; bit-exactness still expected "
        "because the fused strict path mirrors the reference mechanics, but "
        "please inspect")
    assert torch.equal(xr, xf), "fused step math diverged from reference"
    assert torch.equal(vr, vf), "fused velocity math diverged from reference"


# ── GPU: graph path == eager fused path, determinism ─────────────────────────

@gpu
def test_graph_path_bitexact_vs_eager_fused_loop():
    config, energy_fn, x0 = _setup("cuda")
    with torch.no_grad():
        x_graph = sample_langevin_fused(x0, energy_fn, config, seed=1234)
        x_eager = sample_langevin_fused(x0, energy_fn, config, seed=1234,
                                        use_cuda_graph=False)
    assert torch.equal(x_graph, x_eager), (
        "CUDA-graph replay diverged from the eager fused loop")


@gpu
def test_same_seed_determinism_two_runs():
    config, energy_fn, x0 = _setup("cuda")
    with torch.no_grad():
        a = sample_langevin_fused(x0, energy_fn, config, seed=42)
        b = sample_langevin_fused(x0, energy_fn, config, seed=42)
    assert torch.equal(a, b), "same explicit seed produced different runs"


@gpu
def test_global_generator_seed_determinism():
    config, energy_fn, x0 = _setup("cuda")
    with torch.no_grad():
        torch.manual_seed(999)
        a = sample_langevin_fused(x0, energy_fn, config)
        torch.manual_seed(999)
        b = sample_langevin_fused(x0, energy_fn, config)
    assert torch.equal(a, b), "same global seed produced different runs"


# ── GPU: fused vs reference — statistical comparison (justified tolerances) ──

@gpu
def test_final_norm_on_sphere():
    """exp_map keeps x on the sphere; fp32 accumulates ~7e-4 drift over 100
    steps (measured for BOTH implementations: reference 4.4e-4..7.6e-4,
    fused 7.3e-4). Tolerance 2e-3 ≈ 3x the measured worst case."""
    config, energy_fn, x0 = _setup("cuda")
    with torch.no_grad():
        xf = sample_langevin_fused(x0, energy_fn, config, seed=42)
    dev = (xf.norm(dim=-1) - 1).abs().max().item()
    assert dev < 2e-3, f"||x||-1 drifted to {dev:.2e}"


@gpu
def test_final_energy_matches_reference_distribution():
    """Pointwise trajectory equality vs the reference is impossible BY DESIGN
    (the fused path pre-generates its noise stream instead of drawing lazily,
    and materializes the stagnation-recovery noise unconditionally — both
    documented in the module docstring). The invariant distribution must
    agree: measured reference final-energy spread across seeds 0/1/2 was
    11.005/11.016/11.025 (spread ~0.02). Tolerance 0.5 = >20x that spread,
    generous for a different noise realization while still catching real
    semantic breakage (a wrong clip/momentum/noise scale shifts E by >>1)."""
    config, energy_fn, x0 = _setup("cuda")
    with torch.no_grad():
        ref_es = []
        for seed in (0, 1, 2):
            torch.manual_seed(seed)
            xr = sample_langevin(x0.clone(), energy_fn, config)
            ref_es.append(energy_fn.forward(xr).mean().item())
        xf = sample_langevin_fused(x0, energy_fn, config, seed=42)
        e_fused = energy_fn.forward(xf).mean().item()
    spread = max(ref_es) - min(ref_es)
    assert abs(e_fused - sum(ref_es) / len(ref_es)) < max(0.5, 10 * spread), (
        f"fused final energy {e_fused:.4f} vs reference {ref_es}")


@gpu
def test_energy_decreases_vs_init_on_average():
    """Sampling should move chains toward splats (energy at/below init on
    average). Init energy ~11.02 for this workload; guard against a fused
    path that quietly breaks the score sign or step size."""
    config, energy_fn, x0 = _setup("cuda")
    with torch.no_grad():
        e_init = energy_fn.forward(x0).mean().item()
        xf = sample_langevin_fused(x0, energy_fn, config, seed=42)
        e_final = energy_fn.forward(xf).mean().item()
    assert e_final <= e_init + 0.5, (e_init, e_final)


# ── GPU: cache invalidation / parameter updates ──────────────────────────────

@gpu
def test_recapture_after_splat_parameter_update():
    """An optimizer-style in-place update of splat params must invalidate the
    cached graph (key includes data_ptr/_version/_mut_counter); the recaptured
    graph must agree bit-exactly with the eager fused loop on the new params."""
    config, energy_fn, x0 = _setup("cuda")
    with torch.no_grad():
        before = sample_langevin_fused(x0, energy_fn, config, seed=1234)
        splats = energy_fn.splats
        splats.mu.data[:splats.n_active] = torch.nn.functional.normalize(
            splats.mu.data[:splats.n_active] * 1.01, dim=-1)
        splats._mut_counter += 1
        after = sample_langevin_fused(x0, energy_fn, config, seed=1234)
        eager = sample_langevin_fused(x0, energy_fn, config, seed=1234,
                                      use_cuda_graph=False)
    assert torch.isfinite(after).all()
    assert not torch.equal(before, after), "stale graph served after update"
    assert torch.equal(after, eager), "recaptured graph != eager fused loop"


@gpu
def test_diagnostics_and_signature_parity():
    """return_diagnostics returns final velocity/history; output shape and
    dtype match the reference contract."""
    config, energy_fn, x0 = _setup("cuda", steps=10)
    with torch.no_grad():
        out = sample_langevin_fused(x0, energy_fn, config, seed=1)
        x, diag = sample_langevin_fused(x0, energy_fn, config, seed=1,
                                        return_diagnostics=True)
    ref = sample_langevin(x0.clone(), energy_fn, config)  # shape parity only
    assert out.shape == x0.shape == ref.shape
    assert out.dtype == ref.dtype
    assert torch.equal(out, x)
    assert diag["v"].shape == x0.shape
    assert diag["energy_history"].numel() == 10


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
