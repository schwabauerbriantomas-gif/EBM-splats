"""
Fused Langevin sampling — CUDA-graph capture of the full sampling loop.

Problem
-------
``ebm.langevin.sample_langevin`` (reference, untouched) runs ~176 tiny CUDA
kernel launches per step plus one blocking ``.item()`` (stagnation check).
At B=8, 100 steps that is ~17.6k launches and 100 device syncs: on an RTX
3090 the reference loop is dispatch-bound (measured 602 ms median; the GPU
arithmetic itself is microseconds per step).

Approach (chosen by measurement, see benches/bench_langevin.py)
---------------------------------------------------------------
CUDA graph capture of the whole unrolled loop. A single ``graph.replay()``
re-executes every captured kernel from one host call, removing both the
per-kernel launch cost and every host/device sync. Triton kernel fusion was
considered and rejected: the remaining replay time is dominated by kernels
INSIDE ``energy_fn.compute_score`` / the energy terms (reference code — the
``kernels/energy`` agent's territory, intentionally not reimplemented here)
and by ``torch.topk`` in the KNN. Fusing only this module's step math would
save <25% of replay time while breaking the bit-exact equivalence chains
that the tests rely on.

Semantics — what is identical, what differs (honesty notes)
-----------------------------------------------------------
1. Trajectory math (score -> clip -> momentum half-step -> exp_map ->
   second score/clip/momentum -> noise -> tangent projection) uses the
   SAME reference torch ops in the SAME order as ``ebm.langevin``:
   ``energy_fn.compute_score``, ``geometry.exp_map`` /
   ``project_to_tangent`` are called directly. With the optional
   ``langevin_step_fused`` (reference-style lazy RNG), the eager trajectory
   is BIT-IDENTICAL to the reference when the stagnation branch never
   fires (verified by tests; the ``x.mul_(1.0)`` version bump below is
   numerically exact and does not change any value).
2. RNG: the graph path PRE-GENERATES all noise into static buffers before
   capture/replay (2 draws per step: main noise + stagnation-recovery
   noise). Same seed -> bit-identical trajectories, run to run, and no
   RNG-inside-graph pitfalls. This deliberately differs from the
   reference's lazy per-step ``randn_like`` (which draws the recovery
   noise only when stagnated): fused and reference trajectories therefore
   use DIFFERENT noise streams and are NOT pointwise comparable — they are
   statistically equivalent (same invariant distribution); tests compare
   final norms / energies with tolerances justified from the measured
   seed-to-seed spread of the reference itself.
3. Stagnation detection (V2 §7): the reference records
   ``energy_fn.forward(x).mean().item()`` per step — a host sync. The
   fused path keeps an on-device history buffer and computes the same
   max-min window test per step; the recovery noise is materialized
   unconditionally and applied through a multiplicative mask (same math,
   data-dependent VALUE instead of host control flow). Two numerical
   differences, both confined to the stagnation SIGNAL (never to the
   trajectory math):
     a. ``compute_geom_energy`` uses boolean-mask indexing internally
        (``torch.nonzero`` -> host sync -> illegal inside graph capture);
        the fused path computes the identical elementwise mean via
        sum/diagonal arithmetic (same elements, different reduction
        order, possible 1-ulp difference).
     b. The window test runs in fp32 on device instead of float64 on
        host; only a difference within ~1e-7 of ``epsilon`` could flip
        the trigger.
4. KNN memoization: ``SplatStorage.find_neighbors`` caches on host-side
   keys (``data_ptr``/``_version``). Inside a capture the allocator can
   reuse pool addresses across steps, which could serve STALE neighbors
   at capture time and poison the captured graph. Each score/energy call
   is therefore preceded by ``x.mul_(1.0)`` — a numerically exact no-op
   that bumps ``_version`` so every lookup is a cache miss and real KNN
   kernels get captured. The host cache is cleared after capture.
5. Energy/score: the reference ``energy_fn.compute_score`` and the
   reference splat/comp energy terms are called as-is. When a fused
   score kernel lands in ``ebm.kernels.energy``, swap it in here via the
   ``_score_fn`` hook (single place, documented) — this module does not
   reimplement the energy kernel.

Fallbacks
---------
- CPU input (or no CUDA): delegates to the reference
  ``ebm.langevin.sample_langevin`` — bit-identical by construction.
- CUDA but capture fails: eager fused loop (same fused semantics, same
  pre-generated noise) — a warning is emitted.
"""

import math
import warnings

import numpy as np
import torch

from ..config import EBMConfig
from ..geometry import exp_map, project_to_tangent
from ..langevin import LangevinState, langevin_step as _reference_step
from ..langevin import sample_langevin as _reference_sample

__all__ = ["sample_langevin_fused", "langevin_step_fused"]


# ─────────────────────────────────────────────────────────────────────────────
# Score / energy hooks (single swap point for a future fused energy kernel)
# ─────────────────────────────────────────────────────────────────────────────

def _score_fn(energy_fn, x):
    """score(x) = -grad_R E(x). Reference path; swap point for kernels/energy."""
    return energy_fn.compute_score(x)


def _bump(x):
    """Numerically exact no-op (x*1.0) that bumps ``_version``.

    Forces ``SplatStorage.find_neighbors``' LRU cache to miss so that a
    graph capture records real KNN kernels instead of binding a stale
    host-cached tensor. Bit-exact for every finite float, NaN and -0.0.
    """
    return x.mul_(1.0)


def _geom_energy_nosync(x_flat, lam_geom):
    """Sync-free E_geom(x), elementwise-identical to
    ``EnergyFunction.compute_geom_energy`` for B >= 2 (which boolean-mask
    indexes -> torch.nonzero -> host sync, illegal during capture).

    Uses the same elementwise values ``-log(1 - clamp(s, 1-1e-4) + 1e-4)``
    and corrects the mean for the (deterministic) diagonal; only the
    reduction order differs (possible 1-ulp difference).
    """
    sims = torch.mm(x_flat, x_flat.T)
    elem = -torch.log(1.0 - sims.clamp(max=1.0 - 1e-4) + 1e-4)
    n = x_flat.size(0)
    if n < 2:
        return x_flat.new_zeros(())
    off_mean = (elem.sum() - elem.diagonal().sum()) / (n * n - n)
    return off_mean * lam_geom


def _stagnation_energy(energy_fn, config, x_flat):
    """E(x) used by the stagnation detector = reference ``forward`` without
    context vectors: splat + lambda_geom*geom + lambda_comp*comp.
    Splat/comp terms are the reference methods (not reimplemented)."""
    e = energy_fn.compute_splat_energy(x_flat)
    e = e + _geom_energy_nosync(x_flat, config.lambda_geom)
    e = e + energy_fn.compute_comp_energy(x_flat) * config.lambda_comp
    return e


# ─────────────────────────────────────────────────────────────────────────────
# Core step math — shared by every path; identical ops/order to reference
# ─────────────────────────────────────────────────────────────────────────────

def _step_core(x, v, energy_fn, config, noise_main_t, noise_stag_t, history, t):
    """One underdamped Langevin step (V2 stability), graph-safe.

    ``noise_main_t``/``noise_stag_t`` are pre-scaled per-step noise slices
    (already multiplied by their standard deviations) or ``None`` in
    reference-RNG mode (``langevin_step_fused``).
    ``history[t]`` is written with this step's mean energy (stagnation
    window); ``history`` may be ``None`` in reference-RNG mode.
    """
    gamma = config.langevin_gamma
    T = config.langevin_T
    dt = config.langevin_dt

    # score = -grad_R E(x)  (reference call; bump makes KNN cache miss)
    _bump(x)
    score = _score_fn(energy_fn, x)
    grad_e = -score

    # Gradient clipping (P0 fix) — reference expression
    if config.grad_clip > 0:
        grad_norm = grad_e.norm(dim=-1, keepdim=True)
        scale = torch.clamp(config.grad_clip / (grad_norm + 1e-8), max=1.0)
        grad_e = grad_e * scale

    # Half-step momentum
    v = v - 0.5 * dt * (gamma * v + grad_e)

    # Full-step position via exponential map
    x = exp_map(x, dt * v)
    _bump(x)

    # Re-evaluate at new position, clip, finish momentum
    score_new = _score_fn(energy_fn, x)
    grad_e_new = -score_new
    if config.grad_clip > 0:
        grad_norm_new = grad_e_new.norm(dim=-1, keepdim=True)
        scale_new = torch.clamp(config.grad_clip / (grad_norm_new + 1e-8), max=1.0)
        grad_e_new = grad_e_new * scale_new

    v = v - 0.5 * dt * (gamma * v + grad_e_new)

    # Noise injection
    if noise_main_t is not None:
        v = v + noise_main_t
    else:
        v = v + torch.randn_like(v) * np.sqrt(2 * gamma * T * dt)

    # Project velocity to tangent space
    v = project_to_tangent(x, v)

    # Stagnation detection + recovery (V2 §7)
    if noise_stag_t is not None:
        # graph semantics: on-device history, mask instead of host branch
        x_flat = x.reshape(-1, x.size(-1))
        e_mean = _stagnation_energy(energy_fn, config, x_flat).mean()
        history[t] = e_mean
        W = config.energy_stagnation_window
        eps = config.energy_stagnation_epsilon
        if t + 1 >= W:  # static at capture time: unrolled python loop
            recent = history[t + 1 - W: t + 1]
            stagnated = (recent.max() - recent.min()) < eps
            v = v + stagnated.to(v.dtype) * noise_stag_t
    return x, v


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def langevin_step_fused(x, v, energy_fn, config,
                        state: LangevinState | None = None) -> tuple:
    """Single fused-semantics step with REFERENCE rng/stagnation mechanics.

    Identical trajectory math to ``ebm.langevin.langevin_step`` (same ops,
    same order); exists so tests can prove the fused math transcription is
    bit-exact vs the reference. On CPU delegates to the reference step.
    """
    if not x.is_cuda:
        return _reference_step(x, v, energy_fn, config, state)

    x, v = _step_core(x, v, energy_fn, config, None, None, None, 0)

    if state is not None:
        with torch.no_grad():
            x_flat = x.reshape(-1, x.size(-1))
            e_val = energy_fn.forward(x_flat)
            state.record(e_val.mean().item() if e_val.numel() > 1 else e_val.item())
        if state.is_stagnated():
            v = v + torch.randn_like(v) * 0.1
    return x, v


class _FusedGraph:
    """Captured 100%-unrolled Langevin loop + its static buffers."""

    def __init__(self, x_init, energy_fn, config, steps):
        dev = x_init.device
        self.energy_fn = energy_fn
        self.config = config
        self.steps = steps

        # Static I/O buffers (copied into per call, outside the graph)
        self.x_in = x_init.clone()
        # Initial velocity MUST outlive capture: the graph reads it on every
        # replay, and if the only reference dies inside the capture (python
        # rebinding), the allocator recycles its host-pool block and later
        # replays read recycled garbage (observed: replay #1 correct, #2+
        # silently wrong). Same reasoning for noise/history buffers below.
        self._v0 = torch.zeros_like(self.x_in)
        # noise buffers: [steps, B, D] raw randn; scaling happens in-graph
        self.noise_main = torch.zeros(steps, *x_init.shape, device=dev)
        self.noise_stag = torch.zeros(steps, *x_init.shape, device=dev)
        self.noise_std = math.sqrt(2 * config.langevin_gamma * config.langevin_T
                                   * config.langevin_dt)

        splats = energy_fn.splats
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            # Warmup: initialize cuBLAS/topk workspaces + allocator pools.
            # Reads zeros noise; consumes NO rng (all noise is pregenerated
            # outside the graph by design).
            warm_hist = torch.zeros(steps, device=dev)
            xw = self.x_in
            vw = torch.zeros_like(xw)
            for t in range(3):
                xw, vw = _step_core(
                    xw, vw, energy_fn, config,
                    self.noise_main[0] * self.noise_std,
                    self.noise_stag[0] * 0.1, warm_hist, 0)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        # Clear host-side KNN cache left by warmup (it pins warmup tensors)
        for attr in ("_knn_cache_key", "_knn_cache_val", "_knn_cache_x"):
            if hasattr(splats, attr):
                setattr(splats, attr, None)

        history = torch.zeros(steps, device=dev)
        self.history = history
        x = self.x_in
        v = self._v0
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            for t in range(steps):
                x, v = _step_core(
                    x, v, energy_fn, config,
                    self.noise_main[t] * self.noise_std,
                    self.noise_stag[t] * 0.1, history, t)
        self._out = x
        self.v_final = v
        self.history = history

        # Capture filled the host KNN cache with graph-pool tensors; clear
        # so later eager calls cannot pin or serve them.
        for attr in ("_knn_cache_key", "_knn_cache_val", "_knn_cache_x"):
            if hasattr(splats, attr):
                setattr(splats, attr, None)

    def replay(self, x_init, generator):
        """One full sampling run: fresh noise + x, one graph launch."""
        self.x_in.copy_(x_init, non_blocking=True)
        # deterministic per generator state; nothing inside the graph draws
        self.noise_main.normal_(generator=generator)
        self.noise_stag.normal_(generator=generator)
        self.graph.replay()
        return self._out.clone(), self.v_final.clone(), self.history.clone()


# Cache of captured graphs, keyed by everything the captured math depends on.
_GRAPH_CACHE: dict = {}


def _graph_key(x_init, energy_fn, config, steps):
    splats = energy_fn.splats
    w_comp = getattr(energy_fn, "W_comp", None)
    wcomp_state = []
    if w_comp is not None:
        for p in w_comp.parameters():
            wcomp_state.append((p.data_ptr(), p._version))
    return (
        id(energy_fn), tuple(x_init.shape), x_init.dtype, x_init.device.index,
        steps,
        config.langevin_gamma, config.langevin_T, config.langevin_dt,
        config.grad_clip, config.temperature, config.knn_k,
        config.energy_stagnation_window, config.energy_stagnation_epsilon,
        config.lambda_geom, config.lambda_comp, config.latent_dim,
        splats.n_active, splats._mut_counter,
        splats.mu.data_ptr(), splats.mu._version,
        splats.alpha.data_ptr(), splats.alpha._version,
        splats.kappa.data_ptr(), splats.kappa._version,
        tuple(wcomp_state),
    )


def _fill_generator(generator, device):
    if generator is not None:
        return generator
    idx = device.index if device.index is not None else torch.cuda.current_device()
    return torch.cuda.default_generators[idx]


def sample_langevin_fused(x_init: torch.Tensor, energy_fn, config: EBMConfig,
                          *, seed: int | None = None,
                          use_cuda_graph: bool = True,
                          return_diagnostics: bool = False
                          ) -> "torch.Tensor | tuple":
    """Full Langevin sampling loop — fused via CUDA graph capture.

    Same signature and semantics as ``ebm.langevin.sample_langevin``
    (returns the sampled ``x`` of shape ``x_init.shape``); extra keyword
    arguments:
      seed: if given, draws the pre-generated noise from a dedicated
            ``torch.Generator(device)`` seeded with it. If None, the
            current global CUDA generator state is used (so callers that
            ``torch.manual_seed`` get reproducibility too).
      use_cuda_graph: set False to force the eager fused loop (same
            semantics, no capture).
      return_diagnostics: return ``(x, dict)`` with final velocity,
            per-step stagnation-energy history and whether the recovery
            branch fired at the last step.

    Fallbacks: CPU input -> reference ``sample_langevin`` (bit-identical);
    failed capture -> eager fused loop with a warning.
    """
    if not x_init.is_cuda or not torch.cuda.is_available():
        return _reference_sample(x_init, energy_fn, config)

    steps = config.langevin_steps
    with torch.no_grad():
        if not use_cuda_graph:
            x, v, hist = _eager_fused_loop(x_init, energy_fn, config, steps,
                                           seed)
        else:
            key = _graph_key(x_init, energy_fn, config, steps)
            fused = _GRAPH_CACHE.get(key)
            if fused is None:
                try:
                    fused = _FusedGraph(x_init, energy_fn, config, steps)
                except Exception as e:  # noqa: BLE001 — any capture failure
                    torch.cuda.synchronize()  # un-poison the capture context
                    warnings.warn(
                        f"sample_langevin_fused: CUDA graph capture failed "
                        f"({type(e).__name__}: {e}); falling back to the "
                        f"eager fused loop.", RuntimeWarning, stacklevel=2)
                    fused = None
                if fused is not None:
                    # keep exactly one graph per key footprint
                    _GRAPH_CACHE.clear()
                    _GRAPH_CACHE[key] = fused
            if fused is None:
                x, v, hist = _eager_fused_loop(x_init, energy_fn, config,
                                               steps, seed)
            else:
                gen = _fill_generator(
                    _seeded_generator(seed, x_init.device), x_init.device)
                x, v, hist = fused.replay(x_init, gen)

        if return_diagnostics:
            fired = False
            W = config.energy_stagnation_window
            if hist.numel() >= W:
                recent = hist[-W:]
                fired = bool((recent.max() - recent.min()).item()
                             < config.energy_stagnation_epsilon)
            return x, {"v": v, "energy_history": hist,
                       "stagnated_last_window": fired}
        return x


def _seeded_generator(seed, device):
    if seed is None:
        return None  # sentinel: use the device's default generator
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    return g


def _eager_fused_loop(x_init, energy_fn, config, steps, seed):
    """Eager loop with the exact fused semantics (graph oracle / fallback)."""
    dev = x_init.device
    gen = _seeded_generator(seed, dev)
    if gen is None:
        gen = _fill_generator(None, dev)
    noise_main = torch.randn(steps, *x_init.shape, device=dev, generator=gen)
    noise_stag = torch.randn(steps, *x_init.shape, device=dev, generator=gen)
    noise_std = math.sqrt(2 * config.langevin_gamma * config.langevin_T
                          * config.langevin_dt)
    history = torch.zeros(steps, device=dev)
    x = x_init.clone()
    v = torch.zeros_like(x)
    for t in range(steps):
        x, v = _step_core(x, v, energy_fn, config,
                          noise_main[t] * noise_std,
                          noise_stag[t] * 0.1, history, t)
    return x, v, history
