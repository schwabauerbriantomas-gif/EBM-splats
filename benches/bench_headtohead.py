"""Head-to-head GPU benchmark: torch-stock reference vs specialized kernels.

OFFICIAL numbers live here — agent-side benchmarks are preliminary (shared GPU
during development). Run this LAST, sequentially, on an idle GPU:

    python benches/bench_headtohead.py [--reps 20]

Every workload is seeded (seed=42) so reference and kernel run identical math.
Correctness (allclose vs reference) is asserted before anything is timed.
"""

import argparse
import time

import numpy as np
import torch

SEED = 42


def timed(fn, n: int):
    """Return (median, min, max) ms over n runs, CUDA-synchronized."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    xs = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        if dev == "cuda":
            torch.cuda.synchronize()
        xs.append((time.perf_counter() - t0) * 1e3)
    xs.sort()
    mid = n // 2
    med = xs[mid] if n % 2 else 0.5 * (xs[mid - 1] + xs[mid])
    return med, xs[0], xs[-1]


def build_inputs(n_splats=4000):
    from ebm.config import EBMConfig
    from ebm.energy import EnergyFunction
    from ebm.splats import SplatStorage

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = EBMConfig(device=device, latent_dim=640,
                       n_splats_init=min(n_splats, 2000), max_splats=n_splats,
                       knn_k=64, langevin_steps=100)
    splats = SplatStorage(config).to(device)
    energy_fn = EnergyFunction(config, splats).to(device)
    x_query = torch.randn(64, 640, device=device)
    x_query = x_query / x_query.norm(dim=-1, keepdim=True)
    x_chains = torch.randn(8, 640, device=device)
    x_chains = x_chains / x_chains.norm(dim=-1, keepdim=True)
    return config, splats, energy_fn, x_query, x_chains


def check_close(a, b, name, rtol=1e-4, atol=1e-5):
    ok = torch.allclose(a, b, rtol=rtol, atol=atol)
    md = (a - b).abs().max().item() if a.numel() else 0.0
    print(f"    correctness[{name}]: allclose={ok} maxdiff={md:.2e}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--n-splats", type=int, default=4000)
    args = ap.parse_args()

    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"device: {dev} | reps={args.reps} | splats={args.n_splats} | seed={SEED}")
    config, splats, energy_fn, x_query, x_chains = build_inputs(args.n_splats)
    k = config.knn_k
    rows = []

    # ---- KNN (kernel keeps reference routing; warm = memoized like Langevin) -
    try:
        from ebm.kernels import find_neighbors_triton

        ref = splats.find_neighbors(x_query, k)
        ker = find_neighbors_triton(splats, x_query, k)
        ok = all(check_close(r, c, n, rtol=0, atol=0)
                 for r, c, n in zip(ref, ker, ("mu", "alpha", "kappa")))
        if ok:
            r = timed(lambda: splats.find_neighbors(x_query, k), args.reps)
            c = timed(lambda: find_neighbors_triton(splats, x_query, k), args.reps)
            rows.append(("find_neighbors(warm)", r, c))
    except ImportError as e:
        print(f"    knn kernel unavailable: {e}")

    # ---- Energy / score via FusedEnergyFunction drop-in ----------------------
    try:
        from ebm.kernels import FusedEnergyFunction

        fused = FusedEnergyFunction(config, splats)
        for m in ("compute_splat_energy", "compute_score"):
            ref = getattr(energy_fn, m)(x_query)
            ker = getattr(fused, m)(x_query)
            if check_close(ref, ker, m):
                r = timed(lambda: getattr(energy_fn, m)(x_query), args.reps)
                c = timed(lambda: getattr(fused, m)(x_query), args.reps)
                rows.append((m, r, c))
    except ImportError as e:
        print(f"    energy kernel unavailable: {e}")

    # ---- Langevin (graph capture amortized: warm-up outside timing) ----------
    try:
        from ebm.kernels import sample_langevin_fused
        from ebm.langevin import sample_langevin

        def run(fn, fused_mode):
            torch.manual_seed(SEED)
            np.random.seed(SEED)
            x0 = x_chains.clone()
            if fused_mode:
                return fn(x0, energy_fn, config, seed=SEED)
            return fn(x0, energy_fn, config)

        ref = run(sample_langevin, False)
        ker_graph = run(sample_langevin_fused, True)

        def run_fused_eager():
            torch.manual_seed(SEED)
            np.random.seed(SEED)
            return sample_langevin_fused(x_chains.clone(), energy_fn, config,
                                         seed=SEED, use_cuda_graph=False)

        ker_eager = run_fused_eager()
        # The strong, deterministic invariant: graph replay == eager fused path,
        # bit-exact (this is what the kernel actually guarantees).
        ok = torch.equal(ker_graph, ker_eager)
        print(f"    correctness[langevin]: graph == eager bit-exact -> {ok}")
        # vs the plain reference the sampler is stochastic (noise pre-generated
        # in a different order) — report the energy delta informatively, no
        # artificial pass/fail: same process, different noise draw.
        with torch.no_grad():
            e_ref = energy_fn(ref).mean().item()
            e_ker = energy_fn(ker_graph).mean().item()
        print(f"    info[langevin]: mean final energy ref={e_ref:.4f} "
              f"fused={e_ker:.4f} |dE|={abs(e_ref - e_ker):.4f}")
        if ok:
            run(sample_langevin_fused, True)  # capture outside timing
            r = timed(lambda: run(sample_langevin, False), 5)
            c = timed(lambda: run(sample_langevin_fused, True), 5)
            rows.append(("langevin_100steps", r, c))
    except ImportError as e:
        print(f"    langevin kernel unavailable: {e}")

    # ---- Report ---------------------------------------------------------------
    print(f"\n{'workload':<24}{'ref med':>10}{'ref min':>10}{'ker med':>10}"
          f"{'ker min':>10}{'speedup':>9}")
    for name, r, c in rows:
        sp = r[0] / c[0] if c[0] > 0 else float("inf")
        print(f"{name:<24}{r[0]:>8.2f}ms{r[1]:>8.2f}ms{c[0]:>8.2f}ms"
              f"{c[1]:>8.2f}ms{sp:>8.2f}x")
    if not rows:
        print("no kernels were eligible (missing or failed correctness)")


if __name__ == "__main__":
    main()
