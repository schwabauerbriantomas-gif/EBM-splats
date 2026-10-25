#!/usr/bin/env python3
"""
GPU benchmark: Triton fused energy/score kernels vs reference (K2).

Workloads (all on CUDA, seeded, B x 640 queries against the splat store):
  splat_energy_ref    reference compute_splat_energy
  splat_energy_kernel FusedEnergyFunction.compute_splat_energy  (1 kernel)
  score_ref           reference compute_score (manual grad + Riemannian)
  score_kernel        FusedEnergyFunction.compute_score          (1 kernel)
  energy_score_fused  fused_energy_and_score (energy AND score, 1 kernel)
  forward_ref         reference EnergyFunction.forward (splat+geom+comp)
  forward_kernel      FusedEnergyFunction.forward

Notes for honest reading:
  - find_neighbors has an LRU-1 cache keyed on (data_ptr, shape, versions):
    within a timing loop the same x reuses cached neighbors for BOTH paths,
    so what is measured is the post-KNN math (the fused part). In real
    Langevin loops x changes every step and the KNN cost is paid once per
    step by both paths equally.
  - PRELIMINARY numbers: GPU is shared; expect noise. Ratios are more
    meaningful than absolute milliseconds.

Usage:
    python benches/bench_energy.py [--reps 20] [--batches 8,64,1024]
                                   [--json benches/energy_gpu_results.json]
"""

import argparse
import json
import platform
import statistics
import time

import numpy as np
import torch

SEED = 42
LATENT_DIM = 640
N_INIT = 2000
MAX_SPLATS = 4000
KNN_K = 64


def seeded_setup(batch: int):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    from ebm.config import EBMConfig
    from ebm.energy import EnergyFunction
    from ebm.kernels import FusedEnergyFunction
    from ebm.splats import SplatStorage

    config = EBMConfig(latent_dim=LATENT_DIM, n_splats_init=N_INIT,
                       max_splats=MAX_SPLATS, knn_k=KNN_K)
    splats = SplatStorage(config).to("cuda")
    ref = EnergyFunction(config, splats).to("cuda")
    fused = FusedEnergyFunction(config, splats).to("cuda")
    fused.load_state_dict(ref.state_dict())

    x = torch.randn(batch, LATENT_DIM, device="cuda")
    x = x / x.norm(dim=-1, keepdim=True)
    return ref, fused, x


def bench_one(fn, reps: int, warmup: int = 5):
    """Median/min/max ms over `reps` timed runs, CUDA-synced per run."""
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1e3)
    return {
        "median_ms": statistics.median(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "reps": reps,
    }


def make_workloads(ref, fused, x):
    return [
        ("splat_energy_ref", lambda: ref.compute_splat_energy(x)),
        ("splat_energy_kernel", lambda: fused.compute_splat_energy(x)),
        ("score_ref", lambda: ref.compute_score(x)),
        ("score_kernel", lambda: fused.compute_score(x)),
        ("energy_score_fused", lambda: fused.fused_energy_and_score(x)),
        ("forward_ref", lambda: ref.forward(x)),
        ("forward_kernel", lambda: fused.forward(x)),
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--batches", type=str, default="8,64,1024",
                    help="comma-separated batch sizes")
    ap.add_argument("--json", type=str, default=None,
                    help="write full results JSON here")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA no disponible: este bench requiere GPU")

    gpu_name = torch.cuda.get_device_name(0)
    env = {
        "gpu": gpu_name,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "seed": SEED,
        "latent_dim": LATENT_DIM,
        "n_splats_init": N_INIT,
        "max_splats": MAX_SPLATS,
        "knn_k": KNN_K,
        "note": "PRELIMINARY: shared GPU, noisy; ratios > absolutes",
    }
    try:
        import triton
        env["triton"] = triton.__version__
    except Exception:
        env["triton"] = None

    print(f"GPU: {gpu_name} | torch {torch.__version__} | seed {SEED}")
    print(f"Batches: {args.batches} | reps: {args.reps}\n")

    results = {"env": env, "batches": {}}
    for batch in (int(b) for b in args.batches.split(",")):
        ref, fused, x = seeded_setup(batch)
        results["batches"][str(batch)] = {}
        print(f"--- B={batch} ---")
        pairs = {}
        for name, fn in make_workloads(ref, fused, x):
            r = bench_one(fn, args.reps)
            results["batches"][str(batch)][name] = r
            print(f"  {name:22s} median={r['median_ms']:9.3f} ms  "
                  f"min={r['min_ms']:9.3f}  max={r['max_ms']:9.3f}  (n={r['reps']})")
            pairs[name] = r["median_ms"]
        for base, kern in [("splat_energy_ref", "splat_energy_kernel"),
                           ("score_ref", "score_kernel"),
                           ("forward_ref", "forward_kernel")]:
            if pairs.get(base) and pairs.get(kern):
                print(f"  speedup {base.split('_')[0]:14s} {pairs[base]/pairs[kern]:8.2f}x")
        ref_split = pairs["splat_energy_ref"] + pairs["score_ref"]
        print(f"  energy+score: ref split {ref_split:.3f} ms vs fused "
              f"{pairs['energy_score_fused']:.3f} ms "
              f"({ref_split/pairs['energy_score_fused']:.2f}x)")
        print()

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"written: {args.json}")


if __name__ == "__main__":
    main()
