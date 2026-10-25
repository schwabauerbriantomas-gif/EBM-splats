#!/usr/bin/env python3
"""
GPU benchmark for the Langevin sampling loop — reference vs fused.

Workload (the package's heaviest loop): B=8 chains, dim 640, 2000 active
splats, knn_k=64, 100 Langevin steps, seed 42 everywhere.

Workloads:
  reference          ebm.langevin.sample_langevin (untouched baseline)
  fused_eager        sample_langevin_fused(use_cuda_graph=False)
                     — fused semantics without graph capture
  fused_graph        sample_langevin_fused() steady state (graph cached;
                     per-call cost = noise refill + one graph replay)
  fused_graph_cold   measured ONCE: capture + first replay (one-time cost,
                     amortized over subsequent samples of the same shape)

Usage:
    python benches/bench_langevin.py                 # n=5 per workload
    python benches/bench_langevin.py --reps 10
    python benches/bench_langevin.py --json out.json
    python benches/bench_langevin.py --steps 50 --batch 8

NOTE: numbers are PRELIMINARY — this GPU is shared; run on the same machine
and idle GPU for honest before/after comparisons.
"""

import argparse
import json
import statistics
import time

import numpy as np
import torch

SEED = 42


def seeded_setup(args):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    from ebm.config import EBMConfig
    from ebm.energy import EnergyFunction
    from ebm.kernels import sample_langevin_fused
    from ebm.langevin import sample_langevin
    from ebm.splats import SplatStorage

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = EBMConfig(device=device, latent_dim=640,
                       n_splats_init=args.n_splats, max_splats=args.n_splats * 2,
                       knn_k=64, langevin_steps=args.steps)
    splats = SplatStorage(config).to(device)
    energy_fn = EnergyFunction(config, splats).to(device)
    x0 = torch.randn(args.batch, 640, device=device)
    x0 = x0 / x0.norm(dim=-1, keepdim=True)
    return config, energy_fn, x0, sample_langevin, sample_langevin_fused


def bench_one(fn, reps, warmup=1):
    times = []
    for i in range(warmup + reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        if i >= warmup:
            times.append(t1 - t0)
    return {"median_ms": statistics.median(times) * 1e3,
            "min_ms": min(times) * 1e3,
            "max_ms": max(times) * 1e3,
            "reps": reps}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--n-splats", type=int, default=2000)
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("bench_langevin requires a CUDA GPU (the CPU path is "
                         "unchanged reference code; see benches/bench_cpu.py)")

    config, energy_fn, x0, sample_langevin, sample_langevin_fused = \
        seeded_setup(args)

    results = {}

    # reference
    with torch.no_grad():
        results["reference"] = bench_one(
            lambda: sample_langevin(x0.clone(), energy_fn, config), args.reps,
            warmup=1)

        # eager fused (no graph)
        results["fused_eager"] = bench_one(
            lambda: sample_langevin_fused(x0, energy_fn, config,
                                          use_cuda_graph=False), args.reps,
            warmup=1)

        # graph cold start: capture + first replay, measured once
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        sample_langevin_fused(x0, energy_fn, config, seed=SEED)
        torch.cuda.synchronize()
        results["fused_graph_cold"] = {
            "median_ms": (time.perf_counter() - t0) * 1e3,
            "min_ms": (time.perf_counter() - t0) * 1e3,
            "max_ms": (time.perf_counter() - t0) * 1e3,
            "reps": 1, "note": "one-time capture + first replay"}

        # graph steady state (graph cached from the cold call above)
        results["fused_graph"] = bench_one(
            lambda: sample_langevin_fused(x0, energy_fn, config, seed=SEED),
            args.reps, warmup=1)

    name = torch.cuda.get_device_name(0)
    print(f"device: {name} | B={args.batch} steps={args.steps} "
          f"splats={args.n_splats} seed={SEED} (PRELIMINARY, shared GPU)")
    print(f"{'workload':18s} {'median':>10s} {'min':>10s} {'max':>10s}")
    for wl, r in results.items():
        print(f"{wl:18s} {r['median_ms']:9.2f}m {r['min_ms']:9.2f}m "
              f"{r['max_ms']:9.2f}m")
    ref = results["reference"]["median_ms"]
    st = results["fused_graph"]["median_ms"]
    print(f"\nspeedup fused_graph vs reference: {ref / st:.1f}x "
          f"(eager: {ref / results['fused_eager']['median_ms']:.1f}x); "
          f"one-time capture: {results['fused_graph_cold']['median_ms']:.0f}ms")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"device": name, "batch": args.batch,
                       "steps": args.steps, "n_splats": args.n_splats,
                       "seed": SEED, "results": results}, fh, indent=2)
        print(f"\nwritten: {args.json}")


if __name__ == "__main__":
    main()
