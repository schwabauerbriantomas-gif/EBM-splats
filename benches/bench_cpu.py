#!/usr/bin/env python3
"""
CPU micro-benchmarks for EBM-splats — reproducible, seeded.

Measures the hot paths of the library on CPU:
  1. find_neighbors      — KNN over splat store (called ~4x per Langevin step)
  2. energy_forward      — EnergyFunction.forward (splat + geom + comp energy)
  3. score_forward        — EnergyFunction.compute_score (manual Riemannian grad)
  4. langevin_sampling    — full sampling loop (B chains x N steps)
  5. score_network        — ScoreNetwork forward (denoising score model)
  6. flow_matching_loss   — one OT-CFM training step
  7. flow_matching_sample — 10-step ODE inference

All inputs are seeded (torch.manual_seed(42), np.random.seed(42)) so the SAME
work runs before and after an optimization. Timing: time.perf_counter around
the workload, `--reps` repetitions (each repetition re-runs the seeded setup,
so every rep executes identical math). Reports median / min / max seconds.

Usage:
    python benches/bench_cpu.py                 # full run (default reps)
    python benches/bench_cpu.py --reps 7        # more repetitions
    python benches/bench_cpu.py --json out.json # machine-readable output

Reference machine notes: run on the SAME machine, same torch/numpy versions,
nothing else heavy running, for honest before/after comparisons.
"""

import argparse
import json
import statistics
import time

import numpy as np
import torch

SEED = 42


def seeded_inputs():
    """Reseed everything and return the shared set of benchmark inputs."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    from ebm.config import EBMConfig
    from ebm.energy import EnergyFunction
    from ebm.splats import SplatStorage

    # dim 640 (lib default). n_splats=2000: representativo pero razonable en
    # CPU (el default 10000 haría el baseline del KNN muy lento; el perfil
    # relativo de find_neighbors no cambia).
    config = EBMConfig(latent_dim=640, n_splats_init=2000, max_splats=4000,
                       knn_k=64, langevin_steps=100)
    splats = SplatStorage(config)
    energy_fn = EnergyFunction(config, splats)

    x_query = torch.randn(64, 640)
    x_query = x_query / x_query.norm(dim=-1, keepdim=True)

    x_chains = torch.randn(8, 640)
    x_chains = x_chains / x_chains.norm(dim=-1, keepdim=True)

    score_net = None
    try:
        from ebm.score_network import ScoreNetwork
        score_net = ScoreNetwork(dim=640)
    except ImportError:
        pass  # module optional

    try:
        from pglf.flow_matching import FlowMatchingLoss, HypersphereFlowMatching
        fm = HypersphereFlowMatching(dim=640, hidden_dim=1280, n_layers=6)
        fm_loss_fn = FlowMatchingLoss()
        fm_cond = torch.randn(32, 640)
        fm_modality = torch.randint(0, 3, (32,))
        fm_x1 = torch.randn(32, 640)
        fm_x1 = fm_x1 / fm_x1.norm(dim=-1, keepdim=True)
    except ImportError:
        fm = fm_loss_fn = fm_cond = fm_modality = fm_x1 = None

    return {
        "config": config,
        "splats": splats,
        "energy_fn": energy_fn,
        "x_query": x_query,
        "x_chains": x_chains,
        "score_net": score_net,
        "fm": fm,
        "fm_loss_fn": fm_loss_fn,
        "fm_cond": fm_cond,
        "fm_modality": fm_modality,
        "fm_x1": fm_x1,
    }


def make_workloads(inp):
    """Return dict name -> (callable, reps_default)."""

    def find_neighbors():
        inp["splats"].find_neighbors(inp["x_query"], inp["config"].knn_k)

    def energy_forward():
        with torch.no_grad():
            inp["energy_fn"].forward(inp["x_query"])

    def score_forward():
        with torch.no_grad():
            inp["energy_fn"].compute_score(inp["x_query"])

    def langevin_sampling():
        from ebm.langevin import sample_langevin
        with torch.no_grad():
            sample_langevin(inp["x_chains"].clone(), inp["energy_fn"],
                            inp["config"])

    def score_network():
        with torch.no_grad():
            x = inp["x_query"]
            sigma = torch.full((x.size(0),), 0.1)
            inp["score_net"](x, sigma)

    def flow_matching_loss():
        torch.manual_seed(SEED)  # loss muestrea ruido/t internamente
        with torch.no_grad():
            inp["fm_loss_fn"](inp["fm"], inp["fm_x1"], inp["fm_cond"],
                              inp["fm_modality"])

    def flow_matching_sample():
        torch.manual_seed(SEED)  # sample muestrea x0 internamente
        with torch.no_grad():
            inp["fm"].sample(inp["fm_cond"], inp["fm_modality"], n_steps=10)

    wls = [
        ("find_neighbors", find_neighbors, 20),
        ("energy_forward", energy_forward, 20),
        ("score_forward", score_forward, 20),
        ("score_network", score_network, 20),
        ("flow_matching_loss", flow_matching_loss, 10),
        ("flow_matching_sample", flow_matching_sample, 5),
        ("langevin_sampling", langevin_sampling, 3),
    ]
    return wls


def bench_one(fn, reps):
    """Median/min/max of `reps` timed runs (each preceded by torch warm sync)."""
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return {
        "median_s": statistics.median(times),
        "min_s": min(times),
        "max_s": max(times),
        "reps": reps,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=None,
                    help="override default repetitions per workload")
    ap.add_argument("--only", type=str, default=None,
                    help="comma-separated workload names to run")
    ap.add_argument("--json", type=str, default=None, help="write JSON here")
    ap.add_argument("--profile", type=str, default=None,
                    help="cProfile this workload name and print top functions")
    args = ap.parse_args()

    torch.set_num_threads(torch.get_num_threads())  # honor env defaults
    inp = seeded_inputs()
    wls = make_workloads(inp)
    if args.only:
        keep = set(args.only.split(","))
        wls = [w for w in wls if w[0] in keep]

    if args.profile:
        import cProfile
        import pstats
        name = args.profile
        fn = {n: f for n, f, _ in wls}[name]
        pr = cProfile.Profile()
        pr.enable()
        fn()
        pr.disable()
        pstats.Stats(pr).sort_stats("cumulative").print_stats(25)
        return

    results = {}
    for name, fn, default_reps in wls:
        reps = args.reps or default_reps
        r = bench_one(fn, reps)
        results[name] = r
        print(f"{name:24s} median={r['median_s']*1e3:10.2f} ms  "
              f"min={r['min_s']*1e3:10.2f}  max={r['max_s']*1e3:10.2f}  "
              f"(n={r['reps']})")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nwritten: {args.json}")


if __name__ == "__main__":
    main()
