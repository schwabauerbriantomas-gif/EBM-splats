#!/usr/bin/env python3
"""
GPU micro-benchmark: KNN top-k del splat store — stock vs kernels Triton.

Workloads (los mismos de tests/kernels/test_knn.py): B ∈ {8, 64, 1024}
queries × N ∈ {4000, 100000} splats × dim 640, k=64, seeds fijos (42).

Caminos medidos:
  reference_cold    — path torch-stock de SplatStorage.find_neighbors con cache
                      forzadamente frío (x.clone() por llamada → nuevo data_ptr).
                      Es el BASELINE (mm cuBLAS + torch.topk, sin memo hit).
  reference_warm    — find_neighbors con memoización LRU-1 (hit). El uso real
                      dentro de un paso de Langevin es mayormente warm.
  wrapper_warm      — find_neighbors_triton con memo hit (misma clave que ref).
  wrapper_cold      — find_neighbors_triton con cache frío (x.clone()): rutea
                      al ganador medido por shape (ver knn._auto_plan).
  triton_mm_topk    — GEMM d2 Triton (ieee, sin TF32) + torch.topk, directo.
  triton_fused      — kernel fused distancia+topk en 1 lanzamiento (directo).
  mm_solo/topk_solo — componentes del stock, para atribuir costos.

HONESTIDAD: la GPU (RTX 3090) está COMPARTIDA con otros agentes — estos
números son PRELIMINARES y traen dispersión (min/median/max). Los oficiales
saldrán de un head-to-head consolidado posterior en GPU exclusiva. El fused
PIERDE ~40x (el top-k incremental con sort int64 por bloque domina); el
mm_topk Triton gana en B chico y/o N grande y pierde en B grande con N chico.

Usage:
    python benches/bench_knn.py                 # full (n=20 reps)
    python benches/bench_knn.py --reps 5        # más rápido
    python benches/bench_knn.py --json out.json
"""

import argparse
import json
import statistics
import time

import torch

SEED = 42
DIM = 640
K = 64
WORKLOADS = [(8, 4000), (64, 4000), (1024, 4000),
             (8, 100000), (64, 100000), (1024, 100000)]


def seeded_x(b, device):
    g = torch.Generator(device="cpu").manual_seed(SEED)
    x = torch.randn(b, DIM, generator=g).to(device)
    return (x / x.norm(dim=-1, keepdim=True)).contiguous()


def build_store(n_splats, device):
    from ebm.config import EBMConfig
    from ebm.splats import SplatStorage

    torch.manual_seed(SEED)
    cfg = EBMConfig(latent_dim=DIM, n_splats_init=n_splats,
                    max_splats=n_splats, knn_k=K)
    return SplatStorage(cfg).to(device)


def bench_one(fn, reps, setup=None):
    """median/min/max ms de `reps` corridas sincronizadas."""
    times = []
    for _ in range(reps):
        arg = setup() if setup is not None else None
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(arg)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    return {"median_ms": statistics.median(times), "min_ms": min(times),
            "max_ms": max(times), "reps": reps}


def main():
    import ebm.splats as splats_mod
    try:
        # hermético: medir el path torch-stock (baseline), no faiss
        splats_mod._FAISS_MODULE = None
        splats_mod._FAISS_RESOLVED = True
    except Exception:
        pass
    from ebm.kernels.knn import _knn_fused, _knn_mm_topk, find_neighbors_triton

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("bench GPU requiere CUDA")

    dev = "cuda"
    name = torch.cuda.get_device_name(0)
    print(f"GPU: {name} | torch {torch.__version__} | PRELIMINAR: GPU compartida,"
          f" head-to-head oficial posterior\n")

    results = {"meta": {
        "gpu": name, "torch": torch.__version__, "dim": DIM, "k": K,
        "seeds": SEED, "reps": args.reps,
        "nota": "PRELIMINAR en GPU compartida; dispersión incluida; "
                "los oficiales saldrán de un head-to-head consolidado.",
    }, "results": {}}

    for n_splats in sorted({n for _, n in WORKLOADS}):
        try:
            store = build_store(n_splats, dev)
        except torch.cuda.OutOfMemoryError:
            print(f"N={n_splats}: OOM construyendo el store — skipped")
            continue
        mu_active = store.mu[: store.n_active]

        for b_query, n_cfg in WORKLOADS:
            if n_cfg != n_splats:
                continue
            x = seeded_x(b_query, dev)

            def reference_cold(xc):
                store.find_neighbors(xc, K)

            def wrapper_warm(_):
                find_neighbors_triton(store, x, K)

            def wrapper_cold(xc):
                find_neighbors_triton(store, xc, K)

            def mm_topk(_):
                _knn_mm_topk(store, x, K, *( (16, 128, 4, 3) if b_query <= 16
                                            else (64, 64, 4, 3) if b_query <= 64
                                            else (64, 128, 8, 3) ))

            def fused(_):
                _knn_fused(store, x, K, 16 if b_query <= 64 else 32, 128)

            def mm_solo(_):
                with torch.no_grad():
                    torch.mm(x, mu_active.T)

            def topk_solo(_):
                with torch.no_grad():
                    d2 = (x * x).sum(-1, keepdim=True) \
                        + (mu_active * mu_active).sum(-1).unsqueeze(0) \
                        - 2.0 * torch.mm(x, mu_active.T)
                    torch.topk(d2, k=K, dim=-1)

            paths = [
                ("reference_cold", reference_cold, lambda: x.clone()),
                ("reference_warm", lambda _: store.find_neighbors(x, K), None),
                ("wrapper_warm", wrapper_warm, None),
                ("wrapper_cold", wrapper_cold, lambda: x.clone()),
                ("triton_mm_topk", mm_topk, None),
                ("triton_fused", fused, None),
                ("mm_solo", mm_solo, None),
                ("topk_solo", topk_solo, None),
            ]
            key = f"B{b_query}_N{n_splats}"
            results["results"][key] = {}
            print(f"--- B={b_query} × N={n_splats} (k={K}) ---")
            for name_p, fn, setup in paths:
                if name_p == "triton_fused" and b_query == 1024 and n_splats == 100000:
                    continue  # conocido ~40x más lento: no quemar la GPU shared
                try:
                    r = bench_one(fn, args.reps, setup)
                except torch.cuda.OutOfMemoryError:
                    print(f"  {name_p:16s} OOM — skipped")
                    continue
                results["results"][key][name_p] = r
                print(f"  {name_p:16s} median={r['median_ms']:9.3f} ms  "
                      f"min={r['min_ms']:9.3f}  max={r['max_ms']:9.3f}  (n={r['reps']})")
            print()

        del store
        torch.cuda.empty_cache()

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"written: {args.json}")


if __name__ == "__main__":
    main()
