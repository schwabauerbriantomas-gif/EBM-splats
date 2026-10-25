"""
Correctitud GPU del kernel KNN Triton vs la referencia `SplatStorage.find_neighbors`.

Workloads de referencia (los mismos del bench): B ∈ {8, 64, 1024} queries ×
4000 splats × dim 640, k=64, seed 42. Además: ties deliberados (duplicados y
equidistantes), memoización LRU-1, fallbacks (CPU, sin-Triton, k grande).

Semántica verificada (ver docstring de ebm/kernels/knn.py):
  - La referencia (corregida en 2.1.0) selecciona los d2 MÁS CHICOS
    (largest=False), o sea los splats MÁS CERCANOS — consistente con el path
    faiss (mayor IP). El kernel replica eso.
  - Con datos aleatorios (empates de medida nula): índices bit-idénticos para
    B=8/64; para B=1024 el CONJUNTO top-k es idéntico en el 100% de las filas
    y sólo el orden interno de near-ties (gap < ruido fp32 del GEMM, ~1e-6)
    puede diferir — caveat que la propia referencia documenta.
  - Ties EXACTOS: torch.topk CUDA resuelve empates con un bitonic sort no
    estable (orden = artefacto, medido [7,6,4,5,1,0,2,3] con todo empatado);
    el kernel usa índice-ascendente determinista. Con empates exactos se
    compara como multiconjunto (los valores recogidos son los mismos).

Nota: el wrapper devuelve (mu, alpha, kappa) gatherados (como la referencia),
no índices. Los asserts a nivel índice usan los backends internos
(_knn_mm_topk / _knn_fused); los asserts a nivel wrapper comparan salidas.
Como alpha/kappa son uniformes tras normalize() (1.01 / 11.0), la identidad
de índices se verifica vía las filas de mu (únicas por splat).
"""

import pytest
import torch

import ebm.splats as splats_mod
from ebm.config import EBMConfig
from ebm.kernels.knn import (
    _MAX_K_FUSED,
    _knn_fused,
    _knn_mm_topk,
    find_neighbors_triton,
)
from ebm.splats import SplatStorage

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="tests de kernel GPU requieren CUDA"
)

SEED = 42
DIM = 640
N_SPLATS = 4000
K = 64


@pytest.fixture(autouse=True)
def _no_faiss(monkeypatch):
    """Hermético: la referencia debe usar el path torch-stock (baseline medido)."""
    monkeypatch.setattr(splats_mod, "_FAISS_MODULE", None)
    monkeypatch.setattr(splats_mod, "_FAISS_RESOLVED", True)


def build_store(n_splats=N_SPLATS, device="cuda", seed=SEED,
                max_splats=None) -> SplatStorage:
    torch.manual_seed(seed)
    cfg = EBMConfig(latent_dim=DIM, n_splats_init=n_splats,
                    max_splats=max_splats or n_splats, knn_k=K)
    return SplatStorage(cfg).to(device)


def seeded_queries(b, device="cuda", seed=SEED) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(b, DIM, generator=g).to(device)
    return (x / x.norm(dim=-1, keepdim=True)).contiguous()


def ref_indices(store: SplatStorage, x: torch.Tensor, k: int) -> torch.Tensor:
    """Cómputo fresco del path torch-stock de la referencia (sin memoización),
    corregido a NEAREST (largest=False) igual que src/ebm/splats.py en 2.1.0."""
    with torch.no_grad():
        mu_active = store.mu[: store.n_active]
        x_sq = (x * x).sum(dim=-1, keepdim=True)
        mu_sq = (mu_active * mu_active).sum(dim=-1)
        d2 = x_sq + mu_sq.unsqueeze(0) - 2.0 * torch.mm(x, mu_active.T)
        return torch.topk(d2, k=k, dim=-1, largest=False).indices


def gathered_sets_equal(mu_a: torch.Tensor, mu_b: torch.Tensor) -> bool:
    """Multiconjunto de filas igual (las filas de mu son únicas por splat, así
    que esto equivale a conjunto de índices igual, ante cualquier orden)."""
    ua = torch.unique(mu_a.reshape(-1, mu_a.shape[-1]), dim=0, sorted=True)
    ub = torch.unique(mu_b.reshape(-1, mu_b.shape[-1]), dim=0, sorted=True)
    return torch.equal(ua, ub)


# ── Correctitud en los workloads del bench ────────────────────────────────


@pytest.mark.parametrize("b", [8, 64])
def test_exact_bitwise_match_reference(b):
    """B chico/mediano: multiset de vecinos idéntico a la referencia por fila
    (exigible al 100%). El orden interno es bit-idéntico salvo near-ties
    (gap d2 < ruido fp32 del GEMM ~1e-6): cualquier par reordenado debe
    verificar |d2_i - d2_j| < 1e-5."""
    store = build_store()
    x = seeded_queries(b)
    mu_active = store.mu[: store.n_active]
    d2 = (x * x).sum(-1, keepdim=True) + (mu_active * mu_active).sum(-1) \
        - 2.0 * torch.mm(x, mu_active.T)
    ri = ref_indices(store, x, K)
    from ebm.kernels.knn import _knn_mm_topk

    idx_t = _knn_mm_topk(store, x, K, 64, 64, 4, 3)
    mu_t, a_t, k_t = find_neighbors_triton(store, x, K)
    assert torch.equal(mu_t, store.mu[idx_t]), \
        "el wrapper no devuelve el gather de los índices del kernel"
    assert torch.equal(torch.sort(idx_t, dim=1).values,
                       torch.sort(ri, dim=1).values), \
        "conjunto top-k difiere de la referencia"
    d2_t = d2.gather(1, idx_t)
    d2_r = d2.gather(1, ri)
    dord = (d2_t - d2_r).abs()
    assert bool((dord < 1e-5).all()), \
        "diferencia de orden fuera de near-ties (>1e-5 en d2)"
    assert torch.equal(a_t, store.alpha[idx_t]) and torch.equal(k_t, store.kappa[idx_t])


def test_b1024_set_exact_and_order_report():
    """B=1024: conjunto top-k idéntico en TODAS las filas; orden de near-ties
    puede diferir (gap d2 < ruido fp32 del GEMM — caveat de la referencia)."""
    store = build_store()
    x = seeded_queries(1024)
    ri = ref_indices(store, x, K)
    mu_t, _, _ = find_neighbors_triton(store, x, K)
    assert gathered_sets_equal(store.mu[ri], mu_t), \
        "el CONJUNTO top-k debe ser idéntico"

    # Reporte a nivel índice del backend Triton (auto en B=1024/N=4k es la
    # referencia; esto documenta el comportamiento del kernel mm_topk).
    ki = _knn_mm_topk(store, x, K, 64, 128, 8, 3)
    assert torch.equal(torch.sort(ri, dim=1).values, torch.sort(ki, dim=1).values), \
        "conjunto de índices mm_topk debe ser idéntico a la referencia"
    order_diff = int((ri != ki).any(dim=1).sum())
    # informativo, no assert: el orden entre empates casi exactos no está
    # garantizado ni por la referencia (bitonic sort CUDA no estable).
    print(f"\n[B=1024] filas con orden interno distinto (mismo set): "
          f"{order_diff}/1024")


def test_fused_backend_exact_and_set():
    """El kernel fused (1 lanzamiento) coincide con la referencia por fila:
    multiset-exacto en B=8/64/1024. El ORDEN interno puede diferir en
    near-ties (política propia índice-ascendente del fused; con nearest los
    gaps internos del top-k ~1e-6 hacen el orden dependiente del GEMM)."""
    for b in (8, 64, 1024):
        store = build_store()
        x = seeded_queries(b)
        ri = ref_indices(store, x, K)
        idx = _knn_fused(store, x, K, 16 if b < 1024 else 32, 128)
        assert torch.equal(torch.sort(ri, dim=1).values,
                           torch.sort(idx, dim=1).values), \
            f"fused: conjunto top-k difiere en B={b}"


def test_output_shapes_dtypes_detached():
    store = build_store()
    x = seeded_queries(64)
    mu_t, a_t, k_t = find_neighbors_triton(store, x, K)
    assert mu_t.shape == (64, K, DIM)
    assert a_t.shape == (64, K) and k_t.shape == (64, K)
    assert mu_t.dtype == torch.float32
    assert not mu_t.requires_grad and not a_t.requires_grad


def test_selects_smallest_d2_like_reference():
    """Replica la semántica de la referencia corregida (2.1.0): top-k de d2 con
    largest=False (splats MÁS CERCANOS), igual que el path faiss."""
    store = build_store()
    x = seeded_queries(64)
    with torch.no_grad():
        mu_active = store.mu[: store.n_active]
        d2 = (x * x).sum(-1, keepdim=True) + (mu_active * mu_active).sum(-1) \
            - 2.0 * torch.mm(x, mu_active.T)
        idx = _knn_mm_topk(store, x, K, 64, 64, 4, 3)
    sel = d2.gather(1, idx)
    threshold = d2.kthvalue(K, dim=1).values  # K-ésimo menor
    assert bool((sel <= threshold.unsqueeze(1) + 1e-5).all()), \
        "el kernel no está seleccionando los d2 más chicos (nearest)"


# ── Ties deliberados ──────────────────────────────────────────────────────


def test_ties_duplicated_splats_multiset_exact():
    """500 vectores únicos × 8 copias idénticas (mu, alpha, kappa): empates
    exactos masivos. La salida debe coincidir como multiconjunto por fila
    (valores recogidos idénticos; el orden entre copias es política de ties)."""
    n_unique, copies = 500, 8
    torch.manual_seed(SEED)
    base_mu = torch.randn(n_unique, DIM, device="cuda")
    base_mu = base_mu / base_mu.norm(dim=-1, keepdim=True)
    cfg = EBMConfig(latent_dim=DIM, n_splats_init=n_unique * copies,
                    max_splats=n_unique * copies, knn_k=K)
    store = SplatStorage(cfg).to("cuda")
    with torch.no_grad():
        store.mu.data.copy_(base_mu.repeat_interleave(copies, dim=0))
        store.alpha.data.copy_(torch.arange(n_unique, device="cuda",
                                            dtype=torch.float32)
                               .repeat_interleave(copies))
        store.kappa.data.fill_(10.0)
        store.n_active = n_unique * copies
    x = seeded_queries(64)
    ri = ref_indices(store, x, K)
    mu_t, a_t, k_t = find_neighbors_triton(store, x, K)
    assert gathered_sets_equal(store.mu[ri], mu_t)
    # multiconjunto de alphas (identidad del grupo) idéntico, fila a fila
    ar = torch.sort(store.alpha[ri], dim=1).values
    at = torch.sort(a_t, dim=1).values
    assert torch.equal(ar, at)
    kr = torch.sort(store.kappa[ri], dim=1).values
    kt = torch.sort(k_t, dim=1).values
    assert torch.equal(kr, kt)


def test_ties_equidistant_distinct_mu_kernel_deterministic():
    """x=e0 y 64 splats en ±e_j (j<32): TODOS a d2=2.0 exacto con mu distintos.
    torch.topk (bitonic no estable) y el kernel (índice ascendente) pueden
    elegir subconjuntos distintos pero igual de válidos; se verifica validez
    en fp64 + la política determinista documentada del kernel."""
    n = 64
    cfg = EBMConfig(latent_dim=DIM, n_splats_init=n, max_splats=n, knn_k=32)
    store = SplatStorage(cfg).to("cuda")
    with torch.no_grad():
        for j in range(1, 33):  # j≥1: e0 está en la dirección de x y no empata
            store.mu.data[2 * (j - 1), :] = 0
            store.mu.data[2 * (j - 1), j] = 1.0
            store.mu.data[2 * (j - 1) + 1, :] = 0
            store.mu.data[2 * (j - 1) + 1, j] = -1.0
        store.alpha.data[:n] = torch.arange(n, device="cuda", dtype=torch.float32)
    x = torch.zeros(1, DIM, device="cuda")
    x[0, 0] = 1.0

    ri = ref_indices(store, x, 32)
    idx_t = _knn_fused(store, x, 32, 16, 128)

    # validez en fp64: todo lo seleccionado está a d2 == 2.0 (empate global)
    with torch.no_grad():
        mu64 = store.mu.double()[:n]
        x64 = x.double()
        d2_64 = (x64 * x64).sum(-1, keepdim=True) + (mu64 * mu64).sum(-1) \
            - 2.0 * x64 @ mu64.T
    sel_r = d2_64.gather(1, ri)
    sel_t = d2_64.gather(1, idx_t)
    assert torch.allclose(sel_r, torch.full_like(sel_r, 2.0), atol=1e-9)
    assert torch.allclose(sel_t, torch.full_like(sel_t, 2.0), atol=1e-9)
    # política documentada del kernel: empate exacto → índice ascendente
    assert torch.equal(torch.sort(idx_t[0]).values,
                       torch.arange(32, device="cuda"))


def test_kernel_tie_policy_matches_its_docs_small_n():
    """Ties exactos con N=16, k=8, todo equidistante:
    - kernel fused: política documentada índice-ascendente.
    - wrapper auto (ruta a mm_topk): usa torch.topk → réplica BIT-EXACTA de la
      referencia incluido el orden de empates (misma primitiva)."""
    n = 16
    cfg = EBMConfig(latent_dim=64, n_splats_init=n, max_splats=n, knn_k=8)
    store = SplatStorage(cfg).to("cuda")
    with torch.no_grad():
        for i in range(n):
            v = torch.zeros(64, device="cuda")
            v[1] = 1.0 if i % 2 == 0 else -1.0
            store.mu.data[i] = v
    x = torch.zeros(2, 64, device="cuda")
    x[:, 0] = 1.0
    # kernel fused: determinista, índice ascendente en empates
    idx_t = _knn_fused(store, x, 8, 16, 128)
    assert torch.equal(idx_t, torch.arange(8, device="cuda").expand(2, 8))
    # wrapper (mm_topk): bit-idéntico a la referencia, ties incluidos
    ri = ref_indices(store, x, 8)
    mu_t, a_t, k_t = find_neighbors_triton(store, x, 8)
    assert torch.equal(mu_t, store.mu[ri])
    assert torch.equal(a_t, store.alpha[ri])
    assert torch.equal(k_t, store.kappa[ri])


# ── Memoización LRU-1 (misma clave que la referencia) ─────────────────────


def test_memoization_lru1_hit_and_invalidation():
    store = build_store()
    x = seeded_queries(64)
    r1 = find_neighbors_triton(store, x, K)
    r2 = find_neighbors_triton(store, x, K)
    assert r1[0] is r2[0], "segunda llamada con misma clave debe ser cache hit"
    # in-place bump de _version (patrón optimizer step) → recompute
    with torch.no_grad():
        x.add_(0.0)
    r3 = find_neighbors_triton(store, x, K)
    assert r3[0] is not r1[0]
    assert torch.equal(r3[0], r1[0])
    # add_splat bump _mut_counter → recompute, sigue correcto vs referencia
    store = build_store(n_splats=1000, max_splats=1001)
    x = seeded_queries(64)
    find_neighbors_triton(store, x, K)
    center = torch.randn(DIM, device="cuda")
    assert store.add_splat(center / center.norm()) is True
    r4 = find_neighbors_triton(store, x, K)
    ri = ref_indices(store, x, K)
    assert gathered_sets_equal(store.mu[ri], r4[0])
    assert r4[0].shape == (64, K, DIM)


# ── Fallbacks ─────────────────────────────────────────────────────────────


def test_fallback_cpu_store_and_query():
    store = build_store(device="cpu")
    x = seeded_queries(64, device="cpu")
    expected = store.find_neighbors(x, K)
    got = find_neighbors_triton(store, x, K)
    assert torch.equal(got[0], expected[0])
    assert torch.equal(got[1], expected[1])


def test_fallback_when_triton_disabled(monkeypatch):
    import ebm.kernels.knn as knn_mod

    store = build_store()
    x = seeded_queries(64)
    expected = store.find_neighbors(x, K)
    monkeypatch.setattr(knn_mod, "TRITON_FUSED_AVAILABLE", False)
    got = find_neighbors_triton(store, x, K)
    assert torch.equal(got[0], expected[0])


def test_fallback_k_above_fused_limit():
    store = build_store()
    x = seeded_queries(64)
    k = _MAX_K_FUSED + 1
    assert k <= store.n_active
    ri = ref_indices(store, x, k)
    mu_t, a_t, k_t = find_neighbors_triton(store, x, k)
    assert torch.equal(mu_t, store.mu[ri])
    assert torch.equal(a_t, store.alpha[ri])
    assert torch.equal(k_t, store.kappa[ri])


def test_fallback_k_exceeding_n_active_raises_like_reference():
    store = build_store()
    x = seeded_queries(8)
    with pytest.raises(RuntimeError):
        store.find_neighbors(x, store.n_active + 1)
    with pytest.raises(RuntimeError):
        find_neighbors_triton(store, x, store.n_active + 1)


def test_noncontiguous_x_matches_reference_set():
    store = build_store()
    g = torch.Generator(device="cpu").manual_seed(SEED + 1)
    base = torch.randn(64, 2 * DIM, generator=g).to("cuda")
    x = base[:, :DIM]
    assert not x.is_contiguous()
    ri = ref_indices(store, x, K)
    mu_t, _, _ = find_neighbors_triton(store, x, K)
    assert gathered_sets_equal(store.mu[ri], mu_t)


def test_k1_squeeze_semantics_match_reference():
    store = build_store()
    x = seeded_queries(8)
    ri = ref_indices(store, x, 1)
    mu_t, a_t, k_t = find_neighbors_triton(store, x, 1)
    assert a_t.shape == ri.squeeze(-1).shape
    assert torch.equal(mu_t.squeeze(1), store.mu[ri.squeeze(-1)])


def test_float64_inputs_behave_like_reference():
    """El path torch de la referencia NO soporta x double (mm dtype mismatch).
    El wrapper delega y debe propagar el MISMO error (paridad)."""
    store = build_store()
    x = seeded_queries(8).double()
    with pytest.raises(RuntimeError):
        store.find_neighbors(x, K)
    with pytest.raises(RuntimeError):
        find_neighbors_triton(store, x, K)
