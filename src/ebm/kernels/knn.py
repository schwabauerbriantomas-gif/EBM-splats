"""
KNN top-k en Triton para `SplatStorage.find_neighbors` (src/ebm/splats.py).

Semántica replicada EXACTAMENTE de la referencia (`SplatStorage.find_neighbors`,
path torch-stock — el que mide el baseline):

    mu_active = store.mu[:store.n_active]
    d2 = (x*x).sum(-1, keepdim=True) + (mu*mu).sum(-1) - 2 * x @ mu.T
    idx = torch.topk(d2, k=k, dim=-1).indices        # largest=True, sorted=True
    return store.mu[idx], store.alpha[idx], store.kappa[idx]

OJO (verificado empíricamente, probe en RTX 3090 / torch 2.14.1+cu126):
`torch.topk(d2)` usa largest=True por defecto, es decir la referencia selecciona
los d2 MÁS GRANDES (= splats MÁS LEJANOS en métrica euclídea), no los más
cercanos. Además el path faiss (IndexFlatIP) selecciona mayor producto interno
(= más cercanos cuando mu está normalizado): ambos paths de la referencia
DISCREPAN. Este kernel replica el path torch-stock (el baseline medido), no el
path faiss. Si algún día se corrige la referencia, basta invertir el signo del
empaquetado (ver `_pack_keys`) o pasar `largest=False` por el mismo código.

Métrica: distancia euclídea al cuadrado vía identidad del matmul
(||x||² + ||mu||² - 2 x·mu), fp32 con `input_precision="ieee"` en tl.dot
(sin TF32), la misma clase numérica que torch.mm con
`torch.backends.cuda.matmul.allow_tf32=False` (default). El sqrt se omite
porque las distancias sólo se rankean, nunca se devuelven — igual que la ref.

Política de ties (documentada, medida):
  - torch.topk(sorted=True) ordena los k valores en orden descendente. El ORDEN
    y el SUBCONJUNTO elegido entre valores EXACTAMENTE iguales es un artefacto
    del bitonic sort de CUDA, NO estable y NO documentado (medido con todos los
    valores empatados: [7,6,4,5,1,0,2,3] — ni ascendente ni descendente).
  - El backend "mm_topk" (default por auto) delega la selección a torch.topk:
    réplica BIT-EXACTA de la referencia INCLUIDO el orden de empates, por
    construir (misma primitiva sobre el mismo d2... con la única diferencia
    del ruido ~1e-6 del GEMM propio, ver abajo).
  - El kernel "fused" resuelve empates con política propia determinista:
    mismo d2 → índice MENOR primero (empaquetando (d2, 0xFFFFF - idx) en una
    clave int64 y ordenando descendente). Con empates exactos su conjunto
    top-k es válido pero el orden/subconjunto entre empatados puede diferir
    del artefacto bitónico de torch (ver tests/kernels/test_knn.py).
  - d2 se calcula con la MISMA fórmula y composición elemento a elemento que la
    referencia; sólo difiere el orden de reducción del producto punto (GEMM
    tilado), ruido relativo ~1e-6. Puede reordenar empates "casi exactos"
    (mismo caveat que ya documenta la propia referencia).

Fallback: si no hay CUDA, no hay Triton, o el workload sale del encuadre
(fp32, k<=128, n_active>=k, n_active<2^20), `find_neighbors_triton` delega en
`store.find_neighbors` (la referencia) sin error.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - entorno sin triton
    _TRITON_AVAILABLE = False

__all__ = ["find_neighbors_triton", "TRITON_FUSED_AVAILABLE"]

# n_active debe entrar en 20 bits para el empaquetado (índice, d2) -> int64.
_MAX_N_PACK = 1 << 20
# k máximo soportado por el kernel fused (K_PAD = 64 o 128).
_MAX_K_FUSED = 128

TRITON_FUSED_AVAILABLE = bool(_TRITON_AVAILABLE and torch.cuda.is_available())

if _TRITON_AVAILABLE:

    @triton.jit
    def _pack_desc(d2, col_key):
        # Mapa monótono float32 -> uint32 (comparación por bits, soporta
        # negativos): signo 1 -> ~bits, signo 0 -> bits | 0x80000000.
        u = d2.to(tl.uint32, bitcast=True)
        u = tl.where((u >> 31) == 1, ~u, u | 0x80000000)
        # Clave int64: d2 en los bits altos (orden desc = d2 desc),
        # (0xFFFFF - idx) en los bajos (mismo d2 -> índice menor primero).
        return (u.to(tl.int64) << 20) | col_key.to(tl.int64)

    @triton.jit
    def _knn_fused_kernel(
        X, MU, OUT, SCR,
        B, N, D, K,
        stride_x, stride_m,
        BLOCK_Q: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
        K_PAD: tl.constexpr,
        TOTAL: tl.constexpr,
        EVEN_D: tl.constexpr,
    ):
        # Un programa = bloque de BLOCK_Q queries; escanea los N splats en
        # bloques de BLOCK_N manteniendo el top-K_PAD corrido como claves
        # empabetadas (ver _pack_desc). SCRATCH es un buffer por programa:
        # (BLOCK_Q, TOTAL) donde TOTAL = K_PAD + BLOCK_N (potencia de 2).
        pid = tl.program_id(0)
        q_offs = pid * BLOCK_Q + tl.arange(0, BLOCK_Q)
        q_mask = q_offs < B
        q_safe = tl.where(q_mask, q_offs, 0)

        d_offs = tl.arange(0, BLOCK_D)
        n_cols = tl.arange(0, BLOCK_N)
        k_cols = tl.arange(0, K_PAD)
        t_cols = tl.arange(0, TOTAL)

        scr_base = SCR + pid * BLOCK_Q * TOTAL
        run = tl.full((BLOCK_Q, K_PAD), -1, tl.int64)

        for nb in range(0, tl.cdiv(N, BLOCK_N)):
            n_offs = nb * BLOCK_N + n_cols
            n_mask = n_offs < N
            n_safe = tl.where(n_mask, n_offs, 0)

            acc = tl.zeros((BLOCK_Q, BLOCK_N), tl.float32)
            xsq = tl.zeros((BLOCK_Q,), tl.float32)
            msq = tl.zeros((BLOCK_N,), tl.float32)
            for dc in range(0, tl.cdiv(D, BLOCK_D)):
                dcur = dc * BLOCK_D + d_offs
                if EVEN_D:
                    d_safe = tl.where(dcur < D, dcur, 0)
                    a = tl.load(
                        X + q_safe[:, None] * stride_x + d_safe[None, :],
                        mask=q_mask[:, None], other=0.0)
                    bmat = tl.load(
                        MU + n_safe[None, :] * stride_m + d_safe[:, None],
                        mask=n_mask[None, :], other=0.0)
                else:
                    d_mask = dcur < D
                    d_safe = tl.where(d_mask, dcur, 0)
                    a = tl.load(
                        X + q_safe[:, None] * stride_x + d_safe[None, :],
                        mask=q_mask[:, None] & d_mask[None, :], other=0.0)
                    bmat = tl.load(
                        MU + n_safe[None, :] * stride_m + d_safe[:, None],
                        mask=n_mask[None, :] & d_mask[:, None], other=0.0)
                acc = tl.dot(a, bmat, acc, input_precision="ieee")
                xsq += tl.sum(a * a, axis=1)
                msq += tl.sum(bmat * bmat, axis=0)

            # Misma composición elemento a elemento que la referencia:
            # d2 = (x_sq + mu_sq) - (2.0 * dot)
            d2 = (xsq[:, None] + msq[None, :]) - 2.0 * acc
            col_key = tl.where(n_mask[None, :], 0xFFFFF - n_offs[None, :], 0)
            key_new = _pack_desc(d2, col_key)

            # Ping sin pong: escribir running + nuevos, ordenar, recortar.
            tl.store(scr_base + tl.arange(0, BLOCK_Q)[:, None] * TOTAL + k_cols[None, :], run)
            tl.store(scr_base + tl.arange(0, BLOCK_Q)[:, None] * TOTAL + (K_PAD + n_cols)[None, :],
                     key_new)
            tl.debug_barrier()
            merged = tl.load(
                scr_base + tl.arange(0, BLOCK_Q)[:, None] * TOTAL + t_cols[None, :],
                mask=t_cols < K_PAD + BLOCK_N, other=-1)
            srt = tl.sort(merged, 1, descending=True)
            tl.store(scr_base + tl.arange(0, BLOCK_Q)[:, None] * TOTAL + t_cols[None, :], srt)
            tl.debug_barrier()
            run = tl.load(scr_base + tl.arange(0, BLOCK_Q)[:, None] * TOTAL + k_cols[None, :])

        # Desempaquetar: índice = 0xFFFFF - (clave & 0xFFFFF).
        idx_out = (0xFFFFF - (run & 0xFFFFF)).to(tl.int64)
        out_offs = q_safe[:, None] * K + k_cols[None, :]
        tl.store(OUT + out_offs, idx_out, mask=q_mask[:, None] & (k_cols < K)[None, :])

    @triton.jit
    def _d2_gemm_kernel(
        X, MU, OUT,
        B, N, D,
        stride_x, stride_m, stride_o,
        BLOCK_Q: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
        EVEN_D: tl.constexpr,
    ):
        # GEMM d2 en Triton (misma fórmula/composición que la referencia) y
        # torch.topk encima: variante de control para el head-to-head.
        pid_q = tl.program_id(0)
        pid_n = tl.program_id(1)
        q_offs = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
        n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        d_offs = tl.arange(0, BLOCK_D)
        q_mask = q_offs < B
        n_mask = n_offs < N
        q_safe = tl.where(q_mask, q_offs, 0)
        n_safe = tl.where(n_mask, n_offs, 0)

        acc = tl.zeros((BLOCK_Q, BLOCK_N), tl.float32)
        xsq = tl.zeros((BLOCK_Q,), tl.float32)
        msq = tl.zeros((BLOCK_N,), tl.float32)
        for dc in range(0, tl.cdiv(D, BLOCK_D)):
            dcur = dc * BLOCK_D + d_offs
            if EVEN_D:
                d_safe = tl.where(dcur < D, dcur, 0)
                a = tl.load(X + q_safe[:, None] * stride_x + d_safe[None, :],
                            mask=q_mask[:, None], other=0.0)
                bmat = tl.load(MU + n_safe[None, :] * stride_m + d_safe[:, None],
                               mask=n_mask[None, :], other=0.0)
            else:
                d_mask = dcur < D
                d_safe = tl.where(d_mask, dcur, 0)
                a = tl.load(X + q_safe[:, None] * stride_x + d_safe[None, :],
                            mask=q_mask[:, None] & d_mask[None, :], other=0.0)
                bmat = tl.load(MU + n_safe[None, :] * stride_m + d_safe[:, None],
                               mask=n_mask[None, :] & d_mask[:, None], other=0.0)
            acc = tl.dot(a, bmat, acc, input_precision="ieee")
            xsq += tl.sum(a * a, axis=1)
            msq += tl.sum(bmat * bmat, axis=0)

        d2 = (xsq[:, None] + msq[None, :]) - 2.0 * acc
        o_offs = q_safe[:, None] * stride_o + n_safe[None, :]
        tl.store(OUT + o_offs, d2, mask=q_mask[:, None] & n_mask[None, :])


def _reference_neighbors(store, x: torch.Tensor, k: int):
    """Delegación exacta en la referencia (incluye su memoización LRU-1)."""
    return store.find_neighbors(x, k)


def _knn_fused(store, x: torch.Tensor, k: int, block_q: int, block_n: int,
               num_warps: int = 4):
    """Camino Triton fused: 1 lanzamiento de kernel, scratch global interno.

    NOTA DE PERFORMANCE (medido, RTX 3090, torch 2.14.1+cu126): el top-k
    incremental (sort bitónico int64 + barreras + scratch por bloque de
    splats) cuesta ~10-15 ms aunque N sea 4k — pierde ~40x contra
    mm+topk. Se conserva como implementación de referencia del enfoque
    fused y para los tests de correctitud; NO es el camino por defecto.
    """
    n = store.n_active
    b = x.shape[0]
    k_pad = 64 if k <= 64 else 128
    # TOTAL debe ser potencia de 2 para tl.arange/tl.sort; con BLOCK_N=128 y
    # K_PAD<=128 alcanza 256. Las columnas >= K_PAD+BLOCK_N se cargan con
    # other=-1 (clave centinela que ordena al fondo).
    total = 256
    grid = (triton.cdiv(b, block_q),)
    out = torch.empty((b, k), dtype=torch.int64, device=x.device)
    scratch = torch.empty((grid[0], block_q, total), dtype=torch.int64, device=x.device)
    even_d = (x.shape[-1] % 64) == 0
    _knn_fused_kernel[grid](
        x, store.mu, out, scratch,
        b, n, x.shape[-1], k,
        x.stride(0), store.mu.stride(0),
        BLOCK_Q=block_q, BLOCK_N=block_n, BLOCK_D=64,
        K_PAD=k_pad, TOTAL=total, EVEN_D=even_d,
        num_warps=num_warps,
    )
    return out


def _knn_mm_topk(store, x: torch.Tensor, k: int, block_q: int, block_n: int,
                 num_warps: int = 4, num_stages: int = 3):
    """GEMM d2 en Triton (ieee, sin TF32) + torch.topk. Camino ganador."""
    n = store.n_active
    b, d = x.shape
    d2 = torch.empty((b, n), dtype=torch.float32, device=x.device)
    grid = (triton.cdiv(b, block_q), triton.cdiv(n, block_n))
    even_d = (d % 64) == 0
    _d2_gemm_kernel[grid](
        x, store.mu, d2,
        b, n, d,
        x.stride(0), store.mu.stride(0), d2.stride(0),
        BLOCK_Q=block_q, BLOCK_N=block_n, BLOCK_D=64, EVEN_D=even_d,
        num_warps=num_warps, num_stages=num_stages,
    )
    return torch.topk(d2, k=k, dim=-1).indices


def _can_fused(store, x: torch.Tensor, k: int) -> bool:
    if not TRITON_FUSED_AVAILABLE:
        return False
    if x.device != store.mu.device or x.device.type != "cuda":
        return False
    if x.dtype != torch.float32 or store.mu.dtype != torch.float32:
        return False
    if x.dim() != 2:
        return False
    if not (1 <= k <= _MAX_K_FUSED):
        return False
    if store.n_active < k or store.n_active >= _MAX_N_PACK:
        return False
    return True


def _auto_plan(b: int, n: int, k: int):
    """Ruteo por ganador MEDIDO (RTX 3090, torch 2.14.1+cu126, Triton 3.8.0,
    fp32 ieee). PRELIMINAR: GPU compartida con otros agentes; el head-to-head
    consolidado posterior manda. Lo observado en 3 corridas (medianas):

        B=8    N=4k     triton 1.26-1.98x más rápido (ganó en las 3 corridas)
        B=64   N=4k     triton 1.5-2.0x más rápido (ganó en las 3 corridas)
        B=1024 N=4k     stock ~1.1-1.6x más rápido
        B=8    N=100k   INCONCLUSO (0.84x-1.54x según corrida)
        B=64   N=100k   stock ~1.5x más rápido (3/3 corridas del bench)
        B=1024 N=100k   stock ~3.5x más rápido (el "1.16x triton" inicial era
                        artefacto de contención sobre el stock)

    Regla conservadora: B <= 64 y N <= 16384 → mm_topk Triton; cualquier otra
    forma → referencia (stock cuBLAS+topk). La zona de victoria robusta del
    kernel Triton es el régimen Langevin (pocas cadenas, pocos splats).
    """
    if b <= 64 and n <= 16384:
        if b <= 16:
            return ("mm_topk", 16, 128, 4, 3)
        return ("mm_topk", 64, 64, 4, 3)
    return ("reference",)


def find_neighbors_triton(store, x: torch.Tensor, k: int, backend: str = "auto"):
    """Drop-in de `store.find_neighbors(x, k)` con kernel Triton.

    Misma salida exacta que `SplatStorage.find_neighbors` (misma semántica de
    top-k y de ties — ver docstring del módulo para la política de ties).
    Incluye memoización LRU-1 con la MISMA clave que la referencia
    (data_ptr/shape/dtype/versions/k/n_active/_mut_counter), de modo que el
    comportamiento cache/invalidación es idéntico al original.

    backend: "auto" (ganador medido por shape; cae a la referencia si el
             workload sale del encuadre), "fused" (1 kernel, PIERDE en
             benchmarks — se expone para reproducir la medición),
             "mm_topk" (GEMM Triton ieee + torch.topk), "reference".
    """
    plan = _auto_plan(x.shape[0], store.n_active, k) if backend == "auto" else None
    if backend == "reference" or plan is not None and plan[0] == "reference":
        return _reference_neighbors(store, x, k)
    if not _can_fused(store, x, k):
        return _reference_neighbors(store, x, k)

    with torch.no_grad():
        # Memoización LRU-1 — misma clave que la referencia (splats.py).
        cache_key = (x.data_ptr(), tuple(x.shape), x.dtype, x._version, k,
                     store.n_active, store.mu._version, store.alpha._version,
                     store.kappa._version, store._mut_counter)
        if cache_key == getattr(store, "_knn_tri_key", None):
            return store._knn_tri_val

        if not x.is_contiguous():
            x = x.contiguous()

        if plan is not None:
            backend, bq, bn, warps, stages = plan
        else:
            bq, bn, warps, stages = (16 if x.shape[0] <= 64 else 32), 128, 4, 3

        if backend == "mm_topk":
            idx = _knn_mm_topk(store, x, k, bq, bn, warps, stages)
        elif backend == "fused":
            idx = _knn_fused(store, x, k, bq, bn, warps)
        else:
            raise ValueError(f"backend desconocido: {backend}")

        # Mismo post-proceso literal que la referencia (incluye el squeeze(-1),
        # que sólo actúa con k==1) y el gather en (mu, alpha, kappa).
        idx_tensor = idx.squeeze(-1)
        neighbors_mu = store.mu[idx_tensor]
        neighbors_alpha = store.alpha[idx_tensor]
        neighbors_kappa = store.kappa[idx_tensor]

        store._knn_tri_key = cache_key
        store._knn_tri_val = (neighbors_mu, neighbors_alpha, neighbors_kappa)
        store._knn_tri_x = x  # pin: mantiene data_ptr único mientras cachea

    return neighbors_mu, neighbors_alpha, neighbors_kappa
