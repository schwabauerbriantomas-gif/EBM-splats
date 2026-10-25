"""
Triton fused kernels for the EBM splat energy / score hot path (K2).

Replaces the eager chain used by `ebm.energy.EnergyFunction` after KNN:

    reference (per call):
        gather mu/alpha/kappa -> bmm dots -> kappa weights -> log
        -> logsumexp | softmax -> tangent projection [B,K,D]
        -> normalize -> weighted sum -> Riemannian projection

with ONE fused Triton kernel (one program per batch row) that:

    pass 0: row sum of clamped kappa (importance weights)
    pass 1: online (flash-style) logsumexp over the K neighbors
    pass 2: recompute logits, softmax p_k, tangent projection and the
            weighted-sum gradient in registers, then the Riemannian
            projection, storing the final score.

Mathematical semantics are identical to `energy.py` (V2 spec §2.1-2.3):
  - importance = clamp(kappa, 1e-4); w_k = importance / sum(importance)
  - logits_k   = alpha_k * (x·mu_k - 1) / tau + log(clamp(w_k, 1e-8))
  - E_splat    = -logsumexp(logits)                    (energy)
  - p_k        = softmax(logits)
  - t_k        = mu_k - (x·mu_k) x;  t_k /= max(||t_k||, 1e-6)
  - grad       = -sum_k p_k (alpha_k / tau) t_k        (manual, V2 §2.3)
  - score      = -(grad - (x·grad) x)                  (Riemannian, §2.3)

Differences from eager are limited to fp32 summation order (documented
tolerances in tests/kernels/test_energy.py).

CPU / non-fp32 fallback: the same wrapper functions detect CPU tensors,
missing Triton, or non-float32 dtypes and run a pure-torch fallback that
mirrors the reference math on the already-gathered neighbors, so the API
is safe to call unconditionally on any device.
"""

from __future__ import annotations

import torch

from ..config import EBMConfig
from ..energy import EnergyFunction

try:  # Triton ships with torch CUDA builds; absent on CPU-only installs.
    import triton
    import triton.language as tl

    _TRITON_OK = True
except Exception:  # pragma: no cover - exercised on CPU-only environments
    _TRITON_OK = False


if _TRITON_OK:

    @triton.jit
    def _energy_score_kernel(
        x_ptr, mu_ptr, alpha_ptr, kappa_ptr, energy_ptr, out2_ptr,
        K, D, temperature,
        stride_xb, stride_mub, stride_muk, stride_ab, stride_kb, stride_sb,
        BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr,
        KSUM_BLOCK: tl.constexpr, OUTPUT_MODE: tl.constexpr,
    ):
        """One program per batch row: fused weighted-logsumexp energy + score.

        pass 0: sum_k clamp(kappa_k, 1e-4)                     -> sum_imp
        pass 1: online logsumexp of the weighted logits        -> E = -(m + log l)
        pass 2 (OUTPUT_MODE>0): p_k, tangent t_k/||t_k||, weighted sum.
                OUTPUT_MODE 1: + Riemannian projection -> score.
                OUTPUT_MODE 2: raw Euclidean manual gradient (V2 §2.3).

        OUTPUT_MODE: 0 = energy only, 1 = score, 2 = Euclidean gradient.
        Note pass 1 always runs (score needs m, l; mode 0 is energy-only).
        """
        pid = tl.program_id(0)
        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < D
        x = tl.load(x_ptr + pid * stride_xb + offs_d, mask=mask_d, other=0.0)

        # ---- pass 0: importance row sum (clamped kappa) ----
        sum_imp = 0.0
        for k0 in range(0, K, KSUM_BLOCK):
            offs_k = k0 + tl.arange(0, KSUM_BLOCK)
            mask_k = offs_k < K
            kap = tl.load(kappa_ptr + pid * stride_kb + offs_k,
                          mask=mask_k, other=0.0)
            imp = tl.where(mask_k, tl.maximum(kap, 1e-4), 0.0)
            sum_imp += tl.sum(imp, axis=0)

        # ---- pass 1: online logsumexp over neighbor chunks ----
        m = -1e30
        l = 0.0
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < K
            mu = tl.load(mu_ptr + pid * stride_mub + offs_k[:, None] * stride_muk
                         + offs_d[None, :],
                         mask=mask_k[:, None] & mask_d[None, :], other=0.0)
            dots = tl.sum(mu * x[None, :], axis=1)  # [BLOCK_K]
            a = tl.load(alpha_ptr + pid * stride_ab + offs_k, mask=mask_k, other=0.0)
            kap = tl.load(kappa_ptr + pid * stride_kb + offs_k, mask=mask_k, other=0.0)
            w = tl.maximum(kap, 1e-4) / sum_imp
            logits = a * (dots - 1.0) / temperature + tl.log(tl.maximum(w, 1e-8))
            logits = tl.where(mask_k, logits, -1e30)
            m_blk = tl.max(logits, axis=0)
            m_new = tl.maximum(m, m_blk)
            l = l * tl.exp(m - m_new) + tl.sum(tl.exp(logits - m_new), axis=0)
            m = m_new

        tl.store(energy_ptr + pid, -(m + tl.log(l)))

        # ---- pass 2: fused gradient chain ----
        if OUTPUT_MODE > 0:
            grad = tl.zeros([BLOCK_D], dtype=tl.float32)
            for k0 in range(0, K, BLOCK_K):
                offs_k = k0 + tl.arange(0, BLOCK_K)
                mask_k = offs_k < K
                mu = tl.load(mu_ptr + pid * stride_mub + offs_k[:, None] * stride_muk
                             + offs_d[None, :],
                             mask=mask_k[:, None] & mask_d[None, :], other=0.0)
                dots = tl.sum(mu * x[None, :], axis=1)
                a = tl.load(alpha_ptr + pid * stride_ab + offs_k, mask=mask_k, other=0.0)
                kap = tl.load(kappa_ptr + pid * stride_kb + offs_k, mask=mask_k, other=0.0)
                w = tl.maximum(kap, 1e-4) / sum_imp
                logits = a * (dots - 1.0) / temperature + tl.log(tl.maximum(w, 1e-8))
                p = tl.where(mask_k, tl.exp(logits - m) / l, 0.0)
                t = mu - dots[:, None] * x[None, :]          # tangent projection
                tn = tl.sqrt(tl.sum(t * t, axis=1))
                tn = tl.maximum(tn, 1e-6)                    # stability clip (V2 §2.3)
                coeff = -p * (a / temperature) / tn
                grad += tl.sum(coeff[:, None] * t, axis=0)
            if OUTPUT_MODE == 1:                             # Riemannian score
                x_dot_grad = tl.sum(x * grad, axis=0)
                grad = -(grad - x_dot_grad * x)
            tl.store(out2_ptr + pid * stride_sb + offs_d, grad, mask=mask_d)


def _use_triton(*tensors) -> bool:
    """Gate: Triton available, CUDA fp32 tensors, contiguous last dim."""
    if not _TRITON_OK:
        return False
    for t in tensors:
        if not (t.is_cuda and t.dtype == torch.float32):
            return False
    return tensors[0].stride(-1) == 1


# ---------------------------------------------------------------------------
# Pure-torch fallbacks (reference math on already-gathered neighbors).
# ---------------------------------------------------------------------------

def _torch_splat_energy(neighbors_mu, neighbors_alpha, neighbors_kappa, x, temperature):
    dots = torch.bmm(neighbors_mu, x.unsqueeze(-1)).squeeze(-1)
    importance = neighbors_kappa.clamp(min=1e-4)
    weights = importance / importance.sum(dim=-1, keepdim=True)
    exponent = neighbors_alpha * (dots - 1.0) / temperature
    weighted = exponent + torch.log(weights.clamp(min=1e-8))
    return -torch.logsumexp(weighted, dim=-1)


def _torch_splat_energy_grad(neighbors_mu, neighbors_alpha, neighbors_kappa, x, temperature):
    dots = torch.bmm(neighbors_mu, x.unsqueeze(-1)).squeeze(-1)
    importance = neighbors_kappa.clamp(min=1e-4)
    weights = importance / importance.sum(dim=-1, keepdim=True)
    exponents = neighbors_alpha * (dots - 1.0) / temperature
    weighted_logits = exponents + torch.log(weights.clamp(min=1e-8))
    p_k = torch.softmax(weighted_logits, dim=-1)
    x_dot_mu = dots.unsqueeze(-1)
    mu_tangent = neighbors_mu - x_dot_mu * x.unsqueeze(1)
    mu_tangent_norm = mu_tangent.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    mu_tangent_stable = mu_tangent / mu_tangent_norm
    coeff = p_k * (neighbors_alpha / temperature)
    return -(coeff.unsqueeze(-1) * mu_tangent_stable).sum(dim=1)


def _torch_riemannian_project(grad, x):
    x_dot_grad = (x * grad).sum(dim=-1, keepdim=True)
    return -(grad - x_dot_grad * x)


# ---------------------------------------------------------------------------
# Triton launchers
# ---------------------------------------------------------------------------

def _launch(x, mu, alpha, kappa, temperature, output_mode):
    """Run the fused kernel. x [B,D], mu [B,K,D], alpha/kappa [B,K], fp32 CUDA.

    output_mode: 0 = energy, 1 = Riemannian score, 2 = Euclidean manual grad.
    """
    x = x.contiguous()
    mu = mu.contiguous()
    alpha = alpha.contiguous()
    kappa = kappa.contiguous()
    B, K, D = mu.shape
    energy = torch.empty(B, device=x.device, dtype=torch.float32)
    if output_mode > 0:
        out2 = torch.empty(B, D, device=x.device, dtype=torch.float32)
    else:  # dummy tensor: pointer unused when OUTPUT_MODE == 0
        out2 = energy
    block_d = triton.next_power_of_2(D)
    block_k = 16 if block_d <= 1024 else 8
    _energy_score_kernel[(B,)](
        x, mu, alpha, kappa, energy, out2,
        K, D, float(temperature),
        x.stride(0), mu.stride(0), mu.stride(1),
        alpha.stride(0), kappa.stride(0),
        out2.stride(0) if output_mode > 0 else 0,
        BLOCK_K=block_k, BLOCK_D=block_d, KSUM_BLOCK=64,
        OUTPUT_MODE=output_mode, num_warps=8,
    )
    return energy, (out2 if output_mode > 0 else None)


# ---------------------------------------------------------------------------
# Public dispatchers (same semantics as the reference EnergyFunction methods).
# ---------------------------------------------------------------------------

def kernel_splat_energy(neighbors_mu, neighbors_alpha, neighbors_kappa, x, temperature):
    """E_splat from pre-gathered neighbors == reference compute_splat_energy math."""
    if _use_triton(x, neighbors_mu, neighbors_alpha, neighbors_kappa):
        energy, _ = _launch(x, neighbors_mu, neighbors_alpha, neighbors_kappa,
                            temperature, output_mode=0)
        return energy
    return _torch_splat_energy(neighbors_mu, neighbors_alpha, neighbors_kappa,
                               x, temperature)


def kernel_splat_energy_grad(neighbors_mu, neighbors_alpha, neighbors_kappa, x, temperature):
    """Euclidean manual gradient == reference compute_splat_energy_grad math."""
    if _use_triton(x, neighbors_mu, neighbors_alpha, neighbors_kappa):
        _, grad = _launch(x, neighbors_mu, neighbors_alpha, neighbors_kappa,
                          temperature, output_mode=2)
        return grad
    return _torch_splat_energy_grad(neighbors_mu, neighbors_alpha, neighbors_kappa,
                                    x, temperature)


def kernel_score(neighbors_mu, neighbors_alpha, neighbors_kappa, x, temperature):
    """Riemannian score == reference compute_score math (on gathered neighbors)."""
    if _use_triton(x, neighbors_mu, neighbors_alpha, neighbors_kappa):
        _, score = _launch(x, neighbors_mu, neighbors_alpha, neighbors_kappa,
                           temperature, output_mode=1)
        return score
    grad = _torch_splat_energy_grad(neighbors_mu, neighbors_alpha, neighbors_kappa,
                                    x, temperature)
    return _torch_riemannian_project(grad, x)


def kernel_energy_and_score(neighbors_mu, neighbors_alpha, neighbors_kappa, x, temperature):
    """Both outputs from ONE kernel launch (the fused win for Langevin loops)."""
    if _use_triton(x, neighbors_mu, neighbors_alpha, neighbors_kappa):
        energy, score = _launch(x, neighbors_mu, neighbors_alpha, neighbors_kappa,
                                temperature, output_mode=1)
        return energy, score
    energy = _torch_splat_energy(neighbors_mu, neighbors_alpha, neighbors_kappa,
                                 x, temperature)
    grad = _torch_splat_energy_grad(neighbors_mu, neighbors_alpha, neighbors_kappa,
                                    x, temperature)
    return energy, _torch_riemannian_project(grad, x)


# ---------------------------------------------------------------------------
# Drop-in EnergyFunction subclass: identical signatures, kernel-backed.
# ---------------------------------------------------------------------------

class FusedEnergyFunction(EnergyFunction):
    """EnergyFunction with the splat hot path fused into Triton kernels.

    Same constructor/method signatures as `EnergyFunction`; the KNN lookup,
    geom/comp/context energies and all calling conventions are inherited
    unchanged — only the post-KNN splat math runs fused. On CPU (or with
    Triton unavailable) every method transparently falls back to the
    reference torch math.
    """

    def _neighbors(self, x: torch.Tensor):
        original_shape = x.shape[:-1]
        if x.dim() > 2:
            x = x.reshape(-1, x.size(-1))
        mu, alpha, kappa = self.splats.find_neighbors(x, self.config.knn_k)
        return x, mu, alpha, kappa, original_shape

    def compute_splat_energy(self, x: torch.Tensor) -> torch.Tensor:
        x, mu, alpha, kappa, original_shape = self._neighbors(x)
        energy = kernel_splat_energy(mu, alpha, kappa, x, self.config.temperature)
        return energy.view(original_shape)

    def compute_splat_energy_grad(self, x: torch.Tensor) -> torch.Tensor:
        x, mu, alpha, kappa, original_shape = self._neighbors(x)
        grad = kernel_splat_energy_grad(mu, alpha, kappa, x, self.config.temperature)
        return grad.view(original_shape + (self.config.latent_dim,))

    def compute_score(self, x: torch.Tensor) -> torch.Tensor:
        x_flat = x.reshape(-1, self.config.latent_dim)
        x_flat, mu, alpha, kappa, _ = self._neighbors(x_flat)
        return kernel_score(mu, alpha, kappa, x_flat, self.config.temperature)

    def fused_energy_and_score(self, x: torch.Tensor):
        """Single-KNN + single-kernel energy AND score (Langevin hot loop)."""
        x_flat = x.reshape(-1, self.config.latent_dim)
        x_flat, mu, alpha, kappa, _ = self._neighbors(x_flat)
        energy, score = kernel_energy_and_score(mu, alpha, kappa, x_flat,
                                                self.config.temperature)
        return energy, score


def make_fused_energy_fn(config: EBMConfig, splat_store) -> FusedEnergyFunction:
    """Factory mirroring `EnergyFunction(config, splat_store)` construction."""
    return FusedEnergyFunction(config, splat_store)
