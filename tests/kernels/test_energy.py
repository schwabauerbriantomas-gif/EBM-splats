"""
GPU correctness tests: Triton fused energy/score kernels vs reference.

Compares `ebm.kernels.FusedEnergyFunction` against the reference
`ebm.energy.EnergyFunction` on the SAME splat store and the SAME seeded
inputs (seed 42), so the only difference under test is the fused math.

Tolerances (float32, justified):
  The kernel computes dot products and reductions in a different summation
  order than cuBLAS bmm / torch.logsumexp / torch.softmax. For D=640 and
  K=64 fp32 accumulation, measured max diffs on this workload are ~1e-6
  (energy) and ~1e-5 (score/grad); the bounds below leave >=2 orders of
  magnitude of headroom while still catching real semantic drift
  (sign flips, wrong normalization, misapplied masks show up at >=1e-2).
    energy: atol 1e-4, rtol 1e-4   (values O(0.1-10))
    score/grad: atol 5e-4, rtol 5e-4 (values O(0.1-30))

The autograd test uses a controlled case where the reference's manual
stabilized gradient provably equals the true gradient: all neighbor splats
orthogonal to x with equal alpha/kappa (tangent norms are exactly 1 and the
softmax is uniform, so manual grad == tangent-projected autograd grad).
"""

import math

import pytest
import torch

from ebm.config import EBMConfig
from ebm.energy import EnergyFunction
from ebm.kernels import (
    FusedEnergyFunction,
    kernel_energy_and_score,
    kernel_score,
    kernel_splat_energy,
    kernel_splat_energy_grad,
)
from ebm.splats import SplatStorage

SEED = 42
LATENT_DIM = 640
N_INIT = 2000
MAX_SPLATS = 4000
KNN_K = 64

BATCHES = [8, 64, 1024]

ENERGY_ATOL = 1e-4
ENERGY_RTOL = 1e-4
SCORE_ATOL = 5e-4
SCORE_RTOL = 5e-4

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requiere GPU CUDA"
)


def _seeded_setup(batch: int, device: str):
    torch.manual_seed(SEED)
    config = EBMConfig(latent_dim=LATENT_DIM, n_splats_init=N_INIT,
                       max_splats=MAX_SPLATS, knn_k=KNN_K)
    splats = SplatStorage(config).to(device)
    ref = EnergyFunction(config, splats).to(device)
    fused = FusedEnergyFunction(config, splats).to(device)
    fused.load_state_dict(ref.state_dict())  # W_comp parity for full forward

    x = torch.randn(batch, LATENT_DIM, device=device)
    x = x / x.norm(dim=-1, keepdim=True)
    return config, splats, ref, fused, x


@requires_cuda
@pytest.mark.parametrize("batch", BATCHES)
def test_splat_energy_vs_reference(batch):
    _, _, ref, fused, x = _seeded_setup(batch, "cuda")
    with torch.no_grad():
        e_ref = ref.compute_splat_energy(x)
        e_ker = fused.compute_splat_energy(x)
    assert e_ref.shape == e_ker.shape
    torch.testing.assert_close(e_ker, e_ref, atol=ENERGY_ATOL, rtol=ENERGY_RTOL)


@requires_cuda
@pytest.mark.parametrize("batch", BATCHES)
def test_score_vs_reference(batch):
    _, _, ref, fused, x = _seeded_setup(batch, "cuda")
    with torch.no_grad():
        s_ref = ref.compute_score(x)
        s_ker = fused.compute_score(x)
    assert s_ref.shape == s_ker.shape == (batch, LATENT_DIM)
    torch.testing.assert_close(s_ker, s_ref, atol=SCORE_ATOL, rtol=SCORE_RTOL)


@requires_cuda
@pytest.mark.parametrize("batch", BATCHES)
def test_splat_energy_grad_vs_reference(batch):
    _, _, ref, fused, x = _seeded_setup(batch, "cuda")
    with torch.no_grad():
        g_ref = ref.compute_splat_energy_grad(x)
        g_ker = fused.compute_splat_energy_grad(x)
    torch.testing.assert_close(g_ker, g_ref, atol=SCORE_ATOL, rtol=SCORE_RTOL)


@requires_cuda
@pytest.mark.parametrize("batch", BATCHES)
def test_energy_and_score_fused_vs_reference(batch):
    """One fused launch must match both reference outputs simultaneously."""
    _, _, ref, fused, x = _seeded_setup(batch, "cuda")
    with torch.no_grad():
        e_ref = ref.compute_splat_energy(x)
        s_ref = ref.compute_score(x)
        e_ker, s_ker = fused.fused_energy_and_score(x)
    torch.testing.assert_close(e_ker, e_ref, atol=ENERGY_ATOL, rtol=ENERGY_RTOL)
    torch.testing.assert_close(s_ker, s_ref, atol=SCORE_ATOL, rtol=SCORE_RTOL)


@requires_cuda
@pytest.mark.parametrize("batch", BATCHES)
def test_full_forward_vs_reference(batch):
    """Full forward: fused splat energy + identical geom/comp terms."""
    _, _, ref, fused, x = _seeded_setup(batch, "cuda")
    with torch.no_grad():
        e_ref = ref.forward(x)
        e_ker = fused.forward(x)
    torch.testing.assert_close(e_ker, e_ref, atol=ENERGY_ATOL, rtol=ENERGY_RTOL)


@requires_cuda
@pytest.mark.parametrize("batch", [8, 64])
def test_forward_3d_input_shape_preserved(batch):
    """Reference accepts [B, S, D] (D = latent_dim) and views back; ditto wrapper."""
    _, _, ref, fused, x = _seeded_setup(batch * 4, "cuda")  # x: [B*4, D]
    x3 = x.view(batch, 4, LATENT_DIM)                        # [B, S, D]
    with torch.no_grad():
        e_ref = ref.forward(x3)
        e_ker = fused.forward(x3)
        s_ref = ref.compute_score(x3)
        s_ker = fused.compute_score(x3)
    assert e_ref.shape == e_ker.shape == (batch, 4)
    assert s_ref.shape == s_ker.shape == (batch * 4, LATENT_DIM)
    torch.testing.assert_close(e_ker, e_ref, atol=ENERGY_ATOL, rtol=ENERGY_RTOL)
    torch.testing.assert_close(s_ker, s_ref, atol=SCORE_ATOL, rtol=SCORE_RTOL)


@requires_cuda
def test_kernel_dispatchers_on_gathered_neighbors():
    """kernel_* functions match reference math on identical neighbor inputs."""
    _, splats, ref, _, x = _seeded_setup(64, "cuda")
    with torch.no_grad():
        mu, alpha, kappa = splats.find_neighbors(x, KNN_K)
        temp = ref.config.temperature
        torch.testing.assert_close(
            kernel_splat_energy(mu, alpha, kappa, x, temp),
            ref.compute_splat_energy(x), atol=ENERGY_ATOL, rtol=ENERGY_RTOL)
        torch.testing.assert_close(
            kernel_splat_energy_grad(mu, alpha, kappa, x, temp),
            ref.compute_splat_energy_grad(x), atol=SCORE_ATOL, rtol=SCORE_RTOL)
        torch.testing.assert_close(
            kernel_score(mu, alpha, kappa, x, temp),
            ref.compute_score(x), atol=SCORE_ATOL, rtol=SCORE_RTOL)
        e_k, s_k = kernel_energy_and_score(mu, alpha, kappa, x, temp)
        torch.testing.assert_close(e_k, ref.compute_splat_energy(x),
                                   atol=ENERGY_ATOL, rtol=ENERGY_RTOL)
        torch.testing.assert_close(s_k, ref.compute_score(x),
                                   atol=SCORE_ATOL, rtol=SCORE_RTOL)


@requires_cuda
def test_gradient_vs_autograd_controlled_case():
    """Autograd gradient == kernel score in a controlled case.

    Construction: all K neighbor splats are ORTHOGONAL to x with equal
    alpha/kappa. Then ||t_k|| = ||mu_k|| = 1 (tangent normalization is a
    no-op), softmax p_k = 1/K (uniform), and the manual gradient's tangent
    projection is the identity (x·grad = 0), so
        manual grad == autograd grad of E_splat
    exactly, and kernel score must equal -autograd_grad projected.
    """
    device = "cuda"
    dim, k = 8, 4
    torch.manual_seed(SEED)
    config = EBMConfig(latent_dim=dim, n_splats_init=k, max_splats=k, knn_k=k)
    splats = SplatStorage(config).to(device)

    with torch.no_grad():
        rnd = torch.randn(k, dim - 1, device=device)
        rnd = rnd / rnd.norm(dim=-1, keepdim=True)
        mu = torch.zeros(k, dim, device=device)
        mu[:, 1:] = rnd                      # orthogonal to x = e_0
        splats.mu.data.copy_(mu)
        splats.alpha.data.fill_(2.0)
        splats.kappa.data.fill_(10.0)

    x = torch.zeros(1, dim, device=device)
    x[0, 0] = 1.0
    ref = EnergyFunction(config, splats).to(device)
    fused = FusedEnergyFunction(config, splats).to(device)
    fused.load_state_dict(ref.state_dict())

    # (a) sanity of the controlled case: reference manual grad == autograd.
    xg = x.clone().requires_grad_(True)
    with torch.no_grad():
        nb_mu, nb_alpha, nb_kappa = splats.find_neighbors(x, k)
    dots = torch.bmm(nb_mu, xg.unsqueeze(-1)).squeeze(-1)
    importance = nb_kappa.clamp(min=1e-4)
    weights = importance / importance.sum(dim=-1, keepdim=True)
    logits = nb_alpha * (dots - 1.0) / config.temperature \
        + torch.log(weights.clamp(min=1e-8))
    (-torch.logsumexp(logits, dim=-1)).sum().backward()
    g_auto = xg.grad
    g_manual = ref.compute_splat_energy_grad(x)
    torch.testing.assert_close(g_manual, g_auto, atol=1e-5, rtol=1e-4)

    # (b) kernel score == -Riemannian projection of the autograd gradient.
    score_ker = fused.compute_score(x)
    g_riemann = g_auto - (x * g_auto).sum(dim=-1, keepdim=True) * x
    torch.testing.assert_close(score_ker, -g_riemann, atol=SCORE_ATOL,
                               rtol=SCORE_RTOL)


class TestCPUFallback:
    """CPU tensors / no-Triton path must reproduce reference math exactly."""

    def _setup(self, batch):
        torch.manual_seed(SEED)
        config = EBMConfig(latent_dim=64, n_splats_init=32, max_splats=64,
                           knn_k=8)
        splats = SplatStorage(config).to("cpu")
        ref = EnergyFunction(config, splats)
        fused = FusedEnergyFunction(config, splats)
        fused.load_state_dict(ref.state_dict())
        x = torch.randn(batch, 64)
        x = x / x.norm(dim=-1, keepdim=True)
        return ref, fused, x

    def test_cpu_energy(self):
        ref, fused, x = self._setup(16)
        with torch.no_grad():
            torch.testing.assert_close(fused.compute_splat_energy(x),
                                       ref.compute_splat_energy(x),
                                       atol=1e-6, rtol=1e-5)

    def test_cpu_score(self):
        ref, fused, x = self._setup(16)
        with torch.no_grad():
            torch.testing.assert_close(fused.compute_score(x),
                                       ref.compute_score(x),
                                       atol=1e-5, rtol=1e-5)

    def test_cpu_energy_grad(self):
        ref, fused, x = self._setup(16)
        with torch.no_grad():
            torch.testing.assert_close(fused.compute_splat_energy_grad(x),
                                       ref.compute_splat_energy_grad(x),
                                       atol=1e-5, rtol=1e-5)

    def test_cpu_forward(self):
        ref, fused, x = self._setup(16)
        with torch.no_grad():
            torch.testing.assert_close(fused.forward(x), ref.forward(x),
                                       atol=1e-6, rtol=1e-5)

    @requires_cuda
    def test_fp32_cuda_guard(self):
        """Non-fp32 CUDA inputs take the torch fallback at dispatcher level.

        (find_neighbors itself rejects mixed dtypes — reference limitation —
        so the guard is exercised on pre-gathered neighbors.)
        """
        torch.manual_seed(SEED)
        config = EBMConfig(latent_dim=64, n_splats_init=32, max_splats=64,
                           knn_k=8)
        splats = SplatStorage(config).to("cuda")
        ref = EnergyFunction(config, splats).to("cuda")
        x = torch.randn(16, 64, device="cuda")
        x = x / x.norm(dim=-1, keepdim=True)
        with torch.no_grad():
            mu, alpha, kappa = splats.find_neighbors(x, 8)
            s32 = kernel_score(mu, alpha, kappa, x, config.temperature)
            s16 = kernel_score(mu.half(), alpha.half(), kappa.half(),
                               x.half(), config.temperature)
        assert s16.dtype == torch.float16
        assert torch.isfinite(s16).all()
        # same math, fp16 accumulation noise only
        torch.testing.assert_close(s16.float(), s32, atol=1e-2, rtol=1e-2)
