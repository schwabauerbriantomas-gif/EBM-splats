"""Comprehensive pytest tests for EBM-splats core modules."""

import pytest
import torch
import math

# ─── config.py ───

class TestEBMConfig:
    def test_defaults(self):
        from config import EBMConfig
        c = EBMConfig()
        assert c.latent_dim == 640
        assert c.n_splats_init == 10000
        assert c.max_splats == 100000
        assert c.temperature == 0.1
        assert c.vocab_size == 50257
        assert c.device == "cpu"

    def test_custom_config(self):
        from config import EBMConfig
        c = EBMConfig(latent_dim=128, device="cuda", temperature=0.5)
        assert c.latent_dim == 128
        assert c.device == "cuda"
        assert c.temperature == 0.5

    def test_small_dim(self):
        from config import EBMConfig
        c = EBMConfig(latent_dim=2, n_splats_init=5, max_splats=10)
        assert c.latent_dim == 2
        assert c.max_splats == 10

    def test_noise_levels_tuple(self):
        from config import EBMConfig
        c = EBMConfig()
        assert isinstance(c.noise_levels, tuple)
        assert len(c.noise_levels) == 5


# ─── geometry.py ───

class TestGeometry:
    @pytest.fixture
    def points(self):
        torch.manual_seed(42)
        dim = 64
        x = torch.randn(4, dim)
        x = x / x.norm(dim=-1, keepdim=True)
        return x

    def test_exp_map_on_sphere(self, points):
        from geometry import exp_map
        # Project to tangent first for correctness
        v = torch.randn_like(points) * 0.1
        v = v - (v * points).sum(dim=-1, keepdim=True) * points  # project to tangent
        result = exp_map(points, v)
        norms = result.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)

    def test_log_map_tangent(self, points):
        from geometry import log_map, project_to_tangent
        y = torch.randn_like(points)
        y = y / y.norm(dim=-1, keepdim=True)
        log = log_map(points, y)
        # log_map result should be tangent to base point
        dots = (log * points).sum(dim=-1)
        assert torch.allclose(dots, torch.zeros_like(dots), atol=1e-5)

    def test_project_to_tangent_orthogonal(self, points):
        from geometry import project_to_tangent
        v = torch.randn_like(points)
        proj = project_to_tangent(points, v)
        dots = (proj * points).sum(dim=-1)
        assert torch.allclose(dots, torch.zeros_like(dots), atol=1e-5)

    def test_geodesic_symmetry(self, points):
        from geometry import geodesic_distance
        d1 = geodesic_distance(points[:2], points[2:])
        d2 = geodesic_distance(points[2:], points[:2])
        assert torch.allclose(d1, d2, atol=1e-5)

    def test_geodesic_nonneg(self, points):
        from geometry import geodesic_distance
        d = geodesic_distance(points, points)
        assert (d >= 0).all()

    def test_geodesic_identity(self, points):
        from geometry import geodesic_distance
        d = geodesic_distance(points, points)
        assert torch.allclose(d, torch.zeros_like(d), atol=5e-4)

    def test_roundtrip_exp_log(self, points):
        from geometry import exp_map, log_map
        v = torch.randn_like(points) * 0.1
        v = v - (v * points).sum(dim=-1, keepdim=True) * points  # project to tangent
        y = exp_map(points, v)
        v_rec = log_map(points, y)
        cos_sim = (v * v_rec).sum(dim=-1) / (v.norm(dim=-1) * v_rec.norm(dim=-1))
        assert torch.allclose(cos_sim.abs(), torch.ones_like(cos_sim), atol=5e-3)

    def test_1d_input(self):
        from geometry import exp_map, normalize_sphere, project_to_tangent
        torch.manual_seed(0)
        x = torch.randn(64)
        x = normalize_sphere(x)
        v = torch.randn(64) * 0.1
        v = project_to_tangent(x, v)
        result = exp_map(x, v)
        assert result.shape == (64,)
        assert abs(result.norm().item() - 1.0) < 1e-4

    def test_normalize_sphere(self):
        from geometry import normalize_sphere
        x = torch.randn(4, 640)
        result = normalize_sphere(x)
        assert torch.allclose(result.norm(dim=-1), torch.ones(4), atol=1e-5)


# ─── energy.py ───

class TestEnergyFunction:
    @pytest.fixture
    def energy_setup(self):
        torch.manual_seed(42)
        from config import EBMConfig
        from splats import SplatStorage
        from energy import EnergyFunction
        config = EBMConfig(latent_dim=64, n_splats_init=50, max_splats=100,
                          temperature=0.1, knn_k=10)
        splats = SplatStorage(config)
        energy_fn = EnergyFunction(config, splats)
        return config, splats, energy_fn

    def test_near_lower_energy(self, energy_setup):
        config, splats, energy_fn = energy_setup
        center = splats.mu.data[0:1].clone()
        far = torch.randn(1, config.latent_dim)
        far = far / far.norm(dim=-1, keepdim=True)
        e_near = energy_fn.compute_splat_energy(center)
        e_far = energy_fn.compute_splat_energy(far)
        assert e_near.item() <= e_far.item() + 1e-3  # near splat should have low energy

    def test_no_nan(self, energy_setup):
        config, splats, energy_fn = energy_setup
        x = torch.randn(4, config.latent_dim)
        x = x / x.norm(dim=-1, keepdim=True)
        e = energy_fn.compute_splat_energy(x)
        assert not torch.isnan(e).any()

    def test_batch_query(self, energy_setup):
        config, splats, energy_fn = energy_setup
        x = torch.randn(8, config.latent_dim)
        x = x / x.norm(dim=-1, keepdim=True)
        e = energy_fn.compute_splat_energy(x)
        assert e.shape == (8,)

    def test_geom_energy_single_returns_zero(self, energy_setup):
        _, _, energy_fn = energy_setup
        config = energy_fn.config
        x = torch.randn(1, config.latent_dim)
        e = energy_fn.compute_geom_energy(x)
        assert e.item() == 0.0

    def test_context_energy(self, energy_setup):
        config, splats, energy_fn = energy_setup
        B = 4
        x = torch.randn(B, config.latent_dim)
        x = x / x.norm(dim=-1, keepdim=True)
        ctx = {
            'local': torch.randn(B, config.latent_dim),
            'medium': torch.randn(B, config.latent_dim),
            'global': torch.randn(B, config.latent_dim),
        }
        e = energy_fn.compute_context_energy(x, ctx)
        assert e.shape == (B,)

    def test_forward_shape(self, energy_setup):
        config, splats, energy_fn = energy_setup
        x = torch.randn(4, config.latent_dim)
        x = x / x.norm(dim=-1, keepdim=True)
        e = energy_fn(x)
        assert e.shape == (4,)

    def test_score_shape(self, energy_setup):
        config, splats, energy_fn = energy_setup
        x = torch.randn(4, config.latent_dim)
        x = x / x.norm(dim=-1, keepdim=True)
        score = energy_fn.compute_score(x)
        assert score.shape == (4, config.latent_dim)


# ─── splats.py ───

class TestSplatStorage:
    def test_add_and_retrieve(self):
        from config import EBMConfig
        from splats import SplatStorage
        config = EBMConfig(latent_dim=64, n_splats_init=0, max_splats=10)
        store = SplatStorage(config)
        center = torch.randn(64)
        center = center / center.norm()
        added = store.add_splat(center, alpha=2.0, kappa=15.0)
        assert added
        assert store.n_active == 1
        assert store.alpha.data[0].item() == 2.0
        assert store.kappa.data[0].item() == 15.0

    def test_alpha_kappa_defaults(self):
        from config import EBMConfig
        from splats import SplatStorage
        config = EBMConfig(latent_dim=64, n_splats_init=0, max_splats=10, init_alpha=3.0, init_kappa=20.0)
        store = SplatStorage(config)
        store.add_splat(torch.randn(64))
        assert store.alpha.data[0].item() == 3.0
        assert store.kappa.data[0].item() == 20.0

    def test_max_splats(self):
        from config import EBMConfig
        from splats import SplatStorage
        config = EBMConfig(latent_dim=64, n_splats_init=0, max_splats=2)
        store = SplatStorage(config)
        store.add_splat(torch.randn(64))
        store.add_splat(torch.randn(64))
        added = store.add_splat(torch.randn(64))
        assert not added

    def test_find_neighbors(self):
        from config import EBMConfig
        from splats import SplatStorage
        config = EBMConfig(latent_dim=64, n_splats_init=20, max_splats=50, knn_k=5)
        store = SplatStorage(config)
        x = torch.randn(4, 64)
        x = x / x.norm(dim=-1, keepdim=True)
        mu, alpha, kappa = store.find_neighbors(x, 5)
        assert mu.shape == (4, 5, 64)
        assert alpha.shape == (4, 5)
        assert kappa.shape == (4, 5)

    def test_normalize_preserves(self):
        from config import EBMConfig
        from splats import SplatStorage
        config = EBMConfig(latent_dim=64, n_splats_init=10, max_splats=50)
        store = SplatStorage(config)
        store.normalize()
        norms = store.mu.data[:store.n_active].norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_empty_splats(self):
        from config import EBMConfig
        from splats import SplatStorage
        config = EBMConfig(latent_dim=64, n_splats_init=0, max_splats=10)
        store = SplatStorage(config)
        assert store.n_active == 0

    def test_many_splats(self):
        from config import EBMConfig
        from splats import SplatStorage
        config = EBMConfig(latent_dim=64, n_splats_init=0, max_splats=200)
        store = SplatStorage(config)
        for _ in range(200):
            store.add_splat(torch.randn(64))
        assert store.n_active == 200


# ─── model.py ───

class TestEBMModel:
    @pytest.fixture
    def model(self):
        from config import EBMConfig
        from model import EBMModel
        config = EBMConfig(latent_dim=64, n_splats_init=20, max_splats=50,
                          vocab_size=100, hidden_dim=128, langevin_steps=2,
                          knn_k=10)
        return EBMModel(config)

    def test_embed_shape(self, model):
        tokens = torch.randint(0, 100, (2, 4))
        x = model.embed(tokens)
        assert x.shape == (2, 4, 64)

    def test_energy_shape(self, model):
        x = torch.randn(2, 64)
        x = x / x.norm(dim=-1, keepdim=True)
        e = model.compute_energy(x)
        assert e.shape == (2,)

    def test_score_shape(self, model):
        x = torch.randn(2, 64)
        x = x / x.norm(dim=-1, keepdim=True)
        s = model.compute_score(x)
        assert s.shape == (2, 64)

    def test_energy_no_nan(self, model):
        x = torch.randn(4, 64)
        x = x / x.norm(dim=-1, keepdim=True)
        e = model.compute_energy(x)
        assert not torch.isnan(e).any()

    def test_sample_shape(self, model):
        samples = model.sample(n_samples=2)
        assert samples.shape == (2, 64)

    def test_custom_config(self):
        from config import EBMConfig
        from model import EBMModel
        config = EBMConfig(latent_dim=32, n_splats_init=5, max_splats=10,
                          vocab_size=50, hidden_dim=64, langevin_steps=1)
        m = EBMModel(config)
        assert m.config.latent_dim == 32


# ─── score_network.py ───

class TestScoreNetwork:
    @pytest.fixture
    def score_net(self):
        from score_network import ScoreNetwork
        return ScoreNetwork(dim=64, hidden_dim=128, n_layers=2)

    def test_forward_shape(self, score_net):
        x = torch.randn(4, 64)
        sigma = torch.tensor(0.1)
        out = score_net(x, sigma)
        assert out.shape == (4, 64)

    def test_tangent_output(self, score_net):
        from geometry import project_to_tangent
        x = torch.randn(4, 64)
        x = x / x.norm(dim=-1, keepdim=True)
        sigma = torch.tensor(0.5)
        out = score_net(x, sigma)
        # Output should be tangent to x (orthogonal)
        dots = (out * x).sum(dim=-1)
        assert torch.allclose(dots, torch.zeros_like(dots), atol=1e-5)

    def test_gradients_flow(self, score_net):
        x = torch.randn(2, 64, requires_grad=True)
        sigma = torch.tensor(0.1, requires_grad=True)
        out = score_net(x, sigma)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == (2, 64)

    def test_different_batch_sizes(self, score_net):
        sigma = torch.tensor(0.2)
        out1 = score_net(torch.randn(1, 64), sigma)
        out2 = score_net(torch.randn(8, 64), sigma)
        assert out1.shape == (1, 64)
        assert out2.shape == (8, 64)

    def test_sigma_batch(self, score_net):
        x = torch.randn(4, 64)
        sigma = torch.tensor([0.1, 0.2, 0.3, 0.4])
        out = score_net(x, sigma)
        assert out.shape == (4, 64)


# ─── soc.py ───

class TestSOC:
    def test_compute_order_parameter_zero(self):
        from config import EBMConfig
        from splats import SplatStorage
        from soc import compute_order_parameter
        config = EBMConfig(latent_dim=64, n_splats_init=10, max_splats=50)
        splats = SplatStorage(config)
        # No frequency accumulated
        phi = compute_order_parameter(splats, torch.randn(5, 64))
        assert phi == 0.0

    def test_history_buffer(self):
        from soc import HistoryBuffer
        buf = HistoryBuffer(capacity=10, latent_dim=64)
        for i in range(15):
            buf.push(torch.randn(64), torch.tensor(float(i)))
        assert buf.full

    def test_maybe_consolidate_empty(self):
        from config import EBMConfig
        from splats import SplatStorage
        from soc import maybe_consolidate, HistoryBuffer
        config = EBMConfig(latent_dim=64, n_splats_init=10, max_splats=50)
        splats = SplatStorage(config)
        buf = HistoryBuffer(capacity=100, latent_dim=64)
        result = maybe_consolidate(splats, config, buf)
        assert not result  # buffer not full

    def test_langevin_state_stagnation(self):
        from langevin import LangevinState
        state = LangevinState(window=3, epsilon=0.1)
        assert not state.is_stagnated()
        for _ in range(10):
            state.record(1.0)
        assert state.is_stagnated()


# ─── context_hierarchy.py ───

class TestHierarchicalContext:
    @pytest.fixture
    def ctx(self):
        from config import EBMConfig
        from context_hierarchy import HierarchicalContext
        config = EBMConfig(latent_dim=64)
        h = HierarchicalContext(config)
        h.reset(4, torch.device('cpu'))
        return h

    def test_reset(self, ctx):
        assert ctx.c_local.shape == (4, 64)
        assert ctx.c_medium.shape == (4, 64)
        assert ctx.c_global.shape == (4, 64)

    def test_update(self, ctx):
        x = torch.randn(4, 64)
        x = x / x.norm(dim=-1, keepdim=True)
        ctx.update(x)
        assert ctx._step.item() == 1

    def test_get_context(self, ctx):
        result = ctx.get_context()
        assert 'local' in result
        assert 'medium' in result
        assert 'global' in result

    def test_get_combined_context(self, ctx):
        combined = ctx.get_combined_context()
        assert combined.shape == (4, 64)
        norms = combined.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(4), atol=1e-5)

    def test_empty_retrieval(self):
        from config import EBMConfig
        from context_hierarchy import HierarchicalContext
        config = EBMConfig(latent_dim=64)
        h = HierarchicalContext(config)
        # Before reset, buffers exist but have default shape
        result = h.get_context()
        assert isinstance(result, dict)

    def test_context_on_sphere_after_updates(self, ctx):
        from geometry import normalize_sphere
        x = torch.randn(4, 64)
        x = x / x.norm(dim=-1, keepdim=True)
        for _ in range(5):
            ctx.update(x)
        c = ctx.get_combined_context()
        norms = c.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(4), atol=1e-5)


# ─── langevin.py ───

class TestLangevin:
    @pytest.fixture
    def langevin_setup(self):
        torch.manual_seed(42)
        from config import EBMConfig
        from splats import SplatStorage
        from energy import EnergyFunction
        config = EBMConfig(latent_dim=64, n_splats_init=20, max_splats=50,
                          langevin_steps=3, langevin_dt=0.001, langevin_gamma=0.1,
                          knn_k=10)
        splats = SplatStorage(config)
        energy_fn = EnergyFunction(config, splats)
        return config, energy_fn

    def test_sample_no_nan(self, langevin_setup):
        from langevin import sample_langevin
        config, energy_fn = langevin_setup
        x_init = torch.randn(2, config.latent_dim)
        x_init = x_init / x_init.norm(dim=-1, keepdim=True)
        x = sample_langevin(x_init, energy_fn, config)
        assert not torch.isnan(x).any()

    def test_sample_shape(self, langevin_setup):
        from langevin import sample_langevin
        config, energy_fn = langevin_setup
        x_init = torch.randn(4, config.latent_dim)
        x_init = x_init / x_init.norm(dim=-1, keepdim=True)
        x = sample_langevin(x_init, energy_fn, config)
        assert x.shape == (4, config.latent_dim)

    def test_single_step(self, langevin_setup):
        from langevin import langevin_step, LangevinState
        config, energy_fn = langevin_setup
        x = torch.randn(2, config.latent_dim)
        x = x / x.norm(dim=-1, keepdim=True)
        v = torch.zeros_like(x)
        state = LangevinState()
        x_new, v_new = langevin_step(x, v, energy_fn, config, state)
        assert x_new.shape == (2, config.latent_dim)
        assert not torch.isnan(x_new).any()

    def test_different_step_counts(self, langevin_setup):
        from langevin import sample_langevin
        config, energy_fn = langevin_setup
        x_init = torch.randn(2, config.latent_dim)
        x_init = x_init / x_init.norm(dim=-1, keepdim=True)
        config.langevin_steps = 1
        x1 = sample_langevin(x_init, energy_fn, config)
        config.langevin_steps = 5
        x5 = sample_langevin(x_init, energy_fn, config)
        assert x1.shape == x5.shape
        assert not torch.isnan(x5).any()


# ─── evaluation.py ───

class TestEBMEvaluator:
    @pytest.fixture
    def evaluator(self):
        from evaluation import EBMEvaluator
        return EBMEvaluator()

    def test_energy_stats(self, evaluator):
        energy = torch.tensor([1.0, 2.0, 3.0, 4.0])
        stats = evaluator.compute_energy_stats(energy)
        assert stats['energy_mean'] == 2.5
        assert not stats['has_nan']
        assert not stats['has_inf']
        assert stats['is_real']

    def test_energy_stats_nan(self, evaluator):
        energy = torch.tensor([1.0, float('nan'), 3.0])
        stats = evaluator.compute_energy_stats(energy)
        assert stats['has_nan']
        assert not stats['is_real']

    def test_cosine_similarity(self, evaluator):
        samples = torch.randn(5, 64)
        samples = samples / samples.norm(dim=-1, keepdim=True)
        references = samples.clone()
        sim = evaluator.compute_cosine_similarity(samples, references)
        assert 0.0 <= sim <= 1.0
        assert sim > 0.9  # same vectors

    def test_diversity(self, evaluator):
        x = torch.randn(10, 64)
        x = x / x.norm(dim=-1, keepdim=True)
        d = evaluator.compute_diversity(x)
        assert d > 0.0

    def test_diversity_single(self, evaluator):
        x = torch.randn(1, 64)
        d = evaluator.compute_diversity(x)
        assert d == 0.0

    def test_perplexity(self, evaluator):
        energy = torch.tensor([1.0, 2.0])
        ppl = evaluator.estimate_perplexity(energy)
        assert math.isfinite(ppl)
        assert ppl > 0

    def test_coherence_score(self, evaluator):
        tokens = torch.randn(2, 10)
        score = evaluator.compute_coherence_score(tokens)
        assert -1.0 <= score <= 1.0

    def test_coherence_short(self, evaluator):
        tokens = torch.randn(2, 1)  # only 1 token
        score = evaluator.compute_coherence_score(tokens)
        assert score == 0.0

    def test_gradient_health(self, evaluator):
        score = torch.randn(4, 64)
        health = evaluator.check_gradient_health(score)
        assert not health['score_has_nan']
        assert not health['score_is_zero']

    def test_full_report(self, evaluator):
        energy = torch.randn(4)
        score = torch.randn(4, 64)
        samples = torch.randn(4, 64)
        references = torch.randn(10, 64)
        report = evaluator.full_report(energy, score, samples=samples, references=references)
        assert 'energy' in report
        assert 'perplexity' in report
        assert 'gradient' in report
        assert 'diversity' in report

    def test_full_report_minimal(self, evaluator):
        energy = torch.randn(4)
        score = torch.randn(4, 64)
        report = evaluator.full_report(energy, score)
        assert 'energy' in report
        assert 'diversity' not in report
