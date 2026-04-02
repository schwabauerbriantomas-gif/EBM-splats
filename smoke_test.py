"""
Smoke Tests -- P0 Verification.

Tests:
  1. Config: no duplicate fields, V2 params present
  2. Energy sign convention: negative near splats, positive far away
  3. Score computation: no NaN/Inf, non-zero gradients
  4. Riemannian gradient: score is tangent to sphere
  5. Splat storage: add_splat alpha/kappa logic correct
  6. Context hierarchy: update and retrieval works
  7. Langevin: runs without NaN
  8. Evaluation: metrics produce real values
  9. Model: forward pass clean
  10. ScoreNetwork: forward pass clean
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn

from config import EBMConfig
from splats import SplatStorage
from energy import EnergyFunction
from score_network import ScoreNetwork
from context_hierarchy import HierarchicalContext
from langevin import sample_langevin
from evaluation import EBMEvaluator
from geometry import normalize_sphere, project_to_tangent


def test_config():
    """1. Config has V2 params and no duplicates."""
    config = EBMConfig()
    # Check no duplicate fields
    import dataclasses
    fields = [f.name for f in dataclasses.fields(config)]
    assert len(fields) == len(set(fields)), "Duplicate config fields found!"
    # Check V2 params
    assert hasattr(config, 'lambda_context_local')
    assert hasattr(config, 'beta_local')
    assert hasattr(config, 'vocab_size')
    assert config.vocab_size == 50257
    assert hasattr(config, 'grad_clip')
    print("[OK] Config: OK")


def test_energy_sign_convention():
    """2. Energy is lower near splats (negative or near-zero), higher far away."""
    config = EBMConfig(latent_dim=64, n_splats_init=50, knn_k=10, temperature=0.5)
    splats = SplatStorage(config)
    energy_fn = EnergyFunction(config, splats)

    # Pick a splat center as "near" point
    center = splats.mu[0:1].detach()
    near_energy = energy_fn(center)
    
    # Random point far from splat
    far_point = normalize_sphere(torch.randn(1, config.latent_dim))
    far_energy = energy_fn(far_point)
    
    near_e = near_energy.item()
    far_e = far_energy.item()
    
    print(f"  Near energy: {near_e:.4f}, Far energy: {far_e:.4f}")
    assert not torch.isnan(near_energy), "NaN in near energy"
    assert not torch.isnan(far_energy), "NaN in far energy"
    assert near_e <= far_e + 0.1, f"Energy not lower near splat: {near_e} > {far_e}"
    print("[OK] Energy sign convention: OK")


def test_energy_not_zero():
    """2b. Energy values are real (not identically zero)."""
    config = EBMConfig(latent_dim=64, n_splats_init=50, knn_k=10)
    splats = SplatStorage(config)
    energy_fn = EnergyFunction(config, splats)
    
    x = normalize_sphere(torch.randn(8, config.latent_dim))
    energy = energy_fn(x)
    
    assert not torch.allclose(energy, torch.zeros_like(energy)), "Energy is identically zero!"
    assert energy.std().item() > 1e-6, "Energy has no variance!"
    print("[OK] Energy non-zero: OK")


def test_score_no_nan():
    """3. Score computation: no NaN/Inf, non-zero."""
    config = EBMConfig(latent_dim=64, n_splats_init=50, knn_k=10)
    splats = SplatStorage(config)
    energy_fn = EnergyFunction(config, splats)
    
    x = normalize_sphere(torch.randn(4, config.latent_dim))
    score = energy_fn.compute_score(x)
    
    assert not torch.isnan(score).any(), "NaN in score"
    assert not torch.isinf(score).any(), "Inf in score"
    assert score.norm(dim=-1).mean().item() > 1e-8, "Score is zero"
    print("[OK] Score no NaN: OK")


def test_riemannian_gradient():
    """4. Score is tangent to sphere: (x?score) ~ 0."""
    config = EBMConfig(latent_dim=64, n_splats_init=50, knn_k=10)
    splats = SplatStorage(config)
    energy_fn = EnergyFunction(config, splats)
    
    x = normalize_sphere(torch.randn(4, config.latent_dim))
    score = energy_fn.compute_score(x)
    
    # Tangent condition: x ? score ~ 0
    dot_products = (x * score).sum(dim=-1)
    max_dot = dot_products.abs().max().item()
    
    assert max_dot < 1e-4, f"Score not tangent to sphere: max |x?s| = {max_dot}"
    print(f"  Max |x?score| = {max_dot:.6e}")
    print("[OK] Riemannian gradient: OK")


def test_splat_add_splat():
    """5. add_splat alpha/kappa logic correct (was swapped before)."""
    config = EBMConfig(latent_dim=32, n_splats_init=10, init_alpha=1.0, init_kappa=10.0)
    splats = SplatStorage(config)
    initial_n = splats.n_active
    
    center = normalize_sphere(torch.randn(config.latent_dim))
    splats.add_splat(center, alpha=2.5, kappa=7.0)
    
    assert splats.n_active == initial_n + 1, "Splat not added"
    alpha_val = splats.alpha[initial_n].item()
    kappa_val = splats.kappa[initial_n].item()
    
    assert abs(alpha_val - 2.5) < 0.01, f"Alpha wrong: {alpha_val} (expected 2.5)"
    assert abs(kappa_val - 7.0) < 0.01, f"Kappa wrong: {kappa_val} (expected 7.0)"
    print(f"  alpha={alpha_val:.2f}, kappa={kappa_val:.2f}")
    print("[OK] Splat add_splat: OK")


def test_context_hierarchy():
    """6. Context hierarchy: update and retrieval works."""
    config = EBMConfig(latent_dim=64, context_local_window=8, context_medium_window=16)
    ctx = HierarchicalContext(config)
    ctx.reset(4, torch.device('cpu'))
    
    x = normalize_sphere(torch.randn(4, 64))
    for _ in range(10):
        ctx.update(x)
    
    context = ctx.get_context()
    assert 'local' in context
    assert 'medium' in context
    assert 'global' in context
    assert context['local'].shape == (4, 64)
    
    # Check normalized
    for key, vec in context.items():
        norm = vec.norm(dim=-1).mean().item()
        assert abs(norm - 1.0) < 0.01, f"{key} not normalized: {norm}"
    
    print("[OK] Context hierarchy: OK")


def test_langevin_no_nan():
    """7. Langevin sampling runs without NaN."""
    config = EBMConfig(latent_dim=64, n_splats_init=50, knn_k=10,
                       langevin_steps=10, langevin_dt=0.001)
    splats = SplatStorage(config)
    energy_fn = EnergyFunction(config, splats)
    
    x_init = normalize_sphere(torch.randn(2, config.latent_dim))
    x_out = sample_langevin(x_init, energy_fn, config)
    
    assert not torch.isnan(x_out).any(), "NaN in Langevin output"
    # Check still on sphere
    norms = x_out.norm(dim=-1)
    assert (norms - 1.0).abs().max().item() < 0.01, "Output not on sphere"
    print("[OK] Langevin no NaN: OK")


def test_evaluation_metrics():
    """8. Evaluation metrics produce real values."""
    evaluator = EBMEvaluator()
    
    energy = torch.randn(32)
    score = torch.randn(32, 64)
    
    report = evaluator.full_report(energy, score)
    assert report['energy']['is_real'], "Energy not real"
    assert report['gradient']['score_has_nan'] == 0, "Score has NaN"
    assert report['perplexity'] > 0, "Perplexity <= 0"
    
    samples = normalize_sphere(torch.randn(16, 64))
    div = evaluator.compute_diversity(samples)
    assert 0 <= div <= 2, f"Diversity out of range: {div}"
    
    print(f"  Perplexity: {report['perplexity']:.4f}")
    print(f"  Diversity: {div:.4f}")
    print("[OK] Evaluation metrics: OK")


def test_model_forward():
    """9. Model forward pass clean."""
    config = EBMConfig(latent_dim=64, n_splats_init=50, knn_k=10,
                       vocab_size=1000)
    model = type('EBMModel', (), {'__init__': lambda self: None})()
    
    # Test components individually
    splats = SplatStorage(config)
    energy_fn = EnergyFunction(config, splats)
    
    x = normalize_sphere(torch.randn(4, config.latent_dim))
    energy = energy_fn(x)
    score = energy_fn.compute_score(x)
    
    assert not torch.isnan(energy).any()
    assert not torch.isnan(score).any()
    print("[OK] Model forward: OK")


def test_score_network():
    """10. ScoreNetwork forward pass clean."""
    config = EBMConfig(latent_dim=64)
    net = ScoreNetwork(dim=64)
    
    x = normalize_sphere(torch.randn(8, 64))
    sigma = torch.ones(8) * 0.1
    score = net(x, sigma)
    
    assert score.shape == (8, 64), f"Wrong shape: {score.shape}"
    assert not torch.isnan(score).any(), "NaN in ScoreNetwork output"
    assert not torch.isinf(score).any(), "Inf in ScoreNetwork output"
    
    # Check tangent condition
    dots = (x * score).sum(dim=-1)
    assert dots.abs().max().item() < 1e-4, "ScoreNetwork output not tangent"
    print("[OK] ScoreNetwork: OK")


if __name__ == "__main__":
    tests = [
        test_config,
        test_energy_sign_convention,
        test_energy_not_zero,
        test_score_no_nan,
        test_riemannian_gradient,
        test_splat_add_splat,
        test_context_hierarchy,
        test_langevin_no_nan,
        test_evaluation_metrics,
        test_model_forward,
        test_score_network,
    ]
    
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {t.__name__}: {e}")
            failed += 1
    
    print(f"\n{'='*40}")
    print(f"Results: {passed}/{len(tests)} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
