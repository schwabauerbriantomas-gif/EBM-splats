#!/usr/bin/env python3
"""
Test: Verify gradient flow through ScoreNetwork (the fix for the autograd bug).

Before fix: torch.autograd.grad with create_graph=False + .detach() ? zero gradients.
After fix: ScoreNetwork is a regular nn.Module ? backward() propagates gradients.
"""

import sys
import torch
import torch.nn.functional as F

from score_network import ScoreNetwork
from geometry import normalize_sphere, project_to_tangent
from config import EBMConfig


def test_gradients_nonzero():
    """Verify that backward() produces non-zero gradients on ScoreNetwork parameters."""
    dim = 640
    batch = 16
    score_net = ScoreNetwork(dim=dim)
    score_net.train()

    # Simulate training data
    x_0 = normalize_sphere(torch.randn(batch, dim))
    sigma = 0.1
    noise = project_to_tangent(x_0, torch.randn_like(x_0))
    x_t = normalize_sphere(x_0 + sigma * noise)

    # Target score
    score_target = project_to_tangent(x_t, x_0 - x_t) / (sigma ** 2)

    # Forward
    sigma_t = torch.full((batch,), sigma)
    score_pred = score_net(x_t, sigma_t)
    loss = F.mse_loss(score_pred, score_target)

    # Backward
    loss.backward()

    # Check gradients
    has_nonzero = False
    zero_count = 0
    total_count = 0
    for name, param in score_net.named_parameters():
        if param.grad is not None:
            total_count += 1
            if param.grad.abs().max().item() > 0:
                has_nonzero = True
            else:
                zero_count += 1
                print(f"  ? ZERO grad: {name}")
        else:
            zero_count += 1
            print(f"  ? None grad: {name}")

    assert has_nonzero, "No non-zero gradients found! ScoreNetwork is not learning."
    if zero_count > 0:
        print(f"  ??  {zero_count}/{total_count} parameters have zero/None gradients")
    else:
        print(f"  ? All {total_count} parameters have non-zero gradients")
    print(f"  ? Loss: {loss.item():.6f}")


def test_output_shape_and_tangent():
    """Verify output shape and tangent-space constraint."""
    dim = 640
    batch = 8
    score_net = ScoreNetwork(dim=dim)
    score_net.eval()

    x = normalize_sphere(torch.randn(batch, dim))
    sigma = torch.tensor([0.01, 0.05, 0.1, 0.2, 0.5, 0.1, 0.05, 0.01])

    with torch.no_grad():
        score = score_net(x, sigma)

    assert score.shape == (batch, dim), f"Expected ({batch}, {dim}), got {score.shape}"

    # Verify tangent constraint: score ? x ? 0
    dot = (score * x).sum(dim=-1)
    max_dot = dot.abs().max().item()
    assert max_dot < 1e-5, f"Score not in tangent space! max|s?x| = {max_dot}"
    print(f"  ? Output shape: {score.shape}")
    print(f"  ? Tangent constraint: max|s?x| = {max_dot:.2e}")


def test_loss_decreases():
    """Verify loss decreases over a few optimization steps."""
    dim = 128  # smaller for speed
    batch = 32
    steps = 20

    score_net = ScoreNetwork(dim=dim)
    optimizer = torch.optim.Adam(score_net.parameters(), lr=1e-3)

    x_0 = normalize_sphere(torch.randn(batch, dim))
    sigma = 0.1
    noise = project_to_tangent(x_0, torch.randn_like(x_0))
    x_t = normalize_sphere(x_0 + sigma * noise)
    score_target = project_to_tangent(x_t, x_0 - x_t) / (sigma ** 2)

    losses = []
    for _ in range(steps):
        optimizer.zero_grad()
        sigma_t = torch.full((batch,), sigma)
        score_pred = score_net(x_t, sigma_t)
        loss = F.mse_loss(score_pred, score_target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    improvement = losses[0] - losses[-1]
    print(f"  Loss: {losses[0]:.4f} ? {losses[-1]:.4f} (? = {improvement:.4f})")
    assert improvement > 0, f"Loss did not decrease: {losses[0]} ? {losses[-1]}"
    print(f"  ? Loss decreased by {improvement:.4f}")


if __name__ == "__main__":
    print("=" * 60)
    print("ScoreNetwork Gradient Flow Tests")
    print("=" * 60)

    tests = [
        ("Gradients are non-zero", test_gradients_nonzero),
        ("Output shape and tangent constraint", test_output_shape_and_tangent),
        ("Loss decreases with training", test_loss_decreases),
    ]

    passed = 0
    for name, test_fn in tests:
        print(f"\n? {name}")
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"  ? FAILED: {e}")
        except Exception as e:
            print(f"  ? ERROR: {e}")

    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{len(tests)} tests passed")
    print(f"{'=' * 60}")

    sys.exit(0 if passed == len(tests) else 1)
