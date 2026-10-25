"""Unit tests for the pglf package API.

Pure-CPU, torch-only tests: no model downloads, no network, no module-level
side effects (the previous version of this file was a smoke script that
executed training-size models during pytest collection).
"""

import pytest

torch = pytest.importorskip("torch")

import pglf
from pglf import (
    FlowMatchingLoss,
    HypersphereContrastiveLoss,
    HypersphereFlowMatching,
    OmnimodalEncoder,
    ParetoFilter,
    PGLFTrainer,
    TextEncoder,
)

DIM = 40  # divisible by TextEncoder's default n_heads=10
VOCAB = 100
SEQ = 8
BATCH = 4


def _small_encoder() -> TextEncoder:
    torch.manual_seed(0)
    return TextEncoder(
        vocab_size=VOCAB,
        dim=DIM,
        n_layers=1,
        n_heads=2,
        ff_dim=64,
        max_seq_len=SEQ + 8,
        dropout=0.0,
    )


def _batch():
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(1, VOCAB, (BATCH, SEQ), generator=g)
    return ids, torch.ones_like(ids)


def test_public_api_exports_all_documented_names():
    for name in pglf.__all__:
        assert hasattr(pglf, name), f"pglf.{name} missing"


def test_text_encoder_output_shape_and_unit_norm():
    enc = _small_encoder().eval()
    ids, mask = _batch()
    with torch.no_grad():
        emb = enc(ids, mask)
    assert emb.shape == (BATCH, DIM)
    assert torch.isfinite(emb).all()
    norms = emb.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_omnimodal_encoder_routes_text():
    torch.manual_seed(0)
    omni = OmnimodalEncoder(dim=DIM, freeze_pretrained=True)
    omni.eval()
    ids, mask = _batch()
    with torch.no_grad():
        emb, modality = omni.encode_text(ids, mask)
    assert emb.shape == (BATCH, DIM)
    assert modality.shape == (BATCH,)
    assert (modality == OmnimodalEncoder.MODALITY_TEXT).all()


def test_flow_matching_loss_is_finite_and_backprops():
    torch.manual_seed(0)
    flow = HypersphereFlowMatching(dim=DIM, hidden_dim=64, n_layers=2, condition_dim=DIM)
    loss_fn = FlowMatchingLoss()

    x1 = torch.nn.functional.normalize(torch.randn(BATCH, DIM), dim=-1)
    condition = torch.randn(BATCH, DIM)
    modality = torch.zeros(BATCH, dtype=torch.long)

    loss = loss_fn(flow, x1, condition, modality)
    assert loss.dim() == 0
    assert torch.isfinite(loss)

    loss.backward()
    grads = [p.grad for p in flow.parameters() if p.requires_grad]
    assert grads and any(g is not None and g.abs().sum() > 0 for g in grads), (
        "no gradient reached the flow-matching parameters"
    )


def test_pareto_filter_returns_requested_count():
    torch.manual_seed(0)
    emb = torch.nn.functional.normalize(torch.randn(20, DIM), dim=-1)
    pf = ParetoFilter(n_keep=5)
    selected, objectives, fronts = pf.filter(emb)
    assert selected.shape == (5, DIM)
    assert objectives.shape[0] == 5
    assert objectives.shape[1] == pf.n_objectives
    # Fronts partition the whole population
    assert sum(len(f) for f in fronts) == 20
    assert torch.isfinite(selected).all()


def test_contrastive_loss_decreases_for_matching_pairs():
    torch.manual_seed(0)
    loss_fn = HypersphereContrastiveLoss()
    anchor = torch.nn.functional.normalize(torch.randn(6, DIM), dim=-1)
    positive = torch.nn.functional.normalize(anchor + 0.01 * torch.randn(6, DIM), dim=-1)
    negative = torch.nn.functional.normalize(torch.randn(6, DIM), dim=-1)
    loss_pos, _ = loss_fn(anchor, positive)
    loss_neg, _ = loss_fn(anchor, negative)
    assert torch.isfinite(loss_pos) and torch.isfinite(loss_neg)
    assert loss_pos < loss_neg


def test_trainer_class_is_constructible_signature_only():
    # PGLFTrainer wires encoders + ebm components together; building the real
    # object pulls heavy defaults, so only assert the API surface here.
    assert callable(PGLFTrainer)
    for phase in ("init_ebm_components", "phase1_contrastive_training"):
        assert callable(getattr(PGLFTrainer, phase, None)), f"{phase} missing"
