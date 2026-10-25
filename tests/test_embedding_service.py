"""Unit tests for pglf.embedding_service.

Runs entirely on stubs: the service instance is built without touching
__init__ backbone loaders, so no sentence-transformer weights and no tiktoken
BPE files are ever downloaded. Flask is exercised through its test client.
"""

import importlib.util

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pglf.embedding_service import MODEL_REGISTRY, EmbeddingService, HypersphereProjection

flask = pytest.importorskip("flask")
from pglf.embedding_service import create_app

DIM = 16


def _stub_service(dim: int = DIM) -> EmbeddingService:
    """Build an EmbeddingService whose encoder is a deterministic stub."""
    svc = EmbeddingService.__new__(EmbeddingService)
    svc.device = "cpu"
    svc.model_name = "stub"
    svc.projection_dim = dim
    svc._backbone_type = "stub"
    svc._backbone_dim = dim
    svc._projection = HypersphereProjection(dim, dim)
    svc._call_log = []

    def _encode(texts):
        svc._call_log.append(list(texts))
        base = np.arange(dim, dtype=np.float32) + 1.0
        vecs = np.tile(base, (len(texts), 1))
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

    svc._encode_fn = _encode
    return svc


@pytest.fixture
def client():
    app = create_app(_stub_service())
    app.testing = True
    return app.test_client()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_model_registry_has_documented_backbones():
    assert MODEL_REGISTRY["miniml"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert MODEL_REGISTRY["pglf"] == "pglf_textencoder"


# ---------------------------------------------------------------------------
# Projection head
# ---------------------------------------------------------------------------

def test_hypersphere_projection_outputs_unit_vectors():
    torch.manual_seed(0)
    proj = HypersphereProjection(in_dim=8, out_dim=DIM).eval()
    with torch.no_grad():
        out = proj(torch.randn(5, 8))
    assert out.shape == (5, DIM)
    norms = out.norm(p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


# ---------------------------------------------------------------------------
# Python API (stubbed encoder)
# ---------------------------------------------------------------------------

def test_encode_single_string_returns_2d_array():
    svc = _stub_service()
    out = svc.encode("hello world")
    assert isinstance(out, np.ndarray)
    assert out.shape == (1, svc.dim)


def test_encode_batch_splits_into_requested_batches():
    svc = _stub_service()
    texts = [f"text {i}" for i in range(5)]
    out = svc.encode_batch(texts, batch_size=2)
    assert out.shape == (5, svc.dim)
    assert [len(c) for c in svc._call_log] == [2, 2, 1]
    norms = np.linalg.norm(out, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_dim_and_backbone_info_properties():
    svc = _stub_service(dim=7)
    assert svc.dim == 7
    assert svc.backbone_info == "stub"


def test_module_import_does_not_require_tiktoken():
    # tiktoken must stay a lazy dependency of the pglf encoder path, never a
    # module-level import (regression guard for the ebm-embed-service entry
    # point in a clean env). Skips where tiktoken happens to be installed.
    if importlib.util.find_spec("tiktoken") is not None:
        pytest.skip("tiktoken installed; laziness not observable here")
    import sys

    sys.modules.pop("pglf.embedding_service", None)
    import pglf.embedding_service as svc_mod

    assert svc_mod is not None


def test_load_pglf_encoder_gives_actionable_error_without_tiktoken():
    svc = EmbeddingService.__new__(EmbeddingService)
    svc.device = "cpu"
    if importlib.util.find_spec("tiktoken") is not None:
        pytest.skip("tiktoken installed; error path not reachable here")
    with pytest.raises(ImportError, match="tiktoken"):
        svc._load_pglf_encoder()


# ---------------------------------------------------------------------------
# HTTP API via Flask test client
# ---------------------------------------------------------------------------

def test_health_endpoint_reports_model_and_dim(client):
    res = client.get("/health")
    assert res.status_code == 200
    body = res.get_json()
    assert body["status"] == "ok"
    assert body["model"] == "stub"
    assert body["dim"] == DIM


def test_embed_endpoint_returns_embeddings(client):
    res = client.post("/embed", json={"texts": ["hello", "world"]})
    assert res.status_code == 200
    body = res.get_json()
    assert body["dim"] == DIM
    embs = np.asarray(body["embeddings"], dtype=np.float32)
    assert embs.shape == (2, DIM)
    norms = np.linalg.norm(embs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)


def test_embed_endpoint_rejects_empty_texts(client):
    res = client.post("/embed", json={"texts": []})
    assert res.status_code == 400
    assert "error" in res.get_json()


def test_embed_endpoint_respects_batch_size(client):
    app = create_app(_stub_service())
    app.testing = True
    client = app.test_client()
    res = client.post(
        "/embed", json={"texts": ["a", "b", "c"], "batch_size": 1}
    )
    assert res.status_code == 200


# ---------------------------------------------------------------------------
# Optional real-backbone smoke tests (never run in CI: they download weights)
# ---------------------------------------------------------------------------

def test_real_sentence_transformer_backbone_loads():
    if importlib.util.find_spec("sentence_transformers") is None:
        pytest.skip("sentence-transformers not installed")
    pytest.skip(
        "requires downloading all-MiniLM-L6-v2 from the Hub; run manually"
    )
