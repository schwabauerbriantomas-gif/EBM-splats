#!/usr/bin/env python3
"""Smoke test for embedding_service."""
import sys
sys.path.insert(0, "/mnt/c/Users/Brian/Desktop/EBM-splats")
from pglf.embedding_service import EmbeddingService
import numpy as np

print("Creating EmbeddingService with all-MiniLM-L6-v2...")
svc = EmbeddingService(model_name="sentence-transformers/all-MiniLM-L6-v2", projection_dim=384)
print(f"  device={svc.device}  dim={svc.dim}  backbone={svc.backbone_info}")

print()
print("Encoding 2 texts...")
embs = svc.encode(["hello world", "test sentence"])
print(f"  shape={embs.shape}  dtype={embs.dtype}")
norms = np.linalg.norm(embs, axis=1)
print(f"  norms={norms}  (should be ~1.0 on hypersphere)")
assert embs.shape == (2, 384), f"Expected (2, 384), got {embs.shape}"
assert np.allclose(norms, 1.0, atol=1e-5), f"Norms not unit: {norms}"

print()
print("Testing encode_batch with batch_size=1...")
embs2 = svc.encode_batch(["hello world", "test sentence", "another one"], batch_size=1)
print(f"  shape={embs2.shape}")
assert embs2.shape == (3, 384)

print()
print("ALL SMOKE TESTS PASSED")
