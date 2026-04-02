"""
Embedding Service — wraps sentence-transformers backbone with hypersphere projection.

Provides:
  - Python API via EmbeddingService class
  - HTTP server (Flask) on port 8788
  - CLI: python -m pglf.embedding_service --port 8788 --model miniml
"""

import argparse
import json
import logging
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Union

logger = logging.getLogger("pglf.embedding_service")

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    "miniml": "sentence-transformers/all-MiniLM-L6-v2",
    "pglf": "pglf_textencoder",
}

# ---------------------------------------------------------------------------
# Projection head: maps backbone output → configurable dim on the hypersphere
# ---------------------------------------------------------------------------

class HypersphereProjection(nn.Module):
    """Linear projection + L2 normalisation to the unit hypersphere."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.linear(x), p=2, dim=-1)


# ---------------------------------------------------------------------------
# Core embedding service
# ---------------------------------------------------------------------------

class EmbeddingService:
    """
    Wraps a pre-trained sentence-transformers backbone and optionally projects
    to a configurable dimension on the hypersphere.

    Parameters
    ----------
    model_name : str
        Either a sentence-transformers model id (e.g.
        ``"sentence-transformers/all-MiniLM-L6-v2"``) or the literal string
        ``"pglf"`` to load the custom PGLF TextEncoder from ``encoders.py``.
    projection_dim : int
        Target embedding dimensionality.  When it matches the backbone output
        the projection layer is still applied for consistency.
    device : str | None
        ``"cuda"`` if available, otherwise ``"cpu"``.  Pass ``None`` for auto.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        projection_dim: int = 384,
        device: Optional[str] = None,
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model_name = model_name
        self.projection_dim = projection_dim
        self._backbone_type: str = "unknown"

        # ---- Load backbone ----
        if model_name.lower().replace("_", "") == "pglf" or model_name == "pglf_textencoder":
            self._load_pglf_encoder()
        else:
            self._load_sentence_transformer(model_name)

        logger.info(
            "EmbeddingService ready — backbone=%s  dim=%d  device=%s",
            self._backbone_type,
            self.projection_dim,
            self.device,
        )

    # ------------------------------------------------------------------
    # Backbone loaders
    # ------------------------------------------------------------------

    def _load_sentence_transformer(self, model_id: str) -> None:
        from sentence_transformers import SentenceTransformer

        self._backbone_type = model_id
        self._st_model: SentenceTransformer = SentenceTransformer(
            model_id, device=self.device
        )
        backbone_dim: int = self._st_model.get_sentence_embedding_dimension()
        self._projection = HypersphereProjection(backbone_dim, self.projection_dim).to(
            self.device
        )
        self._projection.eval()
        # Store backbone dim for health endpoint
        self._backbone_dim = backbone_dim
        self._encode_fn = self._encode_st

    def _load_pglf_encoder(self) -> None:
        """Load the custom PGLF TextEncoder from pglf.encoders."""
        from pglf.encoders import TextEncoder
        import tiktoken

        self._backbone_type = "pglf_textencoder"
        self._pglf_model = TextEncoder(dim=640).to(self.device)
        self._pglf_model.eval()
        self._tokenizer = tiktoken.get_encoding("p50k_base")
        self._projection = HypersphereProjection(640, self.projection_dim).to(
            self.device
        )
        self._projection.eval()
        self._backbone_dim = 640
        self._encode_fn = self._encode_pglf

    # ------------------------------------------------------------------
    # Internal encode helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _encode_st(self, texts: List[str]) -> np.ndarray:
        features = self._st_model.tokenize(texts)
        features = {k: v.to(self.device) for k, v in features.items()}
        embeddings = self._st_model(features)["sentence_embedding"]
        projected = self._projection(embeddings)
        return projected.cpu().numpy()

    @torch.no_grad()
    def _encode_pglf(self, texts: List[str]) -> np.ndarray:
        max_len = 256
        token_ids = []
        for t in texts:
            ids = self._tokenizer.encode(t)[:max_len]
            token_ids.append(ids)
        # Pad
        max_s = max(len(ids) for ids in token_ids)
        padded = np.zeros((len(token_ids), max_s), dtype=np.int64)
        for i, ids in enumerate(token_ids):
            padded[i, : len(ids)] = ids
        input_ids = torch.tensor(padded, device=self.device)
        attention_mask = input_ids != 0
        embeddings = self._pglf_model(input_ids, attention_mask=attention_mask)
        projected = self._projection(embeddings)
        return projected.cpu().numpy()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Encode one or more texts.  Returns np.ndarray of shape (N, dim)."""
        if isinstance(texts, str):
            texts = [texts]
        return self._encode_fn(texts)

    def encode_batch(
        self, texts: List[str], batch_size: int = 64
    ) -> np.ndarray:
        """Encode texts in batches to limit memory usage."""
        all_embs: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            all_embs.append(self._encode_fn(batch))
        return np.concatenate(all_embs, axis=0)

    @property
    def dim(self) -> int:
        return self.projection_dim

    @property
    def backbone_info(self) -> str:
        return self._backbone_type


# ---------------------------------------------------------------------------
# Flask HTTP server
# ---------------------------------------------------------------------------

def create_app(service: Optional[EmbeddingService] = None, **kwargs) -> "flask.Flask":
    from flask import Flask, request, jsonify

    if service is None:
        service = EmbeddingService(**kwargs)

    app = Flask(__name__)

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify(
            {
                "status": "ok",
                "model": service.backbone_info,
                "dim": service.dim,
            }
        )

    @app.route("/embed", methods=["POST"])
    def embed():
        data = request.get_json(force=True)
        texts = data.get("texts", [])
        if not texts:
            return jsonify({"error": "No texts provided"}), 400
        batch_size = data.get("batch_size", 64)
        embs = service.encode_batch(texts, batch_size=batch_size)
        return jsonify(
            {
                "embeddings": embs.tolist(),
                "dim": service.dim,
            }
        )

    return app


def run_server(port: int = 8788, model_key: str = "miniml", projection_dim: int = 384):
    """Launch the Flask server (blocking)."""
    model_name = MODEL_REGISTRY.get(model_key, model_key)
    service = EmbeddingService(model_name=model_name, projection_dim=projection_dim)
    app = create_app(service)
    logger.info("Starting embedding service on port %d", port)
    app.run(host="0.0.0.0", port=port, threaded=False)


# ---------------------------------------------------------------------------
# CLI — python -m pglf.embedding_service
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PGLF Embedding Service")
    parser.add_argument("--port", type=int, default=8788, help="HTTP port (default 8788)")
    parser.add_argument(
        "--model",
        type=str,
        default="miniml",
        choices=list(MODEL_REGISTRY.keys()),
        help="Model key: miniml (all-MiniLM-L6-v2) or pglf",
    )
    parser.add_argument("--dim", type=int, default=384, help="Projection dimension")
    parser.add_argument("--device", type=str, default=None, help="Force device (cuda/cpu)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )

    model_name = MODEL_REGISTRY[args.model]
    service = EmbeddingService(
        model_name=model_name, projection_dim=args.dim, device=args.device
    )
    app = create_app(service)
    logger.info("Embedding service listening on :%d  model=%s  dim=%d", args.port, model_name, args.dim)
    app.run(host="0.0.0.0", port=args.port, threaded=False)


if __name__ == "__main__":
    main()
