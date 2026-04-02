"""
PGLF Embedding Service — HTTP server for M2M-Rust integration.

Provides a lightweight HTTP API that M2M's MCP server can call
to compute embeddings using the trained PGLF model.
"""

import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer
from flask import Flask, request, jsonify
import numpy as np
from pathlib import Path
import argparse
import logging

from .encoders import TextEncoder
from .flow_matching import HypersphereFlowMatching

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pglf-service")

app = Flask(__name__)

# Global model state
models = {
    "encoder": None,
    "flow_model": None,
    "tokenizer": None,
    "device": "cpu",
    "config": {},
}


def load_models(checkpoint_path: str, device: str = "cuda"):
    """Load trained models for inference."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {"dim": 640})
    dim = config.get("dim", 640)
    
    # Load encoder
    encoder = TextEncoder(dim=dim).to(device)
    if "encoder_state" in checkpoint:
        encoder.load_state_dict(checkpoint["encoder_state"])
    encoder.eval()
    
    # Load flow model
    flow_model = HypersphereFlowMatching(dim=dim).to(device)
    if "flow_state" in checkpoint:
        flow_model.load_state_dict(checkpoint["flow_state"])
    flow_model.eval()
    
    # Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    models["encoder"] = encoder
    models["flow_model"] = flow_model
    models["tokenizer"] = tokenizer
    models["device"] = device
    models["config"] = config
    
    logger.info(f"Models loaded from {checkpoint_path}")
    logger.info(f"Device: {device}, Dim: {dim}")


@app.route("/embed", methods=["POST"])
def embed():
    """Compute embedding for input text."""
    data = request.json
    texts = data.get("texts", [])
    use_flow = data.get("use_flow", False)
    
    if not texts:
        return jsonify({"error": "No texts provided"}), 400
    
    try:
        device = models["device"]
        
        # Tokenize
        encoded = models["tokenizer"](
            texts,
            truncation=True,
            max_length=128,
            padding=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        
        with torch.no_grad():
            embeddings, modality = models["encoder"].encode_text(input_ids, attention_mask)
            
            if use_flow and models["flow_model"] is not None:
                embeddings = models["flow_model"].sample(
                    condition=embeddings,
                    modality=modality,
                    n_steps=models["config"].get("fm_n_steps", 10),
                )
        
        # Convert to list
        embeddings_np = embeddings.cpu().numpy().tolist()
        
        return jsonify({
            "embeddings": embeddings_np,
            "dim": len(embeddings_np[0]),
            "count": len(embeddings_np),
        })
    
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "device": models["device"],
        "dim": models["config"].get("dim", 640),
        "flow_model_loaded": models["flow_model"] is not None,
    })


def main():
    parser = argparse.ArgumentParser(description="PGLF Embedding Service")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to inference checkpoint")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8777)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    load_models(args.checkpoint, args.device)
    
    logger.info(f"Starting PGLF Embedding Service on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=False)


if __name__ == "__main__":
    main()
