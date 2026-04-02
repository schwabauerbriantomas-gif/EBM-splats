#!/usr/bin/env python3
"""
EBM Sample Generator — Generates text from the trained EBM model.

Samples latent vectors using Rectified Flow, decodes to text via GPT-2.
Outputs JSON with samples for external consumption (Telegram, etc.).
"""

import os
import sys
import json
import argparse
import torch
import torch.nn.functional as F

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import EBMConfig
from geometry import normalize_sphere, exp_map
from decoder import EBMDecoder, MoELayer
from score_network import ScoreNetwork

# Import RF velocity net from optimized script
# We redefine it here to avoid import issues
class RectifiedFlowVelocityNet(torch.nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int = 1024):
        super().__init__()
        self.time_mlp = torch.nn.Sequential(
            torch.nn.Linear(1, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
        )
        self.net = torch.nn.Sequential(
            torch.nn.Linear(latent_dim + hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x_t, t):
        from geometry import project_to_tangent
        t_emb = self.time_mlp(t)
        combined = torch.cat([x_t, t_emb], dim=-1)
        v = self.net(combined)
        v = project_to_tangent(x_t, v)
        return v


def sample_rf(vel_net, n_samples, latent_dim, device, n_steps=5):
    """Rectified Flow Euler sampler: noise -> data manifold."""
    x = normalize_sphere(torch.randn(n_samples, latent_dim, device=device))
    dt = 1.0 / n_steps
    with torch.no_grad():
        for step in range(n_steps):
            t_val = 1.0 - step * dt
            t = torch.full((n_samples, 1), t_val, device=device)
            v = vel_net(x, t)
            x = exp_map(x, -dt * v)
            x = normalize_sphere(x)
    return x


def sample_langevin(score_net, n_samples, latent_dim, device, n_steps=200, dt=0.001):
    """Langevin sampler using score network."""
    x = normalize_sphere(torch.randn(n_samples, latent_dim, device=device))
    gamma = 0.1
    with torch.no_grad():
        for step in range(n_steps):
            sigma = torch.full((n_samples,), 0.1, device=device)
            score = score_net(x, sigma)
            noise = torch.randn_like(x) * (2 * gamma * dt) ** 0.5
            x = x - dt * score + noise
            x = normalize_sphere(x)
    return x


def decode_latents(latents, model, tokenizer, device, max_length=64):
    """
    Decode latent vectors to text using nearest-neighbor lookup in embedding space.
    Since the EBM encoder maps tokens -> 640D, we do the inverse:
    find the token whose embedding is closest to each latent vector.
    """
    # Get all token embeddings
    all_embeddings = model.embedding.weight.data  # [vocab_size, latent_dim]
    all_embeddings = F.normalize(all_embeddings, dim=-1)

    results = []
    for i in range(latents.size(0)):
        # latents[i] is a single 640D vector -> find nearest token
        sims = F.cosine_similarity(latents[i:i+1], all_embeddings, dim=-1)  # [vocab_size]
        top_indices = sims.topk(max_length).indices.tolist()
        decoded = tokenizer.decode(top_indices, skip_special_tokens=True)
        results.append(decoded)
    
    return results


def generate_samples(
    checkpoint_dir="checkpoints_optimized",
    vel_checkpoint="vel_net_reflow_1.pt",
    score_checkpoint="score_net_ema_epoch_5.pt",
    n_samples=5,
    n_steps=5,
    max_decode_length=64,
    use_rf=True,
):
    """Generate and return text samples."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = EBMConfig(device="cpu")  # EBMModelCUDA uses device param
    
    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Load models
    # Score network
    score_net = ScoreNetwork(dim=config.latent_dim).to(device)
    score_path = os.path.join(checkpoint_dir, score_checkpoint)
    if os.path.exists(score_path):
        score_net.load_state_dict(torch.load(score_path, map_location=device, weights_only=True))
        print(f"Loaded score net: {score_path}")
    
    # Velocity network
    vel_net = RectifiedFlowVelocityNet(config.latent_dim).to(device)
    vel_path = os.path.join(checkpoint_dir, vel_checkpoint)
    if os.path.exists(vel_path):
        vel_net.load_state_dict(torch.load(vel_path, map_location=device, weights_only=True))
        print(f"Loaded velocity net: {vel_path}")
    
    # EBM model (for decoder + embedding)
    from splats import SplatStorage
    from energy_cuda import EnergyFunctionCUDA
    
    model = type('EBMModel', (), {})()  # lightweight container
    model.config = config
    model.embedding = torch.nn.Embedding(config.vocab_size, config.latent_dim)
    model.embedding.to(device)
    
    # Sample
    if use_rf:
        latents = sample_rf(vel_net, n_samples, config.latent_dim, device, n_steps)
        method = f"RF {n_steps} steps"
    else:
        latents = sample_langevin(score_net, n_samples, config.latent_dim, device)
        method = f"Langevin 200 steps"
    
    # Decode: each latent is a single 640D point -> find nearest tokens
    # For better results, we could use the decoder (MoE), but NN lookup is more reliable
    all_emb = model.embedding.weight.data.detach()
    all_emb_norm = F.normalize(all_emb, dim=-1)
    
    texts = []
    for i in range(n_samples):
        # For each sample, decode as a sequence of tokens
        # We generate a sequence by iteratively finding nearest tokens
        token_ids = []
        current = latents[i:i+1]  # [1, D]
        
        for _ in range(max_decode_length):
            sims = F.cosine_similarity(current, all_emb_norm, dim=-1)  # [vocab_size]
            top_idx = sims.argmax().item()
            token_ids.append(top_idx)
            
            # Move to next position by shifting embedding
            if top_idx == tokenizer.eos_token_id:
                break
        
        text = tokenizer.decode(token_ids, skip_special_tokens=True)
        texts.append(text)
    
    # Also compute stats
    norms = latents.norm(dim=-1)
    pairwise_sims = F.cosine_similarity(
        latents.unsqueeze(1), latents.unsqueeze(0), dim=-1
    )
    avg_sim = (pairwise_sims.sum() - n_samples) / (n_samples * (n_samples - 1))
    
    output = {
        "method": method,
        "n_samples": n_samples,
        "device": str(device),
        "norms_mean": float(norms.mean()),
        "norms_std": float(norms.std()),
        "avg_pairwise_similarity": float(avg_sim),
        "samples": texts,
    }
    
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5, help="Number of samples")
    parser.add_argument("--steps", type=int, default=5, help="RF steps")
    parser.add_argument("--max-len", type=int, default=64, help="Max decode length")
    parser.add_argument("--langevin", action="store_true", help="Use Langevin instead of RF")
    parser.add_argument("--checkpoint-dir", default="checkpoints_optimized")
    parser.add_argument("--output", default=None, help="Output JSON file")
    args = parser.parse_args()
    
    result = generate_samples(
        n_samples=args.n,
        n_steps=args.steps,
        max_decode_length=args.max_len,
        use_rf=not args.langevin,
        checkpoint_dir=args.checkpoint_dir,
    )
    
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
