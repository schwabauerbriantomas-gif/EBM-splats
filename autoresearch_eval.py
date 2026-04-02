#!/usr/bin/env python3
"""EBM Autoresearch Eval - Generation quality evaluation via Langevin sampling."""
import os, sys, time, argparse, math, warnings, logging
sys.stderr = sys.stdout
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, torch.nn as nn, torch.nn.functional as F
from autoresearch_train import TrainConfig, ScoreNetwork, Embedder, evaluate_loss
from geometry import normalize_sphere, project_to_tangent
from dataset_loader import get_dataloader


def langevin_sample(score_net, embedder, n_samples, latent_dim, device,
                    n_steps=50, dt=0.01, gamma=0.1, sigma=0.1):
    """Underdamped Langevin dynamics on S^latent_dim."""
    score_net.eval()
    with torch.no_grad():
        # Random points on sphere
        x = normalize_sphere(torch.randn(n_samples, latent_dim, device=device))
        v = torch.randn_like(x) * 0.01
        v = project_to_tangent(x, v)
        sigma_t = torch.full((n_samples,), sigma, device=device)
        for step in range(n_steps):
            score = score_net(x, sigma_t)
            # Langevin: dv = -gamma*v - grad + noise
            v = v * (1 - gamma) + score * dt + math.sqrt(2 * gamma * dt) * torch.randn_like(v)
            v = project_to_tangent(x, v)
            x = x + v * dt
            x = normalize_sphere(x)
    score_net.train()
    return x


def decode_latents_to_text(latents, embedder, tokenizer, seq_len=32):
    """Decode latents to text via nearest-neighbor matching in embedding space."""
    # Get vocab embeddings [vocab_size, latent_dim]
    vocab_emb = normalize_sphere(embedder.emb.weight.detach())  # [50257, 640]
    latents = normalize_sphere(latents)
    # Cosine similarity (both already normalized)
    sims = torch.matmul(latents, vocab_emb.t())  # [n, vocab_size]
    token_ids = sims.argmax(dim=-1)  # [n]
    # Group into sequences of seq_len and decode
    n = token_ids.shape[0]
    n_seqs = n // seq_len
    texts = []
    for i in range(min(n_seqs, 5)):
        chunk = token_ids[i * seq_len:(i + 1) * seq_len]
        text = tokenizer.decode(chunk.tolist(), skip_special_tokens=True)
        texts.append(text)
    return texts, token_ids


def compute_bpb(score_net, embedder, val_loader, device, cfg, max_batches=20):
    """Bits per byte on validation set."""
    import io
    val_loader.dataset_loader = None  # no-op
    total_loss = 0.0
    total_bytes = 0
    count = 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
            tokens = batch.to(device) if not isinstance(batch, dict) else batch['tokens'].to(device)
            x_0 = embedder(tokens)
            B, S, D = x_0.shape
            x_t_flat = x_0.reshape(-1, D)
            x_0_flat = x_0.reshape(-1, D)
            for sigma in cfg.noise_levels:
                noise = project_to_tangent(x_0_flat, torch.randn_like(x_t_flat))
                x_t = normalize_sphere(x_0_flat + sigma * noise)
                score_target = project_to_tangent(x_t, x_0_flat - x_t) / (sigma ** 2)
                sigma_t = torch.full((x_t.size(0),), sigma, device=device)
                score_pred = score_net(x_t, sigma_t)
                total_loss += F.mse_loss(score_pred, score_target, reduction='sum').item()
            total_bytes += B * S * 2  # approx bytes per token
            count += B * S
    avg_loss = total_loss / count if count > 0 else float('inf')
    # Convert MSE loss to bits per byte (approximate)
    bpb = avg_loss / math.log(2) / 2  # rough conversion
    return bpb


def compute_metrics(texts, token_ids):
    """Compute diversity and repetition metrics."""
    if not texts:
        return 0, 1.0
    unique_tokens = token_ids.unique().numel()
    total_tokens = token_ids.numel()
    # Repetition: ratio of tokens that are same as previous
    if total_tokens < 2:
        return unique_tokens, 0.0
    repeats = (token_ids[1:] == token_ids[:-1]).sum().item()
    rep_ratio = repeats / (total_tokens - 1)
    return unique_tokens, rep_ratio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--time_budget", type=int, default=60)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--n_samples", type=int, default=160)
    parser.add_argument("--n_steps", type=int, default=50)
    parser.add_argument("--n_gen_samples", type=int, default=5)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = TrainConfig()
    start = time.time()

    print(f"Device: {device} | Budget: {args.time_budget}s")

    # Build models
    score_net = ScoreNetwork(cfg).to(device)
    embedder = Embedder(cfg).to(device)

    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if isinstance(ckpt, dict):
            if 'score_net' in ckpt:
                score_net.load_state_dict(ckpt['score_net'])
            elif 'model_state_dict' in ckpt:
                score_net.load_state_dict(ckpt['model_state_dict'])
            else:
                score_net.load_state_dict(ckpt)
            if 'embedder' in ckpt:
                embedder.load_state_dict(ckpt['embedder'])
        else:
            print(f"Warning: unknown checkpoint format, loading fresh model")
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("No checkpoint loaded - evaluating fresh (random) model")

    # Load tokenizer (lightweight - no dataset needed for gen)
    print("Loading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load validation set for BPB
    bpb = float('inf')
    time_left = args.time_budget - (time.time() - start)
    if time_left > 15:
        try:
            print("Loading validation data for BPB...")
            val_loader, _ = get_dataloader('tinystories', seq_len=cfg.seq_len, batch_size=cfg.batch_size,
                                           split='val', max_chars=500_000)
            bpb = compute_bpb(score_net, embedder, val_loader, device, cfg, max_batches=10)
        except Exception as e:
            print(f"BPB computation skipped: {e}")
            bpb = float('inf')

    # Generate samples
    print("Generating samples via Langevin dynamics...")
    seq_len = cfg.seq_len
    total_latents = args.n_gen_samples * seq_len
    latents = langevin_sample(score_net, embedder, total_latents, cfg.latent_dim, device,
                              n_steps=args.n_steps)

    texts, token_ids = decode_latents_to_text(latents, embedder, tokenizer, seq_len=seq_len)
    unique_tokens, rep_ratio = compute_metrics(texts, token_ids)

    # Output
    elapsed = time.time() - start
    print("---")
    print(f"val_bpb:          {bpb:.6f}" if bpb < float('inf') else "val_bpb:          N/A")
    print(f"gen_samples:      {len(texts)}")
    print(f"unique_tokens:    {unique_tokens}")
    print(f"repetition_ratio: {rep_ratio:.2f}")
    print(f"elapsed:          {elapsed:.1f}s")
    print()
    for i, text in enumerate(texts[:args.n_gen_samples]):
        print(f"=== Sample {i+1} ===")
        # Truncate at first very long repetition
        txt = text[:500] if text else "(empty)"
        print(txt.encode('ascii', errors='replace').decode('ascii'))


if __name__ == '__main__':
    main()
