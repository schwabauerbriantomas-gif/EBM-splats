#!/usr/bin/env python3
"""EBM Autoresearch - skip + proj_head + wd=0 + Fourier + layer_residuals"""
import os, sys, time, math, argparse, warnings, logging
from pathlib import Path
# Redirect stderr to stdout to prevent PowerShell from treating warnings as errors
sys.stderr = sys.stdout
# Silence all warnings
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, torch.nn as nn, torch.nn.functional as F
from dataclasses import dataclass
from config import EBMConfig
from geometry import normalize_sphere, project_to_tangent
from dataset_loader import get_dataloader

TIME_BUDGET = int(os.environ.get("EBM_TIME_BUDGET", "300"))

def evaluate_loss(score_net, embedder, val_loader, device, noise_levels, max_batches=50):
    score_net.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_batches: break
            tokens = batch.to(device) if not isinstance(batch, dict) else batch['tokens'].to(device)
            x_0 = embedder(tokens)
            B, S, D = x_0.shape
            loss_sum = 0.0
            for sigma in noise_levels:
                noise = torch.randn_like(x_0)
                noise = project_to_tangent(x_0, noise)
                x_t = normalize_sphere(x_0 + sigma * noise)
                x_t_flat = x_t.reshape(-1, D)
                residual = x_0.reshape(-1, D) - x_t_flat
                score_target = project_to_tangent(x_t_flat, residual) / (sigma ** 2)
                sigma_t = torch.full((x_t_flat.size(0),), sigma, device=device)
                score_pred = score_net(x_t_flat, sigma_t)
                loss_sum += F.mse_loss(score_pred, score_target).item()
            total_loss += loss_sum / len(noise_levels)
            count += 1
    score_net.train()
    return total_loss / max(count, 1)

@dataclass
class TrainConfig:
    latent_dim: int = 640
    sigma_hidden_dim: int = 4096
    sigma_encoder_dim: int = 128
    score_layers: int = 12
    vocab_size: int = 50257
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    batch_size: int = 128
    seq_len: int = 64
    noise_levels: tuple = (0.06, 0.12, 0.24)
    max_chars: int = 5_000_000   # 5MB - fast load (~20s)
    max_val_chars: int = 2_000_000  # 2MB for validation

class SigmaEncoder(nn.Module):
    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.register_buffer('freqs', torch.randn(out_dim // 2) * math.pi)
    def forward(self, sigma):
        sigma = sigma.unsqueeze(-1)
        return torch.cat([torch.sin(sigma * self.freqs), torch.cos(sigma * self.freqs)], dim=-1)

class ScoreNetwork(nn.Module):
    """Score network with skip + projection head + pre-norm layer residuals."""
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.cfg = cfg
        self.sigma_enc = SigmaEncoder(cfg.sigma_encoder_dim)
        in_dim = cfg.latent_dim + cfg.sigma_encoder_dim
        dim = cfg.sigma_hidden_dim
        self.in_proj = nn.Linear(in_dim, dim)
        self.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(cfg.score_layers)])
        self.proj_head = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, cfg.latent_dim))
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(cfg.score_layers)])
        self.skip_proj = nn.Linear(cfg.latent_dim, cfg.latent_dim, bias=False)
    
    def forward(self, x, sigma):
        sigma_emb = self.sigma_enc(sigma)
        h = self.in_proj(torch.cat([x, sigma_emb], dim=-1))
        for layer, norm in zip(self.layers, self.norms):
            h = h + F.gelu(layer(norm(h)))  # Pre-norm residual
        score = self.proj_head(h) + self.skip_proj(x)
        score = score - (score * x).sum(-1, keepdim=True) * x
        return score

class Embedder(nn.Module):
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.emb = nn.Embedding(cfg.vocab_size, cfg.latent_dim)
    def forward(self, tokens):
        return normalize_sphere(self.emb(tokens))

def dsm_loss(score_net, embedder, tokens, cfg, epoch_ratio=1.0):
    x_0 = embedder(tokens)
    B, S, D = x_0.shape
    x_t_flat = x_0.reshape(-1, D)
    x_0_flat = x_0.reshape(-1, D)
    total_loss = torch.tensor(0.0, device=x_0.device)
    for sigma in cfg.noise_levels:
        sigma_eff = sigma * min(1.0, 0.3 + 0.7 * epoch_ratio)
        with torch.no_grad():
            noise = project_to_tangent(x_0_flat, torch.randn_like(x_t_flat))
            x_t = normalize_sphere(x_0_flat + sigma_eff * noise)
            score_target = project_to_tangent(x_t, x_0_flat - x_t) / (sigma_eff ** 2)
        if torch.rand(1).item() < 0.08:
            with torch.no_grad():
                x_t = normalize_sphere(x_t + torch.randn_like(x_t) * 0.01)
        sigma_t = torch.full((x_t.size(0),), sigma_eff, device=x_t.device)
        score_pred = score_net(x_t, sigma_t)
        total_loss = total_loss + F.mse_loss(score_pred, score_target)
    return total_loss / len(cfg.noise_levels)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--time_budget", type=int, default=TIME_BUDGET)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = TrainConfig()
    torch.manual_seed(42)
    print(f"Device: {device} | Budget: {args.time_budget}s | BS: {cfg.batch_size} | LR: {cfg.learning_rate}")
    print(f"Hidden: {cfg.sigma_hidden_dim} | Layers: {cfg.score_layers} | Noise: {cfg.noise_levels}")
    print("Loading data...")
    train_loader, _ = get_dataloader('tinystories', seq_len=cfg.seq_len, batch_size=cfg.batch_size, split='train', max_chars=cfg.max_chars)
    val_loader, _ = get_dataloader('tinystories', seq_len=cfg.seq_len, batch_size=cfg.batch_size, split='val', max_chars=cfg.max_val_chars)
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    score_net = ScoreNetwork(cfg).to(device)
    embedder = Embedder(cfg).to(device)
    n_params = sum(p.numel() for p in score_net.parameters()) + sum(p.numel() for p in embedder.parameters())
    print(f"Parameters: {n_params / 1e6:.2f}M")
    optimizer = torch.optim.AdamW(list(score_net.parameters()) + list(embedder.parameters()), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    print("Training...")
    train_start = time.time()
    best_val_loss = float('inf')
    step = 0
    for epoch in range(args.epochs):
        if time.time() - train_start >= args.time_budget: break
        score_net.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            if time.time() - train_start >= args.time_budget: break
            tokens = batch.to(device) if not isinstance(batch, dict) else batch['tokens'].to(device)
            optimizer.zero_grad(set_to_none=True)
            epoch_ratio = step / max(args.epochs * len(train_loader), 1)
            if scaler:
                with torch.amp.autocast('cuda'):
                    loss = dsm_loss(score_net, embedder, tokens, cfg, epoch_ratio)
                if torch.isnan(loss): continue
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(score_net.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = dsm_loss(score_net, embedder, tokens, cfg, epoch_ratio)
                if torch.isnan(loss): continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(score_net.parameters(), cfg.grad_clip)
                optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
            step += 1
            if step % 200 == 0:
                print(f"  step {step} | loss: {epoch_loss/max(n_batches,1):.2f} | elapsed: {time.time()-train_start:.0f}s")
        print(f"E{epoch+1} | train_loss: {epoch_loss/max(n_batches,1):.2f}")
        # Eval every epoch
        val_loss = evaluate_loss(score_net, embedder, val_loader, device, cfg.noise_levels)
        print(f"E{epoch+1} | val_loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_dir = Path("checkpoints_autoresearch")
            ckpt_dir.mkdir(exist_ok=True)
            torch.save({'score_net': score_net.state_dict(), 'embedder': embedder.state_dict(), 'val_loss': val_loss, 'step': step, 'epoch': epoch+1}, ckpt_dir / "best_model.pt")
            print(f"  >> Best checkpoint saved (val_loss={val_loss:.4f})")
        # Save periodic checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            ckpt_dir = Path("checkpoints_autoresearch")
            ckpt_dir.mkdir(exist_ok=True)
            torch.save({'score_net': score_net.state_dict(), 'embedder': embedder.state_dict(), 'val_loss': val_loss, 'step': step, 'epoch': epoch+1}, ckpt_dir / f"checkpoint_epoch{epoch+1}.pt")
            print(f"  >> Periodic checkpoint saved (epoch {epoch+1})")
        scheduler.step()
    val_loss = evaluate_loss(score_net, embedder, val_loader, device, cfg.noise_levels)

    # Final checkpoint
    ckpt_dir = Path("checkpoints_autoresearch")
    ckpt_dir.mkdir(exist_ok=True)
    torch.save({'score_net': score_net.state_dict(), 'embedder': embedder.state_dict(), 'val_loss': val_loss, 'step': step, 'epoch': args.epochs}, ckpt_dir / "best_model.pt")
    print(f"Final checkpoint saved: checkpoints_autoresearch/best_model.pt")

    print(f"---\nval_loss: {val_loss:.6f}\nbest_val_loss: {best_val_loss:.6f}\nsteps: {step}\nparams_M: {n_params/1e6:.2f}")

if __name__ == '__main__':
    main()
