#!/usr/bin/env python3
"""EBM Training - Memory-efficient streaming version."""

import os
import sys
import time
import logging
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader

from config import EBMConfig
from geometry import normalize_sphere, project_to_tangent, exp_map
from score_network import ScoreNetwork

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger('ebm_train')

device = torch.device('cuda')
config = EBMConfig()
log.info(f'GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

# Config
BATCH_SIZE = 64
SEQ_LEN = 64
N_EPOCHS = 100
MAX_CHARS = 100_000_000  # 100MB limit

# Paths
TINYSTORIES_TRAIN = "D:/datasets/ebm/tinystories_train.txt"
TINYSTORIES_VAL = "D:/datasets/ebm/tinystories_val.txt"
CKPT_DIR = "checkpoints_tinystories"


class StreamingTextDataset(Dataset):
    """Streams text from file and tokenizes in chunks."""
    
    def __init__(self, filepath, tokenizer, seq_len, max_chars=None):
        self.filepath = filepath
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.max_chars = max_chars or float('inf')
    
    def __len__(self):
        return int(os.path.getsize(self.filepath) / 1000)  # Approximate: 1000 chars per line
    
    def __iter__(self):
        with open(self.filepath, 'r', encoding='utf-8') as f:
            char_count = 0
            line_buffer = []
            
            for line in f:
                line = line.rstrip('\n')
                if not line:
                    continue
                
                line_buffer.append(line)
                char_count += len(line_buffer)
                
                # Yield when we have enough for a batch
                if char_count >= BATCH_SIZE * 1000:
                    text_chunk = ''.join(line_buffer)
                    yield text_chunk
                    line_buffer = []
                    char_count = 0


def get_dataloader_streaming(filepath, tokenizer, seq_len, batch_size, split='train'):
    dataset = StreamingTextDataset(filepath, tokenizer, seq_len, max_chars=None)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0, pin_memory=False)
    return dataloader, tokenizer


# Models
class EmbedderWrapper(nn.Module):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        # Lazy-load embeddings
        self.emb = None
        
    def forward(self, token_ids):
        if self.emb is None:
            self.emb = nn.Embedding.from_pretrained('gpt2')
            self.emb = self.emb.to(device)
        return normalize_sphere(self.emb(token_ids))


# Training
def dsm_loss_streaming(score_net, embedder, tokens, epoch, n_epochs, device):
    x_0 = embedder(tokens)
    B, S, D = x_0.shape
    beta = min(1.0, 0.1 + 0.9 * epoch / max(n_epochs * 0.5, 1))
    
    total_loss = torch.tensor(0.0, device=device)
    for sigma in config.noise_levels:
        sigma_eff = sigma * beta
        
        with torch.no_grad():
            noise = torch.randn_like(x_0)
            noise = project_to_tangent(x_0, noise)
            x_t = normalize_sphere(x_0 + sigma_eff * noise)
            x_t_flat = x_t.reshape(-1, D)
            residual = x_0.reshape(-1, D) - x_t_flat
            score_target = project_to_tangent(x_t_flat, residual) / (sigma_eff ** 2)
        
        sigma_t = torch.full((x_t_flat.size(0),), sigma_eff, device=device)
        score_pred = score_net(x_t_flat, sigma_t)
        loss = F.mse_loss(score_pred, score_target)
        total_loss = total_loss + loss
    
    return total_loss / len(config.noise_levels)


def train_streaming(filepath, n_epochs=100):
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Score network
    score_net = ScoreNetwork(dim=config.latent_dim, hidden_dim=2048, sigma_emb_dim=64).to(device)
    
    # EMA
    ema_decay = 0.999
    ema_shadow = {k: v.clone().detach() for k, v in score_net.state_dict().items()}
    
    # Optimizer
    optimizer = torch.optim.AdamW(score_net.parameters(), lr=1e-4, weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda')
    
    # Embedder wrapper
    embedder = EmbedderWrapper(tokenizer)
    
    # Streaming dataloader
    dataloader = get_dataloader_streaming(filepath, tokenizer, SEQ_LEN, BATCH_SIZE, split='train')
    
    os.makedirs(CKPT_DIR, exist_ok=True)
    log.info(f'Streaming training on {filepath}')
    log.info(f'Estimated lines: ~{len(dataloader):,}')
    
    history = []
    total_time_start = time.time()
    
    for epoch in range(n_epochs):
        score_net.train()
        t0 = time.time()
        epoch_loss = 0.0
        nan_count = 0
        
        for bi, tokens in enumerate(dataloader):
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda'):
                loss = dsm_loss_streaming(score_net, embedder, tokens, epoch, n_epochs, device)
                
                if torch.isnan(loss) or torch.isinf(loss):
                    nan_count += 1
                    continue
                
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(score_net.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                
                epoch_loss += loss.item()
            
            # EMA
            with torch.no_grad():
                for k, v in score_net.state_dict().items():
                    if k in ema_shadow:
                        ema_shadow[k].mul_(ema_decay).add_(v.data, alpha=1 - ema_decay)
        
        avg_loss = epoch_loss / max(len(dataloader) - nan_count, 1)
        dur = time.time() - t0
        mem = torch.cuda.max_memory_allocated() / 1e9
        
        log.info(f'E{epoch+1:3d}/{n_epochs} | Loss: {avg_loss:.4f} | NaN: {nan_count} | {dur:.1f}s | {mem:.1f}GB')
        history.append({'epoch': epoch + 1, 'loss': avg_loss, 'duration': dur, 'gpu_gb': mem, 'nan': nan_count})
        
        # Save every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(score_net.state_dict(), f'{CKPT_DIR}/score_net_epoch_{epoch+1}.pt')
            torch.save({k: v.cpu() for k, v in ema_shadow.items()}, f'{CKPT_DIR}/score_net_ema_epoch_{epoch+1}.pt')
            log.info(f'Checkpoint {epoch+1}')
    
    # Final
    torch.save(score_net.state_dict(), f'{CKPT_DIR}/score_net_final.pt')
    torch.save({k: v.cpu() for k, v in ema_shadow.items()}, f'{CKPT_DIR}/score_net_ema_final.pt')
    
    total_time = time.time() - total_time_start
    final_loss = history[-1]['loss']
    log.info(f'Training complete. Total time: {total_time/60:.1f}m | Final loss: {final_loss:.4f}')


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--filepath', type=str, default=TINYSTORIES_TRAIN)
    args = parser.parse_args()
    
    train_streaming(args.filepath, n_epochs=args.epochs)
