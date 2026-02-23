import torch
import torch.nn.functional as F
from dataclasses import dataclass
import json
from config import EBMConfig


def compute_perplexity(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            x, targets = batch.tokens.to(device), batch.targets.to(device)
            
            # Get embeddings
            with torch.no_grad():
                x_embedded = model.embed(x)
            
            # Compute scores
            scores = model.compute_score(x_embedded)
            
            # Convert scores to logits (negative energy = higher probability)
            logits = -scores
            
            # Compute cross-entropy loss
            loss = F.cross_entropy(
                logits.view(-1, model.config.vocab_size),
                targets.view(-1),
                reduction='sum'
            )
            
            total_loss += loss.item()
            total_tokens += targets.numel()
    
    avg_loss = total_loss / len(dataloader)
    ppl = torch.exp(avg_loss)
    
    return {
        'avg_loss': avg_loss,
        'perplexity': ppl.item(),
        'total_tokens': total_tokens,
        'num_batches': len(dataloader)
    }


def compute_energy_metrics(model, dataloader, device):
    model.eval()
    energies = []
    
    with torch.no_grad():
        for batch in dataloader:
            x = batch.tokens.to(device)
            
            with torch.no_grad():
                x_embedded = model.embed(x)
            
            energy = model.compute_energy(x_embedded)
            avg_energy = energy.mean().item()
            energies.append(avg_energy)
    
    return {
        'avg_energy': sum(energies) / len(energies),
        'energy_history': energies,
        'num_batches': len(dataloader)
    }


def compute_convergence_metrics(energies_per_epoch):
    if len(energies_per_epoch) < 2:
        return {
            'trend': 'insufficient_data',
            'stability': 'unknown'
        }
    
    recent_energies = energies_per_epoch[-5:]
    if len(recent_energies) < 3:
        return {
            'trend': 'insufficient_data',
            'stability': 'unknown'
        }
    
    # Check for decreasing energy (good convergence)
    is_decreasing = True
    for i in range(1, len(recent_energies)):
        if recent_energies[i] > recent_energies[i-1] * 1.01:
            is_decreasing = False
            break
    
    if is_decreasing:
        trend = 'converging'
        stability = 'good'
    elif recent_energies[-1] < recent_energies[0] * 0.95:
        trend = 'converged'
        stability = 'excellent'
    else:
        trend = 'diverging'
        stability = 'poor'
    
    return {
        'trend': trend,
        'stability': stability,
        'recent_energies': recent_energies
    }


def compute_diversity_metrics(generated_tokens_list):
    all_tokens = []
    for sample in generated_tokens_list:
        all_tokens.extend(sample)
    
    unique_tokens = set(all_tokens)
    unique_ratio = len(unique_tokens) / len(all_tokens)
    
    # Bigram repetition
    bigram_repetitions = 0
    for i in range(len(all_tokens) - 1):
        bigram = (all_tokens[i], all_tokens[i+1])
        for j in range(len(all_tokens) - 1):
            if (all_tokens[j], all_tokens[j+1]) == bigram:
                bigram_repetitions += 1
    
    bigram_rep_rate = bigram_repetitions / (len(all_tokens) - 1) if len(all_tokens) > 1 else 0
    
    return {
        'total_tokens': len(all_tokens),
        'unique_tokens': len(unique_tokens),
        'unique_ratio': unique_ratio,
        'bigram_rep_rate': bigram_rep_rate
    }


def save_checkpoint(
    epoch,
    model,
    optimizer,
    scheduler,
    metrics,
    output_dir='checkpoints'
):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'metrics': metrics
    }
    
    filename = f"{output_dir}/ebm_epoch_{epoch}.pt"
    torch.save(checkpoint, filename)
    
    return filename


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint['epoch'], checkpoint.get('metrics', {})
