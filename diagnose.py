import torch
from evaluate import compute_perplexity, compute_energy_metrics, compute_convergence_metrics
from logger import TrainingLogger
import json


def diagnose_checkpoint(checkpoint_path, config_path='projects/ebm/config.py', device='cpu'):
    from config import EBMConfig
    from model import EBMModel
    from dataset_utils import get_wikitext_loader
    
    print("=" * 60)
    print("EBM DIAGNOSTIC CHECKER")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print()
    
    # Load config and model
    config = EBMConfig()
    model = EBMModel(config).to(device)
    
    # Load checkpoint
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print()
    
    # Load small validation dataset
    print("Loading validation dataset...")
    try:
        val_loader = get_wikitext_loader(
            max_samples=1000,
            batch_size=16,
            seq_length=config.context_local,
            device=device
        )
        print(f"Validation samples: {len(val_loader.dataset)}")
        print()
    except Exception as e:
        print(f"Warning: Could not load validation dataset: {e}")
        val_loader = None
    
    # Compute perplexity
    if val_loader is not None:
        print("Computing perplexity metrics...")
        ppl_metrics = compute_perplexity(model, val_loader, device)
        print(f"Perplexity: {ppl_metrics['perplexity']:.2f}")
        print(f"Avg Loss: {ppl_metrics['avg_loss']:.4f}")
        print(f"Total Tokens: {ppl_metrics['total_tokens']}")
        print()
    
    # Compute energy metrics
    if val_loader is not None:
        print("Computing energy metrics...")
        energy_metrics = compute_energy_metrics(model, val_loader, device)
        print(f"Avg Energy: {energy_metrics['avg_energy']:.4f}")
        print(f"Energy Trend: {energy_metrics['energy_history'][-5:]}")
        print()
    
    # Analyze convergence
    if 'metrics' in checkpoint:
        if val_loader is not None and 'energy_history' in energy_metrics:
            # Check last 5 energy values
            recent_energies = energy_metrics['energy_history'][-5:]
            convergence = compute_convergence_metrics(recent_energies)
            
            print("Convergence Analysis:")
            print(f"  Trend: {convergence['trend']}")
            print(f"  Stability: {convergence['stability']}")
            print(f"  Recent energies: {convergence['recent_energies']}")
            print()
    
    # Check splat statistics
    print("Splat Store Statistics:")
    n_active = model.splats.n_active
    print(f"  Active splats: {n_active}")
    print(f"  Max splats: {model.splats.max_splats}")
    print(f"  Avg frequency: {model.splats.frequency[:n_active].mean().item():.4f}")
    print(f"  Avg kappa: {model.splats.kappa[:n_active].mean().item():.4f}")
    print(f"  Avg age: {model.splats.age[:n_active].mean().item():.2f}")
    print()
    
    # Recommendations
    print("=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    print()
    
    # Check for issues
    issues = []
    
    if val_loader is not None:
        if ppl_metrics['perplexity'] > 1000:
            issues.append(f"High perplexity ({ppl_metrics['perplexity']:.1f}) - model may need more training")
        elif ppl_metrics['perplexity'] < 10:
            issues.append(f"Very low perplexity ({ppl_metrics['perplexity']:.1f}) - possible overfitting")
        
        if energy_metrics['avg_energy'] > 100:
            issues.append(f"High energy ({energy_metrics['avg_energy']:.1f}) - may indicate convergence issues")
        
        if convergence['trend'] == 'diverging':
            issues.append("Energy diverging - check learning rate and regularization")
    
    if convergence['trend'] == 'converging' and convergence['stability'] != 'excellent':
        issues.append(f"Convergence {convergence['stability']} - may need more epochs or better regularization")
    
    if n_active < 1000:
        issues.append(f"Low splat count ({n_active}/{config.max_splats}) - consider increasing splat capacity")
    
    if issues:
        print("Issues found:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    else:
        print("No critical issues detected")
    
    print()
    print("=" * 60)


def batch_diagnose(checkpoint_dir='checkpoints', config_path='projects/ebm/config.py'):
    print("=" * 60)
    print("BATCH DIAGNOSTIC CHECKER")
    print("=" * 60)
    print()
    
    import os
    from evaluate import compute_convergence_metrics
    import torch
    
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt') and 'ebm_epoch_' in f]
    checkpoint_files.sort()
    
    if not checkpoint_files:
        print("No checkpoints found")
        return
    
    print(f"Found {len(checkpoint_files)} checkpoints")
    print(f"First: {checkpoint_files[0]}")
    print(f"Last: {checkpoint_files[-1]}")
    print()
    
    # Load energy history from all checkpoints
    all_energies = []
    for checkpoint_file in checkpoint_files[-5:]:
        try:
            checkpoint = torch.load(os.path.join(checkpoint_dir, checkpoint_file))
            if 'metrics' in checkpoint and 'avg_energy' in checkpoint['metrics']:
                all_energies.append(checkpoint['metrics']['avg_energy'])
        except Exception as e:
            print(f"Warning: Could not load {checkpoint_file}: {e}")
    
    if len(all_energies) >= 3:
        convergence = compute_convergence_metrics(all_energies)
        print("Energy Convergence Analysis (last 5 checkpoints):")
        print(f"  Trend: {convergence['trend']}")
        print(f"  Energies: {[f'{e:.2f}' for e in all_energies[-5:]]}")
        print("Insufficient checkpoints for convergence analysis")
    
    print()
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="EBM Diagnostic Tool")
    parser.add_argument("--checkpoint", help="Path to specific checkpoint")
    parser.add_argument("--batch", action="store_true", help="Run batch diagnosis on all checkpoints")
    parser.add_argument("--device", default="cpu", help="Device to use (cpu/vulkan)")
    
    args = parser.parse_args()
    
    if args.checkpoint:
        diagnose_checkpoint(args.checkpoint, device=args.device)
    else:
        batch_diagnose()
