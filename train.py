#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EBM Training Script with Phase 1 Improvements
Integrates enhanced splat initialization, curriculum learning, and detailed logging.
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataclasses import dataclass
import logging

# EBM modules
from config import EBMConfig
from model import EBMModel
from splats import SplatStorage
from energy import EnergyFunction
from langevin import sample_langevin
from decoder import EBMDecoder
from geometry import normalize_sphere, geodesic_distance
from soc import HistoryBuffer, maybe_consolidate, compute_order_parameter
from dataset_utils import get_dataloader
from logger import TrainingLogger
from evaluate import compute_perplexity, compute_energy_metrics, compute_convergence_metrics, save_checkpoint

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def get_current_epoch(checkpoint_dir):
    """Find the most recent checkpoint."""
    if not os.path.exists(checkpoint_dir):
        return 0
    
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt') and 'ebm_epoch_' in f]
    if not checkpoints:
        return 0
    
    epochs = [int(f.split('ebm_epoch_')[1].split('.')[0]) for f in checkpoints]
    return max(epochs)


def validate_convergence(metrics_history, config):
    """Validate convergence based on energy trends."""
    if len(metrics_history) < 3:
        return False, {'trend': 'insufficient_data'}
    
    recent_energies = [m['avg_energy'] for m in metrics_history[-5:]]
    
    # Check for decreasing energy (good convergence)
    is_decreasing = True
    for i in range(1, len(recent_energies)):
        if recent_energies[i] >= recent_energies[i-1]:
            is_decreasing = False
            break
    
    if is_decreasing:
        trend = 'converging'
        stability = 'good'
    else:
        trend = 'diverging'
        stability = 'poor'
    
    return True, {
        'trend': trend,
        'stability': stability,
        'recent_energies': recent_energies
    }


def curriculum_adjustment(current_epoch, num_curriculum_epochs, config, splat_store):
    """Adjust splat learning rate based on curriculum phase."""
    if not config.enable_curriculum_learning:
        return 1.0  # No adjustment
    
    phase = min(current_epoch // num_curriculum_epochs, 2)
    
    if phase == 0:
        # Phase 1: Learn basic representations (high temperature)
        return 1.0  # Full exploration
    elif phase == 1:
        # Phase 2: Expand vocabulary (medium temperature)
        return 0.5
    else:
        # Phase 3: Fine-tune (low temperature)
        return 0.25


def train_epoch(model, dataloader, optimizer, scheduler, config, epoch, soc_buffer, logger):
    """Train one epoch with Phase 1 improvements."""
    model.train()
    total_loss = 0
    total_energy = 0
    splat_added = 0
    num_batches = len(dataloader)
    
    start_time = time.time()
    logger.info(f"Starting Epoch {epoch} with {num_batches} batches")
    
    for batch_idx, batch in enumerate(dataloader):
        # Get batch data
        tokens = batch['tokens']
        targets = batch['targets']
        
        # Apply curriculum learning rate adjustment
        learning_rate_adj = curriculum_adjustment(epoch, config.curriculum_epochs, config, model.splats)
        effective_lr = config.learning_rate * learning_rate_adj
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        x = model.embed(tokens)
        energy = model.compute_energy(x)
        
        # Score matching (denoising)
        # Use negative energy to get score
        # Since energy is negative, the score is -energy
        # Minimizing energy means maximizing score
        loss = energy.mean()  # Average over batch
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        for param_group in optimizer.param_groups:
            for param in param_group['params']:
                param.data.mul_(effective_lr / config.learning_rate)
        
        total_loss += loss.item()
        total_energy += energy.mean().item()
        
        # SOC check and consolidation every 100 batches
        if batch_idx % config.soc_check_interval == 0 and batch_idx > 0:
            active_before = model.splats.n_active
            maybe_consolidate(model.splats, config, soc_buffer)
            splat_added += model.splats.n_active - active_before
        
        # Log progress
        if batch_idx % 100 == 0:
            avg_loss = total_loss / (batch_idx + 1)
            avg_energy = total_energy / (batch_idx + 1)
            logger.info(f"Epoch {epoch} | Batch {batch_idx}/{num_batches} | Loss: {avg_loss:.4f} | Energy: {avg_energy:.4f} | Splats: {model.splats.n_active} | Added: {splat_added}")
    
    end_time = time.time()
    epoch_duration = end_time - start_time
    
    # Compute metrics
    avg_loss = total_loss / num_batches
    avg_energy = total_energy / num_batches
    
    metrics = {
        'epoch': epoch,
        'duration_seconds': epoch_duration,
        'avg_loss': avg_loss,
        'avg_energy': avg_energy,
        'num_batches': num_batches,
        'splat_count': model.splats.n_active,
        'splats_added': splat_added,
        'learning_rate_adj': learning_rate_adj
    }
    
    return metrics


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="EBM Training with Phase 1 Improvements")
    parser.add_argument("--device", default="vulkan", choices=["cpu", "vulkan"], help="Device (default: vulkan)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--config", default="projects/ebm/config.py", help="Config file path")
    parser.add_argument("--checkpoint-dir", default="projects/ebm/checkpoints", help="Checkpoint directory")
    parser.add_argument("--resume", help="Resume from checkpoint")
    parser.add_argument("--validate", action="store_true", help="Validate checkpoints only")
    
    args = parser.parse_args()
    
    # Load config
    try:
        from config import EBMConfig
        config = EBMConfig()
        
        if args.resume:
            resume_epoch = get_current_epoch(args.checkpoint_dir)
            config.current_epoch = resume_epoch
            logger.info(f"Resuming from epoch {resume_epoch}")
        else:
            config.current_epoch = 0
        
        # Create model
        model = EBMModel(config).to(config.device)
        
        # Create SOC buffer
        soc_buffer = HistoryBuffer(
            capacity=config.soc_buffer_capacity,
            latent_dim=config.latent_dim
        )
        
        # Create dataloader
        logger.info("Loading training dataset...")
        dataloader = get_dataloader(
            tokenizer_name="gpt2",
            dataset_name="wikitext",
            config_name="wikitext-103-raw-v1",
            split="train",
            batch_size=args.batch_size,
            seq_len=config.context_local,
            max_samples=100000,  # 100k subsets
            device=config.device
        )
        logger.info(f"Dataset loaded: {len(dataloader.dataset)} samples, {len(dataloader)} batches")
        
        # Create optimizer with AdamW
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.reg_weight)
        
        # Create scheduler
        num_training_steps = len(dataloader) * args.epochs
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_training_steps,
            eta_min=0.0,
            eta_max=config.learning_rate
        )
        
        # Create logger
        train_logger = TrainingLogger(output_dir="logs/ebm")
        
        # Validation only
        if args.validate:
            logger.info("Running validation check...")
            checkpoints = [f"projects/ebm/checkpoints/ebm_epoch_{i}.pt" for i in range(args.epochs)]
            
            for checkpoint_path in checkpoints:
                if not os.path.exists(checkpoint_path):
                    logger.warning(f"Checkpoint not found: {checkpoint_path}")
                    continue
                
                logger.info(f"Validating checkpoint: {checkpoint_path}")
                
                try:
                    from diagnose import diagnose_checkpoint
                    diagnose_result = diagnose_checkpoint(
                        checkpoint_path=checkpoint_path,
                        config_path=args.config,
                        device=args.device
                    )
                    logger.info(f"Validation result: {diagnose_result}")
                except Exception as e:
                    logger.error(f"Validation failed: {e}")
            
            logger.info("Validation check complete")
            return
        
        # Resume from checkpoint if requested
        if args.resume:
            checkpoint_path = f"projects/ebm/checkpoints/ebm_epoch_{config.current_epoch}.pt"
            if os.path.exists(checkpoint_path):
                logger.info(f"Loading checkpoint: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path)
                model.load_state_dict(checkpoint['model_state_dict'])
                
                if 'optimizer_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'scheduler_state_dict' in checkpoint:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                if 'soc_buffer' in checkpoint:
                    soc_buffer.load_state_dict(checkpoint['soc_buffer'])
                if 'splats' in checkpoint:
                    model.splats.load_state_dict(checkpoint['splats'])
                
                logger.info(f"Checkpoint loaded successfully")
            else:
                logger.warning(f"Checkpoint not found: {checkpoint_path}")
                config.current_epoch = 0
        
        # Create checkpoint directory
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        
        # Training loop
        logger.info(f"Starting training for {args.epochs} epochs on {config.device}")
        
        all_metrics = []
        metrics_history = []
        
        for epoch in range(config.current_epoch, config.current_epoch + args.epochs):
            metrics = train_epoch(model, dataloader, optimizer, scheduler, config, epoch, soc_buffer, train_logger)
            all_metrics.append(metrics)
            metrics_history.append(metrics)
            
            # Save checkpoint
            ckpt_path = save_checkpoint(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                metrics=metrics,
                output_dir=args.checkpoint_dir
            )
            
            logger.info(f"Epoch {epoch} completed | Loss: {metrics['avg_loss']:.4f} | Energy: {metrics['avg_energy']:.4f}")
            
            # Validate convergence every epoch
            if epoch >= 2:
                is_valid, convergence = validate_convergence(metrics_history[-5:], config)
                logger.info(f"Convergence check: {is_valid} | {convergence}")
                
                if convergence['trend'] == 'diverging':
                    logger.warning(f"Energy diverging! Consider lowering learning rate")
                elif convergence['stability'] != 'excellent':
                    logger.info(f"Convergence not yet excellent (stability: {convergence['stability']})")
        
        # Final validation
        logger.info("Running final validation...")
        final_checkpoint = f"projects/ebm/checkpoints/ebm_epoch_{config.current_epoch + args.epochs}.pt"
        try:
            from diagnose import diagnose_checkpoint
            diagnose_result = diagnose_checkpoint(
                checkpoint_path=final_checkpoint,
                config_path=args.config,
                device=args.device
            )
            logger.info(f"Final validation: {diagnose_result}")
        except Exception as e:
            logger.error(f"Final validation failed: {e}")
        
        # Training summary
        train_logger.save_log()
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info(f"Total Epochs: {args.epochs}")
        logger.info(f"Final Epoch: {config.current_epoch + args.epochs}")
        logger.info(f"Checkpoints saved to: {args.checkpoint_dir}")
        logger.info(f"Logs saved to: {train_logger.output_dir}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Training failed with exception: {e}")
        raise e


if __name__ == "__main__":
    main()
