"""
PGLF Trainer — Orchestrates the three-phase Pareto-Guided Langevin Flow training.

Phase 1: Langevin exploration (map the energy landscape)
Phase 2: Pareto filtering (select optimal trajectories)
Phase 3: Flow Matching distillation (learn deterministic embedding generation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import time
import json
from typing import Optional, Dict, Tuple
from pathlib import Path

# PGLF components
from .encoders import OmnimodalEncoder, TextEncoder
from .flow_matching import HypersphereFlowMatching, FlowMatchingLoss
from .contrastive_head import HypersphereContrastiveLoss, UniformityAlignmentLoss
from .pareto_filter import ParetoFilter

# EBM-splats components (reuse)
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from geometry import project_to_tangent, normalize_sphere
from langevin import sample_langevin
from energy import EnergyFunction
from splats import SplatStorage


class TextEmbeddingDataset(Dataset):
    """Dataset for text embedding training."""
    
    def __init__(self, texts: list, tokenizer, max_len: int = 128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }


class PGLFTrainer:
    """
    Three-phase trainer for Pareto-Guided Langevin Flow embeddings.
    """
    
    def __init__(
        self,
        dim: int = 640,
        device: str = "cuda",
        output_dir: str = "/mnt/c/Users/Brian/Desktop/EBM-splats/pglf_checkpoints",
        # Phase 1: Langevin
        langevin_steps: int = 200,
        langevin_dt: float = 0.001,
        # Phase 2: Pareto
        pareto_n_keep: int = 512,
        # Phase 3: Flow Matching
        fm_hidden_dim: int = 1280,
        fm_n_layers: int = 6,
        fm_n_steps_inference: int = 10,
        # Training
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        batch_size: int = 32,
        amp: bool = True,
    ):
        self.dim = dim
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.langevin_steps = langevin_steps
        self.langevin_dt = langevin_dt
        self.fm_n_steps_inference = fm_n_steps_inference
        self.batch_size = batch_size
        
        # Initialize components
        self.encoder = OmnimodalEncoder(dim=dim).to(device)
        self.flow_model = HypersphereFlowMatching(
            dim=dim, hidden_dim=fm_hidden_dim, n_layers=fm_n_layers,
        ).to(device)
        
        self.contrastive_loss = HypersphereContrastiveLoss(temperature=0.07).to(device)
        self.alignment_loss = UniformityAlignmentLoss(alpha=2.0).to(device)
        self.fm_loss = FlowMatchingLoss(sigma_min=1e-4).to(device)
        self.pareto_filter = ParetoFilter(n_keep=pareto_n_keep)
        
        # Optimizers
        self.encoder_optimizer = AdamW(
            self.encoder.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.flow_optimizer = AdamW(
            self.flow_model.parameters(), lr=lr * 0.5, weight_decay=weight_decay
        )
        
        # Schedulers
        self.encoder_scheduler = None
        self.flow_scheduler = None
        
        # AMP
        self.amp = amp and device == "cuda"
        self.scaler = torch.amp.GradScaler("cuda") if self.amp else None
        
        # EBM components (optional — for energy-guided sampling)
        self.splat_store = None
        self.energy_fn = None
        self.langevin_sampler = None
        
        # Pareto trajectory buffer
        self.golden_trajectories = []
        
        # Stats
        self.stats = {
            "phase": 0,
            "step": 0,
            "encoder_loss": [],
            "fm_loss": [],
            "pareto_front_sizes": [],
        }
    
    def init_ebm_components(self):
        """Initialize EBM-splats components for Langevin exploration."""
        try:
            from config import EBMConfig
            config = EBMConfig()
            self.splat_store = SplatStorage(
                n_splats=config.max_splats,
                dim=config.embedding_dim,
                device=self.device,
            )
            self.energy_fn = EnergyFunction(config).to(self.device)
            self.langevin_sampler = LangevinSampler(
                n_steps=self.langevin_steps,
                dt=self.langevin_dt,
            )
            print(f"[PGLF] EBM components loaded: {config.max_splats} splats, dim={config.embedding_dim}")
        except Exception as e:
            print(f"[PGLF] EBM components not available: {e}")
            print("[PGLF] Training without Langevin energy guidance")
    
    def phase1_contrastive_training(
        self,
        dataloader: DataLoader,
        n_epochs: int = 5,
        phase2_interval: int = 500,
        phase3_interval: int = 2000,
    ):
        """
        Phase 1: Train encoder with contrastive loss.
        Periodically runs Phase 2 (Pareto) and Phase 3 (Flow Matching).
        
        Uses augmented views of the same text as positive pairs.
        """
        self.stats["phase"] = 1
        n_steps = len(dataloader) * n_epochs
        self.encoder_scheduler = CosineAnnealingLR(self.encoder_optimizer, T_max=n_steps)
        
        print(f"[PGLF Phase 1] Starting contrastive training: {n_epochs} epochs, {n_steps} steps")
        print(f"[PGLF Phase 1] Phase 2 every {phase2_interval} steps, Phase 3 every {phase3_interval} steps")
        
        global_step = 0
        
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            epoch_acc = 0.0
            n_batches = 0
            
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                
                # Create two augmented views
                # Augmentation 1: Random token dropout (replace with 0)
                ids_aug = input_ids.clone()
                mask_aug = (torch.rand_like(attention_mask.float()) > 0.15).long() * attention_mask
                ids_aug[mask_aug == 0] = 0
                
                # Disable AMP for first 100 steps (warmup)
                use_amp = self.amp and global_step > 100
                
                with torch.amp.autocast("cuda", enabled=use_amp):
                    emb1, _ = self.encoder.encode_text(input_ids, attention_mask)
                    emb2, _ = self.encoder.encode_text(ids_aug, mask_aug)
                    
                    # Check for NaN
                    if torch.isnan(emb1).any() or torch.isnan(emb2).any():
                        print(f"  WARNING: NaN detected at step {global_step}, skipping")
                        continue
                    
                    # Contrastive loss
                    loss_cl, metrics_cl = self.contrastive_loss(emb1, emb2)
                    
                    # Alignment + uniformity loss
                    loss_au, metrics_au = self.alignment_loss(emb1, emb2)
                    
                    loss = loss_cl + 0.5 * loss_au
                    
                    # NaN check on loss
                    if torch.isnan(loss):
                        print(f"  WARNING: NaN loss at step {global_step}, skipping")
                        continue
                
                # Backward
                self.encoder_optimizer.zero_grad()
                if self.scaler:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.encoder_optimizer)
                    torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
                    self.scaler.step(self.encoder_optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
                    self.encoder_optimizer.step()
                
                self.encoder_scheduler.step()
                global_step += 1
                
                epoch_loss += loss.item()
                epoch_acc += metrics_cl["accuracy"]
                n_batches += 1
                
                self.stats["step"] = global_step
                self.stats["encoder_loss"].append(loss.item())
                
                # Phase 2: Pareto filtering
                if global_step % phase2_interval == 0:
                    self._run_phase2(emb1.detach())
                
                # Phase 3: Flow Matching distillation
                if global_step % phase3_interval == 0 and len(self.golden_trajectories) > 0:
                    self._run_phase3()
                
                if global_step % 100 == 0:
                    avg_loss = epoch_loss / n_batches
                    avg_acc = epoch_acc / n_batches
                    print(f"  Step {global_step}/{n_steps} | Loss: {avg_loss:.4f} | Acc: {avg_acc:.3f} | LR: {self.encoder_scheduler.get_last_lr()[0]:.2e}")
            
            avg_loss = epoch_loss / max(n_batches, 1)
            avg_acc = epoch_acc / max(n_batches, 1)
            print(f"[PGLF Phase 1] Epoch {epoch+1}/{n_epochs} | Loss: {avg_loss:.4f} | Acc: {avg_acc:.3f}")
            
            # Save checkpoint
            self._save_checkpoint(epoch)
        
        print(f"[PGLF Phase 1] Complete. Golden trajectories: {len(self.golden_trajectories)}")
    
    def _run_phase2(self, embeddings: torch.Tensor):
        """
        Phase 2: Pareto filtering of current embeddings.
        Collects golden trajectories for Phase 3.
        """
        self.stats["phase"] = 2
        
        # Compute energy scores if EBM available
        energy_scores = None
        if self.energy_fn is not None:
            with torch.no_grad():
                try:
                    energy_scores = self.energy_fn(embeddings.unsqueeze(1)).squeeze(1)
                except:
                    energy_scores = None
        
        # Pareto filter
        selected, objectives, fronts = self.pareto_filter.filter(
            embeddings, energy_scores=energy_scores
        )
        
        # Add to golden trajectories buffer (keep last 5 rounds)
        self.golden_trajectories.append(selected.cpu())
        if len(self.golden_trajectories) > 5:
            self.golden_trajectories = self.golden_trajectories[-5:]
        
        pareto_info = self.pareto_filter.get_pareto_front(embeddings, objectives)
        self.stats["pareto_front_sizes"].append(pareto_info["pareto_front_size"])
        
        print(f"[PGLF Phase 2] Pareto: {pareto_info['pareto_front_size']}/{len(embeddings)} in front | "
              f"Total golden: {sum(t.shape[0] for t in self.golden_trajectories)}")
    
    def _run_phase3(self, n_steps: int = 200):
        """
        Phase 3: Train Flow Matching on golden trajectories.
        """
        self.stats["phase"] = 3
        
        # Combine all golden trajectories
        all_golden = torch.cat(self.golden_trajectories, dim=0).to(self.device)
        n_golden = all_golden.shape[0]
        
        if n_golden < self.batch_size:
            print(f"[PGLF Phase 3] Not enough golden trajectories ({n_golden}), skipping")
            return
        
        # Initialize scheduler if needed
        if self.flow_scheduler is None:
            self.flow_optimizer = AdamW(self.flow_model.parameters(), lr=5e-4, weight_decay=0.01)
            self.flow_scheduler = CosineAnnealingLR(self.flow_optimizer, T_max=n_steps)
        
        print(f"[PGLF Phase 3] Training Flow Matching on {n_golden} golden embeddings for {n_steps} steps")
        
        self.flow_model.train()
        avg_loss = 0.0
        
        for step in range(n_steps):
            # Sample batch from golden trajectories
            idx = torch.randint(0, n_golden, (min(self.batch_size, n_golden),), device=self.device)
            x_1 = all_golden[idx]
            
            # Use the embedding itself as condition (self-conditioning)
            condition = x_1
            modality = torch.zeros(x_1.shape[0], dtype=torch.long, device=self.device)
            
            with torch.amp.autocast("cuda", enabled=self.amp):
                loss = self.fm_loss(self.flow_model, x_1, condition, modality)
            
            self.flow_optimizer.zero_grad()
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.flow_optimizer)
                torch.nn.utils.clip_grad_norm_(self.flow_model.parameters(), 1.0)
                self.scaler.step(self.flow_optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.flow_model.parameters(), 1.0)
                self.flow_optimizer.step()
            
            self.flow_scheduler.step()
            avg_loss += loss.item()
            
            if (step + 1) % 50 == 0:
                print(f"  FM Step {step+1}/{n_steps} | Loss: {avg_loss / (step+1):.6f}")
        
        self.stats["fm_loss"].append(avg_loss / n_steps)
        print(f"[PGLF Phase 3] Complete. Avg FM loss: {avg_loss / n_steps:.6f}")
    
    def _save_checkpoint(self, epoch: int):
        """Save all model states."""
        checkpoint = {
            "epoch": epoch,
            "step": self.stats["step"],
            "encoder_state": self.encoder.state_dict(),
            "flow_state": self.flow_model.state_dict(),
            "encoder_optimizer": self.encoder_optimizer.state_dict(),
            "flow_optimizer": self.flow_optimizer.state_dict(),
            "stats": self.stats,
        }
        path = self.output_dir / f"pglf_epoch_{epoch+1}.pt"
        torch.save(checkpoint, path)
        
        # Also save as best/latest
        torch.save(checkpoint, self.output_dir / "pglf_latest.pt")
        
        print(f"[PGLF] Checkpoint saved: {path}")
    
    def export_for_inference(self, path: Optional[str] = None):
        """
        Export the trained models for inference.
        Only saves what's needed for embedding generation.
        """
        if path is None:
            path = self.output_dir / "pglf_inference.pt"
        
        inference_state = {
            "encoder_state": self.encoder.state_dict(),
            "flow_state": self.flow_model.state_dict(),
            "config": {
                "dim": self.dim,
                "fm_n_steps": self.fm_n_steps_inference,
            },
        }
        torch.save(inference_state, path)
        print(f"[PGLF] Inference model exported: {path}")
    
    @torch.no_grad()
    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_flow: bool = False,
    ) -> torch.Tensor:
        """
        Generate embeddings for input text.
        
        Args:
            use_flow: If True, use Flow Matching for refined embeddings.
                     If False, use encoder directly (faster).
        """
        self.encoder.eval()
        
        if use_flow:
            self.flow_model.eval()
            condition, modality = self.encoder.encode_text(input_ids, attention_mask)
            embeddings = self.flow_model.sample(
                condition=condition,
                modality=modality,
                n_steps=self.fm_n_steps_inference,
            )
        else:
            embeddings, _ = self.encoder.encode_text(input_ids, attention_mask)
        
        return embeddings


def main():
    """CLI entry point for PGLF training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="PGLF Embedding Training")
    parser.add_argument("--phase", type=int, default=0, help="Start phase (0=all, 1=contrastive, 2=pareto, 3=flow)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default="/mnt/c/Users/Brian/Desktop/EBM-splats/pglf_checkpoints")
    parser.add_argument("--dataset", type=str, default="tinystories", help="Dataset: tinystories, wikitext")
    args = parser.parse_args()
    
    print(f"=== PGLF Training ===")
    print(f"Device: {args.device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Initialize trainer
    trainer = PGLFTrainer(
        device=args.device,
        batch_size=args.batch_size,
        lr=args.lr,
        output_dir=args.output,
    )
    
    # Load dataset
    print(f"\nLoading dataset: {args.dataset}")
    
    if args.dataset == "tinystories":
        # Try local TinyStories files
        data_paths = [
            "/mnt/d/datasets/ebm/tinystories_train.txt",
            "/mnt/d/datasets/ebm/TinyStoriesV2-GPT4-train.txt",
            "/mnt/c/Users/Brian/Desktop/datasets/TinyStories-train.txt",
        ]
        texts = []
        for p in data_paths:
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    texts = [line.strip() for line in f.readlines() if len(line.strip()) > 20][:50000]
                print(f"  Loaded {len(texts)} texts from {p}")
                break
        
        if not texts:
            # Generate synthetic data
            print("  No TinyStories found, using synthetic data")
            texts = [
                f"This is training sample number {i} for the embedding model."
                for i in range(10000)
            ]
    else:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
        texts = [t for t in ds["text"] if len(t) > 50][:50000]
    
    # Tokenizer
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    dataset = TextEmbeddingDataset(texts, tokenizer)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
    )
    
    print(f"  Dataset: {len(dataset)} samples, {len(dataloader)} batches")
    
    # Initialize EBM components
    trainer.init_ebm_components()
    
    # Train
    trainer.phase1_contrastive_training(
        dataloader,
        n_epochs=args.epochs,
        phase2_interval=500,
        phase3_interval=2000,
    )
    
    # Export
    trainer.export_for_inference()
    
    print("\n=== PGLF Training Complete ===")
    print(f"Checkpoints: {args.output}")
    print(f"Stats: {json.dumps(trainer.stats, indent=2, default=str)}")


if __name__ == "__main__":
    main()
