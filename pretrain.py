import os
import torch
import logging

from config import EBMConfig
from model import EBMModel
from train import train_epoch
from soc import HistoryBuffer
from dataset_utils import get_dataloader
from transformers import get_cosine_schedule_with_warmup

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def pretrain():
    # Force localized Vulkan-targeted training dimensions
    config = EBMConfig(
        device="vulkan",
        latent_dim=640,
        n_splats_init=3000,          # Increased for GPU
        max_splats=15000,            # Increased for GPU capacity
        vocab_size=50257,            # GPT2 vocab size precisely
        moe_experts=4,               
        moe_active=2,
        knn_k=64,                    # Increased neighbor searches
    )
    
    logger.info("Initializing pretraining dataset (Wikitext)...")
    try:
        dataloader, tokenizer = get_dataloader(
            tokenizer_name="gpt2",
            dataset_name="wikitext",
            config_name="wikitext-103-raw-v1", # Massive corpus
            split="train",
            batch_size=32,          # Maximize GPU utility
            seq_len=32,             # Larger contextual window
            max_samples=100000      # 100k subsets targeting LLM functional generation bounds
        )
    except Exception as e:
        logger.error(f"Failed to bootstrap data environment: {e}")
        return
        
    logger.info("Instantiating generic EBMModel architecture...")
    active_torch_device = "cpu" if config.device == "vulkan" else config.device
    model = EBMModel(config).to(active_torch_device)
    
    # Establish optimizer utilizing moderate learning rate heuristics for underdamped Langevin
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    # Persistent memory buffers supporting generative metrics
    soc_buffer = HistoryBuffer(capacity=500, latent_dim=config.latent_dim)
    
    num_epochs = 10
    total_steps = len(dataloader) * num_epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=1000, num_training_steps=total_steps
    )
    
    logger.info("======= Commencing EBM Training (Phase 4 Continuous) =======")
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    for epoch in range(num_epochs):
        logger.info(f"--- Epoch {epoch+1} ---")
        avg_loss = train_epoch(model, dataloader, optimizer, config, soc_buffer, scheduler)
        
        logger.info(f"Epoch Completion | Average Denoising Score Loss: {avg_loss:.4f}")
        logger.info(f"Active Parameter Geometries (Splats): {model.splats.n_active} / {model.splats.max_splats}")
        
        # Save model checkpoint
        ckpt_path = os.path.join(checkpoint_dir, f"ebm_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), ckpt_path)
        logger.info(f"Model saved to {ckpt_path}")
        
    logger.info("Execution complete.")

if __name__ == "__main__":
    pretrain()
