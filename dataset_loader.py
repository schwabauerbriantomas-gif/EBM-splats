#!/usr/bin/env python3
"""
Optimized dataset loader for EBM training. Uses local text files
instead of downloading from HuggingFace every time.

Supported datasets:
  - TinyStories: D:/datasets/ebm/tinystories_train.txt (1.9GB, 2.1M stories)
  - TinyStories val: D:/datasets/ebm/tinystories_val.txt (19MB, 22K stories)
  - WikiText-103: cached via HuggingFace (fallback)
"""

import os
import logging
import warnings

# Suppress GPT-2 tokenizer warnings about sequence length
warnings.filterwarnings("ignore", message="Token indices sequence length")

logger = logging.getLogger(__name__)

# Lazy imports for heavy dependencies
def _get_torch_utils():
    """Lazy import torch utilities."""
    import torch
    from torch.utils.data import Dataset, DataLoader
    return torch, Dataset, DataLoader

def _get_transformers():
    """Lazy import transformers."""
    from transformers import AutoTokenizer
    return AutoTokenizer

# Default paths
DATASETS_DIR = "D:/datasets/ebm"
TINYSTORIES_TRAIN = os.path.join(DATASETS_DIR, "tinystories_train.txt")
TINYSTORIES_VAL = os.path.join(DATASETS_DIR, "tinystories_val.txt")


class TextFileDataset:
    """Memory-mapped text file sliced into token chunks."""

    def __init__(self, token_ids: list, seq_len: int):
        self.seq_len = seq_len
        self.tokens = token_ids
        self.total_chunks = len(self.tokens) // seq_len
        # Register with torch Dataset class
        torch, Dataset, _ = _get_torch_utils()
        self._Dataset = Dataset

    def __len__(self):
        return self.total_chunks

    def __getitem__(self, idx):
        torch, _, _ = _get_torch_utils()
        start = idx * self.seq_len
        end = start + self.seq_len
        return torch.tensor(self.tokens[start:end], dtype=torch.long)


def tokenize_file(filepath: str, tokenizer, max_chars: int = None, chunk_size: int = 50_000_000) -> list:
    """Tokenize a text file into a flat list of token IDs, reading in chunks to avoid OOM."""
    logger.info(f"Reading {filepath}...")
    
    all_token_ids = []
    chars_read = 0
    
    with open(filepath, 'r', encoding='utf-8') as f:
        while True:
            if max_chars and chars_read >= max_chars:
                break
            read_size = min(chunk_size, max_chars - chars_read) if max_chars else chunk_size
            text = f.read(read_size)
            if not text:
                break
            chars_read += len(text)
            
            # Tokenize this chunk
            chunk_ids = tokenizer.encode(text, add_special_tokens=False)
            all_token_ids.extend(chunk_ids)
            logger.info(f"  {chars_read:,} chars → {len(all_token_ids):,} tokens")
    
    logger.info(f"  Total: {chars_read:,} characters, {len(all_token_ids):,} tokens")
    return all_token_ids


def get_tinystories_dataloader(
    seq_len: int = 64,
    batch_size: int = 64,
    split: str = "train",
    max_chars: int = None,
    tokenizer_name: str = "gpt2",
    shuffle: bool = True,
):
    """Load TinyStories dataset from local D: drive."""

    if split == "train":
        filepath = TINYSTORIES_TRAIN
    elif split == "val":
        filepath = TINYSTORIES_VAL
    else:
        raise ValueError(f"Unknown split: {split}")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found: {filepath}. Run download script first.")

    # Lazy load transformers and torch
    AutoTokenizer = _get_transformers()
    torch, _, DataLoader = _get_torch_utils()

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    token_ids = tokenize_file(filepath, tokenizer, max_chars)
    dataset = TextFileDataset(token_ids, seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True,
                            num_workers=0, pin_memory=True)

    logger.info(f"TinyStories {split}: {len(dataset):,} chunks of [{batch_size} x {seq_len}]")
    return dataloader, tokenizer


def get_wikitext_dataloader(
    seq_len: int = 32,
    batch_size: int = 64,
    split: str = "train",
    max_samples: int = 5000,
    tokenizer_name: str = "gpt2",
):
    """Fallback: load WikiText-103 from HuggingFace cache."""
    from datasets import load_dataset

    AutoTokenizer = _get_transformers()
    torch, _, DataLoader = _get_torch_utils()

    raw = load_dataset("wikitext", "wikitext-103-raw-v1")
    data = raw[split]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    text_corpus = " ".join(
        row["text"] for row in data.select(range(min(len(data), max_samples)))
    )
    token_ids = tokenizer.encode(text_corpus, add_special_tokens=False)

    dataset = TextFileDataset(token_ids, seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                            num_workers=0, pin_memory=True)
    return dataloader, tokenizer


def get_dataloader(
    dataset_name: str = "tinystories",
    seq_len: int = 64,
    batch_size: int = 64,
    split: str = "train",
    max_chars: int = None,
    tokenizer_name: str = "gpt2",
    **kwargs,
):
    """Unified dataloader factory."""
    if dataset_name == "tinystories":
        return get_tinystories_dataloader(
            seq_len=seq_len, batch_size=batch_size, split=split,
            max_chars=max_chars, tokenizer_name=tokenizer_name,
        )
    elif dataset_name == "wikitext":
        return get_wikitext_dataloader(
            seq_len=seq_len, batch_size=batch_size, split=split,
            max_samples=kwargs.get("max_samples", 5000),
            tokenizer_name=tokenizer_name,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    print("=== TinyStories (100K chars test) ===")
    dl, tok = get_dataloader("tinystories", seq_len=64, batch_size=8, max_chars=100_000)
    batch = next(iter(dl))
    print(f"Batch shape: {batch.shape}")
    print(f"Sample tokens (first 10): {batch[0][:10].tolist()}")
    print(f"Decoded: {tok.decode(batch[0][:10])}")
    
    print("\n=== WikiText fallback ===")
    dl2, tok2 = get_dataloader("wikitext", seq_len=32, batch_size=8, max_samples=100)
    batch2 = next(iter(dl2))
    print(f"Batch shape: {batch2.shape}")
