import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextDataset(Dataset):
    def __init__(self, tokenized_data, seq_len: int = 16):
        """
        Dynamically slices a continuous stream of tokens into chunks of `seq_len`.
        """
        self.seq_len = seq_len
        self.tokens = tokenized_data
        self.total_chunks = len(self.tokens) // seq_len

    def __len__(self):
        return self.total_chunks

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len
        return torch.tensor(self.tokens[start:end], dtype=torch.long)

def get_dataloader(tokenizer_name: str = "gpt2", dataset_name: str = "wikitext", 
                   config_name: str = "wikitext-2-raw-v1", split: str = "train",
                   batch_size: int = 8, seq_len: int = 16, max_samples: int = 5000):
    """
    Downloads, tokenizes, and maps text blocks to PyTorch DataLoader structs.
    """
    logger.info(f"Loading '{dataset_name}' dataset ({split})...")
    
    try:
        raw_datasets = load_dataset(dataset_name, config_name)
        data_split = raw_datasets[split]
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise
        
    logger.info(f"Loading tokenizer '{tokenizer_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Optional pad token setting if required by specific batching, but we utilize 
    # exact chunks here.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    logger.info("Tokenizing active text samples...")
    # Concatenate texts dropping empty lines
    text_corpus = " ".join([row["text"] for row in data_split.select(range(min(len(data_split), max_samples)))])
    
    # Process without truncation targeting a continuous token stream
    tokenized = tokenizer.encode(text_corpus, add_special_tokens=False)
    
    logger.info(f"Generated {len(tokenized)} tokens.")
    
    # Construct sequence slicer dataset
    dataset = TextDataset(tokenized, seq_len=seq_len)
    
    # Drop_last=True ensures all batches are uniformly sized (B, seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    logger.info(f"Dataloader yielding {len(dataloader)} batches of dimension [{batch_size}, {seq_len}]")
    
    return dataloader, tokenizer
