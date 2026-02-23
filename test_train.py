import torch
from config import EBMConfig
from model import EBMModel
from train import train_epoch
from soc import HistoryBuffer

class DummyDataset:
    def __init__(self, size=4, seq_len=16, vocab_size=1000):
        self.data = torch.randint(0, vocab_size, (size, seq_len))
        
    def __iter__(self):
        yield self.data
        
    def __len__(self):
        return 1

def test_dummy_training():
    config = EBMConfig(
        device="cpu",
        n_splats_init=10,
        vocab_size=1000,
        max_splats=100
    )
    
    model = EBMModel(config).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    dataset = DummyDataset()
    
    soc_buffer = HistoryBuffer(capacity=50, latent_dim=config.latent_dim)
    
    print("Testing forward/backward training logic on CPU...")
    loss = train_epoch(model, dataset, optimizer, config, soc_buffer)
    print(f"Dummy epoch complete. Loss: {loss:.4f}")
    
    print(f"Active Splats Post Training: {model.splats.n_active}")
    
if __name__ == "__main__":
    test_dummy_training()
