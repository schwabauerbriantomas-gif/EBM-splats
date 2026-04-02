import torch
from config import EBMConfig
from model import EBMModel
from train import train_epoch
from soc import HistoryBuffer


def test_dummy_training():
    """Test forward/backward training logic on CPU."""
    config = EBMConfig(
        device="cpu",
        n_splats_init=100,
        vocab_size=1000,
        max_splats=200,
        knn_k=50,
    )

    model = EBMModel(config).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    soc_buffer = HistoryBuffer(capacity=50, latent_dim=config.latent_dim)

    class DummyLogger:
        def info(self, msg): pass
        def debug(self, msg): pass
        def warning(self, msg): pass

    # train_epoch expects batches with batch['tokens'] dict format
    class DummyDataset:
        def __init__(self):
            tokens = torch.randint(0, config.vocab_size, (4, 16))
            self.data = {'tokens': tokens, 'targets': tokens}

        def __iter__(self):
            yield self.data

        def __len__(self):
            return 1

    dataset = DummyDataset()

    print("Testing forward/backward training logic on CPU...")
    result = train_epoch(model, dataset, optimizer, scheduler, config, epoch=0,
                       soc_buffer=soc_buffer, logger=DummyLogger())
    print(f"Train epoch result: {result}")
    assert result is not None
    print(f"Active Splats Post Training: {model.splats.n_active}")


if __name__ == "__main__":
    test_dummy_training()
