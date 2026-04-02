import torch
import torch.nn.functional as F
from config import EBMConfig
from model import EBMModel


class DummyTokenizer:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.eos_token_id = vocab_size - 1

    def encode(self, text, add_special_tokens=False):
        return [1, 5, 20]

    def decode(self, tokens):
        return f"Dummy decoded text of length {len(tokens)}"


def test_dummy_generation():
    """Test model.generate() end-to-end on CPU."""
    config = EBMConfig(
        device="cpu",
        n_splats_init=100,
        vocab_size=1000,
        max_splats=200,
        langevin_steps=5,
        knn_k=50,
    )

    model = EBMModel(config).to(config.device)
    model.eval()

    prompt_tokens = torch.tensor([[1, 5, 20]], dtype=torch.long)

    print("Testing model.generate() on CPU...")
    with torch.no_grad():
        output = model.generate(prompt_tokens, max_new_tokens=3)

    assert output is not None, "generate() returned None"
    assert isinstance(output, torch.Tensor), f"Expected Tensor, got {type(output)}"
    print(f"Generated output shape: {output.shape}")


if __name__ == "__main__":
    test_dummy_generation()
