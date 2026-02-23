import torch
from config import EBMConfig
from model import EBMModel
from generate import generate

class DummyTokenizer:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.eos_token_id = vocab_size - 1
        
    def encode(self, text, add_special_tokens=False):
        # dummy encoding: just return a static list of ids
        return [1, 5, 20]
        
    def decode(self, tokens):
        return f"Dummy decoded text of length {len(tokens)}"

def test_dummy_generation():
    config = EBMConfig(
        device="cpu",
        n_splats_init=10,
        vocab_size=1000,
        max_splats=100,
        langevin_steps=10  # Reduced for local fast validation
    )
    
    model = EBMModel(config).to(config.device)
    tokenizer = DummyTokenizer(config.vocab_size)
    
    print("Testing Langevin generation loop on CPU...")
    output = generate(model, tokenizer, "hello world", max_tokens=5, temperature=1.0)
    print(f"Generated text: {output}")

if __name__ == "__main__":
    test_dummy_generation()
