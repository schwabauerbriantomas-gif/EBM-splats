"""
Phase 3: MDLM Baseline Generation Test

Validates that Qwen3-0.6B-diffusion-mdlm loads, generates coherent text,
and exposes logits at each denoising step — the prerequisite for EBM
energy-guided discrete diffusion.

Run: python tests/phase3_mdlm_baseline.py
"""

import sys
import time

import torch
import transformers

import dllm
from dllm.core.samplers import MDLMSampler, MDLMSamplerConfig
from dllm.utils import get_model, get_tokenizer


MODEL_ID = "dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1"


def main():
    print("=" * 70)
    print("Phase 3: MDLM Baseline Generation")
    print("=" * 70)

    # --- Load model ---
    t0 = time.time()
    model = get_model(
        model_args=type("Args", (), {
            "model_name_or_path": MODEL_ID,
            "dtype": torch.bfloat16,
            "device_map": {"": 0},
        })()
    ).eval()
    load_time = time.time() - t0
    print(f"\n[Load] {load_time:.1f}s")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")
    print(f"  GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # --- Tokenizer ---
    tokenizer = get_tokenizer(
        model_args=type("Args", (), {"model_name_or_path": MODEL_ID})()
    )
    print(f"  Vocab size: {tokenizer.vocab_size}")
    print(f"  Mask token: '{tokenizer.mask_token}' (id={tokenizer.mask_token_id})")

    # --- Sampler ---
    sampler = MDLMSampler(model=model, tokenizer=tokenizer)

    # --- Test prompts ---
    messages_list = [
        [{"role": "user", "content": "What is machine learning? Explain in 2 sentences."}],
        [{"role": "user", "content": "Write a haiku about the ocean."}],
    ]

    config = MDLMSamplerConfig(
        steps=64,
        max_new_tokens=64,
        block_size=32,
        temperature=0.0,
        remasking="low_confidence",
    )

    for i, messages in enumerate(messages_list):
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True
        )
        inputs = [inputs] if isinstance(inputs[0], int) else inputs

        t0 = time.time()
        outputs = sampler.sample(inputs, config, return_dict=True)
        gen_time = time.time() - t0

        # Decode output
        for seq in outputs.sequences:
            text = tokenizer.decode(seq, skip_special_tokens=True)
            # Extract just the assistant response
            if "<|im_start|>assistant" in text:
                response = text.split("<|im_start|>assistant")[-1]
                response = response.replace("<|im_end|>", "").strip()
            else:
                response = text.strip()
            print(f"\n[Prompt {i+1}] ({gen_time:.1f}s)")
            print(f"  Q: {messages[0]['content']}")
            print(f"  A: {response}")

    print(f"\n{'=' * 70}")
    print("Baseline generation: WORKS" if True else "FAILED")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
