#!/usr/bin/env python3
"""
Phase 3: Energy-Guided DiffusionGemma

Instead of autoregressive GPT-2 + prefix, use DiffusionGemma's block diffusion
decoder and inject EBM energy guidance directly into the denoising logits.

The key insight: DiffusionGemma's _denoising_step produces logits over the
canvas at EVERY denoising step. We intercept those logits and add an energy
gradient that pushes tokens toward target concepts.

Energy guidance = for each canvas position, compute the MiniLM embedding of
the current canvas, then boost/suppress based on similarity to target topics.
"""

import time, json, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np

device = "cuda"

# ── Energy Logits Processor ──
# This hooks into DiffusionGemma's generation pipeline as a custom logits_processor
# It modifies the logits at each denoising step based on EBM energy guidance

class EnergyGuidanceLogitsProcessor:
    """
    Custom logits processor for DiffusionGemma that injects EBM energy guidance.

    At each denoising step, for each canvas position:
    1. Decode current canvas to text (or use token-level embeddings)
    2. Compute similarity to target topic centers
    3. Boost tokens that increase similarity, suppress tokens that decrease it

    Simplified approach: instead of decoding to text at each step (expensive),
    we operate at the token level. For each token in the vocab, we know its
    MiniLM embedding (via a pre-computed lookup). We boost tokens whose
    embeddings are close to the target topic center.
    """

    def __init__(self, target_embedding, guidance_scale=1.0, token_embeddings=None):
        """
        Args:
            target_embedding: [D] normalized target direction on hypersphere
            guidance_scale: strength of guidance
            token_embeddings: [vocab_size, D] pre-computed MiniLM embeddings for each token
                             (or None to use a simpler approach)
        """
        self.target = target_embedding  # [D]
        self.scale = guidance_scale
        self.token_embs = token_embeddings  # [vocab, D] or None

    def __call__(self, argmax_canvas, logits, **kwargs):
        """
        Args:
            argmax_canvas: [B, canvas_length] current canvas token IDs
            logits: [B, canvas_length, vocab_size] logits from denoiser
        Returns:
            Modified logits [B, canvas_length, vocab_size]
        """
        if self.token_embs is None:
            return logits

        # Compute token-level guidance: how much each token aligns with target
        # token_embs: [vocab, D]
        # target: [D]
        # similarity: [vocab]
        token_sim = self.token_embs @ self.target  # [vocab]

        # Boost/suppress: add scaled similarity to logits
        # This pushes the sampler toward tokens aligned with the target
        guidance = self.scale * token_sim  # [vocab]

        # Apply to all canvas positions uniformly
        # logits: [B, canvas_length, vocab_size]
        logits = logits + guidance.unsqueeze(0).unsqueeze(0)

        return logits


def precompute_token_embeddings(tokenizer, st_model, device, batch_size=512):
    """
    Pre-compute MiniLM embeddings for all tokens in the tokenizer vocab.
    This creates a [vocab_size, D] matrix that maps each token to its
    semantic embedding.

    For multi-token words, we embed the decoded string of each individual token.
    """
    vocab_size = tokenizer.vocab_size
    emb_dim = st_model.get_embedding_dimension()

    print(f"Pre-computing embeddings for {vocab_size} tokens...", flush=True)

    # Decode each token to text
    token_texts = []
    for tid in range(vocab_size):
        try:
            text = tokenizer.decode([tid], skip_special_tokens=True).strip()
            if not text:
                text = tokenizer.decode([tid], skip_special_tokens=False).strip()
            if not text:
                text = "<empty>"
        except:
            text = f"<token_{tid}>"
        token_texts.append(text)

    # Embed in batches
    all_embs = []
    t0 = time.time()
    for start in range(0, vocab_size, batch_size):
        batch = token_texts[start:start+batch_size]
        embs = st_model.encode(batch, batch_size=len(batch),
                              show_progress_bar=False, convert_to_tensor=True,
                              normalize_embeddings=True)
        all_embs.append(embs)

    token_embeddings = torch.cat(all_embs, dim=0)  # [vocab, D]
    print(f"Done in {time.time()-t0:.1f}s. Shape: {token_embeddings.shape}", flush=True)

    return token_embeddings


def test_energy_guided_diffusion():
    """
    Main test: generate text with DiffusionGemma, with and without energy guidance.
    Compare semantic similarity of generated text to target topics.
    """
    from transformers.models.diffusion_gemma import DiffusionGemmaForBlockDiffusion
    from transformers import AutoTokenizer, AutoProcessor, BitsAndBytesConfig
    from sentence_transformers import SentenceTransformer

    torch.cuda.set_per_process_memory_fraction(0.90)

    # ── 1. Load models ──
    print("=== LOAD MODELS ===", flush=True)

    st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    emb_dim = st_model.get_embedding_dimension()
    print(f"MiniLM: {emb_dim}D", flush=True)

    print("Loading DiffusionGemma 26B (4-bit)...", flush=True)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    t0 = time.time()
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        "google/diffusiongemma-26B-A4B-it",
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    print(f"DiffusionGemma loaded in {time.time()-t0:.1f}s", flush=True)
    vram = torch.cuda.memory_allocated() / 1e9
    print(f"VRAM: {vram:.1f} GB / 24 GB", flush=True)

    tokenizer = AutoTokenizer.from_pretrained("google/diffusiongemma-26B-A4B-it")
    print(f"Vocab: {tokenizer.vocab_size}", flush=True)

    # ── 2. Pre-compute token embeddings ──
    print("\n=== TOKEN EMBEDDINGS ===", flush=True)
    # Only embed a subset for efficiency — the full vocab is 262K
    # We'll use the most common tokens
    VOCAB_SUBSET = 50000  # Top 50K tokens covers most of English
    token_embs_full = precompute_token_embeddings(tokenizer, st_model, device)

    # Pad to full vocab size with zeros for unused tokens
    full_vocab = tokenizer.vocab_size
    token_embs = torch.zeros(full_vocab, emb_dim, device=device)
    n_copy = min(VOCAB_SUBSET, token_embs_full.shape[0])
    token_embs[:n_copy] = token_embs_full[:n_copy]
    print(f"Token embeddings ready: {token_embs.shape}", flush=True)

    # ── 3. Define target topics ──
    print("\n=== TARGET TOPICS ===", flush=True)
    topics = {
        "nature": "trees forest animals birds nature park green outdoors hiking",
        "space": "space stars planets galaxy rocket astronaut moon mars cosmos",
        "cooking": "cook kitchen food recipe bake oven delicious meal chef",
        "sadness": "sad lonely cry tears grief loss heartbroken depression",
    }

    topic_embs = {}
    for name, text in topics.items():
        emb = st_model.encode([text], convert_to_tensor=True, normalize_embeddings=True)[0]
        topic_embs[name] = emb
        print(f"  {name}: dim={emb.shape[0]}", flush=True)

    # ── 4. Generation tests ──
    print("\n=== GENERATION TESTS ===", flush=True)

    prompt = "Write a short story."
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    results = []

    # Baseline (no guidance)
    print("\n--- Baseline (no guidance) ---", flush=True)
    t0 = time.time()
    with torch.no_grad():
        out_base = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
        )
    base_text = tokenizer.decode(out_base[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"Generated in {time.time()-t0:.1f}s", flush=True)
    print(f"Text: {base_text[:200]}", flush=True)

    # Compute similarity to all topics
    base_emb = st_model.encode([base_text], convert_to_tensor=True, normalize_embeddings=True)[0]
    base_sims = {name: (base_emb @ emb).item() for name, emb in topic_embs.items()}
    print(f"Topic sims: {base_sims}", flush=True)
    results.append({"condition": "baseline", "text": base_text[:200], "sims": base_sims})

    # Guided generation for each topic
    for topic_name in topics:
        for gs in [2.0, 5.0, 10.0]:
            cond_name = f"guided_{topic_name}_gs{gs}"
            print(f"\n--- {cond_name} ---", flush=True)

            processor = EnergyGuidanceLogitsProcessor(
                target_embedding=topic_embs[topic_name],
                guidance_scale=gs,
                token_embeddings=token_embs,
            )

            # Reset inputs
            inputs_fresh = tokenizer(prompt, return_tensors="pt").to("cuda")

            t0 = time.time()
            with torch.no_grad():
                try:
                    out_guided = model.generate(
                        **inputs_fresh,
                        max_new_tokens=128,
                        do_sample=True,
                        temperature=0.7,
                        logits_processor=[processor],
                    )
                    gen_text = tokenizer.decode(out_guided[0, inputs_fresh["input_ids"].shape[1]:], skip_special_tokens=True)
                    gen_time = time.time() - t0
                    print(f"Generated in {gen_time:.1f}s", flush=True)
                    print(f"Text: {gen_text[:200]}", flush=True)

                    gen_emb = st_model.encode([gen_text], convert_to_tensor=True, normalize_embeddings=True)[0]
                    gen_sims = {name: (gen_emb @ emb).item() for name, emb in topic_embs.items()}
                    print(f"Topic sims: {gen_sims}", flush=True)

                    results.append({"condition": cond_name, "text": gen_text[:200], "sims": gen_sims})
                except Exception as e:
                    print(f"ERROR: {e}", flush=True)
                    results.append({"condition": cond_name, "error": str(e)})

    # ── 5. Summary ──
    print("\n" + "=" * 70)
    print("ENERGY-GUIDED DIFFUSION GEMMA — RESULTS")
    print("=" * 70)

    print(f"\n{'Condition':<30} {'nature':>8} {'space':>8} {'cooking':>8} {'sadness':>8}")
    print("-" * 70)
    for r in results:
        if "sims" in r:
            s = r["sims"]
            print(f"{r['condition']:<30} {s.get('nature',0):>8.4f} {s.get('space',0):>8.4f} {s.get('cooking',0):>8.4f} {s.get('sadness',0):>8.4f}")

    # Check if guidance shifts topic
    base_nature = results[0]["sims"].get("nature", 0) if "sims" in results[0] else 0
    best_nature = max((r["sims"].get("nature", 0) for r in results if "sims" in r and "guided_nature" in r["condition"]), default=0)

    delta = best_nature - base_nature
    print(f"\nNature topic: baseline={base_nature:.4f} -> best guided={best_nature:.4f} (delta={delta:+.4f})")

    if delta > 0.05:
        print("\nVERDICT: Energy guidance WORKS on DiffusionGemma!", flush=True)
        print("  Generated text shifts toward the guided topic at each denoising step.", flush=True)
    elif delta > 0.02:
        print("\nVERDICT: Weak but detectable topic shift.", flush=True)
    else:
        print("\nVERDICT: No significant topic shift from energy guidance.", flush=True)

    # Save results
    with open("/root/EBM-splats/tests/phase3_diffusion_gemma_results.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")

    print("\nDone!", flush=True)


if __name__ == "__main__":
    test_energy_guided_diffusion()
