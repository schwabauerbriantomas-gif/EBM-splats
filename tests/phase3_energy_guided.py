#!/usr/bin/env python3
"""
Phase 3: Energy-Guided DiffusionGemma Block Diffusion

Uses DiffusionGemma's block diffusion decoder and injects EBM energy guidance
directly into the denoising logits at EVERY step.

The key insight from source code analysis:
  _denoising_step calls: logits_processor(input_ids, raw_logits, cur_step=cur_step)
  raw_logits shape: [batch, canvas_length, vocab_size]

We provide a custom LogitsProcessor that, for each canvas position, boosts tokens
whose MiniLM embeddings align with the target topic direction.

Hardware: RTX 3090 24GB, DiffusionGemma AWQ-INT4 (~16GB)
"""

import time, json, sys, os
import torch
import torch.nn.functional as F
import numpy as np

device = "cuda"

# ─── Energy Guidance Logits Processor ──────────────────────────────────────
# Follows HF LogitsProcessor interface: __call__(input_ids, scores, **kwargs)
# scores = raw_logits = [batch, canvas_length, vocab_size]

class EnergyGuidanceLogitsProcessor:
    """
    At each denoising step, modifies the logits to push tokens toward (or away
    from) a target semantic direction.

    For every position in the canvas simultaneously:
      guidance[batch, pos, vocab] = scale * (token_embs[vocab] @ target_dir[batch])
      logits_modified = logits + guidance

    This is analogous to classifier-free guidance but with the EBM energy
    landscape providing the "classifier" signal.
    """

    def __init__(self, target_embedding, guidance_scale=1.0, token_embeddings=None):
        """
        Args:
            target_embedding: [D] normalized target direction on S^(D-1)
            guidance_scale: strength of guidance (0 = no guidance)
            token_embeddings: [vocab_size, D] pre-computed normalized MiniLM embeddings
        """
        self.target = target_embedding  # [D]
        self.scale = guidance_scale
        self.token_embs = token_embeddings  # [vocab, D]

    def __call__(self, input_ids, scores, **kwargs):
        """
        HF-compatible logits processor interface.
        Called as: processor(input_ids, raw_logits, cur_step=cur_step)

        Args:
            input_ids: [batch, seq_len] (the canvas token IDs)
            scores: [batch, canvas_length, vocab_size] raw denoiser logits
        Returns:
            [batch, canvas_length, vocab_size] modified logits
        """
        if self.token_embs is None or self.scale == 0:
            return scores

        # token_embs: [vocab, D], target: [D]
        # token_sim: [vocab] — how much each token aligns with target
        token_sim = self.token_embs @ self.target  # [vocab]

        # Scale and broadcast to all canvas positions
        # scores: [batch, canvas_length, vocab_size]
        guidance = self.scale * token_sim  # [vocab]
        scores = scores + guidance.unsqueeze(0).unsqueeze(0)  # broadcast

        return scores


class EnergyGuidanceLogitsProcessorMulti:
    """
    Multi-target variant: supports blend, suppression, and asymmetric mixing.
    Each target has a weight (positive = boost, negative = suppress).
    """

    def __init__(self, targets_and_weights, token_embeddings):
        """
        Args:
            targets_and_weights: list of (embedding[D], weight) tuples
            token_embeddings: [vocab, D]
        """
        self.token_embs = token_embeddings
        # Pre-compute combined direction
        combined = torch.zeros_like(targets_and_weights[0][0])
        for emb, w in targets_and_weights:
            combined += w * emb
        # Normalize
        self.combined = combined / (combined.norm() + 1e-8)
        self.scale = sum(abs(w) for _, w in targets_and_weights)

    def __call__(self, input_ids, scores, **kwargs):
        if self.scale == 0:
            return scores
        token_sim = self.token_embs @ self.combined  # [vocab]
        guidance = self.scale * token_sim
        return scores + guidance.unsqueeze(0).unsqueeze(0)


def precompute_token_embeddings(tokenizer, st_model, device, max_tokens=20000):
    """
    Pre-compute MiniLM embeddings for the most common tokens.
    DiffusionGemma vocab is 262K — we embed a representative subset.
    Token ID 0-max_tokens covers the most frequent English tokens.
    """
    emb_dim = st_model.get_embedding_dimension()
    print(f"Pre-computing embeddings for {max_tokens} tokens...", flush=True)

    # Decode tokens to text
    token_texts = []
    for tid in range(max_tokens):
        try:
            text = tokenizer.decode([tid], skip_special_tokens=True).strip()
            if not text:
                text = tokenizer.decode([tid], skip_special_tokens=False).strip()
            if not text:
                text = f"token_{tid}"
        except:
            text = f"token_{tid}"
        token_texts.append(text)

    # Embed in batches
    all_embs = []
    batch_size = 512
    t0 = time.time()
    for start in range(0, max_tokens, batch_size):
        batch = token_texts[start:start + batch_size]
        embs = st_model.encode(
            batch, batch_size=len(batch),
            show_progress_bar=False, convert_to_tensor=True,
            normalize_embeddings=True,
        )
        all_embs.append(embs)

    token_embeddings = torch.cat(all_embs, dim=0).to(device)  # [max_tokens, D]
    print(f"Done in {time.time()-t0:.1f}s. Shape: {token_embeddings.shape}", flush=True)
    return token_embeddings


def main():
    from transformers import DiffusionGemmaForBlockDiffusion, AutoTokenizer, AutoProcessor
    from sentence_transformers import SentenceTransformer

    torch.cuda.set_per_process_memory_fraction(0.92)

    # ── 1. Load MiniLM ──
    print("=== LOADING MODELS ===", flush=True)
    st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    emb_dim = st_model.get_embedding_dimension()
    print(f"MiniLM: {emb_dim}D", flush=True)

    # ── 2. Load DiffusionGemma AWQ-INT4 ──
    model_path = "/root/.cache/huggingface/hub/models--cyankiwi--diffusiongemma-26B-A4B-it-AWQ-INT4/snapshots/8756b0a40bce78859a29694529ec9e87cb8066ab"

    print(f"Loading DiffusionGemma from local cache...", flush=True)
    t0 = time.time()
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        model_path,
        device_map={"": 0},
        torch_dtype=torch.float16,
        trust_remote_code=True,
        local_files_only=True,
    )
    load_time = time.time() - t0
    vram = torch.cuda.memory_allocated() / 1e9
    print(f"Loaded in {load_time:.1f}s | VRAM: {vram:.2f} GB / 24 GB", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    print(f"Vocab: {tokenizer.vocab_size}", flush=True)

    # ── 3. Pre-compute token embeddings ──
    print("\n=== TOKEN EMBEDDINGS ===", flush=True)
    VOCAB_SUBSET = 20000  # Top 20K tokens (covers most English)
    token_embs = precompute_token_embeddings(tokenizer, st_model, device, max_tokens=VOCAB_SUBSET)
    # Pad to full vocab size with zeros
    full_vocab = tokenizer.vocab_size
    token_embs_full = torch.zeros(full_vocab, emb_dim, device=device)
    token_embs_full[:VOCAB_SUBSET] = token_embs

    # ── 4. Define target topics ──
    print("\n=== TARGET TOPICS ===", flush=True)
    topics = {
        "nature": "trees forest animals birds nature park green outdoors hiking wilderness",
        "space": "space stars planets galaxy rocket astronaut moon mars cosmos universe",
        "cooking": "cook kitchen food recipe bake oven delicious meal chef ingredients",
        "sadness": "sad lonely cry tears grief loss heartbroken depression mournful",
    }

    topic_embs = {}
    for name, text in topics.items():
        emb = st_model.encode([text], convert_to_tensor=True, normalize_embeddings=True)[0].to(device)
        topic_embs[name] = emb
        print(f"  {name}: ready ({emb_dim}D)", flush=True)

    # ── 5. Baseline generation ──
    print("\n=== BASELINE (no guidance) ===", flush=True)
    prompt = "Write a short story about a character."
    inputs = tokenizer(prompt, return_tensors="pt").to(0)
    print(f"Prompt: '{prompt}'", flush=True)
    print(f"Input shape: {inputs.input_ids.shape}", flush=True)

    t0 = time.time()
    with torch.no_grad():
        out_base = model.generate(
            **inputs,
            max_new_tokens=128,
        )
    gen_time = time.time() - t0
    base_text = tokenizer.decode(out_base[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"Generated in {gen_time:.1f}s", flush=True)
    print(f"Text: {base_text[:300]}", flush=True)

    # Compute topic similarity
    base_emb = st_model.encode([base_text], convert_to_tensor=True, normalize_embeddings=True)[0]
    base_sims = {name: float((base_emb @ emb.cpu()).item()) for name, emb in topic_embs.items()}
    print(f"Topic sims: {base_sims}", flush=True)

    # ── 6. Energy-guided generation ──
    results = [{
        "condition": "baseline",
        "text": base_text[:500],
        "sims": base_sims,
        "gen_time_s": gen_time,
    }]

    guidance_scales = [2.0, 5.0, 10.0]

    for topic_name in topics:
        for gs in guidance_scales:
            cond_name = f"guided_{topic_name}_gs{gs}"
            print(f"\n--- {cond_name} ---", flush=True)

            processor = EnergyGuidanceLogitsProcessor(
                target_embedding=topic_embs[topic_name],
                guidance_scale=gs,
                token_embeddings=token_embs_full,
            )

            inputs_fresh = tokenizer(prompt, return_tensors="pt").to(0)
            t0 = time.time()
            try:
                with torch.no_grad():
                    out_guided = model.generate(
                        **inputs_fresh,
                        max_new_tokens=128,
                        logits_processor=[processor],
                    )
                gen_time = time.time() - t0
                gen_text = tokenizer.decode(
                    out_guided[0, inputs_fresh["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                )
                print(f"Generated in {gen_time:.1f}s", flush=True)
                print(f"Text: {gen_text[:300]}", flush=True)

                gen_emb = st_model.encode([gen_text], convert_to_tensor=True, normalize_embeddings=True)[0]
                gen_sims = {name: float((gen_emb @ emb.cpu()).item()) for name, emb in topic_embs.items()}
                print(f"Topic sims: {gen_sims}", flush=True)

                # Compare to baseline
                delta = gen_sims[topic_name] - base_sims[topic_name]
                print(f"Delta {topic_name}: {delta:+.4f}", flush=True)

                results.append({
                    "condition": cond_name,
                    "text": gen_text[:500],
                    "sims": gen_sims,
                    "delta_target": delta,
                    "gen_time_s": gen_time,
                })
            except Exception as e:
                print(f"ERROR: {type(e).__name__}: {e}", flush=True)
                results.append({"condition": cond_name, "error": str(e)})

    # ── 7. Composition tests ──
    print("\n=== COMPOSITION TESTS ===", flush=True)

    # Blend: nature + space
    for blend_desc, targets in [
        ("blend_nature_space", [(topic_embs["nature"], 0.5), (topic_embs["space"], 0.5)]),
        ("suppress_nature_minus_space", [(topic_embs["nature"], 1.0), (topic_embs["space"], -0.5)]),
        ("triple_nature_space_cooking", [
            (topic_embs["nature"], 0.33), (topic_embs["space"], 0.33), (topic_embs["cooking"], 0.34)
        ]),
    ]:
        print(f"\n--- {blend_desc} ---", flush=True)
        processor = EnergyGuidanceLogitsProcessorMulti(targets, token_embs_full)

        inputs_fresh = tokenizer(prompt, return_tensors="pt").to(0)
        t0 = time.time()
        try:
            with torch.no_grad():
                out = model.generate(
                    **inputs_fresh,
                    max_new_tokens=128,
                    logits_processor=[processor],
                )
            gen_time = time.time() - t0
            gen_text = tokenizer.decode(
                out[0, inputs_fresh["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            print(f"Text: {gen_text[:300]}", flush=True)

            gen_emb = st_model.encode([gen_text], convert_to_tensor=True, normalize_embeddings=True)[0]
            gen_sims = {name: float((gen_emb @ emb.cpu()).item()) for name, emb in topic_embs.items()}
            print(f"Topic sims: {gen_sims}", flush=True)

            results.append({
                "condition": blend_desc,
                "text": gen_text[:500],
                "sims": gen_sims,
                "gen_time_s": gen_time,
            })
        except Exception as e:
            print(f"ERROR: {type(e).__name__}: {e}", flush=True)
            results.append({"condition": blend_desc, "error": str(e)})

    # ── 8. Summary table ──
    print("\n" + "=" * 80)
    print("ENERGY-GUIDED BLOCK DIFFUSION — RESULTS")
    print("=" * 80)

    header = f"{'Condition':<40} {'nature':>8} {'space':>8} {'cooking':>8} {'sadness':>8}"
    print(f"\n{header}")
    print("-" * 80)
    for r in results:
        if "sims" in r:
            s = r["sims"]
            print(f"{r['condition']:<40} {s.get('nature',0):>8.4f} {s.get('space',0):>8.4f} {s.get('cooking',0):>8.4f} {s.get('sadness',0):>8.4f}")

    # Verdict
    print("\n=== VERDICT ===")
    guided_with_data = [r for r in results if "delta_target" in r]
    if guided_with_data:
        best_delta = max(r["delta_target"] for r in guided_with_data)
        avg_delta = sum(r["delta_target"] for r in guided_with_data) / len(guided_with_data)
        print(f"Best delta: {best_delta:+.4f}", flush=True)
        print(f"Avg delta:  {avg_delta:+.4f}", flush=True)
        if best_delta > 0.05:
            print("✅ Energy guidance WORKS — text shifts toward guided topics!", flush=True)
        elif best_delta > 0.02:
            print("⚠️  Weak but detectable topic shift.", flush=True)
        else:
            print("❌ No significant topic shift from energy guidance.", flush=True)

    # Save
    with open("/root/EBM-splats/tests/phase3_energy_guided_results.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")

    print("\nResults saved to phase3_energy_guided_results.jsonl", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
