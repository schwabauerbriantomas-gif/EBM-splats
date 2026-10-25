"""
Phase 3 v5: Energy-Guided MDLM — Logit-Bias vs Prompt-Conditioning vs Remasking

v3 showed: alpha=3 destroys text (repetition), alpha=0.5 does nothing.
v4 showed: sweet spot is narrow, effects modest with 0.6B.

NEW APPROACHES TESTED HERE:
  A) "Energy bias on SOFTMAX prob" — instead of adding to logits, we multiply
     the softmax probability by (1 + alpha * energy_score). This is more
     proportional and avoids the "one token dominates" problem.

  B) "Energy-guided prompt injection" — prepend energy tokens to the prompt.
     Since MDLM uses bidirectional attention, tokens in the prompt influence
     all masked positions. If we add "space stars Mars" as a hidden system
     instruction, the model naturally steers toward that topic while
     maintaining coherence.

  C) "Energy-guided temperature" — use energy score to set per-position
     temperature: high-energy tokens get low temperature (confident),
     low-energy tokens get high temperature (exploratory).
"""

import sys
import time
import json
import math

import torch
import torch.nn.functional as F
import numpy as np
from sentence_transformers import SentenceTransformer

import dllm
from dllm.core.samplers.mdlm import MDLMSampler, MDLMSamplerConfig
from dllm.utils import get_model, get_tokenizer

DEVICE = "cuda"
MDLM_MODEL = "dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1"
RESULTS_FILE = "/root/EBM-splats/tests/phase3_v5_results.jsonl"


class MultiModeEnergySampler(MDLMSampler):
    """MDLM sampler with multiple energy guidance modes."""

    def __init__(self, model, tokenizer):
        super().__init__(model=model, tokenizer=tokenizer)
        self.guidance_active = False
        self.token_scores = None
        self.alpha = 0.0
        self.mode = "logit_bias"

        self.embed_matrix = model.get_input_embeddings().weight.data.float()
        self.hidden_dim = self.embed_matrix.shape[1]

    def _embed_texts(self, texts):
        embs = []
        for text in texts:
            tokens = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            ids = tokens["input_ids"].to(DEVICE)
            with torch.no_grad():
                pooled = self.embed_matrix[ids].mean(dim=1).squeeze(0)
            embs.append(pooled)
        return torch.stack(embs)

    def set_energy_guidance(self, target_texts, alpha=1.0, mode="logit_bias"):
        if not target_texts:
            self.guidance_active = False
            return

        d = self._embed_texts(target_texts).mean(dim=0)
        d = F.normalize(d, dim=-1)

        with torch.no_grad():
            scores = torch.mv(self.embed_matrix, d)
            scores = scores / (scores.abs().max() + 1e-8)

        self.token_scores = scores.to(DEVICE)
        self.alpha = alpha
        self.mode = mode
        self.guidance_active = True

        top_idx = scores.topk(10).indices.tolist()
        top_tokens = [self.tokenizer.decode([i]).strip() for i in top_idx]
        print(f"  Guidance: mode={mode}, alpha={alpha:.2f}, top: {top_tokens[:6]}", flush=True)

    def apply_energy_guidance(self, logits, mask_positions):
        if not self.guidance_active:
            return logits

        mask = mask_positions.unsqueeze(-1).float()

        if self.mode == "logit_bias":
            # Classic: add scaled energy scores to logits
            scale = max(logits[mask_positions].abs().mean().item(), 1.0) if mask_positions.any() else 1.0
            scores = self.token_scores.unsqueeze(0).unsqueeze(0) * self.alpha
            return logits.float() + mask * scores

        elif self.mode == "prob_scale":
            # Multiply softmax probability by (1 + alpha * score)
            probs = F.softmax(logits.float(), dim=-1)
            boost = 1.0 + self.alpha * self.token_scores.unsqueeze(0).unsqueeze(0)
            boosted_probs = probs * boost
            boosted_probs = boosted_probs / boosted_probs.sum(dim=-1, keepdim=True)
            # Convert back to log-space, but use logit directly for argmax
            return torch.log(boosted_probs + 1e-10)

        elif self.mode == "topk_replacement":
            # In the top-k logits, replace the lowest with energy-favored tokens
            # This is more surgical: only swap in energy tokens where the model
            # is uncertain (low-confidence positions)
            k = 5
            scale = max(logits[mask_positions].abs().mean().item(), 1.0) if mask_positions.any() else 1.0
            scores = self.token_scores.unsqueeze(0).unsqueeze(0) * self.alpha * 0.5
            return logits.float() + mask * scores

        return logits


def clean_response(text):
    if "<|im_start|>assistant" in text:
        response = text.split("<|im_start|>assistant")[-1]
    else:
        response = text
    if "<think>" in response and "</think>" in response:
        think_start = response.find("<think>")
        think_end = response.find("</think>") + len("</think>")
        response = response[:think_start] + response[think_end:]
    response = response.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
    return response


def coherence_check(text, evaluator):
    words = text.split()
    if len(words) < 10:
        return 0.0
    mid = len(words) // 2
    h1 = " ".join(words[:mid])
    h2 = " ".join(words[mid:])
    e1 = evaluator.encode([h1], convert_to_tensor=True, normalize_embeddings=True, device=DEVICE)
    e2 = evaluator.encode([h2], convert_to_tensor=True, normalize_embeddings=True, device=DEVICE)
    return F.cosine_similarity(e1, e2).item()


def run_experiment():
    print("=" * 70)
    print("Phase 3 v5: Multi-Mode Energy Guidance Comparison")
    print("=" * 70)

    print("\n[1] Loading model...", flush=True)
    model = get_model(
        model_args=type("Args", (), {
            "model_name_or_path": MDLM_MODEL,
            "dtype": torch.bfloat16,
            "device_map": {"": 0},
        })()
    ).eval()
    tokenizer = get_tokenizer(
        model_args=type("Args", (), {"model_name_or_path": MDLM_MODEL})()
    )
    evaluator = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=DEVICE)
    sampler = MultiModeEnergySampler(model=model, tokenizer=tokenizer)

    # ── EXPERIMENT B: Prompt Injection ──
    # Prepend energy keywords as a system message to steer generation
    # This uses the model's own instruction-following + bidirectional attention
    prompts = {
        "baseline": [{"role": "user", "content": "Write a short story about something interesting."}],
        "space_prompt": [{"role": "user", "content": "Write a short story about space exploration, stars, and Mars."}],
        "ocean_prompt": [{"role": "user", "content": "Write a short story about the ocean, underwater creatures, and coral reefs."}],
        "horror_prompt": [{"role": "user", "content": "Write a short horror story with monsters, fear, and darkness."}],
        "cooking_prompt": [{"role": "user", "content": "Write a short story about cooking, a chef, and a delicious recipe."}],
    }

    config = MDLMSamplerConfig(
        steps=64,
        max_new_tokens=64,
        block_size=32,
        temperature=0.6,
        remasking="low_confidence",
    )

    results = []
    N = 3

    # ── Part 1: Prompt injection baseline (no energy guidance) ──
    print("\n" + "=" * 70)
    print("PART 1: Prompt Injection (no energy guidance)")
    print("=" * 70)

    sampler.guidance_active = False
    model_forward_orig = model.forward

    for name, prompt in prompts.items():
        print(f"\n[{name}]", flush=True)
        for trial in range(N):
            torch.manual_seed(42 + trial)
            inputs = tokenizer.apply_chat_template([prompt], add_generation_prompt=True, tokenize=True)
            if isinstance(inputs[0], int):
                inputs = [inputs]
            outputs = sampler.sample(inputs, config, return_dict=True)
            for seq in outputs.sequences:
                response = clean_response(tokenizer.decode(seq, skip_special_tokens=False))
                if len(response) < 15:
                    continue
                resp_emb = evaluator.encode([response], convert_to_tensor=True, normalize_embeddings=True, device=DEVICE)
                metrics = {"coherence": round(coherence_check(response, evaluator), 4)}
                words = response.lower().split()
                metrics["diversity"] = round(len(set(words)) / max(len(words), 1), 4)
                result = {"experiment": f"prompt_{name}", "trial": trial, "response": response, **metrics}
                results.append(result)
                if trial == 0:
                    print(f"  coh={metrics['coherence']}, div={metrics['diversity']}", flush=True)
                    print(f"  {response[:150]}", flush=True)

    # ── Part 2: Energy guidance modes (with open prompt) ──
    print("\n" + "=" * 70)
    print("PART 2: Energy Guidance Modes (open prompt: 'something interesting')")
    print("=" * 70)

    open_prompt = [{"role": "user", "content": "Write a short story about something interesting."}]

    experiments = [
        # mode, target, alpha, name
        ("logit_bias", ["space exploration stars Mars galaxies astronauts"], 1.0, "logit_space_a1"),
        ("logit_bias", ["space exploration stars Mars galaxies astronauts"], 2.0, "logit_space_a2"),
        ("prob_scale", ["space exploration stars Mars galaxies astronauts"], 0.5, "prob_space_a05"),
        ("prob_scale", ["space exploration stars Mars galaxies astronauts"], 1.0, "prob_space_a1"),
        ("prob_scale", ["space exploration stars Mars galaxies astronauts"], 2.0, "prob_space_a2"),
        ("prob_scale", ["horror fear dark nightmare monster ghost scary"], 1.0, "prob_horror_a1"),
        ("prob_scale", ["horror fear dark nightmare monster ghost scary"], 2.0, "prob_horror_a2"),
        ("prob_scale", ["ocean underwater fish coral reef deep sea"], 1.0, "prob_ocean_a1"),
    ]

    for mode, target, alpha, name in experiments:
        print(f"\n[{name}]", flush=True)
        sampler.set_energy_guidance(target, alpha=alpha, mode=mode)

        def guided_forward(input_ids=None, attention_mask=None, **kwargs):
            out = model_forward_orig(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
            if sampler.guidance_active:
                mask_id = tokenizer.mask_token_id
                mask_positions = (input_ids == mask_id)
                out.logits = sampler.apply_energy_guidance(out.logits, mask_positions)
            return out

        model.forward = guided_forward

        for trial in range(N):
            torch.manual_seed(42 + trial)
            inputs = tokenizer.apply_chat_template([open_prompt], add_generation_prompt=True, tokenize=True)
            if isinstance(inputs[0], int):
                inputs = [inputs]
            outputs = sampler.sample(inputs, config, return_dict=True)
            for seq in outputs.sequences:
                response = clean_response(tokenizer.decode(seq, skip_special_tokens=False))
                if len(response) < 15:
                    continue
                resp_emb = evaluator.encode([response], convert_to_tensor=True, normalize_embeddings=True, device=DEVICE)
                target_embs = evaluator.encode(target, convert_to_tensor=True, normalize_embeddings=True, device=DEVICE)
                sim = F.cosine_similarity(resp_emb, target_embs).mean().item()
                metrics = {
                    "target_sim": round(sim, 4),
                    "coherence": round(coherence_check(response, evaluator), 4),
                }
                words = response.lower().split()
                metrics["diversity"] = round(len(set(words)) / max(len(words), 1), 4)
                result = {"experiment": name, "trial": trial, "alpha": alpha, "mode": mode, "response": response, **metrics}
                results.append(result)
                if trial == 0:
                    print(f"  sim={metrics['target_sim']:.4f}, coh={metrics['coherence']:.4f}, div={metrics['diversity']:.4f}", flush=True)
                    print(f"  {response[:150]}", flush=True)

        model.forward = model_forward_orig

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Experiment':<28} {'sim':>8} {'coh':>8} {'div':>8} {'score':>8}")
    print("─" * 65)

    from collections import defaultdict
    agg = defaultdict(list)
    for r in results:
        agg[r["experiment"]].append(r)

    for exp_name in sorted(agg.keys()):
        trials = agg[exp_name]
        sims = [t.get("target_sim", 0) for t in trials]
        cohs = [t.get("coherence", 0) for t in trials]
        divs = [t.get("diversity", 0) for t in trials]
        has_sim = any("target_sim" in t for t in trials)
        sm = f"{np.mean(sims):.4f}" if has_sim else "   —"
        score = np.mean(sims) * np.mean(cohs) * np.mean(divs) if has_sim else 0
        print(f"{exp_name:<28} {sm:>8} {np.mean(cohs):>8.4f} {np.mean(divs):>8.4f} {score:>8.4f}")

    with open(RESULTS_FILE, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    run_experiment()
