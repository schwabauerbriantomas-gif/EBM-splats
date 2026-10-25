"""
Phase 3 v4: EBM-Guided MDLM — Finding the Sweet Spot

From v3 we learned:
  - alpha=3.0 × scale → semantic domination (text becomes "scary scary scary")
  - alpha=1.0 without scale → too weak (model predictions dominate)
  - The sweet spot is somewhere in between

STRATEGY: Binary search over alpha with FIXED scale factor.
Instead of scaling to logit magnitude, we use a fixed multiplier
that's strong enough to steer but not dominate.

Also: apply energy guidance ONLY at the first few denoising steps
(early steps have the most influence on topic via the cascade effect,
late steps refine grammar which we want the model to handle freely).
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
RESULTS_FILE = "/root/EBM-splats/tests/phase3_v4_results.jsonl"


class EnergyGuidedSampler(MDLMSampler):
    """MDLM sampler with energy guidance, step-limited application."""

    def __init__(self, model, tokenizer):
        super().__init__(model=model, tokenizer=tokenizer)
        self.guidance_active = False
        self.token_scores = None
        self.alpha = 0.0
        self.guidance_steps_ratio = 1.0  # fraction of steps where guidance applies
        self.step_counter = 0
        self.total_steps = 0

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

    def set_energy_guidance(self, target_texts, alpha=1.0, guidance_ratio=1.0):
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
        self.guidance_steps_ratio = guidance_ratio
        self.guidance_active = True

        top_idx = scores.topk(10).indices.tolist()
        top_tokens = [self.tokenizer.decode([i]).strip() for i in top_idx]
        print(f"  Guidance: alpha={alpha:.2f}, ratio={guidance_ratio:.1f}, top: {top_tokens[:6]}", flush=True)

    def reset_step_counter(self, total_steps):
        self.step_counter = 0
        self.total_steps = total_steps

    def apply_energy_guidance(self, logits, mask_positions):
        if not self.guidance_active:
            return logits

        self.step_counter += 1
        progress = self.step_counter / max(self.total_steps, 1)

        # Only apply guidance during early steps if ratio < 1.0
        if progress > self.guidance_steps_ratio:
            return logits

        # Fixed scale: target ~10% of typical logit magnitude at alpha=1.0
        FIXED_SCALE = 2.0  # empirically: logits are ~10-20, so 2.0 × alpha gives 2-20 contribution
        scores = self.token_scores.unsqueeze(0).unsqueeze(0) * self.alpha * FIXED_SCALE
        mask = mask_positions.unsqueeze(-1).float()
        return logits.float() + mask * scores


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
    """Simple coherence proxy: measure average sentence embedding similarity
    between first half and second half of the text. High similarity = coherent topic."""
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
    print("Phase 3 v4: EBM-Guided MDLM — Sweet Spot Search")
    print("=" * 70)

    print("\n[1] Loading model + tokenizer...", flush=True)
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

    evaluator = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2", device=DEVICE
    )

    sampler = EnergyGuidedSampler(model=model, tokenizer=tokenizer)

    base_prompt = [{"role": "user", "content": "Write a short story. Make it interesting."}]

    # ── Grid search: alpha × guidance_ratio ──
    alphas = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    ratios = [1.0, 0.5, 0.25]  # all steps, first half, first quarter

    SPACE_TARGET = ["space exploration stars Mars galaxies astronauts rocket launch"]
    OCEAN_TARGET = ["ocean underwater fish coral reef deep sea waves diving submarine"]
    HORROR_TARGET = ["horror fear dark nightmare monster ghost scary terrifying scream"]

    experiments = []

    # Baseline
    experiments.append({"name": "baseline", "target": None, "alpha": 0.0, "ratio": 1.0})

    # Space sweep
    for alpha in alphas[1:]:
        for ratio in ratios:
            experiments.append({
                "name": f"space_a{alpha}_r{ratio}",
                "target": SPACE_TARGET,
                "alpha": alpha,
                "ratio": ratio,
            })

    # Also test ocean and horror at a few selected configs
    for target_name, target in [("ocean", OCEAN_TARGET), ("horror", HORROR_TARGET)]:
        for alpha in [1.0, 2.0]:
            experiments.append({
                "name": f"{target_name}_a{alpha}_r0.5",
                "target": target,
                "alpha": alpha,
                "ratio": 0.5,
            })

    config_base = MDLMSamplerConfig(
        steps=64,
        max_new_tokens=64,
        block_size=32,
        temperature=0.6,  # moderate temperature for diversity
        remasking="low_confidence",
    )

    results = []
    N_SAMPLES = 2

    for exp in experiments:
        print(f"\n{'─' * 70}")
        print(f"  [{exp['name']}]")
        print(f"{'─' * 70}")

        sampler.set_energy_guidance(
            target_texts=exp["target"],
            alpha=exp["alpha"],
            guidance_ratio=exp["ratio"],
        )

        original_forward = model.forward

        def guided_forward(input_ids=None, attention_mask=None, **kwargs):
            out = original_forward(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
            if sampler.guidance_active:
                mask_id = tokenizer.mask_token_id
                mask_positions = (input_ids == mask_id)
                out.logits = sampler.apply_energy_guidance(out.logits, mask_positions)
            return out

        model.forward = guided_forward
        sampler.reset_step_counter(64)  # 2 blocks × 32 steps

        for trial in range(N_SAMPLES):
            torch.manual_seed(42 + trial)
            inputs = tokenizer.apply_chat_template(
                [base_prompt], add_generation_prompt=True, tokenize=True
            )
            if isinstance(inputs[0], int):
                inputs = [inputs]

            outputs = sampler.sample(inputs, config_base, return_dict=True)

            for seq in outputs.sequences:
                raw = tokenizer.decode(seq, skip_special_tokens=False)
                response = clean_response(raw)

                if not response or len(response) < 15:
                    continue

                resp_emb = evaluator.encode(
                    [response], convert_to_tensor=True,
                    normalize_embeddings=True, device=DEVICE
                )

                metrics = {}
                if exp["target"]:
                    target_embs = evaluator.encode(
                        exp["target"], convert_to_tensor=True,
                        normalize_embeddings=True, device=DEVICE
                    )
                    metrics["target_sim"] = round(
                        F.cosine_similarity(resp_emb, target_embs).mean().item(), 4
                    )

                metrics["coherence"] = round(coherence_check(response, evaluator), 4)

                # Repetition penalty: count unique words / total words
                words = response.lower().split()
                if words:
                    metrics["diversity"] = round(len(set(words)) / len(words), 4)
                else:
                    metrics["diversity"] = 0.0

                result = {
                    "experiment": exp["name"],
                    "trial": trial,
                    "alpha": exp["alpha"],
                    "ratio": exp["ratio"],
                    "response": response,
                    **metrics,
                }
                results.append(result)

        if results:
            latest = results[-N_SAMPLES:]
            for r in latest:
                print(f"  sim={r.get('target_sim', '—')}, coh={r['coherence']}, div={r['diversity']}", flush=True)
                print(f"  → {r['response'][:120]}", flush=True)

        model.forward = original_forward

    model.forward = original_forward

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print("SUMMARY — Sorted by (target_sim × coherence × diversity)")
    print(f"{'=' * 70}")
    print(f"{'Experiment':<25} {'alpha':>6} {'ratio':>6} {'sim':>7} {'coh':>7} {'div':>7} {'score':>7}")
    print("─" * 70)

    # Score = target_sim × coherence × diversity (higher is better)
    for r in results:
        r["score"] = round(
            r.get("target_sim", 0) * r["coherence"] * r["diversity"], 4
        )

    results.sort(key=lambda x: x["score"], reverse=True)
    seen = set()
    for r in results:
        if r["experiment"] in seen:
            continue
        seen.add(r["experiment"])
        ts = f"{r.get('target_sim', 0):.4f}" if "target_sim" in r else "—"
        print(f"{r['experiment']:<25} {r['alpha']:>6.1f} {r['ratio']:>6.2f} {ts:>7} {r['coherence']:>7.4f} {r['diversity']:>7.4f} {r['score']:>7.4f}")

    print(f"\n{'=' * 70}")
    print("BEST CONFIGURATIONS (score = sim × coherence × diversity)")
    print(f"{'=' * 70}")
    for r in results[:5]:
        print(f"\n  [{r['experiment']}] score={r['score']:.4f} (sim={r.get('target_sim', 0):.4f}, coh={r['coherence']:.4f}, div={r['diversity']:.4f})")
        print(f"  {r['response'][:200]}")

    with open(RESULTS_FILE, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nResults saved to {RESULTS_FILE}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    run_experiment()
