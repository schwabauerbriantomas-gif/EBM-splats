"""
Phase 3 FINAL: EBM-Guided LLaDA-8B Masked Diffusion

THE EXPERIMENT: Can EBM energy injection at each denoising step of a large
masked diffusion model (LLaDA-8B) steer generation toward specific topics
while maintaining coherence?

KEY DIFFERENCES FROM v1-v5 (Qwen3-0.6B):
  - LLaDA-8B is 13x larger → much richer token representations
  - LLaDA-8B is instruction-tuned on diverse data → no safety refusals
  - Better baseline coherence → clearer signal when energy guidance works

EXPERIMENT DESIGN:
  1. Open prompt: "Write a short story about something interesting."
  2. Energy guidance toward: space, ocean, horror, cooking, medieval
  3. Measure: target_sim, coherence, diversity, human-readability
  4. Compare: baseline (no guidance) vs guided (various alpha)
  5. Multi-seed for statistical significance
"""

import sys
import time
import json

import torch
import torch.nn.functional as F
import numpy as np
from sentence_transformers import SentenceTransformer

import dllm
from dllm.core.samplers.mdlm import MDLMSampler, MDLMSamplerConfig
from dllm.utils import get_model, get_tokenizer

DEVICE = "cuda"
MODEL_ID = "GSAI-ML/LLaDA-8B-Instruct"
RESULTS_FILE = "/root/EBM-splats/tests/phase3_llada8b_final_results.jsonl"


class EnergyGuidedSampler(MDLMSampler):
    """LLaDA-8B sampler with model-native energy guidance."""

    def __init__(self, model, tokenizer):
        super().__init__(model=model, tokenizer=tokenizer)
        self.guidance_active = False
        self.token_scores = None
        self.alpha = 0.0

        # LLaDA embedding matrix: [vocab, hidden_dim=4096]
        self.embed_matrix = model.get_input_embeddings().weight.data.float()
        self.hidden_dim = self.embed_matrix.shape[1]
        self.vocab_size = self.embed_matrix.shape[0]
        print(f"  Embed matrix: {self.embed_matrix.shape}", flush=True)

    def _embed_texts(self, texts):
        embs = []
        for text in texts:
            tokens = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            ids = tokens["input_ids"].to(DEVICE)
            with torch.no_grad():
                pooled = self.embed_matrix[ids].mean(dim=1).squeeze(0)
            embs.append(pooled)
        return torch.stack(embs)

    def set_energy_guidance(self, target_texts=None, suppress_texts=None, alpha=1.0):
        if not target_texts and not suppress_texts:
            self.guidance_active = False
            return

        d = torch.zeros(self.hidden_dim, device=DEVICE)
        if target_texts:
            d = d + self._embed_texts(target_texts).mean(dim=0)
        if suppress_texts:
            d = d - self._embed_texts(suppress_texts).mean(dim=0)
        d = F.normalize(d, dim=-1)

        with torch.no_grad():
            scores = torch.mv(self.embed_matrix, d)
            # Normalize to [-1, 1]
            scores = scores / (scores.abs().max() + 1e-8)

        self.token_scores = scores.to(DEVICE)
        self.alpha = alpha
        self.guidance_active = True

        top_idx = scores.topk(15).indices.tolist()
        top_tokens = [self.tokenizer.decode([i]).strip() for i in top_idx]
        print(f"  Guidance ON: alpha={alpha:.2f}, top tokens: {top_tokens[:10]}", flush=True)

    def apply_energy_guidance(self, logits, mask_positions):
        if not self.guidance_active:
            return logits

        # Energy scores are in [-1, 1]. Multiply by alpha to get the bias range.
        # At alpha=1.0, energy adds at most ±1.0 to logits (which are typically ±10-20)
        # At alpha=5.0, energy adds ±5.0 — significant but not dominating
        scores = self.token_scores.unsqueeze(0).unsqueeze(0) * self.alpha
        mask = mask_positions.unsqueeze(-1).float()
        return logits.float() + mask * scores


def clean_response(text):
    if "<|im_start|>assistant" in text:
        response = text.split("<|im_start|>assistant")[-1]
    else:
        response = text
    response = response.replace("<|im_end|>", "").strip()
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


def repetition_ratio(text):
    """Measure how repetitive the text is. 1.0 = no repetition, 0.0 = all same word."""
    words = text.lower().split()
    if not words:
        return 0.0
    from collections import Counter
    counts = Counter(words)
    # If the most common word is >50% of text, it's repetitive
    most_common_ratio = counts.most_common(1)[0][1] / len(words)
    return 1.0 - most_common_ratio


def run_experiment():
    print("=" * 70)
    print("Phase 3 FINAL: EBM-Guided LLaDA-8B")
    print("=" * 70)

    # Load model
    print("\n[1] Loading LLaDA-8B-Instruct...", flush=True)
    t0 = time.time()
    model = get_model(
        model_args=type("Args", (), {
            "model_name_or_path": MODEL_ID,
            "dtype": torch.bfloat16,
            "device_map": {"": 0},
        })()
    ).eval()
    print(f"  Loaded in {time.time()-t0:.1f}s, {torch.cuda.memory_allocated()/1e9:.2f} GB VRAM")

    tokenizer = get_tokenizer(
        model_args=type("Args", (), {"model_name_or_path": MODEL_ID})()
    )

    evaluator = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=DEVICE)

    sampler = EnergyGuidedSampler(model=model, tokenizer=tokenizer)

    # ── Experiments ──
    base_prompt = [{"role": "user", "content": "Write a short story about something interesting."}]

    topics = {
        "space":   ["space exploration stars Mars galaxies astronauts rocket launch mission"],
        "ocean":   ["ocean underwater coral reef fish diving deep sea submarine waves"],
        "horror":  ["horror nightmare monster ghost darkness fear terrifying scream blood"],
        "cooking": ["cooking recipe chef kitchen delicious food spices culinary restaurant"],
    }

    experiments = [
        {"name": "baseline",       "target": None,        "alpha": 0.0},
        # Space sweep
        {"name": "space_a2",       "target": topics["space"], "alpha": 2.0},
        {"name": "space_a5",       "target": topics["space"], "alpha": 5.0},
        {"name": "space_a10",      "target": topics["space"], "alpha": 10.0},
        # Ocean
        {"name": "ocean_a5",       "target": topics["ocean"], "alpha": 5.0},
        {"name": "ocean_a10",      "target": topics["ocean"], "alpha": 10.0},
        # Horror
        {"name": "horror_a5",      "target": topics["horror"], "alpha": 5.0},
        {"name": "horror_a10",     "target": topics["horror"], "alpha": 10.0},
        # Cooking
        {"name": "cooking_a5",     "target": topics["cooking"], "alpha": 5.0},
        {"name": "cooking_a10",    "target": topics["cooking"], "alpha": 10.0},
        # Blend: space + ocean
        {"name": "blend_space_ocean", "target": topics["space"] + ["; "] + topics["ocean"], "alpha": 8.0},
    ]

    config = MDLMSamplerConfig(
        steps=128,
        max_new_tokens=128,
        block_size=32,
        temperature=0.6,
        remasking="low_confidence",
    )

    results = []
    N_SAMPLES = 3

    for exp in experiments:
        print(f"\n{'─' * 70}")
        print(f"  [{exp['name']}] alpha={exp['alpha']}")
        print(f"{'─' * 70}")

        sampler.set_energy_guidance(
            target_texts=exp["target"],
            alpha=exp["alpha"],
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

        for trial in range(N_SAMPLES):
            torch.manual_seed(42 + trial)
            inputs = tokenizer.apply_chat_template(
                [base_prompt], add_generation_prompt=True, tokenize=True
            )
            if isinstance(inputs[0], int):
                inputs = [inputs]

            t0 = time.time()
            outputs = sampler.sample(inputs, config, return_dict=True)
            gen_time = time.time() - t0

            for seq in outputs.sequences:
                raw = tokenizer.decode(seq, skip_special_tokens=False)
                response = clean_response(raw)

                if not response or len(response) < 20:
                    continue

                resp_emb = evaluator.encode(
                    [response], convert_to_tensor=True,
                    normalize_embeddings=True, device=DEVICE
                )

                metrics = {
                    "coherence": round(coherence_check(response, evaluator), 4),
                    "diversity": round(len(set(response.lower().split())) / max(len(response.split()), 1), 4),
                    "non_rep": round(repetition_ratio(response), 4),
                    "gen_time": round(gen_time, 1),
                }

                if exp["target"]:
                    target_embs = evaluator.encode(
                        exp["target"], convert_to_tensor=True,
                        normalize_embeddings=True, device=DEVICE
                    )
                    metrics["target_sim"] = round(
                        F.cosine_similarity(resp_emb, target_embs).mean().item(), 4
                    )

                result = {
                    "experiment": exp["name"],
                    "trial": trial,
                    "alpha": exp["alpha"],
                    "response": response,
                    **metrics,
                }
                results.append(result)

                print(f"  [{trial}] sim={metrics.get('target_sim', '—')}, coh={metrics['coherence']}, div={metrics['diversity']}, nonrep={metrics['non_rep']}", flush=True)
                print(f"       {response[:180]}", flush=True)

        model.forward = original_forward

    model.forward = original_forward

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print("AGGREGATE SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Experiment':<22} {'alpha':>6} {'sim_mean':>9} {'sim_std':>9} {'coh':>7} {'div':>7} {'nonrep':>7}")
    print("─" * 70)

    from collections import defaultdict
    agg = defaultdict(list)
    for r in results:
        agg[r["experiment"]].append(r)

    for exp_name in [e["name"] for e in experiments]:
        trials = agg.get(exp_name, [])
        if not trials:
            continue
        sims = [t.get("target_sim", 0) for t in trials]
        cohs = [t.get("coherence", 0) for t in trials]
        divs = [t.get("diversity", 0) for t in trials]
        nonreps = [t.get("non_rep", 0) for t in trials]
        alpha = trials[0]["alpha"]
        has_sim = any("target_sim" in t for t in trials)
        sm = f"{np.mean(sims):.4f}" if has_sim else "   —"
        ss = f"{np.std(sims):.4f}" if has_sim else "   —"
        print(f"{exp_name:<22} {alpha:>6.1f} {sm:>9} {ss:>9} {np.mean(cohs):>7.4f} {np.mean(divs):>7.4f} {np.mean(nonreps):>7.4f}")

    # ── Best examples ──
    print(f"\n{'=' * 70}")
    print("BEST EXAMPLES (highest target_sim with coherence > 0.3)")
    print(f"{'=' * 70}")
    good = [r for r in results if "target_sim" in r and r.get("coherence", 0) > 0.3]
    good.sort(key=lambda x: x["target_sim"], reverse=True)
    for r in good[:8]:
        print(f"\n  [{r['experiment']}] sim={r['target_sim']:.4f}, coh={r['coherence']:.4f}")
        print(f"  {r['response'][:250]}")

    # ── Verdict ──
    print(f"\n{'=' * 70}")
    baseline_sims = []
    guided_sims = [r["target_sim"] for r in results if "target_sim" in r and r["experiment"] != "baseline"]

    if guided_sims:
        print(f"Guided target_sim: mean={np.mean(guided_sims):.4f}, max={np.max(guided_sims):.4f}")
        # Count how many have good coherence
        good_count = sum(1 for r in results if "target_sim" in r and r.get("coherence", 0) > 0.3 and r["target_sim"] > 0.1)
        total_guided = len([r for r in results if "target_sim" in r and r["experiment"] != "baseline"])
        print(f"High-quality guided outputs: {good_count}/{total_guided} ({100*good_count/total_guided:.0f}%)")

        if np.mean(guided_sims) > 0.1 and good_count >= total_guided * 0.3:
            print("\nVERDICT: EBM energy guidance WORKS on LLaDA-8B ✅")
            print("  → Measurable semantic steering with maintained coherence")
        elif np.max(guided_sims) > 0.15:
            print("\nVERDICT: EBM energy guidance shows SIGNAL on LLaDA-8B ⚠️")
            print("  → Effect exists but needs tuning for consistent results")
        else:
            print("\nVERDICT: EBM energy guidance shows WEAK effect ❌")
    print(f"{'=' * 70}")

    with open(RESULTS_FILE, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    run_experiment()
