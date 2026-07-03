"""
Phase 3 v3: EBM-Guided Masked Diffusion — Temperature + Cascade Effect

KEY INSIGHT from v2: Greedy decoding (temperature=0.0) locks the model into its
strongest predictions, making energy guidance ineffective. The fix:

1. temperature > 0 flattens the logit distribution, giving energy guidance room
   to push tokens over the threshold
2. MDLM's confidence-based remasking creates a CASCADE: tokens boosted by energy
   get committed first → they influence the next forward pass via bidirectional
   attention → subsequent tokens are drawn toward the energy direction
3. Multiple steps per block means the cascade compounds

We test: Can energy guidance + temperature steer generation toward a topic
given an OPEN prompt ("Write about something interesting")?
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
RESULTS_FILE = "/root/EBM-splats/tests/phase3_v3_results.jsonl"


class EnergyGuidedSampler(MDLMSampler):
    """MDLM sampler with energy guidance + logit-scale-aware alpha."""

    def __init__(self, model, tokenizer):
        super().__init__(model=model, tokenizer=tokenizer)
        self.guidance_active = False
        self.token_scores = None
        self.alpha = 0.0

        self.embed_matrix = model.get_input_embeddings().weight.data.float()  # [vocab, 1024]
        self.hidden_dim = self.embed_matrix.shape[1]

    def _embed_texts(self, texts):
        """Mean-pool model embeddings for a list of texts."""
        embs = []
        for text in texts:
            tokens = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            ids = tokens["input_ids"].to(DEVICE)
            with torch.no_grad():
                pooled = self.embed_matrix[ids].mean(dim=1).squeeze(0)
            embs.append(pooled)
        return torch.stack(embs)

    def set_energy_guidance(self, target_texts, suppress_texts=None, alpha=1.0):
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
            scores = scores / (scores.abs().max() + 1e-8)

        self.token_scores = scores.to(DEVICE)
        self.alpha = alpha
        self.guidance_active = True

        top_idx = scores.topk(15).indices.tolist()
        top_tokens = [self.tokenizer.decode([i]).strip() for i in top_idx]
        print(f"  Guidance ON: alpha={alpha:.1f}, top tokens: {top_tokens[:8]}", flush=True)

    def apply_energy_guidance(self, logits, mask_positions):
        """Apply energy guidance scaled to logit magnitude."""
        if not self.guidance_active:
            return logits

        # Scale energy scores to match the logit dynamic range
        # This ensures energy is competitive with model predictions
        logit_scale = logits[mask_positions].abs().mean().item() if mask_positions.any() else 1.0
        scale = max(logit_scale, 1.0)

        scores = self.token_scores.unsqueeze(0).unsqueeze(0) * self.alpha * scale
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


def run_experiment():
    print("=" * 70)
    print("Phase 3 v3: EBM-Guided MDLM — Temperature + Scale-Aware Alpha")
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
    print(f"  {torch.cuda.memory_allocated()/1e9:.2f} GB VRAM")

    evaluator = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2", device=DEVICE
    )

    sampler = EnergyGuidedSampler(model=model, tokenizer=tokenizer)

    # ── Open-ended prompt that allows topic steering ──
    base_prompt = [{"role": "user", "content": "Write a short story. Make it interesting."}]

    # ── Experiments ──
    experiments = [
        {"name": "baseline",              "target": None,                          "alpha": 0.0, "temp": 0.6},
        {"name": "space_a3_t06",          "target": ["space exploration stars Mars galaxies astronauts rocket"], "alpha": 3.0, "temp": 0.6},
        {"name": "space_a3_t03",          "target": ["space exploration stars Mars galaxies astronauts rocket"], "alpha": 3.0, "temp": 0.3},
        {"name": "ocean_a3_t06",          "target": ["ocean underwater fish coral reef deep sea waves diving"], "alpha": 3.0, "temp": 0.6},
        {"name": "horror_a3_t06",         "target": ["horror fear dark nightmare monster ghost scary terrifying"], "alpha": 3.0, "temp": 0.6},
        {"name": "cooking_a3_t06",        "target": ["cooking recipe chef kitchen delicious food spices cooking"], "alpha": 3.0, "temp": 0.6},
        {"name": "space_a5_t06",          "target": ["space exploration stars Mars galaxies astronauts rocket"], "alpha": 5.0, "temp": 0.6},
        {"name": "space_a10_t06",         "target": ["space exploration stars Mars galaxies astronauts rocket"], "alpha": 10.0, "temp": 0.6},
    ]

    config_base = MDLMSamplerConfig(
        steps=64,
        max_new_tokens=64,
        block_size=32,
        temperature=0.0,  # overridden per-experiment
        remasking="low_confidence",
    )

    results = []
    N_SAMPLES = 3

    for exp in experiments:
        print(f"\n{'─' * 70}")
        print(f"  [{exp['name']}] alpha={exp['alpha']}, temp={exp['temp']}")
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
        config = MDLMSamplerConfig(**{**config_base.__dict__, "temperature": exp["temp"]})

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
                    sims = F.cosine_similarity(resp_emb, target_embs)
                    metrics["target_sim"] = round(sims.mean().item(), 4)

                result = {
                    "experiment": exp["name"],
                    "trial": trial,
                    "alpha": exp["alpha"],
                    "temp": exp["temp"],
                    "response": response,
                    "gen_time": round(gen_time, 1),
                    **metrics,
                }
                results.append(result)

                if trial == 0:
                    print(f"  [{trial}] {response[:180]}", flush=True)
                    if "target_sim" in metrics:
                        print(f"       sim={metrics['target_sim']:.4f}", flush=True)

        model.forward = original_forward

    model.forward = original_forward

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print("AGGREGATE SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Experiment':<22} {'alpha':>6} {'temp':>5} {'sim_mean':>9} {'sim_std':>9}")
    print("─" * 60)

    from collections import defaultdict
    agg = defaultdict(list)
    for r in results:
        agg[r["experiment"]].append(r)

    experiment_order = [e["name"] for e in experiments]
    for exp_name in experiment_order:
        trials = agg.get(exp_name, [])
        if not trials:
            continue
        sims = [t.get("target_sim", 0) for t in trials]
        alpha = trials[0]["alpha"]
        temp = trials[0]["temp"]
        has_t = any("target_sim" in t for t in trials)
        sm = f"{np.mean(sims):.4f}" if has_t else "—"
        ss = f"{np.std(sims):.4f}" if has_t else "—"
        print(f"{exp_name:<22} {alpha:>6.1f} {temp:>5.1f} {sm:>9} {ss:>9}")

    # ── Best examples ──
    print(f"\n{'=' * 70}")
    print("BEST EXAMPLES (highest target_sim)")
    print(f"{'=' * 70}")
    guided = [r for r in results if "target_sim" in r and r["experiment"] != "baseline"]
    guided.sort(key=lambda x: x["target_sim"], reverse=True)
    for r in guided[:5]:
        print(f"\n  [{r['experiment']}] sim={r['target_sim']:.4f}")
        print(f"  {r['response'][:200]}")

    # ── Verdict ──
    print(f"\n{'=' * 70}")
    baseline_sims = [r.get("target_sim", 0) for r in results if r["experiment"] == "baseline" and "target_sim" in r]
    if not baseline_sims:
        baseline_sims = [0.0]
    guided_sims = [r["target_sim"] for r in guided]

    print(f"Baseline sim:  mean={np.mean(baseline_sims):.4f}")
    print(f"Guided sim:    mean={np.mean(guided_sims):.4f}, max={np.max(guided_sims):.4f}")
    print(f"Delta:         +{np.mean(guided_sims) - np.mean(baseline_sims):.4f}")

    with open(RESULTS_FILE, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nResults saved to {RESULTS_FILE}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    run_experiment()
