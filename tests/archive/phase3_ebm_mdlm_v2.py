"""
Phase 3: EBM-Guided Masked Diffusion (v2 — Model-Native Token Embeddings)

PROBLEM with v1: MiniLM embeddings for individual tokens are noisy because
MiniLM is designed for sentences, not tokens. Also, alpha=1-3 was too small
relative to logit magnitudes of ±10-20.

SOLUTION: Use Qwen3's OWN embedding matrix (1024D) for token-level energy
computation. This gives semantically meaningful per-token scores because
Qwen3's embeddings are designed to represent individual tokens.

ENERGY GUIDANCE MECHANISM:
  1. Define target direction: embed target texts through Qwen3, average pool → d (1024D)
  2. Token scores: scores[v] = dot(embed_matrix[v], d) for all v in vocab
  3. At each denoising step: logits += alpha * mask * scores

ALPHA SCALING:
  We normalize scores to have the same dynamic range as the logits, so alpha=1.0
  means "equal contribution from model and energy." Alpha=0.1 means "10% energy."
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
MDLM_MODEL = "dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1"
RESULTS_FILE = "/root/EBM-splats/tests/phase3_ebm_mdlm_v2_results.jsonl"


class EnergyGuidedSampler(MDLMSampler):
    """MDLM sampler with model-native energy guidance."""

    def __init__(self, model, tokenizer):
        super().__init__(model=model, tokenizer=tokenizer)
        self.guidance_active = False
        self.token_scores = None
        self.alpha = 0.0

        # Get model's embedding matrix: [vocab, hidden_dim]
        self.embed_matrix = model.get_input_embeddings().weight.data.float()  # [vocab, 1024]
        self.hidden_dim = self.embed_matrix.shape[1]
        self.vocab_size = self.embed_matrix.shape[0]
        print(f"  Embed matrix: {self.embed_matrix.shape}", flush=True)

    def _embed_texts_model_native(self, texts):
        """Embed texts using the model's own embedding layer + mean pooling."""
        embs = []
        for text in texts:
            tokens = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            token_ids = tokens["input_ids"].to(DEVICE)
            with torch.no_grad():
                token_embs = self.embed_matrix[token_ids]  # [seq, 1024]
                pooled = token_embs.mean(dim=1)  # [1, 1024]
            embs.append(pooled.squeeze(0))
        return torch.stack(embs)  # [N, 1024]

    def set_energy_guidance(self, target_texts=None, suppress_texts=None, alpha=1.0):
        """Set energy guidance using model-native embeddings."""
        if target_texts is None and suppress_texts is None:
            self.guidance_active = False
            return

        d = torch.zeros(self.hidden_dim, device=DEVICE)

        if target_texts:
            target_embs = self._embed_texts_model_native(target_texts)  # [N, 1024]
            target_dir = target_embs.mean(dim=0)
            d = d + target_dir

        if suppress_texts:
            suppress_embs = self._embed_texts_model_native(suppress_texts)
            suppress_dir = suppress_embs.mean(dim=0)
            d = d - suppress_dir

        # Normalize direction
        d = F.normalize(d, dim=-1)

        # Token scores: dot product of each token's embedding with direction
        # [vocab, 1024] @ [1024] = [vocab]
        with torch.no_grad():
            scores = torch.mv(self.embed_matrix, d)

        # Normalize scores to [-1, 1] range (cosine similarity-like)
        scores = scores / (scores.abs().max() + 1e-8)
        self.token_scores = scores.to(DEVICE)
        self.alpha = alpha
        self.guidance_active = True

        # Show top tokens that energy guidance favors
        top_idx = scores.topk(10).indices.tolist()
        top_tokens = [self.tokenizer.decode([i]) for i in top_idx]
        print(f"  Guidance ON (alpha={alpha:.1f})", flush=True)
        print(f"  Top favored tokens: {top_tokens}", flush=True)

    def apply_energy_guidance(self, logits, mask_positions):
        if not self.guidance_active:
            return logits
        scores = self.token_scores.unsqueeze(0).unsqueeze(0) * self.alpha  # [1, 1, vocab]
        mask = mask_positions.unsqueeze(-1).float()  # [B, T, 1]
        return logits.float() + mask * scores


def clean_response(text, tokenizer):
    """Extract just the assistant's response from the chat template output."""
    if "<|im_start|>assistant" in text:
        response = text.split("<|im_start|>assistant")[-1]
    else:
        response = text
    # Remove think block
    if "<think>" in response:
        response = response.replace(response[response.find("<think>"):response.find("</think>") + len("</think>")], "")
    response = response.replace("<|im_end|>", "").strip()
    return response


def run_experiment():
    print("=" * 70)
    print("Phase 3 v2: EBM-Guided Masked Diffusion (Model-Native)")
    print("=" * 70)

    # Load model
    print("\n[1] Loading Qwen3-0.6B-diffusion-mdlm...", flush=True)
    t0 = time.time()
    model = get_model(
        model_args=type("Args", (), {
            "model_name_or_path": MDLM_MODEL,
            "dtype": torch.bfloat16,
            "device_map": {"": 0},
        })()
    ).eval()
    print(f"  Loaded in {time.time()-t0:.1f}s, {torch.cuda.memory_allocated()/1e9:.2f} GB VRAM")

    tokenizer = get_tokenizer(
        model_args=type("Args", (), {"model_name_or_path": MDLM_MODEL})()
    )

    # Load MiniLM for evaluation (not guidance)
    print("\n[2] Loading MiniLM for evaluation...", flush=True)
    evaluator = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=DEVICE)

    # Build sampler
    print("\n[3] Building EnergyGuidedSampler...", flush=True)
    sampler = EnergyGuidedSampler(model=model, tokenizer=tokenizer)

    # ── Experiments ──
    base_prompt = [{"role": "user", "content": "Tell me a short story about anything."}]

    experiments = [
        {"name": "baseline",         "target": None,                          "suppress": None,                        "alpha": 0.0},
        {"name": "space_a05",        "target": ["space exploration stars Mars galaxies astronauts"], "suppress": None, "alpha": 0.5},
        {"name": "space_a10",        "target": ["space exploration stars Mars galaxies astronauts"], "suppress": None, "alpha": 1.0},
        {"name": "space_a20",        "target": ["space exploration stars Mars galaxies astronauts"], "suppress": None, "alpha": 2.0},
        {"name": "space_a50",        "target": ["space exploration stars Mars galaxies astronauts"], "suppress": None, "alpha": 5.0},
        {"name": "ocean_a20",        "target": ["ocean underwater fish coral reef deep sea waves"], "suppress": None, "alpha": 2.0},
        {"name": "cooking_a20",      "target": ["cooking recipe chef kitchen delicious food spices"], "suppress": None, "alpha": 2.0},
        {"name": "space_supp_ocean", "target": ["space stars Mars galaxies"],   "suppress": ["ocean sea fish underwater"], "alpha": 3.0},
        {"name": "horror_a20",       "target": ["horror fear dark nightmare monster ghost scary"],   "suppress": None, "alpha": 2.0},
    ]

    config = MDLMSamplerConfig(
        steps=64,
        max_new_tokens=64,
        block_size=32,
        temperature=0.0,
        remasking="low_confidence",
    )

    results = []

    for exp in experiments:
        print(f"\n{'─' * 70}")
        print(f"  [{exp['name']}] alpha={exp['alpha']}")
        print(f"{'─' * 70}")

        sampler.set_energy_guidance(
            target_texts=exp["target"],
            suppress_texts=exp["suppress"],
            alpha=exp["alpha"],
        )

        # Monkey-patch model forward
        original_forward = model.forward

        def guided_forward(input_ids=None, attention_mask=None, **kwargs):
            out = original_forward(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
            if sampler.guidance_active:
                mask_id = tokenizer.mask_token_id
                mask_positions = (input_ids == mask_id)
                out.logits = sampler.apply_energy_guidance(out.logits, mask_positions)
            return out

        model.forward = guided_forward

        # Generate N samples per experiment for robustness
        N_SAMPLES = 3
        for trial in range(N_SAMPLES):
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
                response = clean_response(raw, tokenizer)

                if not response or len(response) < 10:
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

                if exp["suppress"]:
                    suppress_embs = evaluator.encode(
                        exp["suppress"], convert_to_tensor=True,
                        normalize_embeddings=True, device=DEVICE
                    )
                    sims_s = F.cosine_similarity(resp_emb, suppress_embs)
                    metrics["suppress_sim"] = round(sims_s.mean().item(), 4)

                result = {
                    "experiment": exp["name"],
                    "trial": trial,
                    "alpha": exp["alpha"],
                    "response": response[:300],
                    "gen_time": round(gen_time, 1),
                    **metrics,
                }
                results.append(result)

                if trial == 0:
                    print(f"  [{trial}] {response[:150]}...", flush=True)
                    if "target_sim" in metrics:
                        print(f"       target_sim={metrics['target_sim']:.4f}", flush=True)
                    if "suppress_sim" in metrics:
                        print(f"       suppress_sim={metrics['suppress_sim']:.4f}", flush=True)

        model.forward = original_forward

    # Restore
    model.forward = original_forward

    # ── Aggregate Summary ──
    print(f"\n{'=' * 70}")
    print("AGGREGATE SUMMARY (mean over trials)")
    print(f"{'=' * 70}")
    print(f"{'Experiment':<25} {'alpha':>6} {'target_sim':>12} {'suppress_sim':>14}")
    print("─" * 70)

    from collections import defaultdict
    agg = defaultdict(list)
    for r in results:
        key = r["experiment"]
        agg[key].append(r)

    for exp_name in [e["name"] for e in experiments]:
        trials = agg.get(exp_name, [])
        if not trials:
            continue
        alpha = trials[0]["alpha"]
        ts = np.mean([t.get("target_sim", 0) for t in trials])
        ss = np.mean([t.get("suppress_sim", 0) for t in trials])
        has_t = any("target_sim" in t for t in trials)
        has_s = any("suppress_sim" in t for t in trials)
        ts_str = f"{ts:.4f}" if has_t else "—"
        ss_str = f"{ss:.4f}" if has_s else "—"
        print(f"{exp_name:<25} {alpha:>6.1f} {ts_str:>12} {ss_str:>14}")

    # Save
    with open(RESULTS_FILE, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nResults saved to {RESULTS_FILE}")

    # ── Verdict ──
    print(f"\n{'=' * 70}")
    baseline_ts = np.mean([r.get("target_sim", 0) for r in results if r["experiment"] == "baseline"])
    guided_ts = [r.get("target_sim", 0) for r in results if "target_sim" in r and r["experiment"] != "baseline"]
    if guided_ts:
        mean_guided = np.mean(guided_ts)
        max_guided = np.max(guided_ts)
        print(f"Baseline target_sim:  {baseline_ts:.4f}")
        print(f"Guided target_sim:    mean={mean_guided:.4f}, max={max_guided:.4f}")
        print(f"Improvement:          +{mean_guided - baseline_ts:.4f} (mean), +{max_guided - baseline_ts:.4f} (max)")

        if mean_guided - baseline_ts > 0.05:
            print("\nVERDICT: EBM energy guidance WORKS — measurable semantic steering ✅")
        elif max_guided - baseline_ts > 0.05:
            print("\nVERDICT: EBM energy guidance shows EFFECT at high alpha — promising ⚠️")
        else:
            print("\nVERDICT: EBM energy guidance shows WEAK effect — needs different approach ❌")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    run_experiment()
