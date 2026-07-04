"""
Phase 3 v6: Confidence-Repair for Energy-Guided Masked Diffusion

ADAPTED FROM DSPARK (DeepSpec):
  DSpark uses a confidence head to determine which proposed tokens are
  trustworthy. Low-confidence tokens are pruned — the target model
  regenerates them.

OUR ADAPTATION FOR ENERGY-GUIDED DIFFUSION:
  In masked diffusion, every committed token has two signals:
    1. Model confidence: softmax(logits_model)[token] — how sure the model is
    2. Energy bias: energy_score[token] × alpha — how much energy pushed it

  PROBLEM: At high alpha, tokens committed by energy force (low model
  confidence) cause repetition and incoherence.

  SOLUTION: After committing tokens at each step, check confidence.
  If a committed token has model_confidence < threshold, RE-MASK it.
  The next denoising step will regenerate it with fresh logits
  (which now include context from the surrounding committed tokens).

  This is analogous to DSpark's _confident_prefix_length, but applied
  bidirectionally in the masked diffusion setting.

ALGORITHM (Confidence-Repair Sampler):
  At each denoising step:
    1. Forward pass → logits [B, T, V]
    2. Apply energy guidance: logits += alpha * mask * energy_scores
    3. Token selection: argmax(logits_with_noise)
    4. Commit high-confidence tokens via topk
    5. REPAIR: for committed tokens where model_softmax < threshold,
       re-mask them (set back to mask_token_id)
    6. Next step regenerates re-masked positions with updated context
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
MODEL_ID = "GSAI-ML/LLaDA-8B-Instruct"
RESULTS_FILE = "/root/EBM-splats/tests/phase3_v6_confidence_repair_results.jsonl"


class ConfidenceRepairSampler(MDLMSampler):
    """
    MDLM sampler with energy guidance + confidence-based repair.

    After committing tokens at each denoising step, tokens with low model
    confidence are re-masked. The next step regenerates them with updated
    bidirectional context.
    """

    def __init__(self, model, tokenizer):
        super().__init__(model=model, tokenizer=tokenizer)
        self.guidance_active = False
        self.token_scores = None
        self.alpha = 0.0
        self.confidence_threshold = 0.0  # 0 = no repair
        self.repair_stats = {"total_committed": 0, "total_repaired": 0}

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

    def set_energy_guidance(self, target_texts, alpha=5.0, confidence_threshold=0.0):
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
        self.confidence_threshold = confidence_threshold
        self.guidance_active = True

        top_idx = scores.topk(10).indices.tolist()
        top_tokens = [self.tokenizer.decode([i]).strip() for i in top_idx]
        print(f"  Guidance: alpha={alpha:.1f}, conf_thresh={confidence_threshold:.2f}, top: {top_tokens[:6]}", flush=True)

    def apply_energy_guidance(self, logits, mask_positions):
        if not self.guidance_active:
            return logits
        scores = self.token_scores.unsqueeze(0).unsqueeze(0) * self.alpha
        mask = mask_positions.unsqueeze(-1).float()
        return logits.float() + mask * scores


def clean_response(text):
    if "<|start_header_id|>assistant<|end_header_id|>" in text:
        response = text.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
    else:
        response = text
    response = response.replace("<|eot_id|>", "").replace("<|endoftext|>", "").strip()
    return response


def coherence_check(text, evaluator):
    words = text.split()
    if len(words) < 10:
        return 0.0
    mid = len(words) // 2
    h1, h2 = " ".join(words[:mid]), " ".join(words[mid:])
    e1 = evaluator.encode([h1], convert_to_tensor=True, normalize_embeddings=True, device=DEVICE)
    e2 = evaluator.encode([h2], convert_to_tensor=True, normalize_embeddings=True, device=DEVICE)
    return F.cosine_similarity(e1, e2).item()


def sample_with_confidence_repair(
    sampler, model, tokenizer, inputs, config,
    alpha, confidence_threshold, target_texts,
):
    """
    Custom sampling loop with confidence repair.

    This replaces sampler.sample() with a modified version that:
    1. Runs the standard MDLM denoising loop
    2. After each token commit, checks model confidence
    3. Re-masks low-confidence committed tokens
    """
    mask_id = tokenizer.mask_token_id
    embed_matrix = model.get_input_embeddings().weight.data.float()

    # Compute energy direction
    embs = []
    for t in target_texts:
        tokens = tokenizer(t, return_tensors="pt", truncation=True, max_length=128)
        ids = tokens["input_ids"].to(DEVICE)
        with torch.no_grad():
            pooled = embed_matrix[ids].mean(dim=1).squeeze(0)
        embs.append(pooled)
    d = F.normalize(torch.stack(embs).mean(dim=0), dim=-1)
    with torch.no_grad():
        token_scores = torch.mv(embed_matrix, d)
        token_scores = token_scores / (token_scores.abs().max() + 1e-8)
        token_scores = token_scores.to(DEVICE)

    # Parse config
    steps = config.steps
    max_new_tokens = config.max_new_tokens
    block_size = config.block_size
    temperature = config.temperature

    # Build canvas
    if isinstance(inputs[0], list):
        inputs = [torch.as_tensor(p, dtype=torch.long, device=DEVICE) for p in inputs]
    prompt_lens = [p.shape[0] for p in inputs]
    max_length = max_new_tokens + max(prompt_lens)
    B = len(inputs)
    T = max_length

    eos_id = tokenizer.eos_token_id
    x = torch.full((B, T), eos_id, dtype=torch.long, device=DEVICE)
    for i, p in enumerate(inputs):
        x[i, :prompt_lens[i]] = p
        x[i, prompt_lens[i]:prompt_lens[i] + max_new_tokens] = mask_id

    attention_mask = torch.zeros((B, T), dtype=torch.long, device=DEVICE)
    for i, pl in enumerate(prompt_lens):
        attention_mask[i, :min(pl + max_new_tokens, T)] = 1

    num_blocks = math.ceil(max_new_tokens / block_size)
    steps_per_block = max(1, math.ceil(steps / num_blocks))

    repair_count = 0
    commit_count = 0

    for b in range(num_blocks):
        block_mask_index = torch.zeros((B, block_size), dtype=torch.bool, device=x.device)
        for j in range(B):
            start = prompt_lens[j] + b * block_size
            end = min(start + block_size, prompt_lens[j] + max_new_tokens, T)
            if start < end:
                block_mask_index[j, :end - start] = x[j, start:end] == mask_id

        from dllm.core.samplers.utils import get_num_transfer_tokens, add_gumbel_noise
        num_transfer_tokens = get_num_transfer_tokens(
            mask_index=block_mask_index, steps=steps_per_block,
            scheduler=sampler.scheduler, stochastic=False,
        )
        effective_steps = num_transfer_tokens.size(1)

        for i in range(effective_steps):
            mask_index = x == mask_id

            # Forward pass
            with torch.no_grad():
                logits = model(x, attention_mask=attention_mask).logits

            # Apply energy guidance
            scores = token_scores.unsqueeze(0).unsqueeze(0) * alpha
            mask = mask_index.unsqueeze(-1).float()
            logits = logits.float() + mask * scores

            # Token selection
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            # Confidence computation from ORIGINAL logits (before energy)
            with torch.no_grad():
                orig_logits = model(x, attention_mask=attention_mask).logits
                model_probs = F.softmax(orig_logits.float(), dim=-1)
            x0_confidence = torch.squeeze(
                torch.gather(model_probs, dim=-1, index=torch.unsqueeze(x0, -1)), -1
            )

            # Low-confidence remasking
            if remasking == "low_confidence":
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                )
            elif remasking == "random":
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            for j in range(B):
                x0_p[j, prompt_lens[j] + (b + 1) * block_size:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True

            # ── CONFIDENCE REPAIR ──
            if confidence_threshold > 0:
                # Among committed tokens, check model confidence
                committed_mask = transfer_index & mask_index
                low_conf = committed_mask & (x0_confidence < confidence_threshold)
                # Re-mask low-confidence tokens
                transfer_index = transfer_index & ~low_conf
                repair_count += low_conf.sum().item()

            commit_count += transfer_index.sum().item()
            x[transfer_index] = x0[transfer_index]

    repair_rate = repair_count / max(commit_count, 1)
    return x, {"repairs": repair_count, "commits": commit_count, "repair_rate": repair_rate}


def run_experiment():
    print("=" * 70)
    print("Phase 3 v6: Confidence-Repair for Energy-Guided Diffusion")
    print("(DSpark-inspired: re-mask low-confidence energy-forced tokens)")
    print("=" * 70)

    print("\n[1] Loading LLaDA-8B...", flush=True)
    model = get_model(
        model_args=type("Args", (), {
            "model_name_or_path": MODEL_ID,
            "dtype": torch.bfloat16,
            "device_map": {"": 0},
        })()
    ).eval()
    tokenizer = get_tokenizer(model_args=type("Args", (), {"model_name_or_path": MODEL_ID})())
    evaluator = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=DEVICE)
    sampler = ConfidenceRepairSampler(model=model, tokenizer=tokenizer)

    prompt = [{"role": "user", "content": "Write a short story about something interesting."}]

    TOPICS = {
        "ocean":   ["ocean underwater coral reef fish diving deep sea submarine waves"],
        "horror":  ["horror nightmare monster ghost darkness fear terrifying scream blood"],
        "space":   ["space exploration stars Mars galaxies astronauts rocket launch mission"],
    }

    config = MDLMSamplerConfig(
        steps=128, max_new_tokens=128, block_size=32,
        temperature=0.6, remasking="low_confidence",
    )

    # ── Experiments ──
    # Compare: no repair vs repair at different thresholds
    experiments = []
    for topic_name, target in TOPICS.items():
        experiments.append({"name": f"{topic_name}_a10_norepair", "target": target, "alpha": 10.0, "threshold": 0.0})
        experiments.append({"name": f"{topic_name}_a10_repair05", "target": target, "alpha": 10.0, "threshold": 0.5})
        experiments.append({"name": f"{topic_name}_a10_repair03", "target": target, "alpha": 10.0, "threshold": 0.3})

    # Baseline (no guidance)
    experiments.insert(0, {"name": "baseline", "target": None, "alpha": 0.0, "threshold": 0.0})

    results = []
    N = 2

    for exp in experiments:
        print(f"\n{'─' * 70}")
        print(f"  [{exp['name']}]")
        print(f"{'─' * 70}")

        inputs = tokenizer.apply_chat_template([prompt], add_generation_prompt=True, tokenize=True)
        if isinstance(inputs[0], int):
            inputs = [inputs]

        for trial in range(N):
            torch.manual_seed(42 + trial)

            if exp["alpha"] == 0:
                # Baseline: standard sampling
                outputs = sampler.sample(inputs, config, return_dict=True)
                repair_stats = {"repairs": 0, "commits": 0, "repair_rate": 0}
            else:
                # Confidence-repair sampling
                global remasking
                remasking = config.remasking
                sequences, repair_stats = sample_with_confidence_repair(
                    sampler, model, tokenizer, inputs, config,
                    alpha=exp["alpha"],
                    confidence_threshold=exp["threshold"],
                    target_texts=exp["target"],
                )
                outputs = type("O", (), {"sequences": sequences})()

            for seq in outputs.sequences:
                raw = tokenizer.decode(seq, skip_special_tokens=False)
                response = clean_response(raw)
                if len(response) < 15:
                    continue

                resp_emb = evaluator.encode([response], convert_to_tensor=True, normalize_embeddings=True, device=DEVICE)
                metrics = {"coherence": round(coherence_check(response, evaluator), 4)}
                words = response.lower().split()
                metrics["diversity"] = round(len(set(words)) / max(len(words), 1), 4)

                if exp["target"]:
                    target_embs = evaluator.encode(exp["target"], convert_to_tensor=True, normalize_embeddings=True, device=DEVICE)
                    metrics["target_sim"] = round(F.cosine_similarity(resp_emb, target_embs).mean().item(), 4)

                metrics["repair_rate"] = round(repair_stats["repair_rate"], 4)

                result = {
                    "experiment": exp["name"],
                    "trial": trial,
                    "alpha": exp["alpha"],
                    "threshold": exp["threshold"],
                    "response": response,
                    **metrics,
                }
                results.append(result)

                print(f"  [{trial}] sim={metrics.get('target_sim', '—')}, coh={metrics['coherence']:.4f}, div={metrics['diversity']:.4f}, repair={metrics['repair_rate']:.4f}", flush=True)
                print(f"       {response[:180]}", flush=True)

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Experiment':<30} {'sim':>8} {'coh':>8} {'div':>8} {'repair':>8}")
    print("─" * 65)

    from collections import defaultdict
    agg = defaultdict(list)
    for r in results:
        agg[r["experiment"]].append(r)

    for exp_name in [e["name"] for e in experiments]:
        trials = agg.get(exp_name, [])
        if not trials:
            continue
        sims = [t.get("target_sim", 0) for t in trials]
        cohs = [t["coherence"] for t in trials]
        divs = [t["diversity"] for t in trials]
        reps = [t["repair_rate"] for t in trials]
        has_sim = any("target_sim" in t for t in trials)
        sm = f"{np.mean(sims):.4f}" if has_sim else "   —"
        print(f"{exp_name:<30} {sm:>8} {np.mean(cohs):>8.4f} {np.mean(divs):>8.4f} {np.mean(reps):>8.4f}")

    with open(RESULTS_FILE, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nResults saved to {RESULTS_FILE}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    run_experiment()
