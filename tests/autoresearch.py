"""
Autoresearch Framework for EBM-Guided Masked Diffusion
Based on Karpathy's methodology: hypothesis → 5-min experiment → measure → keep/revert.

This is the RESEARCH DRIVER. It loads LLaDA-8B once and runs focused experiments.
Each experiment tests ONE hypothesis with a SMALL sample set.
"""

import sys
import time
import json
import argparse
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np
from sentence_transformers import SentenceTransformer

import dllm
from dllm.core.samplers.mdlm import MDLMSampler, MDLMSamplerConfig
from dllm.utils import get_model, get_tokenizer

DEVICE = "cuda"
MODEL_ID = "GSAI-ML/LLaDA-8B-Instruct"
RESULTS_DIR = "/root/EBM-splats/tests/autoresearch_results"

import os
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─── Metrics ───

def clean_response(text):
    if "<|start_header_id|>assistant" in text:
        response = text.split("<|start_header_id|>assistant")[-1]
    elif "<|im_start|>assistant" in text:
        response = text.split("<|im_start|>assistant")[-1]
    else:
        response = text
    for tok in ["<|im_end|>", "<|eot_id|>", "<|endoftext|>", "<|startoftext|>"]:
        response = response.replace(tok, "")
    return response.strip()


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
    words = text.lower().split()
    if not words:
        return 0.0
    from collections import Counter
    counts = Counter(words)
    most_common_ratio = counts.most_common(1)[0][1] / len(words)
    return 1.0 - most_common_ratio


def target_similarity(text, target_texts, evaluator):
    resp_emb = evaluator.encode([text], convert_to_tensor=True, normalize_embeddings=True, device=DEVICE)
    target_embs = evaluator.encode(target_texts, convert_to_tensor=True, normalize_embeddings=True, device=DEVICE)
    return F.cosine_similarity(resp_emb, target_embs).mean().item()


# ─── Guidance Strategies ───

class GuidanceConfig:
    """Configuration for energy guidance — each strategy is a hypothesis."""
    def __init__(self, strategy="logit_additive", alpha=5.0, alpha_schedule="constant",
                 norm_method="abs_max", apply_to="masked_only", score_method="mean_embedding", **kwargs):
        self.strategy = strategy  # logit_additive, logit_blended, prob_multiplicative, prob_additive
        self.alpha = alpha
        self.alpha_schedule = alpha_schedule  # constant, linear_up, linear_down, cosine
        self.norm_method = norm_method  # abs_max, z_score, min_max, softmax
        self.apply_to = apply_to  # masked_only, all
        self.score_method = score_method  # mean_embedding, minilm_similarity, cosine_all
        self.kwargs = kwargs

    def get_alpha_at_step(self, step, total_steps):
        """Alpha schedule: how guidance strength varies over denoising steps."""
        t = step / max(total_steps - 1, 1)
        if self.alpha_schedule == "constant":
            return self.alpha
        elif self.alpha_schedule == "linear_up":
            return self.alpha * t  # 0 → alpha
        elif self.alpha_schedule == "linear_down":
            return self.alpha * (1 - t)  # alpha → 0
        elif self.alpha_schedule == "cosine":
            return self.alpha * 0.5 * (1 - np.cos(np.pi * t))  # 0 → alpha → 0
        return self.alpha


def compute_token_scores(embed_matrix, target_texts, suppress_texts, tokenizer, norm_method,
                         evaluator=None, score_method="mean_embedding"):
    """Compute per-token energy scores from target/suppress text embeddings.

    score_method:
      - 'mean_embedding': project target direction onto each token (original)
      - 'minilm_similarity': use MiniLM to embed target, then compare each token's
        MiniLM embedding (decoded as a word) to the target. More semantically meaningful.
      - 'cosine_all': compute cosine similarity between every target token embedding
        and every vocab token, take the max (soft closest match).
    """
    if score_method == "minilm_similarity" and evaluator is not None:
        # Embed the target texts with MiniLM (sentence-level semantics)
        target_emb = evaluator.encode(
            target_texts, convert_to_tensor=True,
            normalize_embeddings=True, device=DEVICE
        ).mean(dim=0)  # [384]
        target_emb = F.normalize(target_emb, dim=-1)

        # For each vocab token, decode it and compute its MiniLM embedding
        # This is expensive (vocab_size encode calls), so we use a cache
        cache_path = f"/root/EBM-splats/tests/vocab_minilm_cache_{tokenizer.name_or_path.replace('/', '_')}.pt"
        import os
        if os.path.exists(cache_path):
            vocab_embs = torch.load(cache_path, map_location=DEVICE)
        else:
            print(f"  Building MiniLM vocab cache ({embed_matrix.shape[0]} tokens)...", flush=True)
            batch_size = 500
            all_embs = []
            for i in range(0, embed_matrix.shape[0], batch_size):
                batch_ids = list(range(i, min(i + batch_size, embed_matrix.shape[0])))
                words = [tokenizer.decode([tid]).strip() for tid in batch_ids]
                # Only encode non-empty words
                valid = [(j, w) for j, w in enumerate(words) if w and len(w) > 1]
                if valid:
                    valid_words = [w for _, w in valid]
                    embs = evaluator.encode(valid_words, convert_to_tensor=True,
                                           normalize_embeddings=True, device=DEVICE,
                                           show_progress_bar=False)
                    batch_embs = torch.zeros(len(batch_ids), embs.shape[1], device=DEVICE)
                    for idx, (j, _) in enumerate(valid):
                        batch_embs[j] = embs[idx]
                    all_embs.append(batch_embs)
                else:
                    all_embs.append(torch.zeros(len(batch_ids), 384, device=DEVICE))
            vocab_embs = torch.cat(all_embs, dim=0)
            torch.save(vocab_embs, cache_path)
            print(f"  Cached {vocab_embs.shape[0]} token embeddings", flush=True)

        scores = torch.mv(vocab_embs, target_emb)  # [vocab]

    elif score_method == "cosine_all":
        # Compute cosine similarity between each target token embedding and all vocab tokens
        target_embs = []
        for text in (target_texts or []):
            tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            ids = tokens["input_ids"].to(DEVICE)
            with torch.no_grad():
                target_embs.append(embed_matrix[ids].squeeze(0))  # [seq_len, hidden]

        if target_embs:
            target_cat = torch.cat(target_embs, dim=0)  # [total_target_tokens, hidden]
            target_cat = F.normalize(target_cat, dim=-1)
            embed_norm = F.normalize(embed_matrix, dim=-1)
            # Cosine similarity: [vocab, target_tokens]
            sim_matrix = torch.mm(embed_norm, target_cat.T)
            # Take max similarity across target tokens (soft closest match)
            scores = sim_matrix.max(dim=-1).values  # [vocab]
        else:
            scores = torch.zeros(embed_matrix.shape[0], device=DEVICE)

    else:  # mean_embedding (original)
        d = torch.zeros(embed_matrix.shape[1], device=DEVICE)

        for text in (target_texts or []):
            tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            ids = tokens["input_ids"].to(DEVICE)
            with torch.no_grad():
                pooled = embed_matrix[ids].mean(dim=1).squeeze(0)
            d = d + pooled

        for text in (suppress_texts or []):
            tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            ids = tokens["input_ids"].to(DEVICE)
            with torch.no_grad():
                pooled = embed_matrix[ids].mean(dim=1).squeeze(0)
            d = d - pooled

        d = F.normalize(d, dim=-1)
        with torch.no_grad():
            scores = torch.mv(embed_matrix, d)  # [vocab]

    # Normalization
    if norm_method == "abs_max":
        scores = scores / (scores.abs().max() + 1e-8)
    elif norm_method == "z_score":
        scores = (scores - scores.mean()) / (scores.std() + 1e-8)
        scores = torch.tanh(scores)  # bound to [-1, 1]
    elif norm_method == "min_max":
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8) * 2 - 1
    elif norm_method == "softmax_temp":
        temp = 0.1
        scores = F.softmax(scores / temp, dim=-1)
        scores = scores * len(scores) - 1  # center around 0

    return scores


def apply_guidance(logits, mask_positions, token_scores, alpha, config: GuidanceConfig):
    """Apply energy guidance to logits based on strategy."""
    scores = token_scores.unsqueeze(0).unsqueeze(0)  # [1, 1, vocab]
    mask = mask_positions.unsqueeze(-1).float()  # [batch, seq, 1]

    if config.apply_to == "masked_only":
        effective_scores = mask * scores * alpha
    else:
        effective_scores = scores * alpha

    if config.strategy == "logit_additive":
        return logits.float() + effective_scores

    elif config.strategy == "logit_blended":
        # Blend guided and unguided: logits = (1-w)*logits + w*(logits + scores)
        w = min(alpha / 10.0, 0.8)  # cap blend weight
        return logits.float() * (1 - w) + (logits.float() + effective_scores) * w

    elif config.strategy == "prob_multiplicative":
        probs = F.softmax(logits.float(), dim=-1)
        guidance_probs = F.softmax(effective_scores, dim=-1)
        new_probs = probs * (1 + guidance_probs * alpha)
        new_probs = new_probs / new_probs.sum(dim=-1, keepdim=True)
        return torch.log(new_probs + 1e-10)

    elif config.strategy == "prob_additive":
        probs = F.softmax(logits.float(), dim=-1)
        guidance_probs = F.softmax(effective_scores, dim=-1)
        w = min(alpha / 10.0, 0.5)
        new_probs = probs * (1 - w) + guidance_probs * w
        return torch.log(new_probs + 1e-10)

    return logits.float()


# ─── Experiment Runner ───

class EnergyGuidedSampler:
    """Wraps LLaDA-8B with energy guidance using a GuidanceConfig."""

    def __init__(self, model, tokenizer, evaluator, config: GuidanceConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.evaluator = evaluator
        self.config = config

        self.embed_matrix = model.get_input_embeddings().weight.data.float()
        self.hidden_dim = self.embed_matrix.shape[1]

        self.token_scores = None
        self.guidance_active = False
        self._original_forward = model.forward
        self._current_step = 0
        self._total_steps = 128

    def set_guidance(self, target_texts=None, suppress_texts=None):
        if not target_texts and not suppress_texts:
            self.guidance_active = False
            return

        self.token_scores = compute_token_scores(
            self.embed_matrix, target_texts, suppress_texts,
            self.tokenizer, self.config.norm_method,
            evaluator=self.evaluator, score_method=self.config.score_method
        )
        self.guidance_active = True

        top_idx = self.token_scores.topk(15).indices.tolist()
        top_tokens = [self.tokenizer.decode([i]).strip() for i in top_idx]
        print(f"  Guidance ON: {top_tokens[:8]}", flush=True)

    def _guided_forward(self, input_ids=None, attention_mask=None, **kwargs):
        out = self._original_forward(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

        if self.guidance_active and self.token_scores is not None:
            mask_id = self.tokenizer.mask_token_id
            if mask_id is not None:
                mask_positions = (input_ids == mask_id)
                if mask_positions.any():
                    current_alpha = self.config.get_alpha_at_step(
                        self._current_step, self._total_steps
                    )
                    out.logits = apply_guidance(
                        out.logits, mask_positions, self.token_scores,
                        current_alpha, self.config
                    )
        return out

    def install(self):
        self.model.forward = self._guided_forward

    def uninstall(self):
        self.model.forward = self._original_forward

    def generate(self, prompt_text, sampler_config, seed=42, target_texts=None):
        torch.manual_seed(seed)

        self._total_steps = sampler_config.steps
        self._current_step = 0

        # Wrap sample to track steps
        original_sample = None
        from dllm.core.samplers.mdlm import MDLMSampler

        # Use MDLMSampler directly
        mdlm = MDLMSampler(model=self.model, tokenizer=self.tokenizer)

        # Monkey-patch to track steps for alpha schedule
        original_fill = mdlm._fill_mask_fn if hasattr(mdlm, '_fill_mask_fn') else None

        inputs = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            add_generation_prompt=True, tokenize=True
        )
        if isinstance(inputs[0], int):
            inputs = [inputs]

        self.install()

        # Track step counter via forward hook
        step_counter = [0]
        original_fwd = self.model.forward

        def counting_forward(input_ids=None, attention_mask=None, **kwargs):
            step_counter[0] += 1
            self._current_step = step_counter[0]
            return self._guided_forward(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

        self.model.forward = counting_forward

        t0 = time.time()
        outputs = mdlm.sample(inputs, sampler_config, return_dict=True)
        gen_time = time.time() - t0

        self.model.forward = original_fwd

        results = []
        for seq in outputs.sequences:
            raw = self.tokenizer.decode(seq, skip_special_tokens=False)
            response = clean_response(raw)

            if not response or len(response) < 20:
                continue

            metrics = {
                "response": response,
                "coherence": round(coherence_check(response, self.evaluator), 4),
                "diversity": round(len(set(response.lower().split())) / max(len(response.split()), 1), 4),
                "non_rep": round(repetition_ratio(response), 4),
                "gen_time": round(gen_time, 1),
            }

            if target_texts:
                metrics["target_sim"] = round(
                    target_similarity(response, target_texts, self.evaluator), 4
                )

            results.append(metrics)

        return results


# ─── Main ───

def run_experiment(model, tokenizer, evaluator, config: GuidanceConfig, label, n_trials=2,
                   suppress_texts=None):
    """Run a single experiment configuration."""
    print(f"\n{'═' * 70}")
    print(f"  EXPERIMENT: {label}")
    print(f"  Strategy: {config.strategy} | α={config.alpha} | schedule={config.alpha_schedule} | norm={config.norm_method}")
    if suppress_texts:
        print(f"  Suppress: {suppress_texts}")
    print(f"{'═' * 70}")

    sampler = EnergyGuidedSampler(model, tokenizer, evaluator, config)

    base_prompt = "Write a short story about something interesting."

    topics = {
        "space":   ["space exploration stars Mars galaxies astronauts rocket launch mission"],
        "ocean":   ["ocean underwater coral reef fish diving deep sea submarine waves"],
        "horror":  ["horror nightmare monster ghost darkness fear terrifying scream blood"],
        "cooking": ["cooking recipe chef kitchen delicious food spices culinary restaurant"],
    }

    all_results = []

    # Baseline (no guidance)
    sampler.guidance_active = False
    for trial in range(n_trials):
        print(f"\n  [baseline trial {trial}]", flush=True)
        outs = sampler.generate(base_prompt, sampler_config, seed=42+trial)
        for o in outs:
            o["experiment"] = "baseline"
            o["trial"] = trial
            o["alpha"] = 0.0
            all_results.append(o)
            print(f"    coh={o['coherence']}, div={o['diversity']}", flush=True)

    # Guided
    for topic_name, target_texts in topics.items():
        sampler.set_guidance(target_texts=target_texts, suppress_texts=suppress_texts)
        for trial in range(n_trials):
            print(f"\n  [{topic_name} trial {trial}]", flush=True)
            outs = sampler.generate(base_prompt, sampler_config, seed=42+trial, target_texts=target_texts)
            for o in outs:
                o["experiment"] = topic_name
                o["trial"] = trial
                o["alpha"] = config.alpha
                o["strategy"] = config.strategy
                o["schedule"] = config.alpha_schedule
                o["norm"] = config.norm_method
                all_results.append(o)
                print(f"    sim={o.get('target_sim','—')}, coh={o['coherence']}", flush=True)

    sampler.guidance_active = False
    sampler.uninstall()

    # Summary
    print(f"\n{'─' * 70}")
    print(f"  SUMMARY: {label}")
    print(f"{'─' * 70}")
    print(f"{'Topic':<12} {'sim_mean':>9} {'sim_max':>9} {'coh_mean':>9} {'coh>0.3':>8}")
    print("─" * 50)

    agg = defaultdict(list)
    for r in all_results:
        if r["experiment"] != "baseline":
            agg[r["experiment"]].append(r)

    guided_sims = []
    good_count = 0
    total_guided = 0

    for topic_name in ["space", "ocean", "horror", "cooking"]:
        trials = agg.get(topic_name, [])
        if not trials:
            continue
        sims = [t.get("target_sim", 0) for t in trials]
        cohs = [t.get("coherence", 0) for t in trials]
        good = sum(1 for s, c in zip(sims, cohs) if s > 0.15 and c > 0.3)
        total = len(sims)
        guided_sims.extend(sims)
        good_count += good
        total_guided += total
        print(f"{topic_name:<12} {np.mean(sims):>9.4f} {max(sims):>9.4f} {np.mean(cohs):>9.4f} {good}/{total:>5}")

    if guided_sims:
        print(f"\n  OVERALL: sim_mean={np.mean(guided_sims):.4f}, good={good_count}/{total_guided} ({100*good_count/max(total_guided,1):.0f}%)")

    # Save
    fname = f"{RESULTS_DIR}/{label}.jsonl"
    with open(fname, "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")
    print(f"  Saved to {fname}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EBM Autoresearch")
    parser.add_argument("--label", required=True, help="Experiment label")
    parser.add_argument("--strategy", default="logit_additive",
                       choices=["logit_additive", "logit_blended", "prob_multiplicative", "prob_additive"])
    parser.add_argument("--alpha", type=float, default=5.0)
    parser.add_argument("--schedule", default="constant",
                       choices=["constant", "linear_up", "linear_down", "cosine"])
    parser.add_argument("--norm", default="abs_max",
                       choices=["abs_max", "z_score", "min_max", "softmax_temp"])
    parser.add_argument("--score_method", default="mean_embedding",
                       choices=["mean_embedding", "minilm_similarity", "cosine_all"])
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--suppress", type=str, default=None,
                       help="Text to suppress (baseline tokens to penalize)")

    args = parser.parse_args()

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
    tokenizer = get_tokenizer(
        model_args=type("Args", (), {"model_name_or_path": MODEL_ID})()
    )
    print(f"  Loaded in {time.time()-t0:.1f}s, {torch.cuda.memory_allocated()/1e9:.2f} GB VRAM")

    evaluator = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=DEVICE)

    config = GuidanceConfig(
        strategy=args.strategy,
        alpha=args.alpha,
        alpha_schedule=args.schedule,
        norm_method=args.norm,
        score_method=args.score_method,
    )

    sampler_config = MDLMSamplerConfig(
        steps=args.steps,
        max_new_tokens=args.max_new_tokens,
        block_size=args.block_size,
        temperature=args.temperature,
        remasking="low_confidence",
    )

    suppress_texts = [args.suppress] if args.suppress else None

    results = run_experiment(model, tokenizer, evaluator, config, args.label,
                             n_trials=args.trials, suppress_texts=suppress_texts)
