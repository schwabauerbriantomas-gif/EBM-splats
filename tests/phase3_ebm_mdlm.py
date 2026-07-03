"""
Phase 3: EBM-Guided Masked Diffusion Language Model

HYPOTHESIS: Injecting EBM energy at each denoising step of a masked diffusion
language model gives continuous compositional control over generation —
something impossible with autoregressive decoders.

ARCHITECTURE:
  1. Qwen3-0.6B-diffusion-mdlm generates text via iterative unmasking
  2. At each denoising step, the model produces logits [B, T, vocab] for ALL positions
  3. We compute an energy gradient in embedding space using EBM splat centers
  4. We project this gradient onto token logits via the model's embedding matrix
  5. The modified logits steer generation toward target topics / away from suppressed topics

The key difference from autoregressive guidance:
  - AR: energy injected once at the prompt, then collapsed through token sampling
  - MDLM: energy injected at EVERY step, maintaining continuous control throughout

MECHANISM:
  Given target embedding direction d (from EBM splat energy gradient),
  and model embedding matrix E [vocab, hidden_dim]:

  For each position t:
    token_score[v] = dot(E[v], d)   # how aligned is token v with direction d?
    logits[t] += alpha * token_score

  This is classifier-free guidance with EBM as the classifier, but applied
  bidirectionally across all positions simultaneously.
"""

import sys
import time
import json

import torch
import torch.nn.functional as F
import numpy as np
from sentence_transformers import SentenceTransformer

# dllm imports
import dllm
from dllm.core.samplers.mdlm import MDLMSampler, MDLMSamplerConfig
from dllm.utils import get_model, get_tokenizer

# ── Config ──
DEVICE = "cuda"
MDLM_MODEL = "dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1"
EBM_EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"  # 384D, same as EBM-splats

sys.path.insert(0, "/root/EBM-splats/src/ebm")
sys.path.insert(0, "/root/EBM-splats")

RESULTS_FILE = "/root/EBM-splats/tests/phase3_ebm_mdlm_results.jsonl"


class EnergyGuidedSampler(MDLMSampler):
    """
    MDLM sampler with EBM energy injection at each denoising step.

    Instead of subclassing the full sample() method, we monkey-patch the model's
    forward to intercept logits and add energy guidance before token selection.
    """

    def __init__(self, model, tokenizer, embedder):
        super().__init__(model=model, tokenizer=tokenizer)
        self.embedder = embedder
        self.target_directions = {}  # position_range -> (direction, alpha)
        self.global_direction = None
        self.global_alpha = 0.0
        self.guidance_active = False

        # Pre-compute token embeddings projected to MiniLM space
        # We need a bridge: MiniLM(384D) <-> Qwen3(1024D)
        # Approach: embed all vocabulary tokens with MiniLM, then for each token
        # in Qwen's vocab, look up its MiniLM embedding.
        # This creates a [vocab, 384] matrix we can dot with any 384D direction.
        self._build_token_energy_table()

    def _build_token_energy_table(self):
        """Build a [vocab_size, 384] table of MiniLM embeddings for each token."""
        print("  Building token energy table (MiniLM embeddings for all tokens)...", flush=True)
        t0 = time.time()

        # Decode each token ID to text, then embed with MiniLM
        # Use the model's actual embedding size, not the tokenizer's vocab_size
        vocab_size = self.model.lm_head.weight.shape[0]
        token_texts = []
        for tid in range(vocab_size):
            try:
                text = self.tokenizer.decode([tid], skip_special_tokens=True).strip()
            except Exception:
                text = ""
            token_texts.append(text if text else "<pad>")

        # Batch embed
        embeddings = self.embedder.encode(
            token_texts,
            batch_size=512,
            show_progress_bar=False,
            convert_to_tensor=True,
            normalize_embeddings=True,
            device=DEVICE,
        )
        self.token_energy_table = embeddings  # [vocab, 384]
        print(f"  Done in {time.time()-t0:.1f}s. Shape: {self.token_energy_table.shape}", flush=True)

    def set_energy_guidance(self, target_texts=None, suppress_texts=None, alpha=1.0):
        """
        Set energy guidance directions.

        target_texts: list of strings — generation will be pulled toward these topics
        suppress_texts: list of strings — generation will be pushed away from these topics
        alpha: guidance strength (0.0 = no guidance, 1.0 = moderate, 2.0 = strong)

        The net direction is: d = mean(emb(target)) - mean(emb(suppress))
        Then for each token v: logits[v] += alpha * dot(emb(v), d)
        """
        if target_texts is None and suppress_texts is None:
            self.guidance_active = False
            self.global_direction = None
            return

        d = torch.zeros(384, device=DEVICE)

        if target_texts:
            target_embs = self.embedder.encode(
                target_texts, convert_to_tensor=True,
                normalize_embeddings=True, device=DEVICE
            )
            target_dir = target_embs.mean(dim=0)
            target_dir = F.normalize(target_dir, dim=-1)
            d = d + target_dir

        if suppress_texts:
            suppress_embs = self.embedder.encode(
                suppress_texts, convert_to_tensor=True,
                normalize_embeddings=True, device=DEVICE
            )
            suppress_dir = suppress_embs.mean(dim=0)
            suppress_dir = F.normalize(suppress_dir, dim=-1)
            d = d - suppress_dir

        # Normalize the net direction
        norm = d.norm()
        if norm > 1e-6:
            d = d / norm

        self.global_direction = d
        self.global_alpha = alpha
        self.guidance_active = True

        # Pre-compute per-token energy scores: [vocab]
        # token_energy_table is [vocab, 384], direction is [384]
        with torch.no_grad():
            self.token_scores = torch.mv(
                self.token_energy_table.float(), d.float()
            )  # [vocab]
        print(f"  Energy guidance set: alpha={alpha:.1f}, direction norm={norm:.3f}", flush=True)

    def apply_energy_guidance(self, logits, mask_positions):
        """
        Add energy guidance to logits at masked positions only.
        Non-masked positions are left unchanged (they're already committed).

        logits: [B, T, vocab]
        mask_positions: [B, T] boolean — True where token is still masked
        """
        if not self.guidance_active:
            return logits

        # token_scores: [vocab] → broadcast to [1, 1, vocab]
        scores = self.token_scores.unsqueeze(0).unsqueeze(0) * self.global_alpha

        # Only apply at masked positions
        mask = mask_positions.unsqueeze(-1).float()  # [B, T, 1]
        logits = logits + mask * scores

        return logits


def run_experiment():
    print("=" * 70)
    print("Phase 3: EBM-Guided Masked Diffusion")
    print("=" * 70)

    # ── 1. Load embedder ──
    print("\n[1] Loading MiniLM embedder...", flush=True)
    embedder = SentenceTransformer(EBM_EMBEDDER, device=DEVICE)
    print(f"  Embedding dim: {embedder.get_embedding_dimension()}")

    # ── 2. Load MDLM model ──
    print("\n[2] Loading Qwen3-0.6B-diffusion-mdlm...", flush=True)
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

    # ── 3. Build energy-guided sampler ──
    print("\n[3] Building EnergyGuidedSampler...", flush=True)
    sampler = EnergyGuidedSampler(model=model, tokenizer=tokenizer, embedder=embedder)

    # ── 4. Define experiments ──
    base_prompt = [{"role": "user", "content": "Tell me a short story about anything."}]

    experiments = [
        {
            "name": "baseline",
            "target": None,
            "suppress": None,
            "alpha": 0.0,
        },
        {
            "name": "boost_space",
            "target": ["space exploration", "astronauts", "Mars", "galaxies", "stars"],
            "suppress": None,
            "alpha": 1.0,
        },
        {
            "name": "boost_space_strong",
            "target": ["space exploration", "astronauts", "Mars", "galaxies", "stars"],
            "suppress": None,
            "alpha": 3.0,
        },
        {
            "name": "boost_ocean",
            "target": ["ocean", "underwater", "fish", "coral reef", "deep sea"],
            "suppress": None,
            "alpha": 2.0,
        },
        {
            "name": "boost_space_suppress_ocean",
            "target": ["space exploration", "astronauts", "Mars"],
            "suppress": ["ocean", "underwater", "fish", "sea"],
            "alpha": 2.0,
        },
        {
            "name": "boost_cooking",
            "target": ["cooking", "recipe", "chef", "kitchen", "delicious food"],
            "suppress": None,
            "alpha": 2.0,
        },
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
        print(f"  Experiment: {exp['name']}")
        print(f"{'─' * 70}")

        # Set guidance
        sampler.set_energy_guidance(
            target_texts=exp["target"],
            suppress_texts=exp["suppress"],
            alpha=exp["alpha"],
        )

        # Monkey-patch the model forward to inject energy guidance
        original_forward = model.forward

        def guided_forward(input_ids=None, attention_mask=None, **kwargs):
            out = original_forward(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
            logits = out.logits
            if sampler.guidance_active:
                mask_id = tokenizer.mask_token_id
                mask_positions = (input_ids == mask_id)
                logits = sampler.apply_energy_guidance(logits, mask_positions)
                out.logits = logits
            return out

        model.forward = guided_forward

        # Generate
        inputs = tokenizer.apply_chat_template(
            [base_prompt], add_generation_prompt=True, tokenize=True
        )
        # inputs might be a list of lists or a flat list
        if isinstance(inputs[0], int):
            inputs = [inputs]

        t0 = time.time()
        outputs = sampler.sample(inputs, config, return_dict=True)
        gen_time = time.time() - t0

        # Restore original forward
        model.forward = original_forward

        # Decode
        for seq in outputs.sequences:
            text = tokenizer.decode(seq, skip_special_tokens=True)
            if "<|im_start|>assistant" in text:
                response = text.split("<|im_start|>assistant")[-1]
                response = response.replace("<|im_end|>", "").strip()
            else:
                response = text.strip()

            # Measure semantic alignment with targets
            if response:
                resp_emb = embedder.encode(
                    [response], convert_to_tensor=True,
                    normalize_embeddings=True, device=DEVICE
                )

                metrics = {}
                if exp["target"]:
                    target_embs = embedder.encode(
                        exp["target"], convert_to_tensor=True,
                        normalize_embeddings=True, device=DEVICE
                    )
                    sims = F.cosine_similarity(resp_emb, target_embs)
                    metrics["target_sim_mean"] = sims.mean().item()
                    metrics["target_sim_max"] = sims.max().item()

                if exp["suppress"]:
                    suppress_embs = embedder.encode(
                        exp["suppress"], convert_to_tensor=True,
                        normalize_embeddings=True, device=DEVICE
                    )
                    sims_s = F.cosine_similarity(resp_emb, suppress_embs)
                    metrics["suppress_sim_mean"] = sims_s.mean().item()

                result = {
                    "experiment": exp["name"],
                    "alpha": exp["alpha"],
                    "target": exp["target"],
                    "suppress": exp["suppress"],
                    "response": response[:300],
                    "gen_time": round(gen_time, 1),
                    **metrics,
                }
                results.append(result)

                print(f"  Output: {response[:200]}")
                print(f"  Target sim: {metrics.get('target_sim_mean', 'N/A'):.4f}" if "target_sim_mean" in metrics else "")
                print(f"  Suppress sim: {metrics.get('suppress_sim_mean', 'N/A'):.4f}" if "suppress_sim_mean" in metrics else "")
                print(f"  Time: {gen_time:.1f}s")

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Experiment':<30} {'target_sim':>12} {'suppress_sim':>14} {'alpha':>7}")
    print("─" * 70)
    for r in results:
        ts = f"{r.get('target_sim_mean', 0):.4f}" if "target_sim_mean" in r else "—"
        ss = f"{r.get('suppress_sim_mean', 0):.4f}" if "suppress_sim_mean" in r else "—"
        print(f"{r['experiment']:<30} {ts:>12} {ss:>14} {r['alpha']:>7.1f}")

    # Save results
    with open(RESULTS_FILE, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nResults saved to {RESULTS_FILE}")

    # ── Verdict ──
    baseline_target = None
    for r in results:
        if r["experiment"] == "baseline" and "target_sim_mean" not in r:
            baseline_target = 0.0

    guided_sims = [r.get("target_sim_mean", 0) for r in results if r["experiment"] != "baseline" and "target_sim_mean" in r]
    baseline_sims = [r.get("target_sim_mean", 0) for r in results if r["experiment"] == "baseline" and "target_sim_mean" in r]

    if guided_sims:
        print(f"\nMean target similarity (guided): {np.mean(guided_sims):.4f}")
    if baseline_sims:
        print(f"Mean target similarity (baseline): {np.mean(baseline_sims):.4f}")

    print(f"\n{'=' * 70}")
    if guided_sims and np.mean(guided_sims) > 0.3:
        print("VERDICT: EBM energy guidance during masked diffusion WORKS")
    else:
        print("VERDICT: EBM energy guidance shows WEAK effect — needs tuning")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    run_experiment()
