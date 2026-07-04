#!/bin/bash
# Autoresearch chain runner — runs experiments sequentially
# Each experiment takes ~7 min. Total ~21 min.
set -e
cd /root/EBM-splats
source /root/.hermes/hermes-agent/venv/bin/activate

echo "============================================"
echo "CHAIN: 3 experiments (e08, e09, e10)"
echo "============================================"

# e08: Low temperature (0.3) — model more sensitive to logit perturbation
echo "[$(date)] Starting e08_lowtemp..."
python3 tests/autoresearch.py \
  --label "e08_cosall_a10_lowtemp" \
  --strategy logit_additive \
  --alpha 10.0 \
  --schedule cosine \
  --norm abs_max \
  --score_method cosine_all \
  --temperature 0.3 \
  --trials 2 \
  --steps 128 \
  2>&1 | tail -25

echo "[$(date)] e08 done."

# e09: Fewer steps (64) — each step has more weight
echo "[$(date)] Starting e09_fewer_steps..."
python3 tests/autoresearch.py \
  --label "e09_cosall_a10_steps64" \
  --strategy logit_additive \
  --alpha 10.0 \
  --schedule cosine \
  --norm abs_max \
  --score_method cosine_all \
  --steps 64 \
  --trials 2 \
  2>&1 | tail -25

echo "[$(date)] e09 done."

# e10: cosine schedule but with linear_down (strong early, weak late)
echo "[$(date)] Starting e10_lineardown..."
python3 tests/autoresearch.py \
  --label "e10_cosall_a10_lineardown" \
  --strategy logit_additive \
  --alpha 10.0 \
  --schedule linear_down \
  --norm abs_max \
  --score_method cosine_all \
  --trials 2 \
  --steps 128 \
  2>&1 | tail -25

echo "[$(date)] e10 done."
echo "============================================"
echo "ALL CHAIN EXPERIMENTS COMPLETE"
echo "============================================"
