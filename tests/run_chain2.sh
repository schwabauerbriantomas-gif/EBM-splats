#!/bin/bash
# Round 2 of autoresearch: richer target descriptions + hybrid strategies
set -e
cd /root/EBM-splats
source /root/.hermes/hermes-agent/venv/bin/activate

echo "============================================"
echo "CHAIN 2: Richer targets + hybrid approaches"
echo "============================================"

# e11: cosine_all + cosine + α=10 + 3 trials (more statistical power)
echo "[$(date)] Starting e11_3trials..."
python3 tests/autoresearch.py \
  --label "e11_cosall_a10_3trials" \
  --strategy logit_additive \
  --alpha 10.0 \
  --schedule cosine \
  --norm abs_max \
  --score_method cosine_all \
  --trials 3 \
  --steps 128 \
  2>&1 | tail -25
echo "[$(date)] e11 done."

# e12: Higher temp (0.9) — more exploration might let guidance take effect
echo "[$(date)] Starting e12_hightemp..."
python3 tests/autoresearch.py \
  --label "e12_cosall_a10_hightemp" \
  --strategy logit_additive \
  --alpha 10.0 \
  --schedule cosine \
  --norm abs_max \
  --score_method cosine_all \
  --temperature 0.9 \
  --trials 2 \
  --steps 128 \
  2>&1 | tail -25
echo "[$(date)] e12 done."

# e13: logit_blended (interpolates between guided and unguided logits)
echo "[$(date)] Starting e13_blended..."
python3 tests/autoresearch.py \
  --label "e13_cosall_a10_blended" \
  --strategy logit_blended \
  --alpha 10.0 \
  --schedule cosine \
  --norm abs_max \
  --score_method cosine_all \
  --trials 2 \
  --steps 128 \
  2>&1 | tail -25
echo "[$(date)] e13 done."

echo "============================================"
echo "CHAIN 2 COMPLETE"
echo "============================================"
