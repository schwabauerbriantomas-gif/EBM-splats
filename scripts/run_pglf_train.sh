#!/bin/bash
cd /mnt/c/Users/Brian/Desktop/EBM-splats
export PATH="/root/.hermes/hermes-agent/venv/bin:$PATH"

echo "=== PGLF Training ===" 
echo "Starting at: $(date)"

python3 -m pglf.trainer \
    --epochs 2 \
    --batch-size 32 \
    --lr 1e-4 \
    --dataset tinystories \
    --device cuda \
    --output /mnt/c/Users/Brian/Desktop/EBM-splats/pglf_checkpoints \
    2>&1 | tee /tmp/pglf_train.log

echo ""
echo "Finished at: $(date)"
echo "Exit code: $?"
