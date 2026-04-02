#!/usr/bin/env python3
"""Long training run (2h) + evaluation for EBM-splats.
Runs as a single script so it survives session restarts.
Output goes to train_long_output.log
"""
import subprocess, sys, os, time
from datetime import datetime

os.chdir(os.path.dirname(os.path.abspath(__file__)))
log_path = "train_long_output.log"

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")

log("=" * 60)
log("STARTING LONG TRAINING + EVAL")
log("=" * 60)

# Phase 1: Train for 2 hours (7200s)
log("Phase 1: Training (2h budget)")
train_start = time.time()
result = subprocess.run(
    [sys.executable, "autoresearch_train.py", "--time_budget", "7200", "--epochs", "200"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
    timeout=7500  # 2h + buffer
)
train_elapsed = time.time() - train_start

# Log training output
with open(log_path, "a", encoding="utf-8") as f:
    f.write("\n=== TRAIN STDOUT ===\n")
    f.write(result.stdout or "(empty)")
    f.write("\n")

log(f"Training finished in {train_elapsed:.0f}s (rc={result.returncode})")

# Check if checkpoint was saved
ckpt = "checkpoints_autoresearch/best_model.pt"
if os.path.exists(ckpt):
    log(f"Checkpoint exists: {ckpt} ({os.path.getsize(ckpt)/1e6:.1f} MB)")
else:
    log("ERROR: No checkpoint saved!")
    sys.exit(1)

# Phase 2: Evaluate
log("Phase 2: Evaluation")
eval_start = time.time()
result2 = subprocess.run(
    [sys.executable, "autoresearch_eval.py", "--checkpoint", ckpt,
     "--n_samples", "160", "--n_steps", "50", "--n_gen_samples", "5"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
    timeout=600
)

with open(log_path, "a", encoding="utf-8") as f:
    f.write("\n=== EVAL STDOUT ===\n")
    f.write(result2.stdout or "(empty)")
    f.write("\n")

eval_elapsed = time.time() - eval_start
log(f"Evaluation finished in {eval_elapsed:.0f}s (rc={result2.returncode})")

# Extract key results
for line in (result2.stdout or "").split("\n"):
    line = line.strip()
    if any(k in line for k in ["val_bpb", "unique_tokens", "repetition_ratio", "Sample"]):
        log(f"RESULT: {line}")

log("=" * 60)
log(f"ALL DONE. Total: {train_elapsed + eval_elapsed:.0f}s")
log(f"Full log: {os.path.abspath(log_path)}")
log("=" * 60)
