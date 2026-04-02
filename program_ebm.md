# EBM Autoresearch Program

Adapted from Karpathy's autoresearch protocol for EBM-splats.

## Setup

To set up a new experiment run:

1. **Agree on a run tag**: propose a tag based on date (e.g. `mar26`).
2. **Read the in-scope files**:
   - `autoresearch_train.py` — the ONLY file you modify. Model, optimizer, training loop.
   - `config.py` — EBM config (read-only reference).
   - `score_network.py`, `geometry.py`, `dataset_loader.py` — supporting modules (read-only).
3. **Verify data exists**: TinyStories at D:/datasets/ebm/tinystories_*.txt
4. **Initialize results.tsv**: header row only.

## Experimentation

Each experiment runs on single GPU (RTX 3090, 24GB VRAM). Training runs for **5 minutes wall clock**.

**Launch**: `python autoresearch_train.py > run.log 2>&1`

**What you CAN modify**: `autoresearch_train.py` — everything is fair game: score network architecture, hidden dims, number of layers, noise levels, learning rate, optimizer, batch size, loss function, augmentation, etc.

**What you CANNOT modify**: `config.py`, `geometry.py`, `dataset_loader.py`, `score_network.py`, `energy.py`, `splats.py`, `soc.py`, `langevin.py`.

**Goal**: get the lowest `val_loss` (DSM loss on validation TinyStories). Lower is better.

**Constraint**: Must fit in 24GB VRAM. Do not crash with OOM.

## Output format

```
---
val_loss:        0.012345
best_val_loss:   0.011234
training_seconds:300.1
total_seconds:   325.9
peak_vram_mb:    4500.2
total_steps:     953
num_params_M:    5.3
depth:           4
```

Extract results: `grep "^val_loss:\|^peak_vram_mb:" run.log`

## Logging results

Tab-separated `results.tsv`:
```
commit	val_loss	memory_gb	status	description
```

## The experiment loop

LOOP FOREVER:

1. Look at current git state
2. Modify `autoresearch_train.py` with an experimental idea
3. git commit
4. Run: `python autoresearch_train.py > run.log 2>&1`
5. Extract: `grep "^val_loss:\|^peak_vram_mb:" run.log`
6. If crash: read `tail -n 50 run.log`, fix or skip
7. Record in results.tsv (untracked)
8. If val_loss improved: keep the commit
9. If val_loss equal or worse: git reset to previous

**Timeout**: Kill if >10 minutes. Treat as crash.

**NEVER STOP**: Run indefinitely until manually stopped.

## Ideas to try (adapted for EBM)

- Score network depth (3-8 layers)
- Hidden dimension (256, 512, 1024, 2048)
- Learning rate sweep (1e-5 to 1e-2)
- Noise levels (fewer/more, different ranges)
- Noise annealing schedule
- AdamW betas
- Weight decay (0, 0.001, 0.01, 0.1)
- Batch size (32, 64, 128, 256)
- Gradient clipping threshold
- Sigma encoder (Fourier features vs learned MLP)
- Activation function (GELU, SiLU, ReLU squared)
- Layer normalization vs RMS norm
- Residual connections between score layers
- Dropout
- EMA of score network
- Different embedding initialization
- Sequence length (16, 32, 64, 128)
- Curriculum on noise levels
- Input perturbation probability
- Multi-scale score matching
- Skip connections from input to output
- Score network width changes per layer
- Warmup steps
