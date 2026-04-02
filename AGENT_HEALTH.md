# Subagent Health Monitor

## Purpose
Detect when subagents fail, hang, or produce invalid results. Apply corrective measures automatically.

## Failure Modes

### 1. Silent Crash (exit code 1, no output)
**Detection**: Process exits with code 1, result is empty or minimal
**Cause**: Python import error, unhandled exception swallowed by PowerShell stderr
**Correction**: 
- Re-run with explicit error capture: `python -u script.py 2>&1 | Out-File debug.log -Encoding utf8`
- If still no output: `python -c "import traceback; exec(open('script.py').read())"` to get full traceback

### 2. Timeout (agent runs full budget, few/no experiments)
**Detection**: Agent completes with status "timed out", <3 experiments in results.tsv
**Cause**: Data loading too slow (~40s per run eats budget), agent stuck in edit loop
**Correction**:
- Increase time_budget per experiment OR reduce data load time
- Add checkpoint/resume so agent doesn't re-run baseline
- If data load >30s: pre-cache tokenized data

### 3. Git Conflicts (parallel agents editing same file)
**Detection**: Uncommitted changes, divergent branches, val_loss values not comparable
**Cause**: Multiple agents modifying same file simultaneously
**Correction**:
- NEVER run parallel agents on the same file
- Each agent MUST `git checkout -- file.py` before editing
- Use `git stash` if needed

### 4. Val Loss Not Improving (wall-hitting)
**Detection**: 5+ consecutive "discard" entries in results.tsv
**Cause**: Local minimum, wrong direction, architecture fundamentally limited
**Correction**:
- Reset to best-known commit
- Try orthogonal direction (e.g., if trying LR, switch to architecture)
- Reduce search granularity

### 5. Invalid Results (val_loss NaN, Inf, or negative)
**Detection**: Non-finite val_loss values
**Cause**: Numerical instability, exploding gradients
**Correction**: 
- Lower learning rate by 10x
- Add gradient clipping
- Reduce model size
- Check for NaN in intermediate values

## Monitoring Protocol

### Before Spawning
1. Verify no other agent is modifying the same file
2. Ensure clean git state: `git status --short`
3. Record starting commit hash

### During Execution (check every 5 min)
1. Count new git commits: `git log --oneline START..HEAD`
2. Check results.tsv has new entries
3. Verify VRAM usage: `nvidia-smi`

### After Completion
1. Validate val_loss is finite and positive
2. Verify git state is clean (or intentionally dirty with best result)
3. Compare with previous best: is this an actual improvement?
4. Run evaluation script to verify generated text is plausible

## Corrective Actions Table

| Symptom | Auto-Correct | Manual Override |
|---------|-------------|-----------------|
| Agent timeout, <3 experiments | Increase budget 2x, reduce data load | Kill and restart |
| All experiments crash | Debug script, fix imports | Manual debug |
| Parallel git conflicts | Kill all agents, reset to best commit | `git reset --hard` |
| 5+ discards in row | Reset to best, change experiment direction | Change agent prompt |
| val_loss = NaN/Inf | Revert, lower LR 10x | Inspect gradients |
| No commits after 15 min | Check if agent is running, may be hung | Kill and restart |
| val_loss improving but eval shows gibberish | Keep training, may need more time | Increase time_budget |

## Implementation

### Subagent Wrapper Function

When spawning any training subagent:

```
1. Record START_COMMIT
2. Record START_TIME
3. Spawn agent with:
   - Explicit instruction: "git checkout -- autoresearch_train.py before EACH experiment"
   - Explicit instruction: "git add + commit after each IMPROVEMENT only"
   - Explicit instruction: "Record in results.tsv after EACH experiment"
   - Explicit timeout: 3600s (1 hour)
4. On completion:
   a. Count commits: git log START_COMMIT..HEAD --oneline
   b. Count results: wc -l results.tsv
   c. If commits < 3: log failure, apply corrective action
   d. If best_val_loss < previous_best: record success
   e. If best_val_loss >= previous_best: log stagnation
```

### Pre-flight Checklist (run before each spawn)

```powershell
cd C:\Users\Brian\Desktop\EBM-splats
git status --short  # Should be empty or only results.tsv
nvidia-smi --query-gpu=memory.used --format=csv,noheader  # Check VRAM available
```

### Post-flight Validation (run after each completion)

```powershell
git log --oneline -5
type results.tsv
python -u autoresearch_eval.py --time_budget 30
```
