#!/usr/bin/env python3
"""
Subagent Health Monitor - Pre/post flight checks for autoresearch agents.
Usage: python agent_monitor.py [check|preflight|postflight BEST_COMMIT]
"""

import subprocess, sys, os, time

REPO = r"C:\Users\Brian\Desktop\EBM-splats"
RESULTS = os.path.join(REPO, "results.tsv")

def run(cmd, timeout=30):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=REPO, timeout=timeout)
    return r.stdout.strip(), r.stderr.strip(), r.returncode

def preflight():
    """Check conditions before spawning an agent."""
    issues = []

    # 1. Check git state
    out, _, _ = run("git status --short")
    modified = [l for l in out.split('\n') if l.strip() and 'results.tsv' not in l]
    if modified:
        issues.append(f"Dirty working tree: {modified}")
        run("git checkout -- autoresearch_train.py")

    # 2. Check GPU
    out, _, _ = run("nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader")
    if out:
        used, total = out.split(',')
        used_mb = float(used.strip())
        total_mb = float(total.strip())
        if used_mb > total_mb * 0.8:
            issues.append(f"GPU nearly full: {used_mb:.0f}/{total_mb:.0f} MB")
        else:
            print(f"[OK] GPU: {used_mb:.0f}/{total_mb:.0f} MB free")
    else:
        print("[WARN] Could not check GPU status")

    # 3. Check no zombie python processes
    out, _, _ = run('tasklist /FI "IMAGENAME eq python.exe" /FO CSV /NH')
    py_procs = [l for l in out.split('\n') if 'python' in l.lower()]
    print(f"[INFO] Python processes: {len(py_procs)}")

    if issues:
        print(f"[ISSUES] {len(issues)} problem(s) found:")
        for i in issues:
            print(f"  - {i}")
        return False

    print("[OK] Preflight passed. Safe to spawn agent.")
    return True

def postflight(best_commit=None):
    """Validate results after agent completes."""
    print("=== POSTFLIGHT CHECK ===")

    # 1. Count commits since best_commit
    if best_commit:
        out, _, _ = run(f"git log {best_commit}..HEAD --oneline")
        commits = [l for l in out.split('\n') if l.strip()]
        print(f"[INFO] New commits: {len(commits)}")
        if len(commits) < 2:
            print("[WARN] Fewer than 2 new commits - agent may have been ineffective")
            return False
        for c in commits:
            print(f"  {c}")

    # 2. Parse results.tsv
    if os.path.exists(RESULTS):
        with open(RESULTS) as f:
            lines = f.readlines()
        print(f"[INFO] Results entries: {len(lines) - 1}")
        best_loss = float('inf')
        for line in lines[1:]:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                try:
                    loss = float(parts[1])
                    if loss < best_loss:
                        best_loss = loss
                except ValueError:
                    pass
        print(f"[INFO] Best val_loss in results: {best_loss:.4f}")

        # Check for NaN/Inf
        for line in lines[1:]:
            if 'nan' in line.lower() or 'inf' in line.lower():
                print(f"[WARN] Invalid result: {line.strip()}")

    # 3. Check git state
    out, _, _ = run("git status --short")
    modified = [l for l in out.split('\n') if l.strip()]
    untracked = [l for l in modified if l.startswith('?')]
    modified = [l for l in modified if not l.startswith('?') and 'results.tsv' not in l]
    if modified:
        print(f"[WARN] Uncommitted changes: {modified}")
    if untracked:
        print(f"[INFO] Untracked files: {len(untracked)}")

    # 4. Run quick eval
    print("[INFO] Running quick evaluation...")
    out, err, code = run("$env:PYTHONWARNINGS='ignore'; python -u autoresearch_eval.py --time_budget 30", timeout=60)
    if out:
        for line in out.split('\n'):
            if line.strip():
                print(f"  {line}")

    print("[OK] Postflight complete.")
    return True

def check_active():
    """Check if any agents are currently running."""
    out, _, _ = run('tasklist /FI "IMAGENAME eq python.exe" /FO CSV /NH')
    py_procs = [l for l in out.split('\n') if 'python' in l.lower()]
    print(f"Python processes: {len(py_procs)}")

    # Check VRAM
    out, _, _ = run("nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader")
    if out:
        print(f"GPU: {out}")

    # Check latest git state
    out, _, _ = run("git log --oneline -3")
    print(f"Latest commits:\n{out}")

    # Check results
    if os.path.exists(RESULTS):
        with open(RESULTS) as f:
            lines = f.readlines()
        print(f"Results entries: {len(lines) - 1}")
        if len(lines) > 1:
            print(f"Last result: {lines[-1].strip()}")

if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'check'

    if cmd == 'preflight':
        ok = preflight()
        sys.exit(0 if ok else 1)
    elif cmd == 'postflight':
        best = sys.argv[2] if len(sys.argv) > 2 else None
        postflight(best)
    elif cmd == 'check':
        check_active()
    else:
        print(f"Usage: {sys.argv[0]} [preflight|postflight|check]")
