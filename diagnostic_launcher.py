#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EBM Training Diagnostic and Launcher
Bypasses PowerShell command parsing issues by executing Python directly.
"""

import subprocess
import sys
import os
import time
from pathlib import Path


def run_command(cmd, cwd, timeout=3600, verbose=True):
    """Run command with full error handling and visibility."""
    print("=" * 60)
    print("EBM TRAINING DIAGNOSTIC & LAUNCHER")
    print("=" * 60)
    print(f"Command: {cmd}")
    print(f"Working Directory: {cwd}")
    print(f"Timeout: {timeout}s")
    print("=" * 60)
    print()
    
    try:
        # Use shell=False to avoid PowerShell issues
        # Use subprocess.run for better control
        result = subprocess.run(
            cmd,
            shell=False,
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=timeout
        )
        
        if verbose:
            print("[STDOUT]")
            print(result.stdout)
            print("[STDOUT END]")
            print()
            
            if result.stderr:
                print("[STDERR]")
                print(result.stderr)
                print("[STDERR END]")
                print()
        
        print(f"[EXIT CODE] {result.returncode}")
        print("=" * 60)
        
        if result.returncode == 0:
            print("[SUCCESS] Training completed successfully")
            print()
            print("Check logs and checkpoints:")
            print(f"  - Logs: {cwd}/logs/ebm/")
            print(f"  - Checkpoints: {cwd}/checkpoints/ebm/")
            print("=" * 60)
        else:
            print(f"[FAILED] Training failed with exit code {result.returncode}")
            print()
            print("Common issues:")
            print("  - Vulkan not available: Try --device cpu")
            print("  - Missing dependencies: Run pip install -r requirements.txt")
            print("  - Import errors: Check script syntax")
            print("=" * 60)
            
        return result
        
    except subprocess.TimeoutExpired as e:
        print(f"[TIMEOUT] Training timed out after {timeout} seconds")
        print()
        print("The training may still be running in the background.")
        print("Check logs/ebm/ for progress.")
        print("=" * 60)
        return None
        
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        print("=" * 60)
        return None


def check_files(cwd):
    """Check project files and dependencies."""
    print("[FILE CHECK]")
    print(f"Checking files in: {cwd}")
    print()
    
    required_files = [
        "config.py",
        "model.py",
        "splats.py",
        "energy.py",
        "langevin.py",
        "soc.py",
        "decoder.py",
        "geometry.py",
        "vulkan_engine.py",
        "train.py",
        "pretrain.py",
        "evaluate.py",
        "dataset_utils.py"
    ]
    
    missing = []
    present = []
    
    for file in required_files:
        file_path = Path(cwd) / file
        if file_path.exists():
            size = file_path.stat().st_size
            present.append(f"  {file} ({size} bytes)")
        else:
            missing.append(f"  [MISSING] {file}")
    
    print("Present files:")
    for f in present[:10]:
        print(f)
    if len(present) > 10:
        print(f"  ... and {len(present) - 10} more files")
    
    if missing:
        print()
        print("Missing files:")
        for f in missing:
            print(f)
        return False
    
    print()
    
    # Check for Python cache files
    print("[PYTHON CACHE CHECK]")
    cache_dir = cwd / "__pycache__"
    if cache_dir.exists():
        cache_files = list(cache_dir.glob("*.pyc"))
        print(f"Found {len(cache_files)} .pyc files in __pycache__")
        
        # Check for problematic imports
        for file in cache_files:
            if file.stat().st_mtime > time.time() - 300:  # Modified in last 5 minutes
                print(f"  [RECENT] {file.name} (modified: {time.ctime(file.stat().st_mtime)})")
    else:
        print("No __pycache__ directory found")
    
    print()
    
    # Check for logs and checkpoints
    print("[LOGS & CHECKPOINTS CHECK]")
    logs_dir = cwd / "logs"
    checkpoints_dir = cwd / "checkpoints"
    
    if logs_dir.exists():
        log_files = list(logs_dir.glob("**/*.json"))
        print(f"Found {len(log_files)} log files in {logs_dir}/")
        for log_file in sorted(log_files)[-5:]:
            mtime = time.ctime(log_file.stat().st_mtime)
            size = log_file.stat().st_size
            print(f"  {log_file.name} ({size} bytes, {mtime})")
    else:
        print(f"No logs directory found at {logs_dir}/")
    
    if checkpoints_dir.exists():
        ckpt_files = list(checkpoints_dir.glob("**/*.pt"))
        print(f"Found {len(ckpt_files)} checkpoint files in {checkpoints_dir}/")
        for ckpt_file in sorted(ckpt_files)[-5:]:
            mtime = time.ctime(ckpt_file.stat().st_mtime)
            size = ckpt_file.stat().st_size
            print(f"  {ckpt_file.name} ({size} bytes, {mtime})")
    else:
        print(f"No checkpoints directory found at {checkpoints_dir}/")
    
    print()
    return True


def check_dependencies():
    """Check if PyTorch and dependencies are installed."""
    print("[DEPENDENCY CHECK]")
    print()
    
    packages = ["torch", "torchvision", "tokenizers", "transformers", "datasets", "scipy"]
    
    installed = []
    missing = []
    
    for pkg in packages:
        try:
            __import__(pkg)
            installed.append(pkg)
            print(f"  [INSTALLED] {pkg}")
        except ImportError:
            missing.append(pkg)
            print(f"  [MISSING] {pkg}")
    
    print()
    
    # Check PyTorch version and Vulkan
    if "torch" in installed:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"Vulkan available: {hasattr(torch.backends, 'vulkan')}")
        print(f"MPS available: {hasattr(torch.backends, 'mps')}")
        print()
    
    return len(missing) == 0


def main():
    """Main diagnostic function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="EBM Training Diagnostic and Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--check", action="store_true", help="Run diagnostic checks only")
    parser.add_argument("--device", default="vulkan", choices=["cpu", "vulkan"], help="Device (default: vulkan)")
    parser.add_argument("--epochs", type=int, default=12, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    
    args = parser.parse_args()
    
    # Project directory
    project_dir = Path(__file__).parent.absolute()
    
    # Run diagnostics first
    print("=" * 60)
    print("EBM TRAINING DIAGNOSTIC")
    print("=" * 60)
    print(f"Project Directory: {project_dir}")
    print()
    
    if not check_files(project_dir):
        print("[ERROR] Required files are missing. Cannot start training.")
        return
    
    if not check_dependencies():
        print("[ERROR] Required dependencies are missing. Cannot start training.")
        print()
        print("Install dependencies with:")
        print(f"  cd {project_dir}")
        print("  pip install -r requirements.txt")
        return
    
    print("[DIAGNOSTIC CHECKS COMPLETED]")
    print("=" * 60)
    print()
    
    # If only checking, exit now
    if args.check:
        return
    
    # Start training
    cmd = [
        sys.executable,
        "pretrain.py",
        "--device", args.device,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size)
    ]
    
    run_command(" ".join(cmd), cwd=project_dir)


if __name__ == "__main__":
    main()
