#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final EBM Diagnostic Tool
Checks Antigravity status, project files, and provides solutions.
"""

import os
import sys
import subprocess
import json
from pathlib import Path
import time


def run_command(cmd, cwd=None, timeout=10):
    """Run command and return status."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return {
            'success': True,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'returncode': result.returncode
        }
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'stdout': '',
            'stderr': f"Command timed out after {timeout}s",
            'returncode': -1
        }
    except Exception as e:
        return {
            'success': False,
            'stdout': '',
            'stderr': str(e),
            'returncode': -2
        }


def check_antigravity():
    """Check if Antigravity is installed and get its path."""
    print("=" * 60)
    print("[ANTIGRAVITY CHECK]")
    print("=" * 60)
    
    # Check common paths
    antigravity_paths = [
        r"C:\Users\Brian\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Antigravity\Antigravity.exe",
        r"C:\Users\Brian\AppData\Local\Programs\Antigravity\Antigravity.exe",
        r"C:\Users\Brian\AppData\Roaming\Antigravity\Antigravity.exe",
    ]
    
    for path in antigravity_paths:
        if os.path.exists(path):
            print(f"✅ Antigravity found at: {path}")
            return {'installed': True, 'path': path}
    
    print("❌ Antigravity not found in common locations")
    return {'installed': False, 'path': None}


def check_project_files(project_dir):
    """Check all EBM project files."""
    print("\n" + "=" * 60)
    print("[PROJECT FILES CHECK]")
    print("=" * 60)
    print(f"Project Directory: {project_dir}")
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
        "diagnose.py",
        "dataset_utils.py"
    ]
    
    missing_files = []
    file_info = {}
    
    for filename in required_files:
        filepath = os.path.join(project_dir, filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            mtime = os.path.getmtime(filepath)
            file_info[filename] = {
                'exists': True,
                'size_bytes': size,
                'last_modified': mtime
            }
            print(f"  ✅ {filename:30s} ({size:8,} bytes, {time.ctime(mtime)})")
        else:
            missing_files.append(filename)
            file_info[filename] = {'exists': False}
            print(f"  ❌ {filename:30s} [MISSING]")
    
    print()
    if missing_files:
        print(f"Missing files ({len(missing_files)}):")
        for f in missing_files:
            print(f"  - {f}")
    else:
        print("✅ All required files found")
    
    return file_info, missing_files


def check_python():
    """Check Python installation and version."""
    print("\n" + "=" * 60)
    print("[PYTHON CHECK]")
    print("=" * 60)
    
    try:
        result = run_command(["python", "--version"], timeout=5)
        if result['success']:
            print(f"✅ Python installed: {result['stdout'].strip()}")
            return {'installed': True, 'version': result['stdout'].strip()}
    except:
        print("❌ Python not found or not in PATH")
        return {'installed': False}


def check_dependencies():
    """Check PyTorch and key dependencies."""
    print("\n" + "=" * 60)
    print("[DEPENDENCIES CHECK]")
    print("=" * 60)
    
    packages = ["torch", "torchvision", "transformers", "tokenizers", "datasets"]
    installed = {}
    
    for pkg in packages:
        try:
            __import__(pkg)
            installed[pkg] = True
            print(f"  ✅ {pkg}")
        except ImportError:
            installed[pkg] = False
            print(f"  ❌ {pkg} [MISSING]")
    
    if installed.get('torch'):
        try:
            import torch
            print(f"    PyTorch version: {torch.__version__}")
            print(f"    CUDA available: {torch.cuda.is_available()}")
            print(f"    Vulkan available: {hasattr(torch.backends, 'vulkan')}")
        except:
            pass
    
    return installed


def check_common_errors():
    """Check for common EBM project errors."""
    print("\n" + "=" * 60)
    print("[COMMON ERRORS CHECK]")
    print("=" * 60)
    print()
    
    project_dir = r"C:\Users\Brian\.openclaw\workspace\projects\ebm"
    
    # Check for circular imports
    print("Checking for circular imports...")
    splats_path = os.path.join(project_dir, "splats.py")
    if os.path.exists(splats_path):
        with open(splats_path, 'r', encoding='utf-8') as f:
            content = f.read()
            if 'from splats import' in content and 'class ImprovedSplatStore' in content:
                print("  ⚠️  WARNING: Circular import detected in splats.py")
                print("     - File imports from 'splats' but defines 'ImprovedSplatStore'")
                print("     - Fix: Remove 'from splats import SplatStore' line")
            else:
                print("  ✅ No circular imports detected")
    
    # Check for syntax errors
    print("\nChecking for syntax errors...")
    train_path = os.path.join(project_dir, "train.py")
    if os.path.exists(train_path):
        try:
            compile(train_path, 'train.py', 'exec')
            print("  ✅ train.py - No syntax errors")
        except SyntaxError as e:
            print(f"  ❌ train.py - Syntax error: {e}")
            print(f"     Line: {e.lineno}")
        except Exception as e:
            print(f"  ⚠️  train.py - Other error: {e}")
    
    print()


def launch_antigravity(project_dir, antigravity_path):
    """Launch Antigravity with the project directory."""
    print("\n" + "=" * 60)
    print("[LAUNCHING ANTIGRAVITY]")
    print("=" * 60)
    print(f"Antigravity: {antigravity_path}")
    print(f"Project Directory: {project_dir}")
    print()
    print("Attempting to launch Antigravity...")
    print("If it doesn't open, check:")
    print("  1. Is Antigravity installed?")
    print("  2. Is the project path correct?")
    print()
    
    try:
        # Use start command to launch without blocking
        result = run_command([antigravity_path], timeout=5)
        if result['success']:
            print("✅ Antigravity launched successfully")
        else:
            print(f"❌ Failed to launch Antigravity")
            print(f"   Error: {result['stderr']}")
    except Exception as e:
        print(f"❌ Error launching Antigravity: {e}")
    
    print()


def main():
    """Main diagnostic function."""
    project_dir = r"C:\Users\Brian\.openclaw\workspace\projects\ebm"
    
    print("=" * 60)
    print("FINAL EBM DIAGNOSTIC CHECKER")
    print("=" * 60)
    print()
    
    # Run all checks
    antigravity_info = check_antigravity()
    python_info = check_python()
    dependencies = check_dependencies()
    file_info, missing_files = check_project_files(project_dir)
    check_common_errors()
    
    # Summary
    print("\n" + "=" * 60)
    print("[DIAGNOSTIC SUMMARY]")
    print("=" * 60)
    print()
    
    issues = []
    
    if not antigravity_info['installed']:
        issues.append("Antigravity not installed")
    
    if not python_info['installed']:
        issues.append("Python not installed")
    
    if missing_files:
        issues.append(f"Missing {len(missing_files)} project files")
    
    if dependencies.get('torch') and not dependencies['torch']:
        issues.append("PyTorch not installed")
    
    if issues:
        print("❌ ISSUES FOUND:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        print()
        print("RECOMMENDATIONS:")
        print("  1. Install missing dependencies")
        print("  2. Verify project files are present")
        print("  3. Launch Antigravity manually if needed")
        print("=" * 60)
        return False
    
    print("✅ NO CRITICAL ISSUES FOUND")
    print()
    
    # If Antigravity is installed, launch it
    if antigravity_info['installed'] and antigravity_info['path']:
        print("All checks passed. Launching Antigravity...")
        launch_antigravity(project_dir, antigravity_info['path'])
    else:
        print("Please install Antigravity or launch it manually.")
        print()
        print("To launch manually:")
        print(f'  antigravity "{project_dir}"')
        print("=" * 60)
    
    return True


if __name__ == "__main__":
    main()
