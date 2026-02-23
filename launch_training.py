#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Direct Launcher for EBM Training
Bypasses PowerShell command parsing issues by using .NET StartProcess.
"""

import subprocess
import sys
import os

def main():
    """Run EBM training directly."""
    script_dir = r"C:\Users\Brian\.openclaw\workspace\projects\ebm"
    script_path = os.path.join(script_dir, "train.py")
    
    print("=" * 60)
    print("EBM TRAINING LAUNCHER")
    print("=" * 60)
    print(f"Script: {script_path}")
    print(f"Working Directory: {script_dir}")
    print()
    print("Starting training with Vulkan GPU and Phase 1 improvements...")
    print()
    
    # Use StartProcess (PowerShell .NET) to run Python directly
    # This bypasses PowerShell's command parsing issues
    ps_command = f"""
    $process = New-Object System.Diagnostics.ProcessStartInfo
    $process.FileName = "python"
    $process.Arguments = "{script_path} --device vulkan --epochs 12 --batch-size 32"
    $process.WorkingDirectory = "{script_dir}"
    $process.UseShellExecute = $false
    $process.RedirectStandardOutput = $true
    $process.RedirectStandardError = $true
    $process.WindowStyle = "Normal"
    $process.CreateNoWindow = $true

    $process.Start()

    while (-not $process.HasExited) {{
        Start-Sleep -Seconds 1
    }}

    $output = $process.StandardOutput.ReadToEnd()
    $error = $process.StandardError.ReadToEnd()
    $process.WaitForExit()

    Write-Host "Exit Code: " $process.ExitCode

    if ($output) {{
        Write-Host $output
    }}
    if ($error) {{
        Write-Host "ERROR:"
        Write-Host $error
    }}

    Write-Host "Training completed or terminated."
    """
    
    try:
        # Execute PowerShell with the launcher command
        result = subprocess.run(
            ["powershell", "-Command", ps_command],
            capture_output=True,
            text=True,
            cwd=script_dir,
            timeout=600  # 10 minutes timeout
        )
        
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
            
        print(f"Exit code: {result.returncode}")
        
        if result.returncode == 0:
            print()
            print("=" * 60)
            print("TRAINING INITIATED SUCCESSFULLY")
            print("=" * 60)
            print("Check:")
            print(f"  - logs/ebm/training_log_*.json for progress")
            print(f"  - checkpoints/ebm/ for model checkpoints")
            print("=" * 60)
        else:
            print(f"ERROR: Training exited with code {result.returncode}")
            
    except subprocess.TimeoutExpired:
        print("ERROR: Training timed out after 10 minutes")
        print("The training may still be running in the background.")
    except Exception as e:
        print(f"ERROR: {str(e)}")

if __name__ == "__main__":
    main()
