@echo off
setlocal enabledelayedexpansion
setlocal
cd /d "C:\Users\Brian\.openclaw\workspace\projects\ebm"

echo Starting EBM Training with Vulkan GPU...
echo ==========================================================

python -c "import torch; print('PyTorch version:', torch.__version__); print('Vulkan available:', hasattr(torch.backends, 'vulkan'))"

if errorlevel 1 (
    echo [ERROR] PyTorch check failed
    goto :end
)

echo.
echo ==========================================================
echo Starting training...
echo ==========================================================
echo.

python train.py --device vulkan --epochs 12 --batch-size 32

if errorlevel 1 (
    echo.
    echo ==========================================================
    echo [ERROR] Training failed
    echo ==========================================================
    echo.
    echo Checking logs in logs\ebm\...
    dir logs\ebm\*.json /b
    echo.
    echo Common issues:
    echo   - Vulkan not available (use --device cpu)
    echo   - Missing dependencies
    echo   - Script syntax errors
    echo.
    goto :end
)

echo.
echo ==========================================================
echo [SUCCESS] Training completed
echo ==========================================================
echo.
echo Check logs for details: logs\ebm\training_log_*.json
echo.

:end
echo.
echo Press any key to exit...
pause > nul
