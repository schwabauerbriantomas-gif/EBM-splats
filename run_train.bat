@echo off
cd /d "C:\Users\Brian\.openclaw\workspace\projects\ebm"
python train.py --device vulkan --epochs 12 --batch-size 32
if errorlevel 1 (
    echo Training failed with error code %errorlevel%
    pause
) else (
    echo Training completed successfully
)
