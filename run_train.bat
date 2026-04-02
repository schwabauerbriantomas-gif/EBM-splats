@echo off
cd /d C:\Users\Brian\Desktop\EBM-splats
echo [%date% %time%] Starting 2h training + eval >> train_long_output.log
python autoresearch_train.py --time_budget 7200 --epochs 200 >> train_long_output.log 2>&1
echo [%date% %time%] Training done, exit=%ERRORLEVEL% >> train_long_output.log
if exist checkpoints_autoresearch\best_model.pt (
    echo [%date% %time%] Checkpoint found, running eval >> train_long_output.log
    python autoresearch_eval.py --checkpoint checkpoints_autoresearch/best_model.pt --n_samples 160 --n_steps 50 --n_gen_samples 5 >> train_long_output.log 2>&1
    echo [%date% %time%] Eval done, exit=%ERRORLEVEL% >> train_long_output.log
) else (
    echo [%date% %time%] ERROR: No checkpoint saved! >> train_long_output.log
)
echo [%date% %time%] ALL DONE >> train_long_output.log
