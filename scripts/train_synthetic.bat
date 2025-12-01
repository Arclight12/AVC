@echo off
REM Synthetic Training Script
REM Modify EPOCHS below

set EPOCHS=50

python -m src.train --dataset_type synthetic --epochs %EPOCHS%
pause
