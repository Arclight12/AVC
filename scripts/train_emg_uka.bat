@echo off
REM EMG-UKA Training Script
REM Change EPOCHS and DATA_DIR here

set EPOCHS=50
set DATA_DIR=d:/AVC/data/archive/EMG-UKA-Trial-Corpus

python -m src.train --dataset_type emg_uka --data_dir %DATA_DIR% --epochs %EPOCHS%
pause
