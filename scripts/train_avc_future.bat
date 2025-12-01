@echo off
REM Future AVC 6-Sensor Dataset Training

set DATA_DIR=d:/AVC/data/avc_sensor_dataset
set EPOCHS=100

python -m src.train --dataset_type avc_sensor --data_dir %DATA_DIR% --epochs %EPOCHS%
pause
