@echo off
echo =======================================================
echo Starting NeuroCodec Training Pipeline
echo =======================================================
python train.py --config config.yaml --epochs 10 --batch_size 4
pause
