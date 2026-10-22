@echo off
echo =======================================================
echo Exporting NeuroCodec to ONNX & CPU Benchmarking
echo =======================================================
python convert_to_onnx.py --config config.yaml --output_dir onnx_models
pause
