@echo off
echo =======================================================
echo Launching NeuroCodec Explainable AI Web Dashboard
echo =======================================================
python -m streamlit run dashboard.py --server.port 8501
pause
