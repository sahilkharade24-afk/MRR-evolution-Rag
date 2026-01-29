@echo off
echo Starting RAG-MMR Evaluation App...
echo Using virtual environment: %~dp0venv\Scripts\python.exe

call venv\Scripts\activate
venv\Scripts\streamlit run streamlit_app.py --server.address=0.0.0.0 --server.headless=true > app_log.txt 2>&1

type app_log.txt
pause
