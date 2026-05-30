@echo off
echo Starting DeepFake Detector...
call .venv\Scripts\activate.bat
streamlit run app.py
pause
