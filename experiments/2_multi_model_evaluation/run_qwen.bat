@echo off
REM Run TurboQuant validation on Qwen2.5-3B-Instruct (Windows version)
REM Uses original_implementation/validate.py which has proven tokenizer compatibility

setlocal enabledelayedexpansion

echo Evaluating Qwen2.5-3B-Instruct with TurboQuant...
echo.

cd ..\..\original_implementation
python validate.py

cd ..\..\experiments\2_multi_model_evaluation
pause
