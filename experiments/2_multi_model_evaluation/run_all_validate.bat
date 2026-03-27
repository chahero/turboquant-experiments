@echo off
REM Run TurboQuant validation on ALL models using validate.py base
REM This script evaluates all models using the proven validate.py approach

setlocal enabledelayedexpansion

echo.
echo ================================================
echo TurboQuant Multi-Model Evaluation
echo ================================================
echo.

cd ..\..\original_implementation

REM Run main validation (Qwen is hardcoded in validate.py)
echo Running Qwen2.5-3B-Instruct validation...
python validate.py

REM Note: To evaluate other models (Phi-2, LLaMA, Mistral),
REM modify MODEL_NAME in validate.py and run again, or use evaluate_model.py

cd ..\..\experiments\2_multi_model_evaluation
echo.
echo Complete! Results are shown above.
echo For individual model results, check the JSON outputs in model directories.
pause
