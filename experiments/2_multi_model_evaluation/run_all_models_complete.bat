@echo off
REM Run TurboQuant validation on ALL models (Windows version)

setlocal enabledelayedexpansion

echo.
echo ======================================================================
echo TurboQuant: Multi-Model Comprehensive Evaluation
echo ======================================================================
echo.

cd ..\..\original_implementation

REM Evaluate all 4 models
set MODELS=Qwen/Qwen2.5-3B-Instruct meta-llama/Llama-2-7b-hf microsoft/phi-2 mistralai/Mistral-7B-Instruct-v0.1

for %%m in (%MODELS%) do (
    echo.
    echo ================================================
    echo Model: %%m
    echo ================================================
    echo.
    python validate.py --model %%m
    echo.
    echo [COMPLETED] %%m
    echo.
)

echo.
echo ======================================================================
echo All model evaluations complete!
echo ======================================================================
echo.

cd ..\..\experiments\2_multi_model_evaluation
pause
