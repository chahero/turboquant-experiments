@echo off
REM Run TurboQuant validation on ALL models (Windows version)
REM Results are saved to results/ directory

setlocal enabledelayedexpansion

echo.
echo ======================================================================
echo TurboQuant: Multi-Model Comprehensive Evaluation
echo ======================================================================
echo.

REM Create results directory
if not exist "results" mkdir results

REM Save current directory
set SCRIPT_DIR=%cd%

cd ..\..\original_implementation

REM Evaluate 3 models (LLaMA requires HuggingFace authentication - gated repo)
set MODELS=Qwen/Qwen2.5-3B-Instruct microsoft/phi-2 mistralai/Mistral-7B-Instruct-v0.1
set RESULTS_DIR=%SCRIPT_DIR%\results

for %%m in (%MODELS%) do (
    echo.
    echo ================================================
    echo Model: %%m
    echo ================================================
    echo.

    REM Convert model name: replace / with _
    set model_file=%%m
    set model_file=!model_file:/=_!

    REM Run validation and save output
    python validate.py --model %%m > "!RESULTS_DIR!\!model_file!_results.txt" 2>&1

    echo [COMPLETED] %%m
    echo Results saved to: !RESULTS_DIR!\!model_file!_results.txt
    echo.
)

echo.
echo ======================================================================
echo All model evaluations complete!
echo ======================================================================
echo Results saved to: %RESULTS_DIR%
echo.
echo File listing:
dir "%RESULTS_DIR%"
echo.

cd %SCRIPT_DIR%
pause
