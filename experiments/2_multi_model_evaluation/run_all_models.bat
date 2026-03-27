@echo off
REM Run TurboQuant evaluation on multiple models (Windows version)

setlocal enabledelayedexpansion

REM All models: Qwen, LLaMA, Phi-2, Mistral
set MODELS=Qwen/Qwen2.5-3B-Instruct meta-llama/Llama-2-7b-hf microsoft/phi-2 mistralai/Mistral-7B-Instruct-v0.1
set BITS=2 3 4
set CONTEXTS=2048 4096

echo Starting TurboQuant multi-model evaluation...
echo Models: %MODELS%
echo Bits: %BITS%
echo Contexts: %CONTEXTS%
echo.

for %%m in (%MODELS%) do (
    REM Extract model name (last part after /)
    for %%a in (%%m) do set model_name=%%a

    if not exist "!model_name!" mkdir "!model_name!"

    for %%b in (%BITS%) do (
        echo ================================================
        echo Model: %%m ^| Bits: %%b
        echo ================================================

        python evaluate_model.py --model %%m --bits %%b --contexts %CONTEXTS% --output "!model_name!\results_%%bbit.json"

        if %ERRORLEVEL% equ 0 (
            echo [SUCCESS]
        ) else (
            echo [FAILED]
        )
        echo.
    )
)

echo.
echo All evaluations complete!
pause
