#!/bin/bash
# Run TurboQuant evaluation on multiple models

MODELS=(
    "Qwen/Qwen2.5-3B-Instruct"
    "meta-llama/Llama-2-7b-hf"
    "microsoft/phi-2"
    "mistralai/Mistral-7B-Instruct-v0.1"
)

BITS=(2 3 4)
CONTEXTS=(2048 4096)

echo "Starting TurboQuant multi-model evaluation..."
echo "Models: ${MODELS[@]}"
echo "Bits: ${BITS[@]}"
echo "Contexts: ${CONTEXTS[@]}"
echo ""

for model in "${MODELS[@]}"; do
    # Extract model name for directory
    model_name=$(echo "$model" | awk -F'/' '{print tolower($NF)}')
    mkdir -p "$model_name"

    for bits in "${BITS[@]}"; do
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "Model: $model | Bits: $bits"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

        python evaluate_model.py \
            --model "$model" \
            --bits "$bits" \
            --contexts ${CONTEXTS[@]} \
            --output "$model_name/results_${bits}bit.json"

        if [ $? -eq 0 ]; then
            echo "✅ Success"
        else
            echo "❌ Failed"
        fi
        echo ""
    done
done

echo "✅ All evaluations complete!"
