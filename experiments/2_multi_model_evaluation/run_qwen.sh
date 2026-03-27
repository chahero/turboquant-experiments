#!/bin/bash
# Run TurboQuant evaluation on Qwen2.5-3B-Instruct

echo "Evaluating Qwen2.5-3B-Instruct..."

for bits in 2 3 4; do
    echo ""
    echo "Running with $bits bits..."
    python evaluate_model.py \
        --model Qwen/Qwen2.5-3B-Instruct \
        --bits $bits \
        --contexts 2048 4096 8192 \
        --output qwen/results_${bits}bit.json
done

echo ""
echo "✅ Qwen evaluation complete!"
