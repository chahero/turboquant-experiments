#!/bin/bash
# Run TurboQuant validation on ALL models (Linux/Mac version)
# Results are saved to results/ directory

set -e  # Exit on error

echo ""
echo "======================================================================"
echo "TurboQuant: Multi-Model Comprehensive Evaluation"
echo "======================================================================"
echo ""

# Create results directory
mkdir -p results

# Navigate to original_implementation
cd ../../original_implementation

# Models to evaluate (LLaMA requires HuggingFace authentication)
MODELS=(
    "Qwen/Qwen2.5-3B-Instruct"
    "microsoft/phi-2"
    "mistralai/Mistral-7B-Instruct-v0.1"
)

RESULTS_DIR="../experiments/2_multi_model_evaluation/results"

# Evaluate each model
for model in "${MODELS[@]}"; do
    echo ""
    echo "================================================"
    echo "Model: $model"
    echo "================================================"
    echo ""

    # Extract model name for file (replace / with _)
    model_file=$(echo "$model" | tr '/' '_')

    # Run validation and save output
    python validate.py --model "$model" | tee "$RESULTS_DIR/${model_file}_results.txt"

    echo ""
    echo "[COMPLETED] $model"
    echo "Results saved to: $RESULTS_DIR/${model_file}_results.txt"
    echo ""
done

echo ""
echo "======================================================================"
echo "All model evaluations complete!"
echo "======================================================================"
echo "Results saved to: $RESULTS_DIR/"
echo ""
echo "File listing:"
ls -lh "$RESULTS_DIR/"
echo ""

# Navigate back
cd ../../experiments/2_multi_model_evaluation

echo "Done! Press enter to exit."
read -p ""
