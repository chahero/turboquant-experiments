# How to Run TurboQuant Experiments

This guide explains how to run the TurboQuant evaluation experiments.

## Prerequisites

- Python 3.10+
- PyTorch 2.0+ with CUDA (for GPU acceleration)
- Minimum 12GB GPU VRAM (for larger models)
- HuggingFace account (for model access)

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd turboquant-experiments

# Install dependencies
pip install -r original_implementation/requirements.txt

# Install additional evaluation dependencies
pip install transformers bitsandbytes accelerate

# For CUDA support
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## Running Synthetic Tests

Before evaluating on real models, verify the core algorithm works:

```bash
cd experiments/1_paper_reproduction
python ../../original_implementation/test_turboquant.py
```

This tests:
- Lloyd-Max codebook properties
- MSE distortion bounds
- Inner product unbiasedness (QJL correction)
- Needle-in-haystack retrieval
- GPU performance (if CUDA available)

## Running Model Evaluations

### Single Model Test

```bash
cd experiments/2_multi_model_evaluation

# Test on Qwen2.5-3B-Instruct (smallest, ~3.5GB)
python evaluate_model.py \
    --model Qwen/Qwen2.5-3B-Instruct \
    --bits 3 \
    --contexts 2048 4096 \
    --output qwen/results_3bit.json
```

### Test on Different Models

```bash
# LLaMA-2 (requires HuggingFace token)
python evaluate_model.py \
    --model meta-llama/Llama-2-7b-hf \
    --bits 3 \
    --contexts 2048 4096 \
    --output llama/results_3bit.json

# Phi-2 (smaller, GPU-friendly)
python evaluate_model.py \
    --model microsoft/phi-2 \
    --bits 3 \
    --contexts 2048 4096 \
    --output phi/results_3bit.json

# Mistral
python evaluate_model.py \
    --model mistralai/Mistral-7B-Instruct-v0.1 \
    --bits 3 \
    --contexts 2048 4096 \
    --output mistral/results_3bit.json
```

### Batch Evaluation (All Models)

```bash
bash run_all_models.sh
```

This runs evaluations on all configured models with all bit-widths.

## Results Structure

Each evaluation generates a JSON file with:

```json
{
  "model": "model_name",
  "bits": 3,
  "timestamp": "2024-03-27T...",
  "config": {
    "n_layers": 32,
    "hidden_size": 3072,
    "num_heads": 24,
    "head_dim": 128
  },
  "by_context": {
    "2048": {
      "seq_len": 2048,
      "compression_ratio": 5.0,
      "compressed_mb": 58,
      "uncompressed_mb": 290,
      "cosine_similarity": 0.9945,
      "top1_match_pct": 86.0,
      "top5_match_pct": 94.0,
      "n_heads_checked": 72
    },
    ...
  }
}
```

## Analyzing Results

```bash
cd experiments/2_multi_model_evaluation

# Generate comparison table
python analysis.py  # (provided in results analysis script)
```

This creates:
- `comparison_table.csv`: Model-by-model results
- `plots/compression_by_bits.png`: Visualization
- `plots/accuracy_by_bits.png`: Accuracy comparison

## Performance Benchmarking

```bash
cd experiments/3_performance_analysis

# Measure encoding/decoding speed
python benchmark_speed.py

# Analyze memory efficiency
python benchmark_memory.py

# Full benchmark report
python run_benchmarks.sh
```

## Troubleshooting

### Out of Memory (OOM)

Reduce context lengths:
```bash
python evaluate_model.py --model ... --contexts 1024 2048
```

Use smaller models (Phi-2, Qwen-3B instead of Llama-7B).

### Model Download Issues

Ensure HuggingFace token is set:
```bash
huggingface-cli login
```

### CUDA Not Available

Some operations fall back to CPU (slower). Install CUDA 11.8+:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## Configuration

Edit `evaluate_model.py` to customize:
- Default bit-widths
- Context lengths
- Prompt templates
- Comparison metrics

## Citation

If you use this evaluation framework, please cite the original paper:

```bibtex
@article{turboquant2026,
  title={TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate},
  year={2026},
  journal={ICLR},
  url={https://arxiv.org/abs/2504.19874}
}
```
