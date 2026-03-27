# TurboQuant Evaluation Methodology

## Overview

This document describes the experimental methodology used to evaluate TurboQuant across different language models.

## Evaluation Metrics

### 1. Compression Ratio
**Definition**: Uncompressed size / Compressed size

**Calculation**:
```
Keys: (bits-1) × d + 1 × d + 16 + 16 bits per vector
Values: bits × d + 16 bits per vector

FP16 baseline: 16 bits per value
```

**Interpretation**:
- 3.0x = 1 MB KV cache becomes 333 KB
- Higher is better (more compression)

### 2. Cosine Similarity
**Definition**: Similarity between real and estimated attention score distributions

**Calculation**:
```
cos(real_scores, estimated_scores) = <real, estimated> / (||real|| × ||estimated||)
```

**Interpretation**:
- 1.0 = Perfect match
- 0.99 = 99% similar distribution
- Used as primary quality metric (captures overall attention pattern)

### 3. Top-1 Match Percentage
**Definition**: % of attention heads where the most-attended token is the same

**Calculation**:
```
matches = count(argmax(real_scores) == argmax(estimated_scores))
pct = 100 × matches / n_heads
```

**Interpretation**:
- 100% = All heads attend to same token
- 80% = 4/5 heads correct
- Important for next-token prediction accuracy

### 4. Top-5 Match Percentage
**Definition**: % of heads where the real top-1 token appears in estimated top-5

**Calculation**:
```
matches = count(argmax(real) ∈ topk(estimated, k=5))
pct = 100 × matches / n_heads
```

**Interpretation**:
- More lenient than top-1
- 95% = Mostly correct rankings
- Shows if error is "close" or completely wrong

## Experimental Design

### Test Scenarios

1. **Synthetic Validation** (`experiments/1_paper_reproduction/`)
   - Random unit vectors
   - Verifies theoretical bounds
   - No model involved

2. **Single Forward Pass** (`experiments/2_multi_model_evaluation/`)
   - Load pre-trained model
   - One forward pass on long context
   - Capture full KV cache
   - Compress with TurboQuant
   - Compare attention scores

3. **Variable Context Lengths**
   - 2K, 4K, 8K tokens
   - Tests compression stability across lengths
   - Reveals any context-dependent effects

### Model Selection

**Criteria**:
- Open-source (reproducible)
- Varying sizes (3B to 13B)
- Different architectures (LLaMA, Phi, Mistral, Qwen)
- HuggingFace available

**Selected Models**:
1. **Qwen2.5-3B-Instruct** (3.5GB, smallest, easiest)
2. **Microsoft Phi-2** (2.7GB, highly optimized)
3. **Meta LLaMA-2-7B** (13GB, popular baseline)
4. **Mistral-7B-Instruct** (13GB, modern architecture)

### Quantization Config

- **Bit-widths**: 2, 3, 4 bits
- **Stage 1**: (bits-1) for MSE quantization
- **Stage 2**: 1 bit for QJL residual correction
- **Seed**: Fixed per layer for reproducibility

## Data Collection

### Per-Model Evaluation

For each model × bits configuration:

1. **Load Model**: HuggingFace with 4-bit quantization (model weights only)
2. **Prepare Input**:
   - Build long prompt with hidden "needle" fact
   - Filler text repeated to reach target length
   - Tokenize and pad to exact length
3. **Forward Pass**: Single pass, capture KV cache
4. **Compression**:
   - Create compressor for each layer
   - Compress keys (TurboQuantV2) and values (TurboQuantMSE)
5. **Scoring**:
   - Compute real attention scores: `Q @ K^T`
   - Compute estimated scores: asymmetric estimator
   - Compare using metrics above

### Results Storage

Each evaluation outputs:
```json
{
  "model": "Qwen/Qwen2.5-3B-Instruct",
  "bits": 3,
  "by_context": {
    "2048": {
      "seq_len": 2048,
      "compression_ratio": 5.0,
      "cosine_similarity": 0.9945,
      "top1_match_pct": 86.0,
      "top5_match_pct": 94.0
    }
  }
}
```

## Analysis & Visualization

### Aggregation

Results aggregated across:
- All layers (36-80 layers per model)
- All heads (8-96 heads per model)
- All contexts (2-8K tokens)

### Comparison Table

Shows average metrics across contexts for each model × bits combination.

### Plots

1. **Compression vs Bits**: Compression ratio curves
2. **Accuracy vs Bits**: Cosine similarity vs quantization bits
3. **Memory Usage**: Absolute memory savings across models

## Reproducibility

### Requirements

- Python 3.10+
- PyTorch 2.0+ with CUDA
- HuggingFace transformers, bitsandbytes
- 12GB+ GPU VRAM

### Command to Reproduce

```bash
python evaluate_model.py \
    --model Qwen/Qwen2.5-3B-Instruct \
    --bits 3 \
    --contexts 2048 4096 8192 \
    --output qwen/results_3bit.json
```

## Limitations

1. **Single Forward Pass**: Only evaluates one token's attention, not full generation
2. **Fixed Seeds**: Seeds fixed for reproducibility, may not represent all cases
3. **Long Context**: Limited to 8K tokens due to GPU memory
4. **Inference Latency**: Script measures compression, not end-to-end inference speedup
5. **Different Hardware**: Results depend on GPU (VRAM, compute capability)

## Future Improvements

- Full generative sampling (encode all tokens, measure quality)
- Multiple random seeds to estimate variance
- Longer contexts (16K+) with different quantization strategies
- Inference latency benchmarking
- Comparison with other compression methods (KIVI, PolarQuant)
