# TurboQuant Experimental Results - Visualization Charts

This directory contains visualization charts generated from the TurboQuant multi-model evaluation experiments.

All charts compare three models:
- **Qwen2.5-3B** (3.5GB, 72 attention heads)
- **Phi-2** (2.7GB, 1024 attention heads)
- **Mistral-7B** (13GB, 256 heads with GQA)

Evaluated across:
- **Context lengths**: 2K, 4K, 8K tokens
- **Quantization bitwidths**: 2-bit, 3-bit, 4-bit

---

## Chart Descriptions

### 1. Compression Comparison
**File**: `01_compression_comparison.png`

Bar chart showing compression ratios at 8K context across all quantization bitwidths.

**Key Insight**: All models achieve 5.0x-7.3x compression ratios, with 3-bit offering the best balance.

---

### 2. Cosine Similarity by Context Length
**File**: `02_cosine_similarity_context.png`

Three line plots (one per model) showing how cosine similarity to original FP16 attention distributions varies with context length for each quantization bitwidth.

**Key Insight**:
- Qwen and Mistral maintain >0.98 cosine similarity across all contexts
- Phi-2 maintains high cosine similarity despite accuracy degradation
- Cosine similarity is more stable than top-1 accuracy with longer contexts

---

### 3. Top-1 Accuracy Comparison
**File**: `03_top1_accuracy.png`

Bar chart comparing the percentage of attention heads that select the same most-attended token as the original model (3-bit @ 8K context).

**Key Insight**:
- Mistral: 97.7% (excellent consistency)
- Qwen: 86.1% (good performance)
- Phi-2: 28.2% (significant degradation due to 1024 heads)

---

### 4. Context Sensitivity Heatmap
**File**: `04_context_sensitivity_heatmap.png`

Three heatmaps (one per model) showing top-1 match accuracy across all context length and bitwidth combinations.

**Key Insights**:
- **Mistral**: Stable ~97% across all settings
- **Qwen**: Slight degradation at 4K context, recovers at 8K
- **Phi-2**: Sharp degradation as context increases, especially at lower bitwidths

---

### 5. Model Comparison Radar Chart
**File**: `05_model_comparison_radar.png`

Radar chart comparing all models across four dimensions (3-bit @ 8K context):
- Compression ratio (relative to 5.0x reference)
- Cosine similarity (normalized to 0-100)
- Top-1 accuracy
- Top-5 accuracy

**Key Insight**: Mistral excels in accuracy metrics, Qwen in compression and similarity, Phi-2 shows weaker overall performance.

---

### 6. Compression-Accuracy Tradeoff
**File**: `06_compression_accuracy_tradeoff.png`

Scatter plot showing the relationship between compression ratio and top-1 match accuracy at 8K context.

**Key Insight**:
- Higher compression (left side) sacrifices accuracy
- 3-bit offers the best tradeoff for most models
- Clear progression: 2-bit (highest compression, lowest accuracy) → 3-bit → 4-bit (highest accuracy)

---

### 7. Summary Table
**File**: `07_summary_table.png`

Summary table of key metrics at 3-bit quantization @ 8K context.

**Columns**:
- Compression ratio
- Cosine similarity
- Top-1 match accuracy
- Top-5 match accuracy

---

## How Charts Were Generated

All charts were generated using `generate_charts.py` with:
- **matplotlib** for general plotting
- **seaborn** for statistical visualizations
- **numpy** for data processing

To regenerate charts:
```bash
cd experiments/2_multi_model_evaluation
python generate_charts.py
```

---

## Interpreting the Metrics

### Compression Ratio
- **Definition**: Original KV cache size / Compressed KV cache size
- **Example**: 5.0x means KV cache is 20% of original size
- **Higher is better**: More compression = less memory/computation

### Cosine Similarity
- **Definition**: Cosine distance between original and compressed attention distributions
- **Range**: 0 to 1 (1.0 = identical)
- **Why it matters**: Measures how similar the attention patterns remain
- **Example**: 0.9945 means 99.45% similarity

### Top-1 Match %
- **Definition**: Percentage of attention heads selecting the same most-attended token
- **Range**: 0% to 100%
- **Why it matters**: Direct measure of attention accuracy
- **Example**: 97.7% means 250/256 heads select the same top token

### Top-5 Match %
- **Definition**: Percentage of attention heads where the true top token is in the top-5 predictions
- **Range**: 0% to 100%
- **Why it matters**: More lenient metric; captures "close enough" predictions

---

## Key Findings Summary

### By Model

**Mistral-7B** ⭐ Best Overall
- Consistent 97%+ top-1 accuracy across all contexts
- GQA architecture makes it more quantization-friendly
- Recommended for long-context applications

**Qwen2.5-3B**
- Highest cosine similarity (0.9945)
- Good compression-quality balance
- Ideal for similarity-critical tasks
- Stable across contexts

**Phi-2**
- Smallest model (2.7GB)
- High cosine similarity (0.9924) despite accuracy issues
- Sharp context length degradation
- Better for short-context applications only

### By Quantization Bitwidth

**2-bit**: Highest compression (7.3x), lowest accuracy
**3-bit**: Best balance (5.0x compression, ~85-97% top-1)
**4-bit**: Best accuracy (3.8x compression, ~90-99% top-1)

### Practical Implications

On a 12GB GPU:
- **FP16 baseline**: ~8K token context max
- **TurboQuant 3-bit**: ~40K token context possible
- **Context improvement**: 5x increase with minimal quality loss

---

For detailed analysis, see [../RESULTS.md](../RESULTS.md)
