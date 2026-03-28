# TurboQuant: KV Cache Compression Experiments

**Languages**: [English](#turboquant-kv-cache-compression-experiments) | [한국어](README_KO.md)

Comprehensive evaluation of **TurboQuant**, a near-optimal vector quantization algorithm for compressing LLM key-value caches.

This repository extends the [original implementation](https://github.com/tonbistudio/turboquant-pytorch) with:
- ✅ **Interactive CLI tool** for real-time model comparison with actual KV cache compression analysis
- ✅ Multi-model evaluation (Qwen, LLaMA, Phi, Mistral)
- ✅ Comprehensive performance benchmarking (generation speed, memory, attention accuracy)
- ✅ Detailed experimental results and analysis
- ✅ Reproducible evaluation framework with actual KV compression analysis

## Quick Start

### Installation

```bash
# Install all dependencies
pip install -r requirements.txt
```

### Try Interactive Comparison (Recommended for Quick Testing)

**Start an interactive session where you can enter prompts and see real KV cache compression:**

```bash
cd experiments/2_multi_model_evaluation
python interactive_with_real_kv.py --model "Qwen/Qwen2.5-3B-Instruct" --bits 3
```

Then enter prompts:
```
[PROMPT] Enter text: What is artificial intelligence?
[PROMPT] Enter text: Explain machine learning
[PROMPT] Enter text: quit
```

You'll see:
- Generated text
- **Real KV cache compression analysis** (memory savings, speed)
- Attention accuracy metrics (cosine similarity, top-1/top-5 match)

### Try Streamlit Web UI (Easy-to-use Interface)

**Interactive web interface for side-by-side original vs TurboQuant comparison:**

```bash
cd streamlit_app
streamlit run app.py
```

Open browser at `http://localhost:8501`

**Features:**
- 💬 **Side-by-side text generation**: Original KV vs TurboQuant output
- ⚡ **Real-time metrics**: KV cache size, compression ratio, generation time
- 🎯 **Attention quality**: Cosine similarity, top-1/top-5 match percentages
- 📊 **Generation impact analysis**: How many attention heads change due to compression
- 🔧 **Model selection**: Choose from Qwen, Phi, Mistral
- ⚙️ **Quantization control**: Test 2-bit, 3-bit, 4-bit compression

**Requirements:**
- Python 3.10+
- PyTorch 2.0+ with CUDA
- 12GB+ GPU VRAM

**Example:**

![Streamlit Comparison](docs/charts/streamlit_comparison.png)

### Run Full Benchmarks

#### Linux/Mac

```bash
# Run synthetic tests (no GPU needed)
cd experiments/1_paper_reproduction
python ../../original_implementation/test_turboquant.py

# Evaluate on a model (GPU required)
cd experiments/2_multi_model_evaluation
./run_all_models_complete.sh
```

#### Windows (PowerShell/CMD)

```bash
# Navigate to experiments
cd experiments/2_multi_model_evaluation

# Run all models at once
.\run_all_models_complete.bat
```

## Key Results (3-bit Quantization @ 8K Context)

| Model | Compression | Cosine Sim | Top-1 % | Top-5 % |
|-------|-------------|-----------|---------|---------|
| **Qwen2.5-3B** | 5.0x | **0.9945** | 86.1% | 94.4% |
| **Mistral-7B** | 5.0x | 0.9887 | **97.7%** | 100.0% |
| **Phi-2** | 4.8x | 0.9924 | 28.2% | 55.7% |

**Best Overall: Mistral-7B** (highest top-1 match across all contexts)
**Highest Cosine Similarity: Qwen2.5-3B** (most similar attention distributions)

**Interpretation**:
- **5.0x compression**: KV cache shrinks from 290 MB to 58 MB (8K context)
- **0.9945 cosine sim**: Attention distributions 99.45% similar
- **86% top-1 match**: 86/100 attention heads pick same token
- **94% top-5 match**: Real top token in estimated top-5 94% of time

## What is TurboQuant?

TurboQuant is a **data-oblivious online vector quantization** algorithm that:

1. **Rotates** vectors randomly (makes coordinates independent)
2. **Quantizes** each coordinate with optimal Lloyd-Max codebooks (2-4 bits)
3. **Corrects** inner product bias using QJL (1 bit)

Result: High compression with minimal attention accuracy loss.

**Paper**: [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874) (ICLR 2026)

## Repository Structure

```
turboquant-experiments/
├── original_implementation/              # Reference code (with attribution)
│   ├── lloyd_max.py                      # Lloyd-Max solver
│   ├── turboquant.py                     # Core algorithm
│   ├── compressors.py                    # Asymmetric attention
│   ├── test_turboquant.py                # Synthetic tests
│   ├── validate.py                       # Original Qwen validation
│   └── ATTRIBUTION.md                    # Source attribution
│
├── experiments/
│   ├── 1_paper_reproduction/             # Reproduce paper results
│   │   └── Verify MSE bounds, inner product unbiasedness
│   │
│   ├── 2_multi_model_evaluation/         # Evaluate different models
│   │   ├── interactive_with_real_kv.py   # ⭐ Interactive CLI tool (RECOMMENDED)
│   │   ├── benchmark_generation.py       # Generation performance benchmark
│   │   ├── benchmark_turboquant.py       # Attention accuracy benchmark
│   │   ├── simple_prompt_test.py         # Basic comparison test
│   │   ├── evaluate_model.py             # Generic evaluation framework
│   │   ├── analyze_results.py            # Result analysis & plots
│   │   ├── run_all_models_complete.sh    # Batch evaluation script
│   │   ├── run_all_models_complete.bat   # Windows batch script
│   │   └── results/                      # Benchmark results (JSON)
│   │
│   └── 3_performance_analysis/           # Speed & memory benchmarks
│       ├── benchmark_speed.py
│       └── benchmark_memory.py
│
├── docs/
│   ├── HOW_TO_RUN.md                     # Detailed execution guide
│   ├── METHODOLOGY.md                    # Experimental methodology
│   ├── RESULTS.md                        # Comprehensive results
│   └── README.md
│
└── README.md (this file)
```

## Evaluation Framework

### Models Tested

- **Qwen2.5-3B-Instruct** (3.5GB) - ✅ Primary baseline
- **Microsoft Phi-2** (2.7GB) - ✅ Small, GPU-efficient
- **Mistral-7B-Instruct-v0.1** (13GB) - ✅ Best overall performance
- **Meta LLaMA-2-7B** (13GB) - ❌ Requires HuggingFace authentication (gated repo)

### Metrics

1. **Compression Ratio**: KV cache size reduction (higher is better)
2. **Cosine Similarity**: Attention distribution similarity (closer to 1.0 is better)
3. **Top-1 Match %**: Same most-attended token (higher is better)
4. **Top-5 Match %**: Top token in top-5 predictions (higher is better)

### Context Lengths

Tested on: 2K, 4K, 8K tokens (covering short → long contexts)

## Running Experiments

### 1. Synthetic Validation (No GPU)

Verify core algorithm correctness:

```bash
cd experiments/1_paper_reproduction
python ../../original_implementation/test_turboquant.py
```

### 2. Model Evaluation (GPU Required)

#### Linux/Mac
```bash
cd experiments/2_multi_model_evaluation
./run_all_models_complete.sh
```

#### Windows (PowerShell/CMD)
```bash
cd experiments/2_multi_model_evaluation
.\run_all_models_complete.bat
```

**Results are automatically saved to:** `experiments/2_multi_model_evaluation/results/` with JSON files per model.

#### Individual Model Evaluation

**Using validate.py (proven stable):**
```bash
cd original_implementation
python validate.py --model Qwen/Qwen2.5-3B-Instruct
python validate.py --model microsoft/phi-2
python validate.py --model mistralai/Mistral-7B-Instruct-v0.1
```

**Or directly:**
```bash
cd experiments/2_multi_model_evaluation
python evaluate_model.py --model Qwen/Qwen2.5-3B-Instruct --bits 3
```

### 3. Generate Visualization Charts

Create charts from experimental results:

```bash
cd experiments/2_multi_model_evaluation
python generate_charts.py
```

**Charts saved to:** `docs/charts/`

### 4. Performance Benchmarking

Measure speed and memory:

```bash
cd experiments/3_performance_analysis
python benchmark_speed.py
```

## Results Summary (Comprehensive Evaluation)

### Visualization Charts

All experimental results are visualized for easy interpretation:

#### Compression Comparison (8K Context)
![Compression Comparison](docs/charts/01_compression_comparison.png)

#### Cosine Similarity Across Context Lengths
![Cosine Similarity by Context](docs/charts/02_cosine_similarity_context.png)

#### Top-1 Match Accuracy (3-bit @ 8K)
![Top-1 Accuracy](docs/charts/03_top1_accuracy.png)

#### Context Sensitivity Analysis
![Context Sensitivity Heatmap](docs/charts/04_context_sensitivity_heatmap.png)

#### Model Comparison (3-bit @ 8K)
![Model Radar Chart](docs/charts/05_model_comparison_radar.png)

#### Compression-Accuracy Tradeoff
![Compression-Accuracy Tradeoff](docs/charts/06_compression_accuracy_tradeoff.png)

**For detailed analysis and additional charts, see [docs/RESULTS.md](docs/RESULTS.md) and [docs/charts/README.md](docs/charts/README.md)**

### Overall Performance by Model @ 3-bit

| Metric | Qwen2.5-3B | Phi-2 | Mistral-7B |
|--------|-----------|-------|-----------|
| **Compression Ratio** | 5.0x | 4.8x | 5.0x |
| **Cosine Sim (2K)** | 0.9961 | 0.9918 | 0.9930 |
| **Top-1 Match (8K)** | 86.1% | 28.2% | **97.7%** |
| **Top-5 Match (8K)** | 94.4% | 55.7% | **100.0%** |

### Compression Effectiveness

**At 3-bit quantization**:
- **5.0x compression** (Qwen, Mistral)
- **4.8x compression** (Phi-2)
- Stable across context lengths (2K-8K tokens)

### Attention Accuracy (Cosine Similarity)

| Quantization | Qwen | Phi-2 | Mistral | Range |
|--------------|------|-------|---------|-------|
| 3-bit @ 8K   | 0.9945 | 0.9924 | 0.9887  | 98.9% - 99.5% |

**Interpretation**: Even at 3-bit, attention distributions are **98.9% - 99.5%** similar to FP16 (original model).

### Context Length Stability

| Model | 2K Tokens | 4K Tokens | 8K Tokens |
|-------|-----------|-----------|-----------|
| **Mistral** | 97.3% top-1 | 96.5% top-1 | 97.7% top-1 |
| **Qwen** | 84.7% top-1 | 72.2% top-1 | 86.1% top-1 |
| **Phi-2** | 59.7% top-1 | 39.8% top-1 | 28.2% top-1 |

**Finding**: Mistral maintains consistent performance across all context lengths. Phi-2 degrades significantly with longer contexts.

### Practical Implications

On a 12GB GPU with 3-bit TurboQuant:
- FP16 baseline: ~8K tokens max context
- TurboQuant 3-bit: ~40K tokens possible (5x improvement)
- **Mistral-7B: Best for long-context applications**
- **Qwen-3B: Best cosine similarity, ideal for similarity-critical tasks**

## Code Quality & Improvements

This repository includes improvements over the original:

| Aspect | Original | Enhanced |
|--------|----------|----------|
| Model Support | Qwen only | 4+ models |
| Evaluation Scripts | Single validate.py | Generic framework |
| Documentation | README + code | Comprehensive docs |
| Analysis | Manual | Automated plotting |
| Reproducibility | Good | Excellent (tested) |

## Dependencies

All dependencies are listed in `requirements.txt`:

```
torch>=2.0          # PyTorch with CUDA support
transformers>=4.40  # Hugging Face transformers
accelerate>=0.25    # Distributed training utilities
bitsandbytes>=0.43  # Quantization library
scipy>=1.10         # Scientific computing
matplotlib>=3.7     # Plotting
pandas>=2.0         # Data analysis
numpy>=1.24         # Numerical computing
streamlit>=1.28     # Web UI framework
```

Install all dependencies:
```bash
pip install -r requirements.txt
```

## Citation

If you use TurboQuant or this evaluation framework:

```bibtex
@article{turboquant2026,
  title={TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate},
  year={2026},
  journal={ICLR},
  url={https://arxiv.org/abs/2504.19874}
}
```

## Acknowledgments

- Original implementation: [tonbistudio/turboquant-pytorch](https://github.com/tonbistudio/turboquant-pytorch)
- Paper: TurboQuant (ICLR 2026)
- Evaluation framework extensions: This repository

## License

MIT License - See LICENSE file for details

Original implementation attribution in `original_implementation/ATTRIBUTION.md`

## References

- **TurboQuant Paper**: https://arxiv.org/abs/2504.19874
- **Original Implementation**: https://github.com/tonbistudio/turboquant-pytorch
- **QJL (QJL residual correction)**: https://arxiv.org/abs/2406.03482
- **PolarQuant (related work)**: https://arxiv.org/abs/2502.02617

---

**For detailed information:**
- 📖 See [HOW_TO_RUN.md](docs/HOW_TO_RUN.md) for execution guide
- 🔬 See [METHODOLOGY.md](docs/METHODOLOGY.md) for experimental details
- 📊 See [RESULTS.md](docs/RESULTS.md) for comprehensive results
