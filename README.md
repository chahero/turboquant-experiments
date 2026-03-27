# TurboQuant: KV Cache Compression Experiments

Comprehensive evaluation of **TurboQuant**, a near-optimal vector quantization algorithm for compressing LLM key-value caches.

This repository extends the [original implementation](https://github.com/tonbistudio/turboquant-pytorch) with:
- ✅ Multi-model evaluation (Qwen, LLaMA, Phi, Mistral)
- ✅ Comprehensive performance benchmarking
- ✅ Detailed experimental results and analysis
- ✅ Reproducible evaluation framework

## Quick Start

```bash
# Install dependencies
pip install torch transformers bitsandbytes accelerate scipy

# Run synthetic tests (no GPU needed)
cd experiments/1_paper_reproduction
python ../../original_implementation/test_turboquant.py

# Evaluate on a model (GPU required)
cd experiments/2_multi_model_evaluation
python evaluate_model.py --model Qwen/Qwen2.5-3B-Instruct --bits 3
```

## Key Results

| Model | Compression | Cosine Sim | Top-1 % | Top-5 % |
|-------|-------------|-----------|---------|---------|
| Qwen 3B (3-bit) | 5.0x | 0.9945 | 86% | 94% |
| Qwen 3B (4-bit) | 3.8x | 0.9983 | 92% | 96% |
| (More models evaluated in `RESULTS.md`) | ... | ... | ... | ... |

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
│   │   ├── evaluate_model.py             # Generic evaluation framework
│   │   ├── analyze_results.py            # Result analysis & plots
│   │   ├── run_*.sh                      # Model-specific scripts
│   │   └── {qwen,llama,phi}/             # Results per model
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
- **Microsoft Phi-2** (2.7GB) - Small, GPU-efficient
- **Meta LLaMA-2-7B** (13GB) - Popular baseline
- **Mistral-7B-Instruct** (13GB) - Modern architecture

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

# Output: Verify MSE bounds, inner product unbiasedness, needle-in-haystack
```

### 2. Model Evaluation (GPU Required)

Evaluate on different models:

```bash
cd experiments/2_multi_model_evaluation

# Single model, single bit-width
python evaluate_model.py --model Qwen/Qwen2.5-3B-Instruct --bits 3

# All models, all bit-widths (requires significant GPU time)
bash run_all_models.sh

# Analyze results
python analyze_results.py --dir .
```

### 3. Performance Benchmarking

Measure speed and memory:

```bash
cd experiments/3_performance_analysis
python benchmark_speed.py
```

## Results Summary

### Compression Effectiveness

**At 3-bit quantization**:
- 5.0x - 5.3x compression across all tested models
- Stable across context lengths (2K-8K tokens)
- Minimal sensitivity to model architecture

### Attention Accuracy

**Cosine Similarity** (primary metric):
- 3-bit: 0.994 - 0.996 (99.4% - 99.6% similar)
- 4-bit: 0.998 - 0.999 (99.8% - 99.9% similar)
- 2-bit: 0.985 - 0.990 (98.5% - 99.0% similar)

**Interpretation**: Even at 3-bit, attention distributions are 99.5% similar to FP16.

### Practical Implications

On a 12GB GPU with 3-bit TurboQuant:
- FP16 baseline: ~8K tokens max context
- TurboQuant 3-bit: ~40K tokens possible (5x improvement)

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

```
torch>=2.0
transformers>=4.35
bitsandbytes>=0.41
accelerate>=0.20
scipy>=1.10
matplotlib>=3.7
pandas>=2.0
```

Install all:
```bash
pip install -r original_implementation/requirements.txt
pip install transformers bitsandbytes accelerate scipy matplotlib pandas
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
