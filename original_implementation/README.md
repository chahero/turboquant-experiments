# Original TurboQuant Implementation

This directory contains the original PyTorch implementation of TurboQuant from [tonbistudio/turboquant-pytorch](https://github.com/tonbistudio/turboquant-pytorch) under MIT License.

See `ATTRIBUTION.md` for full attribution and citation information.

## Files Overview

| File | Purpose |
|------|---------|
| `lloyd_max.py` | Lloyd-Max optimal scalar quantizer for Beta distribution coordinates |
| `turboquant.py` | Core TurboQuant algorithm: Stage 1 (MSE) + Stage 2 (QJL inner product correction) |
| `compressors.py` | Production-ready compressors with asymmetric attention score computation |
| `test_turboquant.py` | Comprehensive tests: MSE bounds, inner product unbiasedness, needle-in-haystack |
| `validate.py` | Real model validation on Qwen2.5-3B-Instruct with attention score accuracy analysis |
| `__init__.py` | Package exports |
| `requirements.txt` | Dependencies |
| `README_ORIGINAL.md` | Original repository README |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run synthetic tests
python test_turboquant.py

# Validate on real model (requires GPU)
python validate.py
```

## Key Classes

- **`TurboQuantMSE`**: Stage 1 - Random rotation + Lloyd-Max quantization
- **`TurboQuantProd`**: Stage 1 + Stage 2 - Adds QJL correction for unbiased inner products
- **`TurboQuantCompressorV2`**: Asymmetric attention with direct score computation
- **`LloydMaxCodebook`**: Precomputed optimal quantizer codebooks

## Paper Reference

> TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate (ICLR 2026)
> https://arxiv.org/abs/2504.19874
