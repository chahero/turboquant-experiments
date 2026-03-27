# Original Implementation Attribution

## Source

This directory contains the original PyTorch implementation of TurboQuant from:

**Repository**: [tonbistudio/turboquant-pytorch](https://github.com/tonbistudio/turboquant-pytorch)

**License**: MIT

## Reference Paper

> **TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate**
>
> Published at ICLR 2026
>
> ArXiv: https://arxiv.org/abs/2504.19874

## Description

This original implementation includes:
- `lloyd_max.py`: Lloyd-Max optimal scalar quantizer solver for Beta distributions
- `turboquant.py`: Core TurboQuant algorithm (Stage 1 MSE + Stage 2 QJL)
- `compressors.py`: Production-oriented compressors with asymmetric attention
- `test_turboquant.py`: Synthetic algorithm validation tests
- `validate.py`: Real model validation on Qwen2.5-3B-Instruct
- `requirements.txt`: Dependencies

## Citation

If you use this implementation, please cite:

```bibtex
@article{turboquant2026,
  title={TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate},
  year={2026},
  journal={ICLR},
  url={https://arxiv.org/abs/2504.19874}
}
```

## Our Modifications

The main repository extends this original implementation with:
- Multi-model evaluation (Qwen, LLaMA, Phi, Mistral)
- Performance benchmarking and analysis
- Comprehensive experimental results
- Bug fixes and code improvements
- Detailed documentation and reproducibility guides

See the parent README.md for details on our contributions.
