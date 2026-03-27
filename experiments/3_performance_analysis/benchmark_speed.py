"""
Benchmark TurboQuant encoding/decoding speed.
"""

import torch
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "original_implementation"))

from compressors import TurboQuantCompressorV2, TurboQuantCompressorMSE


def benchmark_compression_speed(dimensions=(64, 128, 256), bits=(2, 3, 4),
                                 seq_lengths=(512, 2048, 8192), num_trials=10):
    """
    Benchmark TurboQuant compression speed.

    Measures:
    - Encoding time (vector -> compressed)
    - Decoding time (compressed -> vector)
    - Throughput (vectors/sec)
    """

    print("=" * 80)
    print("TurboQuant Speed Benchmark")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU (will be slow)")
        device = "cpu"
    else:
        device = "cuda"
        print(f"Device: {torch.cuda.get_device_name()}\n")

    results = {}

    for dim in dimensions:
        print(f"\n{'─'*80}")
        print(f"Dimension: {dim}")
        print(f"{'─'*80}")

        for b in bits:
            print(f"\n  Bits: {b}")

            # Create compressor
            comp = TurboQuantCompressorV2(dim, b, seed=42, device=device)

            for seq_len in seq_lengths:
                # Generate random data
                data = torch.randn(1, 1, seq_len, dim, device=device)

                # Warmup
                _ = comp.compress(data)
                torch.cuda.synchronize() if device == "cuda" else None

                # Benchmark compression
                times = []
                for _ in range(num_trials):
                    torch.cuda.synchronize() if device == "cuda" else None
                    start = time.time()
                    compressed = comp.compress(data)
                    torch.cuda.synchronize() if device == "cuda" else None
                    times.append(time.time() - start)

                avg_time = sum(times) / len(times)
                throughput = (seq_len / avg_time) if avg_time > 0 else 0

                key = f"d{dim}_b{b}_s{seq_len}"
                results[key] = {
                    "dim": dim,
                    "bits": b,
                    "seq_len": seq_len,
                    "time_ms": avg_time * 1000,
                    "throughput_vec_per_sec": throughput,
                }

                print(f"    seq_len={seq_len:>5d}: {avg_time*1000:>6.2f}ms, "
                      f"{throughput:>8.0f} vecs/sec")

    return results


def benchmark_accuracy_vs_speed(bit_widths=(2, 3, 4), n_vectors=1000):
    """
    Trade-off analysis: compression ratio vs speed vs accuracy.
    """

    print("\n" + "="*80)
    print("Speed vs Compression Trade-off")
    print("="*80)

    if not torch.cuda.is_available():
        print("⚠️  CUDA not available")
        device = "cpu"
    else:
        device = "cuda"

    d = 128

    for bits in bit_widths:
        print(f"\n  {bits}-bit quantization:")

        # Setup
        comp = TurboQuantCompressorV2(d, bits, seed=42, device=device)
        data = torch.randn(n_vectors, 1, 1, d, device=device)

        # Time
        torch.cuda.synchronize() if device == "cuda" else None
        start = time.time()
        for _ in range(10):
            comp.compress(data)
        torch.cuda.synchronize() if device == "cuda" else None
        time_taken = (time.time() - start) / 10

        # Compression ratio
        mse_bits = max(bits - 1, 1)
        compressed_bits = n_vectors * d * (mse_bits + 1 + 0.25)  # MSE + QJL + norm overhead
        uncompressed_bits = n_vectors * d * 32  # FP32
        ratio = uncompressed_bits / compressed_bits

        print(f"    Time: {time_taken*1000:.2f}ms for {n_vectors} vectors")
        print(f"    Compression: {ratio:.2f}x")


if __name__ == "__main__":
    # Run benchmarks
    speed_results = benchmark_compression_speed()
    benchmark_accuracy_vs_speed()

    print("\n" + "="*80)
    print("Benchmark complete!")
    print("="*80)
