"""
Analyze and visualize TurboQuant evaluation results.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


def load_results(results_dir: str = "."):
    """Load all JSON result files from a directory."""
    results_dir = Path(results_dir)
    all_results = {}

    for json_file in results_dir.rglob("results_*.json"):
        with open(json_file, 'r') as f:
            data = json.load(f)
            model = data['model'].split('/')[-1].lower()
            bits = data['bits']
            key = f"{model}_{bits}bit"
            all_results[key] = data

    return all_results


def create_comparison_table(results_dir: str = "."):
    """Create a comprehensive comparison table."""

    results = load_results(results_dir)

    if not results:
        print("No results found!")
        return None

    # Organize by model and bits
    table_data = []

    for key, result in results.items():
        model = result['model'].split('/')[-1]
        bits = result['bits']

        # Average across all contexts
        contexts = result['by_context']
        if not contexts:
            continue

        avg_compression = np.mean([c['compression_ratio'] for c in contexts.values()])
        avg_cosine = np.mean([c['cosine_similarity'] for c in contexts.values()])
        avg_top1 = np.mean([c['top1_match_pct'] for c in contexts.values()])
        avg_top5 = np.mean([c['top5_match_pct'] for c in contexts.values()])

        table_data.append({
            'Model': model,
            'Bits': bits,
            'Compression Ratio': f"{avg_compression:.2f}x",
            'Cosine Similarity': f"{avg_cosine:.6f}",
            'Top-1 Match %': f"{avg_top1:.1f}%",
            'Top-5 Match %': f"{avg_top5:.1f}%",
        })

    df = pd.DataFrame(table_data)

    print("\n" + "="*100)
    print("COMPARISON TABLE: TurboQuant Performance Across Models")
    print("="*100)
    print(df.to_string(index=False))
    print("="*100)

    # Save as CSV
    output_csv = Path(results_dir) / "comparison_table.csv"
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Saved to {output_csv}")

    return df


def plot_compression_by_bits(results_dir: str = "."):
    """Plot compression ratio vs bit-width."""

    results = load_results(results_dir)

    if not results:
        print("No results found!")
        return

    # Organize data
    models = {}
    for key, result in results.items():
        model = result['model'].split('/')[-1]
        bits = result['bits']

        contexts = result['by_context']
        avg_compression = np.mean([c['compression_ratio'] for c in contexts.values()])

        if model not in models:
            models[model] = {}
        models[model][bits] = avg_compression

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    for model, compression_by_bits in models.items():
        bits = sorted(compression_by_bits.keys())
        ratios = [compression_by_bits[b] for b in bits]
        ax.plot(bits, ratios, marker='o', linewidth=2, label=model)

    ax.set_xlabel('Bit-width', fontsize=12)
    ax.set_ylabel('Compression Ratio', fontsize=12)
    ax.set_title('TurboQuant: Compression Ratio vs Bit-width', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_xticks([2, 3, 4])

    output_file = Path(results_dir) / "plots" / "compression_by_bits.png"
    output_file.parent.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"✅ Saved plot to {output_file}")
    plt.close()


def plot_accuracy_by_bits(results_dir: str = "."):
    """Plot accuracy (cosine similarity) vs bit-width."""

    results = load_results(results_dir)

    if not results:
        return

    # Organize data
    models = {}
    for key, result in results.items():
        model = result['model'].split('/')[-1]
        bits = result['bits']

        contexts = result['by_context']
        avg_cosine = np.mean([c['cosine_similarity'] for c in contexts.values()])

        if model not in models:
            models[model] = {}
        models[model][bits] = avg_cosine

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    for model, cosine_by_bits in models.items():
        bits = sorted(cosine_by_bits.keys())
        similarities = [cosine_by_bits[b] for b in bits]
        ax.plot(bits, similarities, marker='s', linewidth=2, label=model)

    ax.set_xlabel('Bit-width', fontsize=12)
    ax.set_ylabel('Cosine Similarity', fontsize=12)
    ax.set_title('TurboQuant: Attention Score Accuracy vs Bit-width', fontsize=14, fontweight='bold')
    ax.set_ylim([0.98, 1.0])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_xticks([2, 3, 4])

    output_file = Path(results_dir) / "plots" / "accuracy_by_bits.png"
    output_file.parent.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"✅ Saved plot to {output_file}")
    plt.close()


def plot_memory_efficiency(results_dir: str = "."):
    """Plot memory usage reduction."""

    results = load_results(results_dir)

    if not results:
        return

    # Organize data
    models = {}
    for key, result in results.items():
        model = result['model'].split('/')[-1]
        bits = result['bits']

        contexts = result['by_context']
        # Use first context length
        if contexts:
            first_context = list(contexts.values())[0]
            uncompressed = first_context['uncompressed_mb']
            compressed = first_context['compressed_mb']

            if model not in models:
                models[model] = {}
            models[model][bits] = (uncompressed, compressed)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(models))
    width = 0.2
    bits_list = [2, 3, 4]

    for i, bits in enumerate(bits_list):
        compressed_sizes = []
        for model in models.keys():
            if bits in models[model]:
                _, compressed = models[model][bits]
                compressed_sizes.append(compressed)
            else:
                compressed_sizes.append(0)

        ax.bar(x + i*width, compressed_sizes, width, label=f'{bits}-bit')

    # Add uncompressed baseline
    uncompressed_baseline = []
    for model in models.keys():
        if list(models[model].values()):
            uncompressed, _ = list(models[model].values())[0]
            uncompressed_baseline.append(uncompressed)
        else:
            uncompressed_baseline.append(0)

    ax.axhline(y=np.mean(uncompressed_baseline), color='r', linestyle='--',
               linewidth=2, label='FP16 (uncompressed)')

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('KV Cache Memory (MB)', fontsize=12)
    ax.set_title('TurboQuant: Memory Usage Reduction', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(models.keys())
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    output_file = Path(results_dir) / "plots" / "memory_by_model.png"
    output_file.parent.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"✅ Saved plot to {output_file}")
    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze TurboQuant results")
    parser.add_argument("--dir", type=str, default=".", help="Results directory")

    args = parser.parse_args()

    print("Analyzing results...")
    create_comparison_table(args.dir)
    print("\nGenerating plots...")
    plot_compression_by_bits(args.dir)
    plot_accuracy_by_bits(args.dir)
    plot_memory_efficiency(args.dir)

    print("\n✅ Analysis complete!")
