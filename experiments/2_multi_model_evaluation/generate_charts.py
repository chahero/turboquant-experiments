#!/usr/bin/env python3
"""
Generate visualization charts for TurboQuant experimental results.
Creates bar charts, line plots, and comparison visualizations.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

# Results directory
RESULTS_DIR = Path(__file__).parent / "results"
CHARTS_DIR = Path(__file__).parent.parent.parent / "docs" / "charts"
CHARTS_DIR.mkdir(exist_ok=True)

# Color scheme
COLORS = {
    "Qwen2.5-3B": "#FF6B6B",
    "Phi-2": "#4ECDC4",
    "Mistral-7B": "#45B7D1",
}

CONTEXT_COLORS = {
    "2K": "#FFA07A",
    "4K": "#FFD700",
    "8K": "#98D8C8",
}

def parse_results(model_name: str, filename: str) -> Dict:
    """Parse results from text file."""
    filepath = RESULTS_DIR / filename
    if not filepath.exists():
        print(f"Warning: {filepath} not found")
        return {}

    with open(filepath, 'r') as f:
        content = f.read()

    results = {
        "model": model_name,
        "contexts": {}
    }

    # Extract context sections
    context_pattern = r"Context:\s*([\d,]+)\s*tokens.*?\n={20,}(.*?)(?=={20,}|$)"
    matches = re.finditer(context_pattern, content, re.DOTALL)

    for match in matches:
        context_tokens = match.group(1).replace(',', '')
        section = match.group(2)

        # Determine context label
        ctx_int = int(context_tokens)
        if ctx_int <= 2500:
            ctx_label = "2K"
        elif ctx_int <= 4500:
            ctx_label = "4K"
        else:
            ctx_label = "8K"

        results["contexts"][ctx_label] = {}

        # Extract TQ-2bit, TQ-3bit, TQ-4bit metrics
        for bitwidth in ["2bit", "3bit", "4bit"]:
            pattern = rf"TQ-{bitwidth}:.*?(?=TQ-|\Z)"
            bit_match = re.search(pattern, section, re.DOTALL)

            if bit_match:
                bit_section = bit_match.group(0)

                # Extract metrics
                compression = extract_float(bit_section, r"Compression:\s*([\d.]+)x")
                cosine_sim = extract_float(bit_section, r"Score cosine sim:\s*([\d.]+)")
                top1 = extract_float(bit_section, r"Top-1 match:\s*([\d.]+)%")
                top5 = extract_float(bit_section, r"Top-5 match:\s*([\d.]+)%")

                results["contexts"][ctx_label][bitwidth] = {
                    "compression": compression,
                    "cosine_sim": cosine_sim,
                    "top1": top1,
                    "top5": top5
                }

    return results

def extract_float(text: str, pattern: str) -> float:
    """Extract float value from text using regex pattern."""
    match = re.search(pattern, text)
    if match:
        try:
            return float(match.group(1))
        except (ValueError, IndexError):
            return 0.0
    return 0.0

def plot_compression_comparison():
    """Create bar chart comparing compression ratios."""
    fig, ax = plt.subplots(figsize=(12, 6))

    models_data = {
        "Qwen2.5-3B": parse_results("Qwen2.5-3B", "Qwen_Qwen2.5-3B-Instruct_results.txt"),
        "Phi-2": parse_results("Phi-2", "microsoft_phi-2_results.txt"),
        "Mistral-7B": parse_results("Mistral-7B", "mistralai_Mistral-7B-Instruct-v0.1_results.txt"),
    }

    x = np.arange(len(models_data))
    width = 0.25

    bitwidths = ["2bit", "3bit", "4bit"]
    bitwidth_labels = ["2-bit", "3-bit", "4-bit"]

    for i, bitwidth in enumerate(bitwidths):
        compressions = []
        for model_name in models_data.keys():
            # Use 8K context
            if "8K" in models_data[model_name]["contexts"]:
                ctx_data = models_data[model_name]["contexts"]["8K"]
                if bitwidth in ctx_data:
                    compressions.append(ctx_data[bitwidth]["compression"])
                else:
                    compressions.append(0)
            else:
                compressions.append(0)

        ax.bar(x + i*width, compressions, width, label=bitwidth_labels[i], alpha=0.8)

    ax.set_xlabel("Model", fontsize=12, fontweight='bold')
    ax.set_ylabel("Compression Ratio (Higher is Better)", fontsize=12, fontweight='bold')
    ax.set_title("Compression Ratio Comparison (8K Context)", fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(models_data.keys())
    ax.legend(title="Quantization", fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "01_compression_comparison.png", dpi=300, bbox_inches='tight')
    print("[CREATED] 01_compression_comparison.png")
    plt.close()

def plot_cosine_similarity_by_context():
    """Create line plot showing cosine similarity across context lengths."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    models_data = {
        "Qwen2.5-3B": parse_results("Qwen2.5-3B", "Qwen_Qwen2.5-3B-Instruct_results.txt"),
        "Phi-2": parse_results("Phi-2", "microsoft_phi-2_results.txt"),
        "Mistral-7B": parse_results("Mistral-7B", "mistralai_Mistral-7B-Instruct-v0.1_results.txt"),
    }

    contexts = ["2K", "4K", "8K"]
    bitwidths = ["2bit", "3bit", "4bit"]
    bitwidth_labels = ["2-bit", "3-bit", "4-bit"]
    bitwidth_markers = ["o", "s", "^"]

    for idx, (model_name, model_data) in enumerate(models_data.items()):
        ax = axes[idx]

        for bw_idx, bitwidth in enumerate(bitwidths):
            cosine_sims = []
            for ctx in contexts:
                if ctx in model_data["contexts"] and bitwidth in model_data["contexts"][ctx]:
                    cosine_sims.append(model_data["contexts"][ctx][bitwidth]["cosine_sim"])
                else:
                    cosine_sims.append(0)

            ax.plot(contexts, cosine_sims, marker=bitwidth_markers[bw_idx],
                   label=bitwidth_labels[bw_idx], linewidth=2, markersize=8)

        ax.set_xlabel("Context Length", fontsize=11, fontweight='bold')
        ax.set_ylabel("Cosine Similarity", fontsize=11, fontweight='bold')
        ax.set_title(f"{model_name}", fontsize=12, fontweight='bold')
        ax.set_ylim(0.96, 1.00)
        ax.grid(True, alpha=0.3)
        ax.legend(title="Quantization", fontsize=9)

    plt.suptitle("Cosine Similarity Across Context Lengths", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "02_cosine_similarity_context.png", dpi=300, bbox_inches='tight')
    print("[CREATED] 02_cosine_similarity_context.png")
    plt.close()

def plot_top1_accuracy():
    """Create grouped bar chart for top-1 match accuracy."""
    fig, ax = plt.subplots(figsize=(14, 6))

    models_data = {
        "Qwen2.5-3B": parse_results("Qwen2.5-3B", "Qwen_Qwen2.5-3B-Instruct_results.txt"),
        "Phi-2": parse_results("Phi-2", "microsoft_phi-2_results.txt"),
        "Mistral-7B": parse_results("Mistral-7B", "mistralai_Mistral-7B-Instruct-v0.1_results.txt"),
    }

    x = np.arange(len(models_data))
    width = 0.25

    # Focus on 3-bit @ 8K (primary metric)
    top1_scores = []
    for model_name in models_data.keys():
        if "8K" in models_data[model_name]["contexts"]:
            if "3bit" in models_data[model_name]["contexts"]["8K"]:
                top1_scores.append(models_data[model_name]["contexts"]["8K"]["3bit"]["top1"])
            else:
                top1_scores.append(0)
        else:
            top1_scores.append(0)

    bars = ax.bar(x, top1_scores, width*3, color=[COLORS.get(m, "#999999") for m in models_data.keys()], alpha=0.8)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

    ax.set_xlabel("Model", fontsize=12, fontweight='bold')
    ax.set_ylabel("Top-1 Match Accuracy (%)", fontsize=12, fontweight='bold')
    ax.set_title("Top-1 Match Accuracy (3-bit @ 8K Context)", fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models_data.keys())
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "03_top1_accuracy.png", dpi=300, bbox_inches='tight')
    print("[CREATED] 03_top1_accuracy.png")
    plt.close()

def plot_context_sensitivity():
    """Create heatmap showing context sensitivity."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    models_data = {
        "Qwen2.5-3B": parse_results("Qwen2.5-3B", "Qwen_Qwen2.5-3B-Instruct_results.txt"),
        "Phi-2": parse_results("Phi-2", "microsoft_phi-2_results.txt"),
        "Mistral-7B": parse_results("Mistral-7B", "mistralai_Mistral-7B-Instruct-v0.1_results.txt"),
    }

    contexts = ["2K", "4K", "8K"]
    bitwidths = ["2bit", "3bit", "4bit"]

    for idx, (model_name, model_data) in enumerate(models_data.items()):
        ax = axes[idx]

        # Build matrix: rows=bitwidths, cols=contexts, values=top1%
        matrix = np.zeros((len(bitwidths), len(contexts)))

        for i, bitwidth in enumerate(bitwidths):
            for j, ctx in enumerate(contexts):
                if ctx in model_data["contexts"] and bitwidth in model_data["contexts"][ctx]:
                    matrix[i, j] = model_data["contexts"][ctx][bitwidth]["top1"]

        sns.heatmap(matrix, annot=True, fmt=".1f", cmap="RdYlGn", vmin=0, vmax=100,
                   xticklabels=contexts, yticklabels=["2-bit", "3-bit", "4-bit"],
                   cbar_kws={'label': 'Top-1 Accuracy (%)'}, ax=ax, linewidths=1)

        ax.set_title(f"{model_name}", fontsize=12, fontweight='bold')
        ax.set_ylabel("Quantization", fontsize=11, fontweight='bold')
        ax.set_xlabel("Context Length", fontsize=11, fontweight='bold')

    plt.suptitle("Top-1 Match Accuracy Heatmap", fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "04_context_sensitivity_heatmap.png", dpi=300, bbox_inches='tight')
    print("[CREATED] 04_context_sensitivity_heatmap.png")
    plt.close()

def plot_model_comparison_radar():
    """Create radar chart comparing models at 3-bit @ 8K."""
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    models_data = {
        "Qwen2.5-3B": parse_results("Qwen2.5-3B", "Qwen_Qwen2.5-3B-Instruct_results.txt"),
        "Phi-2": parse_results("Phi-2", "microsoft_phi-2_results.txt"),
        "Mistral-7B": parse_results("Mistral-7B", "mistralai_Mistral-7B-Instruct-v0.1_results.txt"),
    }

    # Metrics to compare (normalized to 0-100)
    metrics = ["Compression\n(5.0x reference)", "Cosine Sim\n(×100)", "Top-1 Accuracy\n(%)", "Top-5 Accuracy\n(%)"]
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    # Normalize compression: use 5.0x as reference (100)
    compression_ref = 5.0

    for model_name in models_data.keys():
        values = []

        if "8K" in models_data[model_name]["contexts"] and "3bit" in models_data[model_name]["contexts"]["8K"]:
            data = models_data[model_name]["contexts"]["8K"]["3bit"]

            # Compression (normalize to reference)
            compression_norm = (data["compression"] / compression_ref) * 100
            values.append(min(compression_norm, 100))  # Cap at 100

            # Cosine similarity (multiply by 100)
            values.append(data["cosine_sim"] * 100)

            # Top-1 accuracy
            values.append(data["top1"])

            # Top-5 accuracy
            values.append(data["top5"])
        else:
            values = [0, 0, 0, 0]

        values += values[:1]  # Complete the circle

        ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=COLORS.get(model_name, "#999999"))
        ax.fill(angles, values, alpha=0.15, color=COLORS.get(model_name, "#999999"))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylim(0, 105)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_rlabel_position(0)
    ax.grid(True)

    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    plt.title("Model Comparison (3-bit @ 8K Context)", fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "05_model_comparison_radar.png", dpi=300, bbox_inches='tight')
    print("[CREATED] 05_model_comparison_radar.png")
    plt.close()

def plot_bitwidth_tradeoff():
    """Create scatter plot showing compression vs accuracy tradeoff."""
    fig, ax = plt.subplots(figsize=(12, 7))

    models_data = {
        "Qwen2.5-3B": parse_results("Qwen2.5-3B", "Qwen_Qwen2.5-3B-Instruct_results.txt"),
        "Phi-2": parse_results("Phi-2", "microsoft_phi-2_results.txt"),
        "Mistral-7B": parse_results("Mistral-7B", "mistralai_Mistral-7B-Instruct-v0.1_results.txt"),
    }

    for model_name in models_data.keys():
        if "8K" not in models_data[model_name]["contexts"]:
            continue

        compressions = []
        top1_accs = []
        bitwidth_labels = []

        for bitwidth in ["2bit", "3bit", "4bit"]:
            if bitwidth in models_data[model_name]["contexts"]["8K"]:
                data = models_data[model_name]["contexts"]["8K"][bitwidth]
                compressions.append(data["compression"])
                top1_accs.append(data["top1"])
                bitwidth_labels.append(bitwidth)

        ax.scatter(compressions, top1_accs, s=300, alpha=0.7,
                  label=model_name, color=COLORS.get(model_name, "#999999"))

        # Add labels for each point
        for comp, acc, label in zip(compressions, top1_accs, bitwidth_labels):
            bit_num = label.replace("bit", "")
            ax.annotate(f"{bit_num}-bit", (comp, acc), xytext=(5, 5),
                       textcoords='offset points', fontsize=9, fontweight='bold')

    ax.set_xlabel("Compression Ratio (Higher is Better)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Top-1 Match Accuracy (%, Higher is Better)", fontsize=12, fontweight='bold')
    ax.set_title("Compression-Accuracy Tradeoff (8K Context)", fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='lower left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "06_compression_accuracy_tradeoff.png", dpi=300, bbox_inches='tight')
    print("[CREATED] 06_compression_accuracy_tradeoff.png")
    plt.close()

def plot_summary_table():
    """Create a summary comparison table as image."""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('tight')
    ax.axis('off')

    models_data = {
        "Qwen2.5-3B": parse_results("Qwen2.5-3B", "Qwen_Qwen2.5-3B-Instruct_results.txt"),
        "Phi-2": parse_results("Phi-2", "microsoft_phi-2_results.txt"),
        "Mistral-7B": parse_results("Mistral-7B", "mistralai_Mistral-7B-Instruct-v0.1_results.txt"),
    }

    # Build table data for 3-bit @ 8K
    table_data = [["Model", "Compression", "Cosine Sim", "Top-1 %", "Top-5 %"]]

    for model_name in models_data.keys():
        if "8K" in models_data[model_name]["contexts"] and "3bit" in models_data[model_name]["contexts"]["8K"]:
            data = models_data[model_name]["contexts"]["8K"]["3bit"]
            table_data.append([
                model_name,
                f"{data['compression']:.1f}x",
                f"{data['cosine_sim']:.4f}",
                f"{data['top1']:.1f}%",
                f"{data['top5']:.1f}%"
            ])

    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    colWidths=[0.25, 0.15, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)

    # Style header row
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(table_data)):
        for j in range(len(table_data[0])):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
            table[(i, j)].set_text_props(weight='bold')

    plt.title("TurboQuant Results Summary (3-bit @ 8K Context)",
             fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "07_summary_table.png", dpi=300, bbox_inches='tight')
    print("[CREATED] 07_summary_table.png")
    plt.close()

def main():
    """Generate all charts."""
    print("=" * 70)
    print("TurboQuant: Generating Result Visualizations")
    print("=" * 70)
    print()

    # Check if results directory exists
    if not RESULTS_DIR.exists():
        print(f"Error: Results directory not found at {RESULTS_DIR}")
        return

    print(f"Reading results from: {RESULTS_DIR}")
    print(f"Saving charts to: {CHARTS_DIR}")
    print()

    try:
        print("Generating charts...")
        print()

        plot_compression_comparison()
        plot_cosine_similarity_by_context()
        plot_top1_accuracy()
        plot_context_sensitivity()
        plot_model_comparison_radar()
        plot_bitwidth_tradeoff()
        plot_summary_table()

        print()
        print("=" * 70)
        print("All charts generated successfully!")
        print("=" * 70)
        print(f"Charts saved to: {CHARTS_DIR}")
        print()

        # List generated files
        chart_files = sorted(CHARTS_DIR.glob("*.png"))
        if chart_files:
            print("Generated files:")
            for f in chart_files:
                print(f"  - {f.name}")

    except Exception as e:
        print(f"Error generating charts: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
