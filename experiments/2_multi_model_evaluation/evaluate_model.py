"""
Generic TurboQuant evaluation framework for different LLM models.

This script evaluates TurboQuant KV cache compression on various models:
- Compression ratio
- Attention score accuracy (cosine similarity, top-1 match, top-5 match)
- Memory usage

Usage:
    python evaluate_model.py --model qwen/Qwen2.5-3B-Instruct --bits 3 --output results.json
"""

import torch
import torch.nn.functional as F
import json
import sys
import os
from pathlib import Path
from datetime import datetime
import argparse

# Add original_implementation to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "original_implementation"))

from compressors import TurboQuantCompressorV2, TurboQuantCompressorMSE
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def build_prompt(tokenizer, target_tokens=2048, needle="test", needle_pos=0.5):
    """Build a prompt with needle hidden in haystack."""
    filler = """The quarterly financial review covered budget allocations, spending reports,
and revenue projections. The committee discussed infrastructure upgrades and maintenance schedules.
Several action items were assigned to team leads for follow-up.\n\n"""

    filler_len = len(tokenizer.encode(filler))
    n_reps = max(1, target_tokens // filler_len)
    needle_idx = int(n_reps * needle_pos)

    parts = []
    for i in range(n_reps):
        if i == needle_idx:
            parts.append(f"\n--- Important ---\n{needle}\n--- End ---\n\n")
        parts.append(filler)

    haystack = "".join(parts)
    return f"Context:\n{haystack}\nQuestion: Find the important fact."


def evaluate_model(model_name: str, bits: int = 3, context_lengths: list = None,
                   device: str = "cuda", output_file: str = None):
    """
    Evaluate TurboQuant on a specific model.

    Args:
        model_name: HuggingFace model identifier
        bits: Quantization bits (2, 3, or 4)
        context_lengths: List of context lengths to test
        device: torch device
        output_file: Path to save results JSON

    Returns:
        dict: Results dictionary with compression and accuracy metrics
    """

    if context_lengths is None:
        context_lengths = [2048, 4096]

    print(f"\n{'='*70}")
    print(f"Model: {model_name}")
    print(f"Bits: {bits}, Contexts: {context_lengths}")
    print(f"{'='*70}\n")

    # Load model and tokenizer
    print("Loading model...", flush=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except ValueError as e:
        if "sentencepiece or tiktoken" in str(e):
            print("Retrying with use_fast=False...")
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        else:
            raise

    # Detect model size and adjust quantization config accordingly
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4"
            ),
            device_map="auto",
            dtype=torch.float16,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"Warning: 4-bit quantization failed, trying FP16: {e}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            dtype=torch.float16,
            trust_remote_code=True,
        )

    model.eval()

    # Get model info
    # Handle different model architectures
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        n_layers = len(model.model.layers)
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        n_layers = len(model.transformer.h)  # GPT2, GPT-like models
    else:
        n_layers = getattr(model.config, 'num_hidden_layers', 0)

    hidden_size = model.config.hidden_size
    num_heads = model.config.num_attention_heads
    num_kv_heads = getattr(model.config, 'num_key_value_heads', num_heads)
    head_dim = hidden_size // num_heads

    print(f"Model loaded. Config:")
    print(f"  Layers: {n_layers}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  KV heads: {num_kv_heads}")
    print(f"  Head dim: {head_dim}")
    print(f"  Memory: {torch.cuda.memory_allocated() // 1024 // 1024} MB\n")

    results = {
        "model": model_name,
        "bits": bits,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "n_layers": n_layers,
            "hidden_size": hidden_size,
            "num_heads": num_heads,
            "head_dim": head_dim,
        },
        "by_context": {}
    }

    # Evaluate on each context length
    for context_len in context_lengths:
        print(f"\n{'─'*70}")
        print(f"Context length: {context_len} tokens")
        print(f"{'─'*70}")

        try:
            # Build prompt and tokenize
            needle = f"Secret code is {context_len}-NEEDLE-42."
            prompt = build_prompt(tokenizer, target_tokens=context_len, needle=needle)
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=context_len + 256
            ).to(device)

            seq_len = inputs["input_ids"].shape[1]

            # Forward pass to get KV cache
            with torch.no_grad():
                outputs = model(**inputs, use_cache=True, output_attentions=False)

            try:
                cache = outputs.past_key_values
            except:
                print(f"  ⚠️  Model doesn't use standard cache format, skipping")
                continue

            # Compress and evaluate
            total_compressed_bytes = 0
            total_uncompressed_bytes = 0
            cosine_sims = []
            top1_matches = 0
            top5_matches = 0
            n_checks = 0

            print(f"  Processing {n_layers} layers...", end="", flush=True)

            for layer_idx in range(min(n_layers, len(cache.layers) if hasattr(cache, 'layers') else len(cache))):
                try:
                    if hasattr(cache, 'layers'):
                        keys = cache.layers[layer_idx].keys      # (1, num_kv_heads, seq, head_dim)
                        values = cache.layers[layer_idx].values
                    else:
                        keys, values = cache[layer_idx]  # Tuple format
                        # Reshape if needed
                        if keys.dim() == 3:
                            keys = keys.unsqueeze(1)  # Add head dim
                            values = values.unsqueeze(1)

                    B, H, S, D = keys.shape

                    # Compress
                    key_comp = TurboQuantCompressorV2(D, bits, seed=layer_idx * 1000, device=device)
                    val_comp = TurboQuantCompressorMSE(D, bits, seed=layer_idx * 1000 + 500, device=device)

                    compressed_k = key_comp.compress(keys)
                    compressed_v = val_comp.compress(values)

                    # Memory accounting
                    n_key_vecs = B * H * S
                    mse_bits = max(bits - 1, 1)
                    k_bits = n_key_vecs * D * mse_bits  # MSE indices
                    k_bits += n_key_vecs * D * 1         # QJL signs
                    k_bits += n_key_vecs * 16             # residual norms
                    k_bits += n_key_vecs * 16             # vector norms

                    v_bits = n_key_vecs * D * bits        # Values MSE indices
                    v_bits += n_key_vecs * 16              # vector norms

                    total_compressed_bytes += (k_bits + v_bits) / 8
                    total_uncompressed_bytes += (keys.numel() + values.numel()) * 2

                    # Compare attention scores
                    query = keys[:, :, -1:, :]  # Last token
                    real_scores = torch.matmul(query.float(), keys.float().transpose(-2, -1)).squeeze(-2)
                    tq_scores = key_comp.asymmetric_attention_scores(query, compressed_k).squeeze(-2)

                    # Per-head metrics
                    for h in range(H):
                        rs = real_scores[0, h]
                        ts = tq_scores[0, h]

                        # Cosine similarity
                        cos = F.cosine_similarity(rs.unsqueeze(0), ts.unsqueeze(0)).item()
                        cosine_sims.append(cos)

                        # Top-1 match
                        if rs.argmax().item() == ts.argmax().item():
                            top1_matches += 1

                        # Top-5 match
                        if rs.argmax().item() in ts.topk(5).indices.tolist():
                            top5_matches += 1

                        n_checks += 1

                except Exception as e:
                    print(f"\n  ⚠️  Layer {layer_idx} error: {e}")
                    continue

            print(" [OK]")

            # Aggregate results
            compression_ratio = total_uncompressed_bytes / max(total_compressed_bytes, 1)
            avg_cosine = sum(cosine_sims) / len(cosine_sims) if cosine_sims else 0
            top1_pct = 100 * top1_matches / max(n_checks, 1)
            top5_pct = 100 * top5_matches / max(n_checks, 1)

            context_results = {
                "seq_len": seq_len,
                "compression_ratio": float(compression_ratio),
                "compressed_mb": float(total_compressed_bytes / 1024 / 1024),
                "uncompressed_mb": float(total_uncompressed_bytes / 1024 / 1024),
                "cosine_similarity": float(avg_cosine),
                "top1_match_pct": float(top1_pct),
                "top5_match_pct": float(top5_pct),
                "n_heads_checked": n_checks,
            }

            # Print results
            print(f"\n  Results:")
            print(f"    Compression:  {compression_ratio:.2f}x")
            print(f"    Cosine sim:   {avg_cosine:.6f}")
            print(f"    Top-1 match:  {top1_pct:.1f}%")
            print(f"    Top-5 match:  {top5_pct:.1f}%")

            results["by_context"][str(context_len)] = context_results

        except Exception as e:
            print(f"\n  [ERROR] Error at context {context_len}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save results
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n[SUCCESS] Results saved to {output_file}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate TurboQuant on a model")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model ID")
    parser.add_argument("--bits", type=int, default=3, choices=[1, 2, 3, 4], help="Quantization bits")
    parser.add_argument("--contexts", type=int, nargs="+", default=[2048, 4096], help="Context lengths")
    parser.add_argument("--device", type=str, default="cuda", help="torch device")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")

    args = parser.parse_args()

    results = evaluate_model(
        args.model,
        bits=args.bits,
        context_lengths=args.contexts,
        device=args.device,
        output_file=args.output
    )
