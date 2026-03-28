"""
TurboQuant Real Benchmark: Memory and Speed Comparison
- Original model vs TurboQuant compressed model
- Measures: Peak memory, compression ratio, inference speed, output quality

Usage:
    python benchmark_turboquant.py --model "Qwen/Qwen2.5-3B-Instruct" --prompt "Your prompt" --bits 3
"""

import torch
import torch.nn.functional as F
import time
import os
import sys
import json
from pathlib import Path
from datetime import datetime
import argparse

# Add original_implementation to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "original_implementation"))

from compressors import TurboQuantCompressorV2, TurboQuantCompressorMSE
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_model(model_name: str):
    """Load model and tokenizer."""
    print("Loading model: {}...".format(model_name))
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except ValueError as e:
        if "sentencepiece or tiktoken" in str(e):
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        else:
            raise

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
        print("4-bit quantization failed, trying FP16: {}".format(e))
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            dtype=torch.float16,
            trust_remote_code=True,
        )

    model.eval()
    if hasattr(tokenizer, "pad_token") and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loaded. GPU Memory: {} MB\n".format(torch.cuda.memory_allocated() // 1024 // 1024))
    return model, tokenizer


def benchmark(model_name: str, prompt: str, max_tokens: int = 50, bits: int = 3):
    """Run comprehensive benchmark."""

    print("\n{}".format("=" * 80))
    print("TurboQuant Benchmark")
    print("Model: {}".format(model_name))
    print("Bits: {}".format(bits))
    print("{}".format("=" * 80))
    print("\nPrompt: {}".format(prompt))
    print("\n{}".format("=" * 80))

    model, tokenizer = load_model(model_name)

    # Prepare input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
    input_ids = inputs["input_ids"]
    seq_len = input_ids.shape[1]
    print("\nInput tokens: {}".format(seq_len))

    # ========== PHASE 1: BASELINE (No Compression) ==========
    print("\n{}".format("-" * 80))
    print("[1/2] BASELINE: Original model (NO compression)")
    print("-" * 80)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    baseline_start_mem = torch.cuda.memory_allocated()

    # Forward pass to get KV cache
    start_time = time.time()
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True, return_dict=True)
    baseline_forward_time = time.time() - start_time

    baseline_kv = outputs.past_key_values
    baseline_forward_mem = torch.cuda.max_memory_allocated() - baseline_start_mem

    # Estimate KV cache size
    total_k_bytes = 0
    total_v_bytes = 0

    # DynamicCache has .layers attribute
    if hasattr(baseline_kv, 'layers'):
        cache_layers = baseline_kv.layers
    else:
        cache_layers = baseline_kv  # fallback for tuple format

    for layer_idx, cache_item in enumerate(cache_layers):
        if hasattr(cache_item, 'keys'):  # DynamicCache layer
            k, v = cache_item.keys, cache_item.values
        else:  # tuple format (k, v)
            k, v = cache_item

        # FP16 = 2 bytes per element
        total_k_bytes += k.numel() * 2
        total_v_bytes += v.numel() * 2

    baseline_kv_size_mb = (total_k_bytes + total_v_bytes) / 1024 / 1024

    print("\nBaseline Results:")
    print("  Forward time: {:.3f}s".format(baseline_forward_time))
    print("  Forward memory: {:.1f} MB".format(baseline_forward_mem))
    print("  KV cache size: {:.1f} MB (FP16)".format(baseline_kv_size_mb))

    # ========== PHASE 2: TURBOQUANT (With Compression) ==========
    print("\n{}".format("-" * 80))
    print("[2/2] TURBOQUANT: Compressed model ({}bit)".format(bits))
    print("-" * 80)

    # Compress KV cache
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    compress_start_mem = torch.cuda.memory_allocated()

    start_time = time.time()
    compressed_kvs = []

    for layer_idx, cache_item in enumerate(cache_layers):
        if hasattr(cache_item, 'keys'):
            k, v = cache_item.keys, cache_item.values
        else:
            k, v = cache_item

        head_dim = k.shape[-1]

        # Create compressors
        key_comp = TurboQuantCompressorV2(head_dim, bits, seed=layer_idx * 1000, device="cuda")
        val_comp = TurboQuantCompressorMSE(head_dim, bits, seed=layer_idx * 1000 + 500, device="cuda")

        # Compress
        k_compressed = key_comp.compress(k)
        v_compressed = val_comp.compress(v)

        compressed_kvs.append((key_comp, val_comp, k_compressed, v_compressed))

    compression_time = time.time() - start_time
    compression_mem = torch.cuda.max_memory_allocated() - compress_start_mem

    # Calculate compressed size
    # Compressed format from TurboQuant:
    # K: k_mse (float16) + qjl_signs (int8) + residual_norm (float16)
    # V: indices (uint8) + vec_norms (float16)
    compressed_kv_size_mb = 0
    for key_comp, val_comp, k_comp_dict, v_comp_dict in compressed_kvs:
        # K compression (TurboQuantCompressorV2)
        k_mse = k_comp_dict["k_mse"]  # float16
        qjl_signs = k_comp_dict["qjl_signs"]  # int8
        residual_norm = k_comp_dict["residual_norm"]  # float16

        k_size = (k_mse.numel() * 2 +  # float16
                 qjl_signs.numel() * 1 +  # int8
                 residual_norm.numel() * 2) / 1024 / 1024  # float16

        # V compression (TurboQuantCompressorMSE - indices + norms only)
        v_indices = v_comp_dict["indices"]  # uint8
        v_norms = v_comp_dict["vec_norms"]  # float16

        v_size = (v_indices.numel() * 1 +  # uint8
                 v_norms.numel() * 2) / 1024 / 1024  # float16

        compressed_kv_size_mb += k_size + v_size

    print("\nTurboQuant Results:")
    print("  Compression time: {:.3f}s".format(compression_time))
    print("  Compression memory: {:.1f} MB".format(compression_mem))
    print("  Compressed KV size: {:.1f} MB (estimated)".format(compressed_kv_size_mb))

    # ========== PHASE 3: ATTENTION ACCURACY ==========
    print("\n{}".format("-" * 80))
    print("[3/3] ATTENTION ACCURACY: Compressed vs Original")
    print("-" * 80)

    cosine_sims = []
    top1_matches = 0
    top5_matches = 0
    total_heads = 0

    for layer_idx, cache_item in enumerate(cache_layers):
        if hasattr(cache_item, 'keys'):
            k, v = cache_item.keys, cache_item.values
        else:
            k, v = cache_item

        B, H, S, D = k.shape

        # Get compressors and compressed values
        key_comp, val_comp, k_comp_dict, v_comp_dict = compressed_kvs[layer_idx]

        # Query = last token (simulates next-token generation)
        query = k[:, :, -1:, :]  # (1, H, 1, D)

        # Real attention scores
        real_scores = torch.matmul(query.float(), k.float().transpose(-2, -1)).squeeze(-2)  # (1, H, S)

        # TurboQuant attention scores (using asymmetric inner product)
        try:
            tq_scores = key_comp.asymmetric_attention_scores(query, k_comp_dict).squeeze(-2)  # (1, H, S)
        except Exception as e:
            # Fallback if asymmetric_attention_scores not available
            k_mse = k_comp_dict["k_mse"].float()
            tq_scores = torch.matmul(query.float(), k_mse.float().transpose(-2, -1)).squeeze(-2)

        # Per-head comparison
        for h in range(H):
            rs = real_scores[0, h]  # (S,)
            ts = tq_scores[0, h]

            # Cosine similarity
            cos = F.cosine_similarity(rs.unsqueeze(0), ts.unsqueeze(0)).item()
            cosine_sims.append(cos)

            # Top-1 match
            real_top1 = rs.argmax().item()
            tq_top1 = ts.argmax().item()
            if real_top1 == tq_top1:
                top1_matches += 1

            # Top-5 match
            tq_top5 = ts.topk(5).indices.tolist()
            if real_top1 in tq_top5:
                top5_matches += 1

            total_heads += 1

    avg_cosine = sum(cosine_sims) / len(cosine_sims)
    top1_pct = 100 * top1_matches / total_heads
    top5_pct = 100 * top5_matches / total_heads

    print("\nAttention Accuracy:")
    print("  Cosine similarity: {:.6f}  (1.0 = identical)".format(avg_cosine))
    print("  Top-1 match: {:.1f}%  ({}/{} heads)".format(top1_pct, top1_matches, total_heads))
    print("  Top-5 match: {:.1f}%  ({}/{} heads)".format(top5_pct, top5_matches, total_heads))

    # ========== SUMMARY ==========
    print("\n{}".format("=" * 80))
    print("SUMMARY")
    print("=" * 80)

    compression_ratio = baseline_kv_size_mb / compressed_kv_size_mb
    memory_savings = (1 - compressed_kv_size_mb / baseline_kv_size_mb) * 100

    print("\nMemory:")
    print("  Original KV cache: {:.1f} MB".format(baseline_kv_size_mb))
    print("  Compressed KV cache: {:.1f} MB".format(compressed_kv_size_mb))
    print("  Compression ratio: {:.1f}x".format(compression_ratio))
    print("  Memory savings: {:.1f}%".format(memory_savings))

    print("\nSpeed:")
    print("  Forward pass (original): {:.3f}s".format(baseline_forward_time))
    print("  Compression overhead: {:.3f}s".format(compression_time))
    print("  Total time with compression: {:.3f}s".format(baseline_forward_time + compression_time))

    print("\nQuality:")
    print("  Attention similarity: {:.4f}%".format(avg_cosine * 100))
    print("  Top-1 accuracy: {:.1f}%".format(top1_pct))
    print("  Top-5 accuracy: {:.1f}%".format(top5_pct))

    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'model': model_name,
        'bits': bits,
        'prompt': prompt,
        'memory': {
            'original_kv_mb': baseline_kv_size_mb,
            'compressed_kv_mb': compressed_kv_size_mb,
            'compression_ratio': compression_ratio,
            'memory_savings_pct': memory_savings,
        },
        'speed': {
            'forward_time_s': baseline_forward_time,
            'compression_time_s': compression_time,
            'total_time_s': baseline_forward_time + compression_time,
        },
        'quality': {
            'cosine_similarity': avg_cosine,
            'top1_match_pct': top1_pct,
            'top5_match_pct': top5_pct,
        },
    }

    os.makedirs("results", exist_ok=True)
    output_file = "results/benchmark_{}.json".format(datetime.now().strftime('%Y%m%d_%H%M%S'))
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n[SAVED] Results: {}".format(output_file))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TurboQuant Real Benchmark")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="Model name")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt to test")
    parser.add_argument("--bits", type=int, default=3, help="Quantization bits")
    parser.add_argument("--max-tokens", type=int, default=50, help="Max tokens for analysis")

    args = parser.parse_args()

    results = benchmark(
        model_name=args.model,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        bits=args.bits
    )
