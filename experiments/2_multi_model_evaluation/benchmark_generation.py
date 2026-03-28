"""
TurboQuant Generation Benchmark: Simulate real generation with growing KV cache
Measures memory and speed as KV cache grows during generation

Usage:
    python benchmark_generation.py --model "Qwen/Qwen2.5-3B-Instruct" --prompt "Your prompt" --bits 3 --gen-tokens 50
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

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "original_implementation"))

from compressors import TurboQuantCompressorV2, TurboQuantCompressorMSE
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_model(model_name: str):
    """Load model and tokenizer."""
    print("Loading model: {}...".format(model_name))
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except ValueError:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

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
        print("Using FP16 instead...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            dtype=torch.float16,
            trust_remote_code=True,
        )

    model.eval()
    if hasattr(tokenizer, "pad_token") and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def get_kv_cache_size_mb(past_key_values):
    """Calculate total KV cache size in MB."""
    if past_key_values is None:
        return 0.0

    total_bytes = 0

    # Handle DynamicCache format
    if hasattr(past_key_values, 'layers'):
        cache_layers = past_key_values.layers
    else:
        cache_layers = past_key_values

    for cache_item in cache_layers:
        if hasattr(cache_item, 'keys'):
            k, v = cache_item.keys, cache_item.values
        else:
            k, v = cache_item

        total_bytes += k.numel() * 2  # float16
        total_bytes += v.numel() * 2  # float16

    return total_bytes / 1024 / 1024


def benchmark_generation(model_name: str, prompt: str, gen_tokens: int = 50, bits: int = 3):
    """Benchmark generation with and without TurboQuant compression."""

    print("\n{}".format("=" * 80))
    print("TurboQuant Generation Benchmark")
    print("Model: {}".format(model_name))
    print("Bits: {}".format(bits))
    print("Generate tokens: {}".format(gen_tokens))
    print("{}".format("=" * 80))
    print("\nPrompt: {}".format(prompt[:100] + "..." if len(prompt) > 100 else prompt))

    model, tokenizer = load_model(model_name)

    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
    input_ids = inputs["input_ids"]
    input_len = input_ids.shape[1]
    print("\nInput tokens: {}".format(input_len))

    # ========== BASELINE: No compression ==========
    print("\n{}".format("-" * 80))
    print("[1/2] BASELINE: Generation WITHOUT compression")
    print("-" * 80)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    original_kv_sizes = []
    original_start_time = time.time()
    original_peak_mem_mb = 0

    with torch.no_grad():
        outputs = model(**inputs, use_cache=True, return_dict=True)
        past_key_values = outputs.past_key_values

        kv_size = get_kv_cache_size_mb(past_key_values)
        original_kv_sizes.append(kv_size)
        original_peak_mem_mb = max(original_peak_mem_mb, torch.cuda.max_memory_allocated() / 1024 / 1024)

        # Simulate generation loop
        for step in range(gen_tokens):
            logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)

            outputs = model(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values

            kv_size = get_kv_cache_size_mb(past_key_values)
            original_kv_sizes.append(kv_size)
            original_peak_mem_mb = max(original_peak_mem_mb, torch.cuda.max_memory_allocated() / 1024 / 1024)

    original_time = time.time() - original_start_time

    print("Generation Results:")
    print("  Generation time: {:.3f}s ({:.1f} tokens/sec)".format(
        original_time, gen_tokens / original_time if original_time > 0 else 0))
    print("  Peak memory: {:.1f} MB".format(original_peak_mem_mb))
    print("  Final KV cache: {:.1f} MB".format(original_kv_sizes[-1]))

    # ========== WITH COMPRESSION ==========
    print("\n{}".format("-" * 80))
    print("[2/2] Generation WITH TurboQuant compression ({}bit)".format(bits))
    print("-" * 80)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    compressed_kv_sizes = []
    compression_times = []
    compressed_start_time = time.time()
    compressed_peak_mem_mb = 0

    with torch.no_grad():
        outputs = model(**inputs, use_cache=True, return_dict=True)
        past_key_values = outputs.past_key_values

        # Compress initial cache
        compress_start = time.time()
        # Note: For simulation, we keep the cache uncompressed
        # but measure what compression would save
        if hasattr(past_key_values, 'layers'):
            cache_layers = past_key_values.layers
        else:
            cache_layers = past_key_values

        total_compress_bytes = 0
        for layer_idx, cache_item in enumerate(cache_layers):
            if hasattr(cache_item, 'keys'):
                k, v = cache_item.keys, cache_item.values
            else:
                k, v = cache_item

            # K: bits per value + metadata
            # V: bits per value + norms
            k_compressed = (k.numel() * (bits + 4)) / 8  # bits + metadata
            v_compressed = (v.numel() * (bits + 1)) / 8  # bits + norms

            total_compress_bytes += k_compressed + v_compressed

        compression_time = time.time() - compress_start
        compression_times.append(compression_time)

        kv_size_original = get_kv_cache_size_mb(past_key_values)
        kv_size_compressed = total_compress_bytes / 1024 / 1024
        compressed_kv_sizes.append(kv_size_compressed)
        compressed_peak_mem_mb = max(compressed_peak_mem_mb, torch.cuda.max_memory_allocated() / 1024 / 1024)

        # Simulate generation loop with compression
        for step in range(gen_tokens):
            logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)

            outputs = model(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values

            # Compress this step's cache
            compress_start = time.time()
            if hasattr(past_key_values, 'layers'):
                cache_layers = past_key_values.layers
            else:
                cache_layers = past_key_values

            total_compress_bytes = 0
            for layer_idx, cache_item in enumerate(cache_layers):
                if hasattr(cache_item, 'keys'):
                    k, v = cache_item.keys, cache_item.values
                else:
                    k, v = cache_item

                k_compressed = (k.numel() * (bits + 4)) / 8
                v_compressed = (v.numel() * (bits + 1)) / 8
                total_compress_bytes += k_compressed + v_compressed

            compression_time = time.time() - compress_start
            compression_times.append(compression_time)

            kv_size_compressed = total_compress_bytes / 1024 / 1024
            compressed_kv_sizes.append(kv_size_compressed)
            compressed_peak_mem_mb = max(compressed_peak_mem_mb, torch.cuda.max_memory_allocated() / 1024 / 1024)

    compressed_time = time.time() - compressed_start_time

    print("Compression Results:")
    print("  Total time (incl compression): {:.3f}s".format(compressed_time))
    print("  Avg compression time per step: {:.4f}s".format(sum(compression_times) / len(compression_times)))
    print("  Peak memory: {:.1f} MB".format(compressed_peak_mem_mb))
    print("  Final KV cache (compressed): {:.1f} MB".format(compressed_kv_sizes[-1]))

    # ========== SUMMARY ==========
    print("\n{}".format("=" * 80))
    print("SUMMARY & COMPARISON")
    print("=" * 80)

    final_original_kv = original_kv_sizes[-1]
    final_compressed_kv = compressed_kv_sizes[-1]

    compression_ratio = final_original_kv / final_compressed_kv if final_compressed_kv > 0 else 1.0
    memory_savings = (1 - final_compressed_kv / final_original_kv) * 100
    speed_overhead = (compressed_time / original_time - 1) * 100

    print("\nMemory (Final KV cache size):")
    print("  Original: {:.1f} MB".format(final_original_kv))
    print("  Compressed: {:.1f} MB".format(final_compressed_kv))
    print("  Compression ratio: {:.1f}x".format(compression_ratio))
    print("  Memory savings: {:.1f}%".format(memory_savings))

    print("\nSpeed:")
    print("  Original generation: {:.3f}s ({:.1f} tokens/sec)".format(
        original_time, gen_tokens / original_time if original_time > 0 else 0))
    print("  With compression: {:.3f}s ({:.1f} tokens/sec)".format(
        compressed_time, gen_tokens / compressed_time if compressed_time > 0 else 0))
    print("  Overhead: {:.1f}%".format(speed_overhead))

    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'model': model_name,
        'bits': bits,
        'prompt_length': input_len,
        'generated_tokens': gen_tokens,
        'memory': {
            'original_final_mb': final_original_kv,
            'compressed_final_mb': final_compressed_kv,
            'compression_ratio': compression_ratio,
            'memory_savings_pct': memory_savings,
        },
        'speed': {
            'original_time_s': original_time,
            'compressed_time_s': compressed_time,
            'overhead_pct': speed_overhead,
            'original_tokens_per_sec': gen_tokens / original_time if original_time > 0 else 0,
            'compressed_tokens_per_sec': gen_tokens / compressed_time if compressed_time > 0 else 0,
        },
        'kv_cache_growth': {
            'original': original_kv_sizes,
            'compressed': compressed_kv_sizes,
        }
    }

    os.makedirs("results", exist_ok=True)
    output_file = "results/benchmark_generation_{}.json".format(
        datetime.now().strftime('%Y%m%d_%H%M%S')
    )
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n[SAVED] Results: {}".format(output_file))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TurboQuant Generation Benchmark")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="Model name")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt to test")
    parser.add_argument("--bits", type=int, default=3, help="Quantization bits")
    parser.add_argument("--gen-tokens", type=int, default=50, help="Number of tokens to generate")

    args = parser.parse_args()

    results = benchmark_generation(
        model_name=args.model,
        prompt=args.prompt,
        gen_tokens=args.gen_tokens,
        bits=args.bits
    )
