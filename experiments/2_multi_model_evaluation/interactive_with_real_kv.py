"""
Interactive TurboQuant Comparison - REAL KV Compression Analysis
- Generate text normally (both models identical)
- Then analyze: compress actual KV cache and measure impact

This shows:
1. Text generation (identical since no KV compression applied)
2. REAL KV compression effect: size reduction, attention accuracy
"""

import torch
import torch.nn.functional as F
import time
import sys
import os
from pathlib import Path
from typing import Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "original_implementation"))

from compressors import TurboQuantCompressorV2, TurboQuantCompressorMSE
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class InteractiveWithRealKV:
    def __init__(self, model_name: str, bits: int = 3, max_new_tokens: int = 128):
        self.model_name = model_name
        self.bits = bits
        self.max_new_tokens = max_new_tokens
        self.compressors = {}

        print("\n[LOADING] Model: {}".format(model_name))
        print("[LOADING] Bits: {}".format(bits))

        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except ValueError:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

        # Load model
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
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
        except Exception:
            print("[WARNING] 4-bit failed, using FP16")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                dtype=torch.float16,
                trust_remote_code=True,
            )

        self.model.eval()
        if hasattr(self.tokenizer, "pad_token") and self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        gpu_mem = torch.cuda.memory_allocated() // 1024 // 1024
        print("[LOADED] GPU Memory: {} MB".format(gpu_mem))

    def generate_text(self, prompt: str) -> Tuple[str, float]:
        """Generate text and return with time."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        input_len = inputs["input_ids"].shape[1]

        start_time = time.time()

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        elapsed_time = time.time() - start_time
        text = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)

        return text, elapsed_time

    def analyze_kv_compression(self, prompt: str) -> dict:
        """Analyze KV compression on actual cache from forward pass."""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
        input_len = inputs["input_ids"].shape[1]

        # Forward pass to get KV cache
        with torch.no_grad():
            outputs = self.model(**inputs, use_cache=True, return_dict=True)

        cache = outputs.past_key_values

        # Handle DynamicCache
        if hasattr(cache, 'layers'):
            cache_layers = cache.layers
        else:
            cache_layers = cache

        # Calculate original KV size
        original_kv_bytes = 0
        for cache_item in cache_layers:
            if hasattr(cache_item, 'keys'):
                k, v = cache_item.keys, cache_item.values
            else:
                k, v = cache_item

            original_kv_bytes += k.numel() * 2 + v.numel() * 2

        original_kv_mb = original_kv_bytes / 1024 / 1024

        # Compress and analyze
        cosine_sims = []
        top1_matches = 0
        top5_matches = 0
        total_heads = 0
        compressed_kv_bytes = 0

        start_compress = time.time()

        for layer_idx, cache_item in enumerate(cache_layers):
            if hasattr(cache_item, 'keys'):
                k, v = cache_item.keys, cache_item.values
            else:
                k, v = cache_item

            B, H, S, D = k.shape

            # Create compressors
            if layer_idx not in self.compressors:
                key_comp = TurboQuantCompressorV2(
                    D, self.bits, seed=layer_idx * 1000, device="cuda"
                )
                val_comp = TurboQuantCompressorMSE(
                    D, self.bits, seed=layer_idx * 1000 + 500, device="cuda"
                )
                self.compressors[layer_idx] = (key_comp, val_comp)

            key_comp, val_comp = self.compressors[layer_idx]

            # Compress
            k_compressed = key_comp.compress(k)
            v_compressed = val_comp.compress(v)

            # Calculate compressed size
            k_mse = k_compressed["k_mse"]
            qjl_signs = k_compressed["qjl_signs"]
            residual_norm = k_compressed["residual_norm"]

            v_indices = v_compressed["indices"]
            v_norms = v_compressed["vec_norms"]

            k_comp_bytes = (k_mse.numel() * 2 + qjl_signs.numel() * 1 + residual_norm.numel() * 2)
            v_comp_bytes = (v_indices.numel() * 1 + v_norms.numel() * 2)
            compressed_kv_bytes += k_comp_bytes + v_comp_bytes

            # Attention accuracy analysis
            query = k[:, :, -1:, :]  # Last token query

            # Real scores
            real_scores = torch.matmul(query.float(), k.float().transpose(-2, -1)).squeeze(-2)

            # TurboQuant scores
            tq_scores = key_comp.asymmetric_attention_scores(query, k_compressed).squeeze(-2)

            # Per-head comparison
            for h in range(H):
                rs = real_scores[0, h]
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

        compress_time = time.time() - start_compress
        compressed_kv_mb = compressed_kv_bytes / 1024 / 1024

        avg_cosine = sum(cosine_sims) / len(cosine_sims)
        top1_pct = 100 * top1_matches / total_heads
        top5_pct = 100 * top5_matches / total_heads
        compression_ratio = original_kv_mb / compressed_kv_mb if compressed_kv_mb > 0 else 1.0
        memory_savings = (1 - compressed_kv_mb / original_kv_mb) * 100

        return {
            'original_kv_mb': original_kv_mb,
            'compressed_kv_mb': compressed_kv_mb,
            'compression_ratio': compression_ratio,
            'memory_savings_pct': memory_savings,
            'compress_time_s': compress_time,
            'cosine_similarity': avg_cosine,
            'top1_match_pct': top1_pct,
            'top5_match_pct': top5_pct,
        }

    def compare(self, prompt: str):
        """Compare both models."""
        print("\n{}".format("=" * 80))
        print("INPUT PROMPT:")
        print("{}".format(prompt))
        print("=" * 80)

        # Generate text
        print("\n[GENERATING] Generating text...")
        text, gen_time = self.generate_text(prompt)

        # Analyze KV compression
        print("[ANALYZING] Analyzing KV cache compression...")
        analysis = self.analyze_kv_compression(prompt)

        # Display results
        print("\n{}".format("=" * 80))
        print("MODEL OUTPUT:")
        print("=" * 80)
        print(text)

        print("\n{}".format("=" * 80))
        print("GENERATION METRICS:")
        print("=" * 80)
        print("Generation time: {:.2f}s".format(gen_time))
        print("Output length: {} chars, {} words".format(len(text), len(text.split())))

        print("\n{}".format("=" * 80))
        print("KV CACHE COMPRESSION ANALYSIS ({}bit):".format(self.bits))
        print("=" * 80)
        print("\nMemory Impact:")
        print("  Original KV cache: {:.1f} MB".format(analysis['original_kv_mb']))
        print("  Compressed KV cache: {:.1f} MB".format(analysis['compressed_kv_mb']))
        print("  Compression ratio: {:.1f}x".format(analysis['compression_ratio']))
        print("  Memory savings: {:.1f}%".format(analysis['memory_savings_pct']))

        print("\nAttention Accuracy:")
        print("  Cosine similarity: {:.6f}  (1.0 = identical)".format(analysis['cosine_similarity']))
        print("  Top-1 match: {:.1f}%".format(analysis['top1_match_pct']))
        print("  Top-5 match: {:.1f}%".format(analysis['top5_match_pct']))

        print("\nCompression Speed:")
        print("  Compression time: {:.3f}s".format(analysis['compress_time_s']))

    def interactive_loop(self):
        """Main interactive loop."""
        print("\n{}".format("=" * 80))
        print("INTERACTIVE TURBOQUANT ANALYSIS")
        print("{}".format("=" * 80))
        print("\nEnter prompts to:")
        print("  1. Generate text")
        print("  2. Analyze real KV cache compression")
        print("  3. See compression impact on attention")
        print("\nType 'quit' to exit, 'help' for help")
        print("")

        while True:
            try:
                prompt = input("\n[PROMPT] Enter text: ").strip()

                if prompt.lower() == "quit":
                    print("\nGoodbye!")
                    break

                if prompt.lower() == "help":
                    print("\nCommands:")
                    print("  quit     - Exit program")
                    print("  help     - Show this help")
                    print("  settings - Show settings")
                    print("")
                    continue

                if prompt.lower() == "settings":
                    print("\nSettings:")
                    print("  Model: {}".format(self.model_name))
                    print("  Bits: {}".format(self.bits))
                    print("  Max tokens: {}".format(self.max_new_tokens))
                    print("")
                    continue

                if not prompt:
                    print("[ERROR] Empty prompt")
                    continue

                self.compare(prompt)

            except KeyboardInterrupt:
                print("\n\nInterrupted")
                break
            except Exception as e:
                print("[ERROR] {}".format(str(e)))
                continue


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Interactive TurboQuant with Real KV Analysis")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="Model name")
    parser.add_argument("--bits", type=int, default=3, help="Compression bits")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max generation tokens")

    args = parser.parse_args()

    tool = InteractiveWithRealKV(args.model, args.bits, args.max_tokens)
    tool.interactive_loop()


if __name__ == "__main__":
    main()
