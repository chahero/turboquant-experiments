"""
Simple prompt comparison: Run the same prompt twice and compare outputs.
This version doesn't implement compression yet, but serves as baseline.

Usage:
    python simple_prompt_test.py --model Qwen/Qwen2.5-3B-Instruct --prompt "Your prompt"
"""

import torch
import sys
import os
import json
from pathlib import Path
from datetime import datetime
import argparse
from typing import Tuple, List, Dict

# Add original_implementation to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "original_implementation"))

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_model(model_name: str):
    """Load model and tokenizer."""
    print("Loading model: {}...".format(model_name))
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except ValueError as e:
        if "sentencepiece or tiktoken" in str(e):
            print("Using use_fast=False for tokenizer...")
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

    print("GPU Memory: {} MB".format(torch.cuda.memory_allocated() // 1024 // 1024))
    return model, tokenizer


def generate_text(model, tokenizer, prompt: str, max_length: int = 256,
                  temperature: float = 0.7, top_p: float = 0.9, seed: int = None) -> Tuple[str, str, List[int]]:
    """Generate text from prompt with optional seed."""
    if seed is not None:
        torch.manual_seed(seed)

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    input_ids = inputs["input_ids"]
    input_len = input_ids.shape[1]

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=False,
        )

    # Ensure generated_ids is on CPU for decoding
    if generated_ids.is_cuda:
        generated_ids = generated_ids.cpu()

    generated_ids_list = generated_ids[0].tolist() if generated_ids.dim() > 1 else generated_ids.tolist()
    generated_text = tokenizer.decode(generated_ids_list, skip_special_tokens=True)

    # Extract only the generated part (skip input)
    generated_only = tokenizer.decode(generated_ids_list[input_len:], skip_special_tokens=True)

    return generated_text, generated_only, generated_ids_list


def compute_similarity(text1: str, text2: str) -> Dict[str, float]:
    """Compute various similarity metrics."""
    tokens1 = text1.lower().split()
    tokens2 = text2.lower().split()

    # Exact match
    exact_match = 1.0 if text1 == text2 else 0.0

    # Token-level metrics
    if len(tokens1) == 0 or len(tokens2) == 0:
        jaccard = 0.0
        bleu_like = 0.0
    else:
        # Jaccard similarity
        set1 = set(tokens1)
        set2 = set(tokens2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        jaccard = intersection / union if union > 0 else 0.0

        # BLEU-like: how many tokens from text2 are in text1
        matches = sum(1 for t in tokens2 if t in set1)
        bleu_like = matches / len(tokens2)

    # Character-level similarity
    char1 = set(text1)
    char2 = set(text2)
    char_intersection = len(char1 & char2)
    char_union = len(char1 | char2)
    char_similarity = char_intersection / char_union if char_union > 0 else 0.0

    return {
        'exact_match': exact_match,
        'jaccard_similarity': jaccard,
        'bleu_like': bleu_like,
        'char_similarity': char_similarity,
    }


def run_test(model_name: str, prompt: str, max_length: int = 256,
             temperature: float = 0.7, output_file: str = None):
    """Run prompt test."""

    print("\n{}".format("=" * 80))
    print("Model: {}".format(model_name))
    print("{}".format("=" * 80))
    print("\nPrompt:\n{}".format(prompt))
    print("\n{}".format("=" * 80))

    # Load model
    model, tokenizer = load_model(model_name)

    # Generate 3 times to see if there's variance
    print("\n[GENERATION RUNS]")
    print("\nRun #1 (seed=42)...")
    text1_full, text1_gen, ids1 = generate_text(
        model, tokenizer, prompt, max_length=max_length,
        temperature=temperature, seed=42
    )
    print("[OK] Generated {} chars / {} tokens".format(len(text1_gen), len(ids1)))
    print("Output:\n{}\n".format(text1_gen[:200]))

    print("Run #2 (seed=42 - should be identical)...")
    text2_full, text2_gen, ids2 = generate_text(
        model, tokenizer, prompt, max_length=max_length,
        temperature=temperature, seed=42
    )
    print("[OK] Generated {} chars / {} tokens".format(len(text2_gen), len(ids2)))
    print("Output:\n{}\n".format(text2_gen[:200]))

    print("Run #3 (seed=None - randomized)...")
    text3_full, text3_gen, ids3 = generate_text(
        model, tokenizer, prompt, max_length=max_length,
        temperature=temperature, seed=None
    )
    print("[OK] Generated {} chars / {} tokens".format(len(text3_gen), len(ids3)))
    print("Output:\n{}\n".format(text3_gen[:200]))

    # Comparison
    print("\n{}".format("=" * 80))
    print("COMPARISON RESULTS")
    print("{}".format("=" * 80))

    print("\n--- Run #1 vs Run #2 (same seed - should be identical) ---")
    sim_1_2 = compute_similarity(text1_gen, text2_gen)
    print("Exact match: {}".format(sim_1_2['exact_match']))
    print("Jaccard similarity: {:.4f}".format(sim_1_2['jaccard_similarity']))
    print("BLEU-like: {:.4f}".format(sim_1_2['bleu_like']))
    print("Character similarity: {:.4f}".format(sim_1_2['char_similarity']))

    print("\n--- Run #1 vs Run #3 (different seeds - should differ) ---")
    sim_1_3 = compute_similarity(text1_gen, text3_gen)
    print("Exact match: {}".format(sim_1_3['exact_match']))
    print("Jaccard similarity: {:.4f}".format(sim_1_3['jaccard_similarity']))
    print("BLEU-like: {:.4f}".format(sim_1_3['bleu_like']))
    print("Character similarity: {:.4f}".format(sim_1_3['char_similarity']))

    print("\n--- Token-level analysis ---")
    tokens1 = text1_gen.split()
    tokens3 = text3_gen.split()
    matching = sum(1 for i, (t1, t3) in enumerate(zip(tokens1, tokens3)) if t1 == t3)
    min_len = min(len(tokens1), len(tokens3))
    if min_len > 0:
        token_match_rate = matching / min_len
        print("Token match rate (Run #1 vs #3): {:.2%}".format(token_match_rate))

    # Prepare output
    results = {
        'timestamp': datetime.now().isoformat(),
        'model': model_name,
        'prompt': prompt,
        'runs': [
            {
                'run': 1,
                'seed': 42,
                'text': text1_gen,
                'tokens_count': len(ids1),
            },
            {
                'run': 2,
                'seed': 42,
                'text': text2_gen,
                'tokens_count': len(ids2),
            },
            {
                'run': 3,
                'seed': None,
                'text': text3_gen,
                'tokens_count': len(ids3),
            },
        ],
        'comparisons': {
            'run1_vs_run2': sim_1_2,
            'run1_vs_run3': sim_1_3,
        },
    }

    # Save results
    if output_file is None:
        output_file = "results/simple_prompt_test_{}.json".format(
            datetime.now().strftime('%Y%m%d_%H%M%S')
        )

    os.makedirs("results", exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n[SAVED] Results: {}".format(output_file))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple prompt comparison test")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct",
                        help="Model name")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Prompt to test")
    parser.add_argument("--max-length", type=int, default=256,
                        help="Max generation length")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path")

    args = parser.parse_args()

    results = run_test(
        model_name=args.model,
        prompt=args.prompt,
        max_length=args.max_length,
        temperature=args.temperature,
        output_file=args.output
    )
