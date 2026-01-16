#!/usr/bin/env python3
"""
Count tokens for JP-TL-Bench evaluation pairs using Google GenAI.

This script uses actual data from the baseset to calculate accurate
token counts for both input (prompt + formatted_data) and output (analysis).

Features:
- Caches results to avoid re-counting
- Supports resuming from interruption
- Saves progress after each batch
"""

import json
import os
from pathlib import Path
from statistics import mean, stdev

from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()


def load_compare_prompt(prompt_path: Path) -> str:
    """Load the compare prompt template."""
    return prompt_path.read_text()


def load_judgments(judgments_path: Path) -> list[dict]:
    """Load judgments from JSONL (contains both input and output)."""
    judgments = []
    with open(judgments_path) as f:
        for line in f:
            if line.strip():
                judgments.append(json.loads(line))
    return judgments


def load_cache(cache_path: Path) -> dict:
    """Load cached token counts."""
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)
    return {"input_tokens": {}, "output_tokens": {}}


def save_cache(cache_path: Path, cache: dict):
    """Save token counts to cache."""
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)


def count_tokens_for_text(client, model: str, text: str) -> int:
    """Count tokens for a given text."""
    content = types.Content(
        role="user",
        parts=[types.Part(text=text)]
    )
    result = client.models.count_tokens(model=model, contents=[content])
    return result.total_tokens


def get_judgment_key(judgment: dict, idx: int) -> str:
    """Generate a unique key for a judgment."""
    name = judgment.get("name", f"judgment_{idx}")
    # Include a hash of formatted_data for uniqueness
    data_hash = hash(judgment.get("formatted_data", "")[:100])
    return f"{name}_{idx}_{data_hash}"


def main():
    # Setup
    base_dir = Path(__file__).parent
    prompt_path = base_dir / "prompts" / "compare_prompt.txt"
    judgments_path = base_dir / "baseset" / "v1.0" / "base_set.gemini-2.5-flash.jsonl"
    cache_path = base_dir / "token_counts_cache.json"

    # Load data
    compare_prompt_template = load_compare_prompt(prompt_path)
    judgments = load_judgments(judgments_path)
    cache = load_cache(cache_path)

    # Initialize client
    client = genai.Client()
    model = "gemini-2.0-flash"  # Use for token counting

    print(f"Loaded {len(judgments)} judgments from baseset")
    print(f"Compare prompt template: {len(compare_prompt_template)} chars")
    print(f"Cache file: {cache_path}")
    print(f"Cached entries: {len(cache['input_tokens'])} input, {len(cache['output_tokens'])} output")
    print()

    # Count tokens for each judgment (both input and output)
    input_tokens = []
    output_tokens = []

    new_counts = 0
    cached_counts = 0
    batch_size = 50  # Save cache every N new counts

    print(f"Counting tokens for all {len(judgments)} judgments...")
    print("(Using cache where available, Ctrl+C to interrupt safely)")
    print()

    try:
        for i, judgment in enumerate(judgments):
            key = get_judgment_key(judgment, i)

            # Check cache first
            if key in cache["input_tokens"] and key in cache["output_tokens"]:
                input_tok = cache["input_tokens"][key]
                output_tok = cache["output_tokens"][key]
                cached_counts += 1
            else:
                # INPUT: Build the full prompt (same as what was sent to judge)
                formatted_data = judgment.get("formatted_data", "")
                full_prompt = compare_prompt_template.replace("{{formatted_data}}", formatted_data)

                # OUTPUT: Get the actual analysis response
                analysis = judgment.get("analysis", "")

                # Count tokens via API
                input_tok = count_tokens_for_text(client, model, full_prompt)
                output_tok = count_tokens_for_text(client, model, analysis)

                # Update cache
                cache["input_tokens"][key] = input_tok
                cache["output_tokens"][key] = output_tok
                new_counts += 1

                # Save cache periodically
                if new_counts % batch_size == 0:
                    save_cache(cache_path, cache)
                    print(f"  [Cache saved: {len(cache['input_tokens'])} entries]")

            input_tokens.append(input_tok)
            output_tokens.append(output_tok)

            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(judgments)} (new: {new_counts}, cached: {cached_counts})")

    except KeyboardInterrupt:
        print("\n\nInterrupted! Saving cache...")
        save_cache(cache_path, cache)
        print(f"Cache saved with {len(cache['input_tokens'])} entries.")
        print("Run again to resume from where you left off.")
        return

    # Final cache save
    save_cache(cache_path, cache)
    print(f"\nDone! New counts: {new_counts}, From cache: {cached_counts}")
    print()

    # Calculate statistics
    avg_input = mean(input_tokens)
    std_input = stdev(input_tokens) if len(input_tokens) > 1 else 0
    min_input = min(input_tokens)
    max_input = max(input_tokens)

    avg_output = mean(output_tokens)
    std_output = stdev(output_tokens) if len(output_tokens) > 1 else 0
    min_output = min(output_tokens)
    max_output = max(output_tokens)

    # Calculate costs for multiple models
    # Full evaluation: 1400 judgments = 70 items × 20 anchors
    num_judgments = 1400
    total_input_tokens = avg_input * num_judgments
    total_output_tokens = avg_output * num_judgments

    # Model pricing (as of late 2025)
    models = {
        "gemini-2.5-flash": {
            "input": 0.30,   # $/1M tokens
            "output": 2.50,  # $/1M tokens
            "note": "thinking_budget=0, no thinking token cost"
        },
        "gemini-3-flash": {
            "input": 0.50,   # $/1M tokens
            "output": 3.00,  # $/1M tokens
            "note": "thinking_level=MINIMAL (~0 thinking tokens)"
        }
    }

    # Calculate costs for each model
    model_costs = {}
    for model_name, pricing in models.items():
        per_judgment_input = (avg_input / 1_000_000) * pricing["input"]
        per_judgment_output = (avg_output / 1_000_000) * pricing["output"]
        per_judgment_total = per_judgment_input + per_judgment_output
        full_eval_cost = per_judgment_total * num_judgments
        model_costs[model_name] = {
            "per_judgment_input": per_judgment_input,
            "per_judgment_output": per_judgment_output,
            "per_judgment_total": per_judgment_total,
            "full_eval_cost": full_eval_cost,
            "pricing": pricing
        }

    judge_model = os.getenv("JUDGE_MODEL", judgments[0].get("judge_model", "gemini-2.5-flash"))
    judge_model_key = judge_model.split("/")[-1]
    if judge_model_key not in models:
        judge_model_key = "gemini-2.5-flash"

    default_model = judge_model_key
    default_pricing = models[default_model]
    default_costs = model_costs[default_model]
    default_note = default_pricing.get("note")

    input_price_per_1m = default_pricing["input"]
    output_price_per_1m = default_pricing["output"]
    cost_per_judgment_input = default_costs["per_judgment_input"]
    cost_per_judgment_output = default_costs["per_judgment_output"]
    cost_per_judgment = default_costs["per_judgment_total"]
    total_cost = default_costs["full_eval_cost"]

    # Print results
    print("=" * 60)
    print("JP-TL-Bench Token Count Analysis (Actual Data)")
    print("=" * 60)
    print()
    print("INPUT TOKENS (per judgment)")
    print("-" * 40)
    print(f"  (includes prompt + anchor translation + candidate translation)")
    print(f"  Average:  {avg_input:,.0f} tokens")
    print(f"  Std Dev:  {std_input:,.0f} tokens")
    print(f"  Min:      {min_input:,} tokens")
    print(f"  Max:      {max_input:,} tokens")
    print()
    print("OUTPUT TOKENS (per judgment) - ACTUAL")
    print("-" * 40)
    print(f"  Average:  {avg_output:,.0f} tokens")
    print(f"  Std Dev:  {std_output:,.0f} tokens")
    print(f"  Min:      {min_output:,} tokens")
    print(f"  Max:      {max_output:,} tokens")
    print()
    print(f"COST ESTIMATE ({default_model})")
    print("-" * 40)
    print(f"  Input:  ${input_price_per_1m:.2f}/1M tokens")
    print(f"  Output: ${output_price_per_1m:.2f}/1M tokens")
    if default_note:
        print(f"  ({default_note})")
    print()
    print("PER JUDGMENT")
    print(f"  Input tokens:   {avg_input:,.0f}")
    print(f"  Output tokens:  {avg_output:,.0f}")
    print(f"  Input cost:     ${cost_per_judgment_input:.6f}")
    print(f"  Output cost:    ${cost_per_judgment_output:.6f}")
    print(f"  Total:          ${cost_per_judgment:.6f}")
    print()
    print(f"FULL EVALUATION ({num_judgments:,} judgments = 70 items × 20 anchors)")
    print("-" * 40)
    print(f"  Total input tokens:  {total_input_tokens:,.0f}")
    print(f"  Total output tokens: {total_output_tokens:,.0f}")
    print(f"  Input cost:          ${(total_input_tokens / 1_000_000) * input_price_per_1m:.2f}")
    print(f"  Output cost:         ${(total_output_tokens / 1_000_000) * output_price_per_1m:.2f}")
    print(f"  TOTAL COST:          ${total_cost:.2f}")
    print()
    print("COST ESTIMATES (all models)")
    print("-" * 70)
    print(f"{'Model':<18} {'Input $/1M':>12} {'Output $/1M':>12} {'Per judgment':>14} {'Full eval':>12}")
    print("-" * 70)
    for model_name in models:
        cost = model_costs[model_name]
        pricing = cost["pricing"]
        print(
            f"{model_name:<18} {pricing['input']:>12.2f} {pricing['output']:>12.2f} "
            f"{cost['per_judgment_total']:>14.6f} {cost['full_eval_cost']:>12.2f}"
        )
    print("-" * 70)
    print()
    print("=" * 60)

    # Detailed breakdown table
    print()
    print("Sample Token Counts (first 30 judgments)")
    print("-" * 70)
    print(f"{'#':<4} {'Name':<25} {'Input':>12} {'Output':>12} {'Total':>12}")
    print("-" * 70)
    for i, (judgment, inp, out) in enumerate(zip(judgments[:30], input_tokens[:30], output_tokens[:30])):
        name = judgment.get("name", f"judgment_{i}")[:24]
        print(f"{i+1:<4} {name:<25} {inp:>12,} {out:>12,} {inp+out:>12,}")
    print("-" * 70)
    print(f"{'AVG':<4} {'(all ' + str(len(judgments)) + ' judgments)':<25} {avg_input:>12,.0f} {avg_output:>12,.0f} {avg_input+avg_output:>12,.0f}")

    # Save final results to JSON for reference
    results_path = base_dir / "token_counts_results.json"
    results = {
        "num_judgments_analyzed": len(judgments),
        "input_tokens": {
            "average": avg_input,
            "std_dev": std_input,
            "min": min_input,
            "max": max_input
        },
        "output_tokens": {
            "average": avg_output,
            "std_dev": std_output,
            "min": min_output,
            "max": max_output
        },
        "cost_estimate": {
            "model": default_model,
            "input_price_per_1m": input_price_per_1m,
            "output_price_per_1m": output_price_per_1m,
            "note": default_note,
            "per_judgment": cost_per_judgment,
            "per_judgment_input": cost_per_judgment_input,
            "per_judgment_output": cost_per_judgment_output,
            "full_evaluation_1400": total_cost
        },
        "cost_estimates_by_model": model_costs
    }
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
