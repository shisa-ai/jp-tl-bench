#!/usr/bin/env python3
"""
Generate base set scores with easy/hard difficulty splits.
This script processes base_set.{judge}.jsonl files and generates
comprehensive scores including easy/hard splits for the score visualizer.
"""

import click
import json
from pathlib import Path
from choix_analyzer import LLMRanker, build_ranked_slice_rows, load_comparisons_from_file


def generate_base_scores(judge_model: str, baseset_version: str, task=None):
    """Generate comprehensive base set scores including easy/hard splits."""
    safe_judge_name = judge_model.replace("/", "__")
    snapshot_dir = Path(baseset_version)
    base_file = snapshot_dir / f"base_set.{safe_judge_name}.jsonl"

    if not base_file.exists():
        print(f"Error: Base set file not found at {base_file}")
        return False

    print(f"Processing base set file: {base_file}")

    # Load comparisons
    comparisons = load_comparisons_from_file(base_file, task=task)
    if not comparisons:
        print("Error: No valid comparisons found")
        return False

    print(f"Loaded {len(comparisons)} comparisons")

    # Fit the ranker
    ranker = LLMRanker(task=task)
    ranker.fit(comparisons)

    all_scores = build_ranked_slice_rows(ranker, task=task)

    # Write output
    reports_dir = snapshot_dir / "reports"
    reports_dir.mkdir(exist_ok=True)
    output_file = reports_dir / f"base_set.{safe_judge_name}_scores.json"

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_scores, f, ensure_ascii=False, indent=2)

    print(f"\nGenerated {len(all_scores)} score entries")
    print(f"Scores saved to: {output_file}")

    # Print summary
    models = set(s['model'] for s in all_scores if s['difficulty'] == 'all' and s['language'] == 'all')
    print(f"Models in base set: {len(models)}")

    return True


@click.command()
@click.option('--judge-model', '-j', default='gemini-2.5-flash', help='Judge model name')
@click.option('--baseset-version', default='baseset/v1.0', help='Path to baseset snapshot directory')
@click.option('--task', envvar='TASK_CONFIG', help='Task config path or name under benchmark_tasks/.')
def main(judge_model, baseset_version, task):
    """Generate base set scores with easy/hard difficulty splits."""
    success = generate_base_scores(judge_model, baseset_version, task=task)
    if not success:
        exit(1)


if __name__ == "__main__":
    main()
