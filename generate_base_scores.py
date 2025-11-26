#!/usr/bin/env python3
"""
Generate base set scores with easy/hard difficulty splits.
This script processes base_set.{judge}.jsonl files and generates
comprehensive scores including easy/hard splits for the score visualizer.
"""

import click
import json
import os
import re
from pathlib import Path
from choix_analyzer import LLMRanker, load_comparisons_from_file


def generate_base_scores(judge_model: str, baseset_version: str):
    """Generate comprehensive base set scores including easy/hard splits."""
    safe_judge_name = judge_model.replace("/", "__")
    snapshot_dir = Path(baseset_version)
    base_file = snapshot_dir / f"base_set.{safe_judge_name}.jsonl"

    if not base_file.exists():
        print(f"Error: Base set file not found at {base_file}")
        return False

    print(f"Processing base set file: {base_file}")

    # Load comparisons
    comparisons = load_comparisons_from_file(base_file)
    if not comparisons:
        print("Error: No valid comparisons found")
        return False

    print(f"Loaded {len(comparisons)} comparisons")

    # Fit the ranker
    ranker = LLMRanker()
    ranker.fit(comparisons)

    # Generate scores for all slices
    all_scores = []

    # Slices to generate: overall, by language, by difficulty, and combined
    slices = [
        ("all", "all", "overall"),           # Overall
        ("all", "english", "overall"),       # EN->JA overall
        ("all", "japanese", "overall"),      # JA->EN overall
        ("easy", "all", "overall"),          # Easy overall
        ("hard", "all", "overall"),          # Hard overall
        ("easy", "english", "easy"),         # EN->JA easy
        ("hard", "english", "hard"),         # EN->JA hard
        ("easy", "japanese", "easy"),        # JA->EN easy
        ("hard", "japanese", "hard"),        # JA->EN hard
    ]

    for difficulty, language, slice_label in slices:
        try:
            rankings = ranker.get_rankings(difficulty, language)
        except ValueError as e:
            print(f"Warning: Could not generate rankings for {difficulty}/{language}: {e}")
            continue

        # Convert safe names back to display names
        rankings['llm'] = rankings['llm'].str.replace('__', '/')

        # Add each model's scores to the output
        for _, row in rankings.iterrows():
            all_scores.append({
                "slice": slice_label,
                "model": row['llm'],
                "safe_name": row['llm'].replace('/', '__'),
                "score": float(row['score']),
                "wins": int(row['wins']),
                "matches": int(row['total_matches']),
                "win_rate": float(row['wins']) / float(row['total_matches']) * 100 if row['total_matches'] > 0 else 0.0,
                "EN": float(row['EN']),
                "LT": float(row['LT']),
                "difficulty": difficulty,
                "language": language,
            })

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
def main(judge_model, baseset_version):
    """Generate base set scores with easy/hard difficulty splits."""
    success = generate_base_scores(judge_model, baseset_version)
    if not success:
        exit(1)


if __name__ == "__main__":
    main()
