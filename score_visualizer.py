#!/usr/bin/env python3
"""
Compact score visualizer for translation benchmark results.
Displays EN->JA and JA->EN scores in separate rich tables with Easy/Hard/Overall LT scores.
"""

import json
import glob
from typing import Dict, List, Tuple
import click
from rich.console import Console
from rich.table import Table
from rich.text import Text

def load_scores_from_file(filepath: str) -> Dict:
    """Load scores from a single JSONL file and organize by difficulty/language."""
    scores = {}

    with open(filepath, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            difficulty = data['difficulty']
            language = data['language']

            if difficulty not in scores:
                scores[difficulty] = {}
            if language not in scores[difficulty]:
                scores[difficulty][language] = {}

            scores[difficulty][language] = {
                'llm': data['llm'],
                'score': data['score'],
                'LT': data['LT'],
                'EN': data['EN'],
                'wins': data['wins'],
                'total_matches': data['total_matches']
            }

    return scores

def extract_model_name(full_name: str) -> str:
    """Extract a clean model name from the full LLM identifier."""
    # Keep the full name since we'll use overflow=fold
    return full_name

def is_tested_model(model_name: str) -> bool:
    """Check if this is one of the tested models (has translations in translations/ dir)."""
    import os
    # Convert model name to filename format
    filename = model_name.replace('/', '__') + '.jsonl'
    return os.path.exists(f'translations/{filename}')

def create_direction_table(scores_data: List[Tuple], title: str, console: Console) -> Table:
    """Create a rich table for either EN->JA or JA->EN direction."""
    table = Table(title=title, show_header=True, header_style="bold magenta")

    table.add_column("Model", style="cyan", no_wrap=False, max_width=80, overflow="fold")
    table.add_column("Easy LT", justify="right", style="green", min_width=8)
    table.add_column("Hard LT", justify="right", style="yellow", min_width=8)
    table.add_column("Overall LT", justify="right", style="bold red", min_width=10)
    table.add_column("Win Rate", justify="right", style="blue", min_width=8)

    for model_data in scores_data:
        model_name = extract_model_name(model_data['model'])
        easy_lt = f"{model_data.get('easy_lt', 0):.2f}" if model_data.get('easy_lt') is not None else "N/A"
        hard_lt = f"{model_data.get('hard_lt', 0):.2f}" if model_data.get('hard_lt') is not None else "N/A"
        overall_lt = f"{model_data['overall_lt']:.2f}"

        wins = model_data['wins']
        total = model_data['total_matches']
        win_rate = f"{wins/total*100:.1f}%" if total > 0 else "N/A"

        # Highlight tested models with bright magenta/pink
        if is_tested_model(model_data['model']):
            model_style = "bright_magenta"
            model_name = f"[{model_style}]{model_name}[/{model_style}]"

        table.add_row(model_name, easy_lt, hard_lt, overall_lt, win_rate)

    return table

@click.command()
@click.option('--path', '-p', default='scores/', help='Path to scores directory')
@click.option('--filter', '-f', default='', help='Filter models by name substring')
@click.option('--top', '-t', type=int, help='Show only top N models')
def main(path: str, filter: str, top: int):
    """Display translation benchmark scores in compact rich tables."""
    console = Console()

    # Find all score files
    score_files = glob.glob(f"{path}/*_tl_bench_scores.jsonl")

    if not score_files:
        console.print(f"[red]No score files found in {path}[/red]")
        return

    # Collect all model data from all files
    all_models = {}

    for file_path in score_files:
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    data = json.loads(line.strip())
                    model_name = data['llm']
                    difficulty = data['difficulty']
                    language = data['language']

                    # Apply filter if specified
                    if filter and filter.lower() not in model_name.lower():
                        continue

                    # Initialize model entry if not exists
                    if model_name not in all_models:
                        all_models[model_name] = {
                            'model': model_name,
                            'en_ja': {},  # EN->JA (english language in data)
                            'ja_en': {},  # JA->EN (japanese language in data)
                            'overall_lt': 0,
                            'wins': 0,
                            'total_matches': 0
                        }

                    # Store overall scores (all difficulty, all language)
                    if difficulty == 'all' and language == 'all':
                        all_models[model_name]['overall_lt'] = data['LT']
                        all_models[model_name]['wins'] = data['wins']
                        all_models[model_name]['total_matches'] = data['total_matches']

                    # EN->JA scores (english language entries)
                    elif language == 'english':
                        if difficulty == 'all':
                            all_models[model_name]['en_ja']['overall_lt'] = data['LT']
                            all_models[model_name]['en_ja']['wins'] = data['wins']
                            all_models[model_name]['en_ja']['total_matches'] = data['total_matches']
                        elif difficulty == 'easy':
                            all_models[model_name]['en_ja']['easy_lt'] = data['LT']
                        elif difficulty == 'hard':
                            all_models[model_name]['en_ja']['hard_lt'] = data['LT']

                    # JA->EN scores (japanese language entries)
                    elif language == 'japanese':
                        if difficulty == 'all':
                            all_models[model_name]['ja_en']['overall_lt'] = data['LT']
                            all_models[model_name]['ja_en']['wins'] = data['wins']
                            all_models[model_name]['ja_en']['total_matches'] = data['total_matches']
                        elif difficulty == 'easy':
                            all_models[model_name]['ja_en']['easy_lt'] = data['LT']
                        elif difficulty == 'hard':
                            all_models[model_name]['ja_en']['hard_lt'] = data['LT']

        except Exception as e:
            console.print(f"[yellow]Warning: Could not process {file_path}: {e}[/yellow]")
            continue

    if not all_models:
        console.print("[red]No valid model data found[/red]")
        return

    # Sort by overall LT score (descending)
    sorted_models = sorted(all_models.values(), key=lambda x: x.get('overall_lt', 0), reverse=True)

    # Apply top limit if specified
    if top:
        sorted_models = sorted_models[:top]

    # Prepare data for EN->JA table
    en_ja_data = []
    for model in sorted_models:
        if 'en_ja' in model and 'overall_lt' in model['en_ja']:
            row_data = {
                'model': model['model'],
                'easy_lt': model['en_ja'].get('easy_lt'),
                'hard_lt': model['en_ja'].get('hard_lt'),
                'overall_lt': model['en_ja']['overall_lt'],
                'wins': model['en_ja']['wins'],
                'total_matches': model['en_ja']['total_matches']
            }
            en_ja_data.append(row_data)

    # Prepare data for JA->EN table
    ja_en_data = []
    for model in sorted_models:
        if 'ja_en' in model and 'overall_lt' in model['ja_en']:
            row_data = {
                'model': model['model'],
                'easy_lt': model['ja_en'].get('easy_lt'),
                'hard_lt': model['ja_en'].get('hard_lt'),
                'overall_lt': model['ja_en']['overall_lt'],
                'wins': model['ja_en']['wins'],
                'total_matches': model['ja_en']['total_matches']
            }
            ja_en_data.append(row_data)

    # Sort each direction by its own overall LT score
    en_ja_data.sort(key=lambda x: x['overall_lt'], reverse=True)
    ja_en_data.sort(key=lambda x: x['overall_lt'], reverse=True)

    # Display tables
    console.print()
    en_ja_table = create_direction_table(en_ja_data, "🇺🇸 → 🇯🇵 English to Japanese Translation", console)
    console.print(en_ja_table)

    console.print()
    ja_en_table = create_direction_table(ja_en_data, "🇯🇵 → 🇺🇸 Japanese to English Translation", console)
    console.print(ja_en_table)

    console.print(f"\n[dim]Found {len(all_models)} models total[/dim]")

if __name__ == "__main__":
    main()
