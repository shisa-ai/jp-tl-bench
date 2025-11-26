#!/usr/bin/env python3
"""
Canonical score visualizer for JP TL Bench.
- Reads canonical scores.json files under results/<base>/<model>/<judge>/.
- Shows base-set anchors (teal) alongside test models.
- Allows small gaps: only skips if >20% pairs missing; otherwise shows with a missing% column.
"""

import json
import glob
import os
from pathlib import Path
from typing import Dict, List

import click
from rich.console import Console
from rich.table import Table


def extract_model_name(full_name: str) -> str:
    return full_name


def is_tested_model(model_name: str) -> bool:
    safe = model_name.replace("/", "__") + ".jsonl"
    return (Path("translations") / safe).exists()


def load_score_file(filepath: str) -> Dict:
    with open(filepath, "r") as f:
        return json.load(f)


def load_base_anchor_scores(baseset_version: str, judge_model: str) -> Dict[str, Dict]:
    """Load base anchor scores if available, keyed by model and slice label."""
    judge_safe = judge_model.replace("/", "__")
    report = Path(f"baseset/{baseset_version}/reports/base_set.{judge_safe}_scores.json")
    if not report.exists():
        return {}

    try:
        data = json.load(report.open())
        # Build a lookup structure: model -> slice -> stats
        model_stats: Dict[str, Dict[str, Dict]] = {}
        rows = data if isinstance(data, list) else []
        for row in rows:
            model = row.get("model") or row.get("Model") or row.get("llm")
            slice_label = row.get("slice") or row.get("Slice")
            if not model or not slice_label:
                continue

            lt_val = row.get("LT")
            if lt_val is None:
                lt_val = row.get("lt")

            wins_val = row.get("wins")
            if wins_val is None:
                wins_val = row.get("Wins")

            matches_val = row.get("matches")
            if matches_val is None:
                matches_val = row.get("Matches") or row.get("total_matches") or row.get("total")

            if model not in model_stats:
                model_stats[model] = {}
            model_stats[model][slice_label] = {
                "lt": float(lt_val) if lt_val is not None else None,
                "wins": int(wins_val) if wins_val is not None else 0,
                "matches": int(matches_val) if matches_val is not None else 0,
            }

        return model_stats
    except Exception:
        return {}


def create_direction_table(scores_data: List[Dict], title: str, console: Console) -> Table:
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

        # teal for base anchors, magenta for tested translations
        if model_data.get("is_base"):
            model_style = "bright_cyan"
        elif is_tested_model(model_data['model']):
            model_style = "bright_magenta"
        else:
            model_style = None

        if model_style:
            model_name = f"[{model_style}]{model_name}[/{model_style}]"

        table.add_row(model_name, easy_lt, hard_lt, overall_lt, win_rate)

    return table


@click.command()
@click.option('--path', '-p', default='results/', help='Path to results root (default results/)')
@click.option('--filter', '-f', default='', help='Filter models by name substring')
@click.option('--top', '-t', type=int, help='Show only top N models per direction')
@click.option('--baseset-version', default=None, help='Baseset version to display (default v1.0 or BASESET_SNAPSHOT_DIR name)')
@click.option('--judge', default=None, help='Judge to display (safe name or raw). Defaults to gemini-2.5-flash.')
def main(path: str, filter: str, top: int, baseset_version: str, judge: str):
    """Display translation benchmark scores from canonical results files."""
    console = Console()
    if not baseset_version:
        baseset_version = os.getenv("BASESET_VERSION") or Path(os.getenv("BASESET_SNAPSHOT_DIR", "baseset/v1.0")).name
    if not judge:
        judge = os.getenv("DEFAULT_JUDGE", "gemini-2.5-flash")

    score_files = glob.glob(f"{path}/**/scores.json", recursive=True)
    if not score_files:
        console.print(f"[red]No score files found under {path}[/red]")
        return

    en_ja_data = []
    ja_en_data = []
    skipped = 0

    # Optional base anchors
    anchor_rows = load_base_anchor_scores(baseset_version, judge) if judge else {}

    for file_path in score_files:
        try:
            data = load_score_file(file_path)
        except Exception as e:
            console.print(f"[yellow]Warning: Could not process {file_path}: {e}[/yellow]")
            continue

        if data.get("baseset_version") != baseset_version:
            skipped += 1
            continue

        if judge:
            safe_j = judge.replace("/", "__")
            if safe_j not in file_path and judge not in file_path:
                skipped += 1
                continue

        missing_pairs = data.get("missing_pairs")
        expected_pairs = data.get("expected_pairs") or 0
        missing_ratio = (missing_pairs / expected_pairs) if expected_pairs else 0
        if missing_pairs is not None and missing_ratio > 0.20:
            console.print(f"[yellow]Skipping {file_path}: missing_pairs={missing_pairs} ({missing_ratio:.1%})[/yellow]")
            continue

        model_name = data.get("model")
        if filter and filter.lower() not in (model_name or "").lower():
            continue

        enja = data.get("en_ja", {}) or {}
        jaen = data.get("ja_en", {}) or {}
        if enja.get("overall"):
            en_ja_data.append({
                "model": model_name,
                "easy_lt": enja.get("easy", {}).get("lt"),
                "hard_lt": enja.get("hard", {}).get("lt"),
                "overall_lt": enja.get("overall", {}).get("lt", 0),
                "wins": enja.get("overall", {}).get("wins", 0),
                "total_matches": enja.get("overall", {}).get("total", 0),
                "is_base": False,
                "missing_ratio": missing_ratio if expected_pairs else 0,
            })
        if jaen.get("overall"):
            ja_en_data.append({
                "model": model_name,
                "easy_lt": jaen.get("easy", {}).get("lt"),
                "hard_lt": jaen.get("hard", {}).get("lt"),
                "overall_lt": jaen.get("overall", {}).get("lt", 0),
                "wins": jaen.get("overall", {}).get("wins", 0),
                "total_matches": jaen.get("overall", {}).get("total", 0),
                "is_base": False,
                "missing_ratio": missing_ratio if expected_pairs else 0,
            })

    # Add base anchors to both directions if present
    for model, slices in anchor_rows.items():
        en_overall = slices.get("en_ja")
        ja_overall = slices.get("ja_en")

        if en_overall:
            en_overall_lt = en_overall.get("lt")
            en_ja_data.append({
                "model": model,
                "easy_lt": (slices.get("en_ja_easy") or {}).get("lt"),
                "hard_lt": (slices.get("en_ja_hard") or {}).get("lt"),
                "overall_lt": en_overall_lt if en_overall_lt is not None else 0,
                "wins": en_overall.get("wins", 0),
                "total_matches": en_overall.get("matches", 0),
                "is_base": True,
                "missing_ratio": 0.0,
            })

        if ja_overall:
            ja_overall_lt = ja_overall.get("lt")
            ja_en_data.append({
                "model": model,
                "easy_lt": (slices.get("ja_en_easy") or {}).get("lt"),
                "hard_lt": (slices.get("ja_en_hard") or {}).get("lt"),
                "overall_lt": ja_overall_lt if ja_overall_lt is not None else 0,
                "wins": ja_overall.get("wins", 0),
                "total_matches": ja_overall.get("matches", 0),
                "is_base": True,
                "missing_ratio": 0.0,
            })

    if not en_ja_data and not ja_en_data:
        console.print("[red]No valid model data found[/red]")
        return

    en_ja_data.sort(key=lambda x: x['overall_lt'], reverse=True)
    ja_en_data.sort(key=lambda x: x['overall_lt'], reverse=True)

    if top:
        en_ja_data = en_ja_data[:top]
        ja_en_data = ja_en_data[:top]

    console.print()
    en_ja_table = create_direction_table(en_ja_data, "🇺🇸 → 🇯🇵 English to Japanese Translation", console)
    console.print(en_ja_table)

    console.print()
    ja_en_table = create_direction_table(ja_en_data, "🇯🇵 → 🇺🇸 Japanese to English Translation", console)
    console.print(ja_en_table)

    console.print(f"\n[dim]Displayed {len(en_ja_data)} EN→JA and {len(ja_en_data)} JA→EN models (skipped {skipped})[/dim]")


if __name__ == "__main__":
    main()
