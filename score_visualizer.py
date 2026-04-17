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
from artifact_metadata import load_score_with_sidecar
from baseset.legacy_boundary import legacy_candidate_paths
from benchmark_tasks import load_task_config


def extract_model_name(full_name: str) -> str:
    return full_name


def is_tested_model(model_name: str) -> bool:
    safe = model_name.replace("/", "__") + ".jsonl"
    return (Path("translations") / safe).exists()


def load_score_file(filepath: str) -> Dict:
    return load_score_with_sidecar(filepath)


def canonical_anchor_slice(row: Dict, task_config) -> str | None:
    slice_label = row.get("slice") or row.get("Slice")
    if not slice_label:
        return None

    difficulty = (row.get("difficulty") or row.get("Difficulty") or "all").lower()
    language = row.get("language") or row.get("Language")
    if slice_label in {"overall", "easy", "hard"} and language and str(language).lower() != "all":
        language_text = str(language)
        try:
            direction = task_config.direction_by_key(language_text)
        except ValueError:
            try:
                direction = task_config.direction_for_record({"language": language_text})
            except ValueError:
                direction = None
        if direction is not None:
            if slice_label == "overall" or difficulty == "all":
                return direction.key
            if difficulty in {"easy", "hard"}:
                return f"{direction.key}_{difficulty}"

    valid_slices = {"overall"}
    for direction_key in task_config.scoring_direction_order:
        valid_slices.add(direction_key)
        valid_slices.add(f"{direction_key}_easy")
        valid_slices.add(f"{direction_key}_hard")
    if slice_label in valid_slices:
        return slice_label

    if slice_label not in {"overall", "easy", "hard"} or not language or str(language).lower() == "all":
        return slice_label

    language_text = str(language)
    try:
        direction = task_config.direction_by_key(language_text)
    except ValueError:
        try:
            direction = task_config.direction_for_record({"language": language_text})
        except ValueError:
            return slice_label

    if slice_label == "overall" or difficulty == "all":
        return direction.key
    if difficulty not in {"easy", "hard"}:
        return direction.key
    return f"{direction.key}_{difficulty}"


def load_base_anchor_scores(baseset_version: str, judge_model: str, task=None) -> Dict[str, Dict]:
    """Load base anchor scores if available, keyed by model and slice label."""
    task_config = load_task_config(task)
    judge_safe = judge_model.replace("/", "__")
    legacy_report = Path(f"baseset/{baseset_version}/reports/base_set.{judge_safe}_scores.json")
    report = None
    for candidate in legacy_candidate_paths(legacy_report, Path(f"baseset/{baseset_version}")):
        if candidate.exists():
            report = candidate
            break
    if report is None:
        return {}

    try:
        data = json.load(report.open())
        # Build a lookup structure: model -> slice -> stats
        model_stats: Dict[str, Dict[str, Dict]] = {}
        rows = data if isinstance(data, list) else []
        for row in rows:
            model = row.get("model") or row.get("Model") or row.get("llm")
            slice_label = canonical_anchor_slice(row, task_config)
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


def extract_candidate_direction_rows(summary: Dict, task_config=None, task=None, missing_ratio: float = 0.0) -> Dict[str, Dict]:
    if task_config is None:
        task_config = load_task_config(task)
    model_name = summary.get("model")
    rows: Dict[str, Dict] = {}
    for direction_key in task_config.scoring_direction_order:
        direction_summary = summary.get(direction_key, {}) or {}
        overall = direction_summary.get("overall")
        if not overall:
            continue
        rows[direction_key] = {
            "model": model_name,
            "easy_lt": (direction_summary.get("easy") or {}).get("lt"),
            "hard_lt": (direction_summary.get("hard") or {}).get("lt"),
            "overall_lt": overall.get("lt", 0),
            "wins": overall.get("wins", 0),
            "total_matches": overall.get("total", 0),
            "is_base": False,
            "missing_ratio": missing_ratio,
        }
    return rows


def build_anchor_direction_row(model: str, slices: Dict[str, Dict], direction_key: str) -> Dict | None:
    overall = slices.get(direction_key)
    if not overall:
        return None
    overall_lt = overall.get("lt")
    return {
        "model": model,
        "easy_lt": (slices.get(f"{direction_key}_easy") or {}).get("lt"),
        "hard_lt": (slices.get(f"{direction_key}_hard") or {}).get("lt"),
        "overall_lt": overall_lt if overall_lt is not None else 0,
        "wins": overall.get("wins", 0),
        "total_matches": overall.get("matches", 0),
        "is_base": True,
        "missing_ratio": 0.0,
    }


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
@click.option('--task', envvar='TASK_CONFIG', help='Task config path or name under benchmark_tasks/.')
def main(path: str, filter: str, top: int, baseset_version: str, judge: str, task: str):
    """Display translation benchmark scores from canonical results files."""
    console = Console()
    task_config = load_task_config(task)
    if not baseset_version:
        baseset_version = os.getenv("BASESET_VERSION") or Path(os.getenv("BASESET_SNAPSHOT_DIR", "baseset/v1.0")).name
    if not judge:
        judge = os.getenv("DEFAULT_JUDGE", "gemini-2.5-flash")

    score_files = glob.glob(f"{path}/**/scores.json", recursive=True)
    if not score_files:
        console.print(f"[red]No score files found under {path}[/red]")
        return

    direction_rows = {direction_key: [] for direction_key in task_config.scoring_direction_order}
    skipped = 0

    # Optional base anchors
    anchor_rows = load_base_anchor_scores(baseset_version, judge, task=task) if judge else {}

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

        for direction_key, row in extract_candidate_direction_rows(
            data,
            task_config=task_config,
            missing_ratio=missing_ratio if expected_pairs else 0,
        ).items():
            direction_rows[direction_key].append(row)

    # Add base anchors to all directions if present
    for model, slices in anchor_rows.items():
        for direction_key in task_config.scoring_direction_order:
            row = build_anchor_direction_row(model, slices, direction_key)
            if row:
                direction_rows[direction_key].append(row)

    if not any(direction_rows.values()):
        console.print("[red]No valid model data found[/red]")
        return

    displayed_counts = []
    for direction_key in task_config.scoring_direction_order:
        rows = direction_rows[direction_key]
        rows.sort(key=lambda x: x['overall_lt'], reverse=True)
        if top:
            rows = rows[:top]
        direction = task_config.direction_by_key(direction_key)
        console.print()
        console.print(create_direction_table(rows, f"{direction.display_name} Translation", console))
        displayed_counts.append(f"{direction.display_name}: {len(rows)}")

    console.print(f"\n[dim]Displayed {'; '.join(displayed_counts)} (skipped {skipped})[/dim]")


if __name__ == "__main__":
    main()
