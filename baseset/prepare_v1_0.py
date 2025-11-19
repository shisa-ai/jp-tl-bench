#!/usr/bin/env python3
"""
Helper for preparing, building, and validating the base-set v1.0 snapshot from prior runs.

Pipeline:
1. Read baseset manifest and copy each translation file into baseset/v1.0/translations/.
2. Generate every pairwise comparison JSONL using the copied translations.
3. Emit coverage stats.
4. Locate an existing judged base_set file (or a user-supplied one) and print/save the LT table.

The script never overwrites repo-level files outside baseset/v1.0/.
"""

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from io import StringIO
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import click

REPO_ROOT = Path(__file__).resolve().parents[1]
BASESET_DIR = Path(__file__).resolve().parent
SNAPSHOT_DIR = BASESET_DIR / "v1.0"
TRANSLATION_DIR = SNAPSHOT_DIR / "translations"
ARTIFACT_DIR = SNAPSHOT_DIR
REPORT_DIR = SNAPSHOT_DIR / "reports"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from choix_analyzer import LLMRanker, load_comparisons_from_file  # type: ignore  # noqa: E402


def safe_name(model: str) -> str:
    return model.replace("/", "__")


def load_manifest(path: Path) -> List[dict]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if "models" not in data:
        raise click.ClickException(f"Manifest missing 'models' list: {path}")
    return data["models"]


def copy_translations(entries: List[dict]) -> List[str]:
    TRANSLATION_DIR.mkdir(parents=True, exist_ok=True)
    copied = []
    click.echo("\n[1/4] Copying translation dumps into baseset/v1.0/translations/")
    for entry in entries:
        model = entry["model"]
        source = entry.get("source")
        if not source:
            click.echo(f"  - Skipping {model}: no 'source' path in manifest.", err=True)
            continue
        src_path = (REPO_ROOT / source).resolve()
        dest_path = (TRANSLATION_DIR / f"{safe_name(model)}.jsonl").resolve()
        if not src_path.exists():
            click.echo(f"  - MISSING {model}: {src_path}", err=True)
            continue
        shutil.copy2(src_path, dest_path)
        copied.append(dest_path)
        click.echo(f"  - Copied {src_path.relative_to(REPO_ROOT)} -> baseset/v1.0/translations/{dest_path.name}")
    if not copied:
        raise click.ClickException("No translation files were copied. Check manifest sources.")
    return [path.name for path in copied]


def load_jsonl(path: Path) -> List[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def format_translation_pair(conv_a: dict, conv_b: dict) -> str:
    buf = StringIO()
    buf.write(f"## Name: {conv_a.get('name', 'Unnamed')}\n\n")
    buf.write("## Source Text:\n")
    buf.write(f"{conv_a.get('source_text', '')}\n\n")
    buf.write("## Translation A\n")
    buf.write(f"{conv_a.get('translation', '')}\n\n")
    buf.write("## Translation B\n")
    buf.write(f"{conv_b.get('translation', '')}\n\n")
    buf.write("---\n")
    return buf.getvalue()


def ensure_alignment(convs_a: List[dict], convs_b: List[dict], file_a: str, file_b: str) -> Iterable[Tuple[dict, dict]]:
    if len(convs_a) != len(convs_b):
        raise ValueError(f"File length mismatch: {file_a} ({len(convs_a)}) vs {file_b} ({len(convs_b)})")
    dict_a = {item["name"]: item for item in convs_a}
    dict_b = {item["name"]: item for item in convs_b}
    names_a, names_b = set(dict_a), set(dict_b)
    if names_a != names_b:
        missing_in_a = sorted(names_b - names_a)
        missing_in_b = sorted(names_a - names_b)
        raise ValueError(
            f"Name mismatch between {file_a} and {file_b}.\n"
            f"    Missing in {file_a}: {missing_in_a or 'none'}\n"
            f"    Missing in {file_b}: {missing_in_b or 'none'}"
        )
    for name in sorted(names_a):
        yield dict_a[name], dict_b[name]


def write_pair_settings(file_a: str, file_b: str, example_name: str) -> Dict[str, str]:
    return {
        "id": hashlib.md5(f"{file_a}_{file_b}_{example_name}".encode()).hexdigest(),
        "llm_a": file_a,
        "llm_b": file_b,
    }


def generate_pairs(outputs: List[str]) -> Path:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    pair_path = ARTIFACT_DIR / "base_conversation_pairs.v1.0.jsonl"
    click.echo("\n[2/4] Generating pairwise comparison file")
    safe_names = sorted([Path(name).stem for name in outputs])
    translations = {safe: load_jsonl(TRANSLATION_DIR / f"{safe}.jsonl") for safe in safe_names}
    total_pairs = 0
    with pair_path.open("w", encoding="utf-8") as handle:
        for idx, safe_a in enumerate(safe_names):
            for safe_b in safe_names[idx + 1 :]:
                for conv_a, conv_b in ensure_alignment(translations[safe_a], translations[safe_b], safe_a, safe_b):
                    settings = write_pair_settings(safe_a, safe_b, conv_a["name"])
                    payload = {
                        "id": settings["id"],
                        "llm_a": settings["llm_a"],
                        "llm_b": settings["llm_b"],
                        "formatted_data": format_translation_pair(conv_a, conv_b),
                        "name": conv_a["name"],
                        "english": conv_a["english"],
                        "difficulty": conv_a["difficulty"],
                    }
                    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    total_pairs += 1
    click.echo(f"  - Wrote {total_pairs:,} rows to {pair_path}")
    return pair_path


def summarize_pairs(pair_file: Path, manifest_models: Dict[str, str]) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "pair_coverage.json"
    stats = {safe: {"matches": 0, "opponents": set()} for safe in manifest_models}
    with pair_file.open(encoding="utf-8") as handle:
        for line in handle:
            data = json.loads(line)
            a, b = data["llm_a"], data["llm_b"]
            if a not in stats or b not in stats:
                continue
            stats[a]["matches"] += 1
            stats[b]["matches"] += 1
            stats[a]["opponents"].add(b)
            stats[b]["opponents"].add(a)
    payload = []
    for safe, info in stats.items():
        payload.append(
            {
                "model": manifest_models[safe],
                "safe_name": safe,
                "matches": info["matches"],
                "unique_opponents": len(info["opponents"]),
            }
        )
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    click.echo("\n[3/4] Coverage summary")
    for row in payload:
        click.echo(
            f"  - {row['model']}: {row['matches']} matches vs {row['unique_opponents']} opponents"
        )
    click.echo(f"  - Saved JSON report to {report_path}")
    return report_path


def resolve_analysis_file(explicit: str, judge_model: str) -> Optional[Path]:
    if explicit:
        candidate = Path(explicit)
        if not candidate.is_absolute():
            base_candidate = BASESET_DIR / candidate
            repo_candidate = REPO_ROOT / candidate
            candidate = base_candidate if base_candidate.exists() else repo_candidate
        return candidate.resolve()

    safe_judge = safe_name(judge_model)
    preferred = [
        ARTIFACT_DIR / f"base_set.{safe_judge}.jsonl",
        REPO_ROOT / "base_sets" / f"base_set.{safe_judge}.jsonl",
    ]
    for path in preferred:
        if path.exists():
            return path
    candidates = sorted((REPO_ROOT / "base_sets").glob("base_set.*.jsonl"))
    return candidates[0].resolve() if candidates else None


def collect_present_models(analysis_file: Optional[Path]) -> set:
    present = set()
    if not analysis_file or not analysis_file.exists():
        return present
    with analysis_file.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            present.update({data.get("llm_a"), data.get("llm_b")})
    present.discard(None)
    return present


ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE)


def find_missing_answer_ids(analysis_file: Optional[Path]) -> List[str]:
    if not analysis_file or not analysis_file.exists():
        return []
    missing = []
    with analysis_file.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            analysis = data.get("analysis", "")
            if not ANSWER_RE.search(analysis or ""):
                missing.append(data.get("id"))
    return [m for m in missing if m]


def run_auto_judge(
    pair_file: Path,
    judge_model: str,
    base_url: str,
    api_key_env: str,
    max_workers: int,
    concurrency_limit: int,
) -> Path:
    click.echo("\n[3b] Running translation comparer to fill missing judges")
    repo_pairs = REPO_ROOT / "base_conversation_pairs.jsonl"
    backup = None
    if repo_pairs.exists():
        backup = repo_pairs.with_suffix(".setgen.bak")
        shutil.move(repo_pairs, backup)
    shutil.copy2(pair_file, repo_pairs)
    try:
        cmd = [
            sys.executable,
            "translation_comparer_any_model.py",
            "--base-url",
            base_url,
            "--judge-model",
            judge_model,
            "--generate-base-set",
            "--max-workers",
            str(max_workers),
            "--concurrency-limit",
            str(concurrency_limit),
            "--api-key-env",
            api_key_env,
        ]
        click.echo(f"  - Executing: {' '.join(cmd)}")
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    finally:
        repo_pairs.unlink(missing_ok=True)
        if backup:
            shutil.move(backup, repo_pairs)

    safe_judge = safe_name(judge_model)
    base_file = REPO_ROOT / "base_sets" / f"base_set.{safe_judge}.jsonl"
    if not base_file.exists():
        raise click.ClickException(f"translation comparer finished but {base_file} was not created.")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    dest = ARTIFACT_DIR / base_file.name
    shutil.copy2(base_file, dest)
    click.echo(f"  - Copied {base_file.relative_to(REPO_ROOT)} -> {dest.relative_to(BASESET_DIR)}")
    return dest


def build_rows(ranker: LLMRanker, manifest_models: Dict[str, str], difficulty: str = "all", language: Optional[str] = None) -> List[dict]:
    try:
        rankings = ranker.get_rankings(difficulty, language)
    except ValueError:
        return []
    rows = []
    for _, row in rankings.iterrows():
        safe = row["llm"]
        matches = int(row["total_matches"])
        wins = int(row["wins"])
        rows.append(
            {
                "model": manifest_models.get(safe, safe.replace("__", "/")),
                "safe_name": safe,
                "score": row["score"],
                "wins": wins,
                "matches": matches,
                "win_rate": wins / matches * 100 if matches else 0.0,
                "EN": row["EN"],
                "LT": row["LT"],
                "difficulty": difficulty,
                "language": language or "all",
            }
        )
    return rows


def print_table(rows: List[dict], title: str) -> None:
    if not rows:
        click.echo(f"\n{title}\n  (no data)")
        return
    click.echo(f"\n{title}")
    header = f"{'Rank':>4} {'Model':<55} {'Score':>7} {'Wins':>6} {'Matches':>8} {'Win%':>7} {'EN':>6} {'LT':>6}"
    click.echo(header)
    click.echo("-" * len(header))
    for idx, entry in enumerate(rows, start=1):
        click.echo(
            f"{idx:>4} {entry['model']:<55} "
            f"{entry['score']:>7.3f} {entry['wins']:>6} {entry['matches']:>8} "
            f"{entry['win_rate']:>7.2f} {entry['EN']:>6.2f} {entry['LT']:>6.2f}"
        )


def analyze_wins(analysis_file: Path, manifest_models: Dict[str, str], missing_models: Optional[List[str]] = None) -> None:
    comparisons = load_comparisons_from_file(str(analysis_file))
    filtered = [
        (a, b, winner, difficulty, is_english)
        for (a, b, winner, difficulty, is_english) in comparisons
        if a in manifest_models and b in manifest_models
    ]
    if missing_models is None:
        present = {c[0] for c in comparisons} | {c[1] for c in comparisons}
        missing_models = sorted(safe for safe in manifest_models if safe not in present)
    if missing_models:
        click.echo("\nWARNING: The following models have no judged comparisons in this file:", err=True)
        for model in missing_models:
            click.echo(f"  - {manifest_models[model]}", err=True)
        click.echo(
            "Run translation_comparer_any_model.py --generate-base-set using "
            "baseset/v1.0/base_conversation_pairs.v1.0.jsonl and your judge of choice.",
            err=True,
        )
    if not filtered:
        click.echo("  - No judged comparisons found for manifest models; skipping scoring.", err=True)
        return

    ranker = LLMRanker()
    ranker.fit(filtered)
    click.echo("\n[4/4] Bradley–Terry scores")
    slice_rows = []
    overall_rows = build_rows(ranker, manifest_models, "all", None)
    print_table(overall_rows, "Overall (all directions)")
    slice_rows.extend({"slice": "overall", **row} for row in overall_rows)

    en_rows = build_rows(ranker, manifest_models, "all", "english")
    print_table(en_rows, "EN→JA (judge saw english inputs)")
    slice_rows.extend({ "slice": "en_ja", **row} for row in en_rows)

    ja_rows = build_rows(ranker, manifest_models, "all", "japanese")
    print_table(ja_rows, "JA→EN (judge saw japanese inputs)")
    slice_rows.extend({"slice": "ja_en", **row} for row in ja_rows)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    scores_path = REPORT_DIR / f"{analysis_file.stem}_scores.json"
    with scores_path.open("w", encoding="utf-8") as handle:
        json.dump(slice_rows, handle, indent=2)
    click.echo(f"  - Saved table to {scores_path}")


@click.command()
@click.option(
    "--manifest",
    default="v1.0/manifest.json",
    show_default=True,
    help="Relative path to the manifest JSON.",
)
@click.option(
    "--analysis-file",
    default="",
    help="Optional explicit path to a judged base_set JSONL. Defaults to baseset/v1.0/ or base_sets/.",
)
@click.option(
    "--judge-model",
    default="gemini-2.5-flash",
    show_default=True,
    help="Judge model name used to look up base_set files.",
)
@click.option(
    "--judge-base-url",
    default=os.getenv("JUDGE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"),
    show_default=True,
    help="Base URL for the judge API when auto-running missing comparisons (mirrors run_translation_bench defaults).",
)
@click.option(
    "--judge-api-key-env",
    default=os.getenv("JUDGE_API_KEY_ENV", "GEMINI_API_KEY"),
    show_default=True,
    help="Environment variable containing the judge API key (mirrors run_translation_bench defaults).",
)
@click.option("--max-workers", default=40, show_default=True, help="Max workers for the judge run.")
@click.option("--concurrency-limit", default=40, show_default=True, help="Concurrency limit for the judge run.")
@click.option("--auto-judge/--no-auto-judge", default=True, show_default=True, help="Automatically run the judge when comparisons are missing.")
@click.option(
    "--max-judge-attempts",
    default=3,
    show_default=True,
    help="Maximum times to run the judge when trying to fill missing data (including retries for missing answers).",
)
def main(
    manifest: str,
    analysis_file: str,
    judge_model: str,
    judge_base_url: str,
    judge_api_key_env: str,
    max_workers: int,
    concurrency_limit: int,
    auto_judge: bool,
    max_judge_attempts: int,
) -> None:
    click.echo("=== Base-set v1.0 preparer (from prior runs) ===")
    manifest_path = (BASESET_DIR / manifest).resolve()
    if not manifest_path.exists():
        raise click.ClickException(f"Manifest not found: {manifest_path}")

    click.echo(f"Loading manifest from {manifest_path}")
    entries = load_manifest(manifest_path)
    manifest_models = {safe_name(entry["model"]): entry["model"] for entry in entries}

    copied = copy_translations(entries)
    pair_file = generate_pairs(copied)
    summarize_pairs(pair_file, manifest_models)

    analysis_path = resolve_analysis_file(analysis_file, judge_model)
    present_models = collect_present_models(analysis_path)
    missing = sorted([safe for safe in manifest_models if safe not in present_models])

    missing_answers = find_missing_answer_ids(analysis_path)

    if missing:
        click.echo("\nThe following models have no judged comparisons yet:")
        for safe in missing:
            click.echo(f"  - {manifest_models[safe]}")
    if missing_answers:
        click.echo(f"\nFound {len(missing_answers)} comparisons without <answer> tags in {analysis_path}")

    attempts = 0
    while auto_judge and (missing or missing_answers) and attempts < max_judge_attempts:
        if not judge_base_url:
            raise click.ClickException(
                "Need --judge-base-url (or JUDGE_URL env) to auto-run the judge for missing models."
            )
        attempts += 1
        if attempts == 1:
            click.echo("\nSince --no-auto-judge was not supplied, running translation comparisons now...")
        else:
            click.echo(f"\nRetrying judge run to backfill missing data (attempt {attempts}/{max_judge_attempts})...")
        analysis_path = run_auto_judge(
            pair_file,
            judge_model,
            judge_base_url,
            judge_api_key_env,
            max_workers,
            concurrency_limit,
        )
        present_models = collect_present_models(analysis_path)
        missing = sorted([safe for safe in manifest_models if safe not in present_models])
        missing_answers = find_missing_answer_ids(analysis_path)
        if missing:
            click.echo("Still missing models after judge run:")
            for safe in missing:
                click.echo(f"  - {manifest_models[safe]}")
        if missing_answers:
            click.echo(f"Still {len(missing_answers)} comparisons without <answer>; will retry if attempts remain.")

    if (missing or missing_answers) and not auto_judge:
        click.echo(
            "\nSome data are missing. Re-run with --auto-judge "
            "or execute translation_comparer_any_model.py manually.",
            err=True,
        )
    elif (missing or missing_answers) and auto_judge and attempts >= max_judge_attempts:
        click.echo(
            f"\nReached max judge attempts ({max_judge_attempts}) but still have missing data."
            " You may re-run later or inspect API failures.",
            err=True,
        )

    if analysis_path and analysis_path.exists():
        click.echo(f"\nScoring judged comparisons from {analysis_path}")
        analyze_wins(analysis_path, manifest_models, missing)
    else:
        click.echo("\nNo judged base_set file found. Run translation_comparer_any_model.py --generate-base-set first.")


if __name__ == "__main__":
    main()

