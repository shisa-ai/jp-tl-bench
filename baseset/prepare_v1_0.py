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

from baseset.legacy_boundary import (
    legacy_candidate_paths,
    schema_v2_path,
    write_snapshot_report_sidecar,
    write_legacy_jp_v1_boundary_metadata,
)
from benchmark_tasks import load_task_config
from pair_contract import (
    PAIR_ID_SCHEMA_V1,
    compute_pair_fingerprint,
    compute_pair_id_v1,
)

from choix_analyzer import (  # type: ignore  # noqa: E402
    LLMRanker,
    build_ranked_slice_rows,
    iter_score_slice_specs,
    load_comparisons_from_file,
)


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
        if dest_path.exists():
            copied.append(dest_path)
            click.echo(f"  - Reusing frozen snapshot copy for {model}: baseset/v1.0/translations/{dest_path.name}")
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
        "id": compute_pair_id_v1(file_a, file_b, example_name),
        "pair_id_schema": PAIR_ID_SCHEMA_V1,
        "llm_a": file_a,
        "llm_b": file_b,
    }


def generate_pairs(outputs: List[str]) -> Path:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    pair_path = ARTIFACT_DIR / "base_conversation_pairs.v1.0.jsonl"
    if pair_path.exists():
        click.echo(f"  - Reusing frozen legacy pair file at {pair_path}")
        return pair_path
    pair_path = schema_v2_path(pair_path)
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
                        "pair_id_schema": settings["pair_id_schema"],
                        "llm_a": settings["llm_a"],
                        "llm_b": settings["llm_b"],
                        "formatted_data": format_translation_pair(conv_a, conv_b),
                        "name": conv_a["name"],
                        "english": conv_a["english"],
                        "difficulty": conv_a["difficulty"],
                    }
                    payload["pair_fingerprint"] = compute_pair_fingerprint(payload)
                    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    total_pairs += 1
    click.echo(f"  - Wrote {total_pairs:,} rows to {pair_path}")
    return pair_path


def summarize_pairs(pair_file: Path, manifest_models: Dict[str, str]) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "pair_coverage.json"
    if report_path.exists():
        report_path = schema_v2_path(report_path)
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
    preferred = legacy_candidate_paths(ARTIFACT_DIR / f"base_set.{safe_judge}.jsonl", SNAPSHOT_DIR)
    for path in preferred:
        if path.exists():
            return path
    return None


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
            "--pairs-file",
            str(pair_file),
        ]
        click.echo(f"  - Executing: {' '.join(cmd)}")
        env = os.environ.copy()
        env["BASESET_SNAPSHOT_DIR"] = str(SNAPSHOT_DIR)
        subprocess.run(cmd, cwd=REPO_ROOT, check=True, env=env)
    finally:
        pass

    safe_judge = safe_name(judge_model)
    candidate_sources = legacy_candidate_paths(ARTIFACT_DIR / f"base_set.{safe_judge}.jsonl", SNAPSHOT_DIR)
    for source in candidate_sources:
        if not source.exists():
            continue
        try:
            relative_source = source.relative_to(REPO_ROOT)
        except ValueError:
            relative_source = source
        click.echo(f"  - Using judged base set at {relative_source}")
        return source
    raise click.ClickException(
        "translation comparer finished but no judged base_set artifact was created in the snapshot."
    )


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


def analyze_wins(
    analysis_file: Path,
    manifest_models: Dict[str, str],
    judge_model: str,
    pair_file: Path,
    missing_models: Optional[List[str]] = None,
    task=None,
) -> None:
    comparisons = load_comparisons_from_file(str(analysis_file), task=task)
    filtered = [
        (a, b, winner, difficulty, direction_key)
        for (a, b, winner, difficulty, direction_key) in comparisons
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

    ranker = LLMRanker(task=task)
    ranker.fit(filtered)
    click.echo("\n[4/4] Bradley–Terry scores")
    slice_rows = build_ranked_slice_rows(ranker, manifest_models=manifest_models, task=task)
    for spec in iter_score_slice_specs(task=task):
        rows = [row for row in slice_rows if row["slice"] == spec["slice"]]
        if rows:
            print_table(rows, spec["title"])

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    scores_path = REPORT_DIR / f"{analysis_file.stem}_scores.json"
    legacy_stem = analysis_file.stem.replace(".schema-v2", "")
    legacy_scores_path = REPORT_DIR / f"{legacy_stem}_scores.json"
    if legacy_scores_path.exists():
        scores_path = schema_v2_path(legacy_scores_path)
    with scores_path.open("w", encoding="utf-8") as handle:
        json.dump(slice_rows, handle, indent=2)
    task_config = load_task_config(task)
    sidecar_path = write_snapshot_report_sidecar(
        scores_path,
        snapshot_dir=SNAPSHOT_DIR,
        manifest_payload={
            "snapshot_version": SNAPSHOT_DIR.name,
            "task_id": task_config.task_id,
            "task_type": task_config.task_type,
            "task_version": task_config.task_version,
            "task_config_digest": task_config.task_config_digest,
            "dataset_repo": task_config.dataset.repo,
            "dataset_config": task_config.dataset.config,
            "dataset_split": task_config.dataset.split,
            "dataset_revision": task_config.dataset.revision,
            "manifest_file": "manifest.json",
        },
        task_config=task_config,
        judge_model=judge_model,
        pair_file=pair_file,
        analysis_file=analysis_file,
    )
    click.echo(f"  - Saved table to {scores_path}")
    click.echo(f"  - Saved report metadata to {sidecar_path}")


@click.command()
@click.option(
    "--manifest",
    default="v1.0/manifest.json",
    show_default=True,
    help="Relative path to the manifest JSON.",
)
@click.option("--task", envvar="TASK_CONFIG", help="Task config path or name under benchmark_tasks/.")
@click.option(
    "--analysis-file",
    default="",
    help="Optional explicit path to a judged base_set JSONL. Defaults to baseset/v1.0/.",
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
    task: str,
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
    boundary_path = write_legacy_jp_v1_boundary_metadata(SNAPSHOT_DIR, judge_model)
    click.echo(f"  - Saved legacy boundary metadata to {boundary_path}")

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
        analyze_wins(
            analysis_path,
            manifest_models,
            judge_model,
            pair_file,
            missing,
            task=task,
        )
    else:
        click.echo("\nNo judged base_set file found. Run translation_comparer_any_model.py --generate-base-set first.")


if __name__ == "__main__":
    main()
