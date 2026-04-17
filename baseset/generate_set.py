#!/usr/bin/env python3
"""
Generic helper for generating base-set artifacts from a prepared snapshot.

Assumptions for a snapshot directory (e.g., baseset/v1.0/ or baseset/v2.0/):
- It contains a manifest JSON with a "models" list (at <snapshot>/manifest.json by default).
- It contains a translations/ subdirectory with one JSONL per model, named using the
  safe model name ("/" replaced by "__"), e.g. translations/openai__gpt-4o.jsonl.

This script:
1. Generates a pairwise comparison JSONL for all manifest models.
2. Emits coverage stats.
3. Locates (or generates) a judged base_set file and prints/saves the LT table.

Unlike prepare_v1_0.py, this script does not copy or sync translation dumps for you;
it expects the snapshot directory to already be populated.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from io import StringIO
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import click

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baseset.legacy_boundary import (
    is_legacy_jp_v1_snapshot,
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


TASK_IDENTITY_FIELDS = (
    "item_id",
    "name",
    "task_id",
    "task_type",
    "task_version",
    "source_text",
    "difficulty",
    "source_language",
    "target_language",
    "english",
)
TASK_SLICE_TAG_FIELDS = ("category", "tags", "slice_tags")


def safe_name(model: str) -> str:
    return model.replace("/", "__")


def load_manifest_document(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if "models" not in data:
        raise click.ClickException(f"Manifest missing 'models' list: {path}")
    return data


def load_manifest(path: Path) -> List[dict]:
    return load_manifest_document(path)["models"]


def ensure_manifest_metadata(path: Path, manifest_payload: dict, snapshot_dir: Path, task_config) -> dict:
    if is_legacy_jp_v1_snapshot(snapshot_dir):
        return manifest_payload
    expected = {
        "snapshot_version": snapshot_dir.name,
        "task_id": task_config.task_id,
        "task_type": task_config.task_type,
        "task_version": task_config.task_version,
        "task_config_digest": task_config.task_config_digest,
        "dataset_repo": task_config.dataset.repo,
        "dataset_config": task_config.dataset.config,
        "dataset_split": task_config.dataset.split,
        "dataset_revision": task_config.dataset.revision,
    }
    updated = dict(manifest_payload)
    changed = False
    for key, value in expected.items():
        existing = updated.get(key)
        if existing is not None and existing != value:
            raise click.ClickException(
                f"Manifest metadata mismatch for {key}: expected {value!r}, found {existing!r} in {path}"
            )
        if existing is None:
            updated[key] = value
            changed = True
    if changed:
        path.write_text(json.dumps(updated, indent=2) + "\n", encoding="utf-8")
    return updated


def load_jsonl(path: Path) -> List[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def record_match_key(item: dict) -> str:
    return item.get("item_id") or item.get("name")


def task_slice_metadata(item: dict) -> dict:
    metadata = {}
    for key in TASK_SLICE_TAG_FIELDS:
        if key in item:
            metadata[key] = item[key]
    return metadata


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
    dict_a = {record_match_key(item): item for item in convs_a}
    dict_b = {record_match_key(item): item for item in convs_b}
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


def validate_task_identity_fields(conv_a: dict, conv_b: dict, file_a: str, file_b: str) -> None:
    candidate_fields = list(TASK_IDENTITY_FIELDS)
    candidate_fields.extend(key for key in TASK_SLICE_TAG_FIELDS if key in conv_a or key in conv_b)
    for key in candidate_fields:
        if conv_a.get(key) != conv_b.get(key):
            raise ValueError(
                f"Task-defining field mismatch for '{conv_a.get('item_id') or conv_a.get('name')}' "
                f"between {file_a} and {file_b}: {key}={conv_a.get(key)!r} vs {conv_b.get(key)!r}"
            )


def write_pair_settings(file_a: str, file_b: str, example_name: str) -> Dict[str, str]:
    return {
        "id": compute_pair_id_v1(file_a, file_b, example_name),
        "pair_id_schema": PAIR_ID_SCHEMA_V1,
        "llm_a": file_a,
        "llm_b": file_b,
    }


def generate_pairs(translation_dir: Path, snapshot_dir: Path, models: List[dict], pair_filename: str, task=None) -> Path:
    artifact_dir = snapshot_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)
    pair_path = artifact_dir / pair_filename
    if is_legacy_jp_v1_snapshot(snapshot_dir) and pair_path.exists():
        click.echo(f"  - Reusing frozen legacy pair file at {pair_path}")
        return pair_path
    if is_legacy_jp_v1_snapshot(snapshot_dir):
        pair_path = schema_v2_path(pair_path)

    click.echo("\n[2/4] Generating pairwise comparison file")
    task_config = load_task_config(task)
    snapshot_version = snapshot_dir.name
    safe_names = sorted([safe_name(entry["model"]) for entry in models])
    translations: Dict[str, List[dict]] = {}
    for safe in safe_names:
        src = translation_dir / f"{safe}.jsonl"
        if not src.exists():
            raise click.ClickException(f"Missing translation dump for {safe}: {src}")
        translations[safe] = [task_config.normalize_record(item) for item in load_jsonl(src)]

    total_pairs = 0
    with pair_path.open("w", encoding="utf-8") as handle:
        for idx, safe_a in enumerate(safe_names):
            for safe_b in safe_names[idx + 1 :]:
                for conv_a, conv_b in ensure_alignment(translations[safe_a], translations[safe_b], safe_a, safe_b):
                    validate_task_identity_fields(conv_a, conv_b, safe_a, safe_b)
                    settings = write_pair_settings(safe_a, safe_b, conv_a["name"])
                    payload = {
                        "id": settings["id"],
                        "pair_id_schema": settings["pair_id_schema"],
                        "llm_a": settings["llm_a"],
                        "llm_b": settings["llm_b"],
                        "formatted_data": format_translation_pair(conv_a, conv_b),
                        "item_id": conv_a["item_id"],
                        "name": conv_a["name"],
                        "task_id": conv_a["task_id"],
                        "task_type": conv_a["task_type"],
                        "task_version": conv_a["task_version"],
                        "snapshot_version": snapshot_version,
                        "source_language": conv_a["source_language"],
                        "target_language": conv_a["target_language"],
                        "difficulty": conv_a["difficulty"],
                    }
                    if "english" in conv_a:
                        payload["english"] = conv_a["english"]
                    payload.update(task_slice_metadata(conv_a))
                    payload["pair_fingerprint"] = compute_pair_fingerprint(payload)
                    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    total_pairs += 1
    click.echo(f"  - Wrote {total_pairs:,} rows to {pair_path}")
    return pair_path


def summarize_pairs(pair_file: Path, report_dir: Path, manifest_models: Dict[str, str]) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "pair_coverage.json"
    if is_legacy_jp_v1_snapshot(report_dir.parent) and report_path.exists():
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


def resolve_analysis_file(explicit: str, snapshot_dir: Path, judge_model: str) -> Optional[Path]:
    if explicit:
        candidate = Path(explicit)
        if candidate.is_absolute():
            return candidate

        # Try a few plausible roots for relative paths:
        # - relative to snapshot dir (common when passing "base_set...jsonl")
        # - relative to snapshot parent (e.g., running from baseset/ and passing "v1.0/...")
        # - relative to repo root
        for root in (snapshot_dir, snapshot_dir.parent, REPO_ROOT):
            candidate_path = (root / candidate).resolve()
            if candidate_path.exists():
                return candidate_path
        # If none matched, fall through and return None to signal "not found"
        return None

    safe_judge = safe_name(judge_model)
    preferred = legacy_candidate_paths(snapshot_dir / f"base_set.{safe_judge}.jsonl", snapshot_dir)
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
            analysis = data.get("analysis", "") or ""
            match = ANSWER_RE.search(analysis)
            if not match:
                missing.append(data.get("id"))
                continue
            answer = "".join(c for c in match.group(1) if c.isalpha()).lower()
            if answer not in {"a", "b"}:
                missing.append(data.get("id"))
    return [m for m in missing if m]


def collect_judged_ids(analysis_file: Optional[Path]) -> set:
    """Collect IDs of all comparisons that have been judged (valid <answer>A/B</answer>)."""
    if not analysis_file or not analysis_file.exists():
        return set()
    judged = set()
    with analysis_file.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            analysis = data.get("analysis", "") or ""
            match = ANSWER_RE.search(analysis)
            if not match:
                continue
            answer = "".join(c for c in match.group(1) if c.isalpha()).lower()
            if answer not in {"a", "b"}:
                continue
            item_id = data.get("id")
            if item_id:
                judged.add(item_id)
    return judged


def count_total_pairs(pair_file: Path) -> int:
    """Count total number of pairs in the pair file."""
    count = 0
    with pair_file.open(encoding="utf-8") as handle:
        for line in handle:
            count += 1
    return count


def collect_pair_ids(pair_file: Path) -> set:
    """Collect all pair IDs from the pair file to detect missing rows."""
    ids = set()
    with pair_file.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                data = json.loads(line)
                pid = data.get("id")
                if pid:
                    ids.add(pid)
            except json.JSONDecodeError:
                continue
    return ids


def run_auto_judge(
    pair_file: Path,
    snapshot_dir: Path,
    task: str | None,
    judge_model: str,
    judge_profile: str,
    base_url: str,
    api_key_env: str,
    max_workers: int,
    concurrency_limit: int,
    skip_ids: Optional[set] = None,
    use_gemini: bool = False,
) -> Path:
    click.echo("\n[3b] Running translation comparer to fill missing judges")
    skip_ids_file = None
    try:
        cmd = [
            sys.executable,
            "translation_comparer_any_model.py",
            "--judge-profile",
            judge_profile,
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
        if task:
            cmd[2:2] = ["--task", task]
        if use_gemini:
            cmd.append("--gemini-judge")
        if skip_ids:
            # Write skip_ids to a temporary file to avoid "argument list too long" errors
            skip_ids_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
            for skip_id in sorted(skip_ids):
                skip_ids_file.write(f"{skip_id}\n")
            skip_ids_file.close()
            cmd.extend(["--skip-ids-file", skip_ids_file.name])
            click.echo(f"  - Writing {len(skip_ids):,} skip IDs to temp file: {skip_ids_file.name}")
        click.echo(f"  - Executing: {' '.join(cmd)}")
        env = os.environ.copy()
        env["BASESET_SNAPSHOT_DIR"] = str(snapshot_dir)
        subprocess.run(cmd, cwd=REPO_ROOT, check=True, env=env)
    finally:
        if skip_ids_file and os.path.exists(skip_ids_file.name):
            os.unlink(skip_ids_file.name)

    safe_judge = safe_name(judge_model)
    candidate_sources = legacy_candidate_paths(snapshot_dir / f"base_set.{safe_judge}.jsonl", snapshot_dir)
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
    snapshot_dir: Path,
    report_dir: Path,
    manifest_models: Dict[str, str],
    manifest_payload: dict,
    judge_model: str,
    pair_file: Path,
    missing_models: Optional[List[str]] = None,
    task=None,
) -> Path:
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
    if not filtered:
        click.echo("  - No judged comparisons found for manifest models; skipping scoring.", err=True)
        return report_dir / f"{analysis_file.stem}_scores.json"

    ranker = LLMRanker(task=task)
    ranker.fit(filtered)
    click.echo("\n[4/4] Bradley–Terry scores")
    slice_rows = build_ranked_slice_rows(ranker, manifest_models=manifest_models, task=task)
    for spec in iter_score_slice_specs(task=task):
        rows = [row for row in slice_rows if row["slice"] == spec["slice"]]
        if rows:
            print_table(rows, spec["title"])

    report_dir.mkdir(parents=True, exist_ok=True)
    scores_path = report_dir / f"{analysis_file.stem}_scores.json"
    if is_legacy_jp_v1_snapshot(snapshot_dir):
        legacy_stem = analysis_file.stem.replace(".schema-v2", "")
        legacy_scores_path = report_dir / f"{legacy_stem}_scores.json"
        if legacy_scores_path.exists():
            scores_path = schema_v2_path(legacy_scores_path)
    with scores_path.open("w", encoding="utf-8") as handle:
        json.dump(slice_rows, handle, indent=2)
    task_config = load_task_config(task)
    sidecar_path = write_snapshot_report_sidecar(
        scores_path,
        snapshot_dir=snapshot_dir,
        manifest_payload={
            **manifest_payload,
            "manifest_file": "manifest.json",
        },
        task_config=task_config,
        judge_model=judge_model,
        pair_file=pair_file,
        analysis_file=analysis_file,
    )
    click.echo(f"  - Saved table to {scores_path}")
    click.echo(f"  - Saved report metadata to {sidecar_path}")
    return scores_path


@click.command()
@click.option(
    "--snapshot-dir",
    default="baseset/v1.0",
    show_default=True,
    help="Snapshot directory (contains manifest.json, translations/, reports/). Relative to repo root.",
)
@click.option(
    "--manifest",
    default="",
    help="Path to the manifest JSON. Defaults to <snapshot-dir>/manifest.json.",
)
@click.option(
    "--pair-filename",
    default="base_conversation_pairs.jsonl",
    show_default=True,
    help="Filename for the generated pair file inside the snapshot directory.",
)
@click.option("--task", envvar="TASK_CONFIG", help="Task config path or name under benchmark_tasks/.")
@click.option(
    "--judge-profile",
    envvar="JUDGE_PROFILE",
    default="default",
    show_default=True,
    help="Judge profile path or name under judge_profiles/. Forwarded to the auto-judge comparer.",
)
@click.option(
    "--analysis-file",
    default="",
    help="Optional explicit path to a judged base_set JSONL. Defaults to <snapshot-dir>/base_set.<judge>.jsonl.",
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
@click.option(
    "--auto-judge/--no-auto-judge",
    default=True,
    show_default=True,
    help="Automatically run the judge when comparisons are missing.",
)
@click.option(
    "--max-judge-attempts",
    default=3,
    show_default=True,
    help="Maximum times to run the judge when trying to fill missing data (including retries for missing answers).",
)
@click.option(
    "--rerun/--no-rerun",
    default=False,
    show_default=True,
    help="Force re-run all judgments even if they already exist (expensive!).",
)
@click.option(
    "--gemini-judge/--no-gemini-judge",
    default=False,
    show_default=True,
    help="Use native Gemini API instead of OpenAI-compatible endpoint. Bypasses safety filtering and may avoid API errors.",
)
def main(
    snapshot_dir: str,
    manifest: str,
    pair_filename: str,
    task: str,
    judge_profile: str,
    analysis_file: str,
    judge_model: str,
    judge_base_url: str,
    judge_api_key_env: str,
    max_workers: int,
    concurrency_limit: int,
    auto_judge: bool,
    max_judge_attempts: int,
    rerun: bool,
    gemini_judge: bool,
) -> None:
    click.echo("=== Generic base-set generator ===")

    snapshot_path = Path(snapshot_dir)
    if not snapshot_path.is_absolute():
        snapshot_path = (REPO_ROOT / snapshot_path).resolve()
    if not snapshot_path.exists():
        raise click.ClickException(f"Snapshot directory not found: {snapshot_path}")

    translation_dir = snapshot_path / "translations"
    report_dir = snapshot_path / "reports"

    if manifest:
        manifest_path = Path(manifest)
        if not manifest_path.is_absolute():
            manifest_path = (REPO_ROOT / manifest_path).resolve()
    else:
        manifest_path = snapshot_path / "manifest.json"

    if not manifest_path.exists():
        raise click.ClickException(f"Manifest not found: {manifest_path}")

    click.echo(f"Loading manifest from {manifest_path}")
    task_config = load_task_config(task)
    manifest_payload = ensure_manifest_metadata(
        manifest_path,
        load_manifest_document(manifest_path),
        snapshot_path,
        task_config,
    )
    entries = manifest_payload["models"]
    manifest_models = {safe_name(entry["model"]): entry["model"] for entry in entries}

    pair_file = generate_pairs(translation_dir, snapshot_path, entries, pair_filename, task=task)
    summarize_pairs(pair_file, report_dir, manifest_models)
    if is_legacy_jp_v1_snapshot(snapshot_path):
        boundary_path = write_legacy_jp_v1_boundary_metadata(snapshot_path, judge_model)
        click.echo(f"  - Saved legacy boundary metadata to {boundary_path}")

    analysis_path = resolve_analysis_file(analysis_file, snapshot_path, judge_model)
    explicit_analysis = bool(analysis_file)
    if explicit_analysis and (not analysis_path or not analysis_path.exists()) and not auto_judge:
        raise click.ClickException(
            f"Analysis file not found: {analysis_file}. Tried relative to {snapshot_path}, its parent, and repo root."
        )

    # Collect statistics about what's been judged
    pair_ids = collect_pair_ids(pair_file)
    total_pairs = len(pair_ids)
    judged_ids = set() if rerun else collect_judged_ids(analysis_path)
    missing_answers_ids = find_missing_answer_ids(analysis_path)
    missing_rows = pair_ids - (judged_ids | set(missing_answers_ids))

    present_models = collect_present_models(analysis_path)
    missing = sorted([safe for safe in manifest_models if safe not in present_models])

    # Show statistics
    click.echo("\n[Stats] Judging progress:")
    click.echo(f"  - Total pairs: {total_pairs:,}")
    if rerun:
        click.echo(f"  - Already judged: 0 (--rerun flag set, will re-run all)")
        click.echo(f"  - Need to judge: {total_pairs:,}")
    else:
        click.echo(f"  - Already judged: {len(judged_ids):,} ({len(judged_ids)/total_pairs*100:.1f}%)")
        click.echo(f"  - Missing answers: {len(missing_answers_ids):,}")
        if missing_rows:
            click.echo(f"  - Missing rows (not present in base_set): {len(missing_rows):,}")
        need_to_judge = total_pairs - len(judged_ids)
        click.echo(f"  - Need to judge: {need_to_judge:,} ({need_to_judge/total_pairs*100:.1f}%)")

    if missing:
        click.echo("\nThe following models have no judged comparisons yet:")
        for safe in missing:
            click.echo(f"  - {manifest_models[safe]}")

    attempts = 0
    skip_ids = None if rerun else judged_ids
    while auto_judge and (missing or missing_answers_ids or missing_rows or rerun or (total_pairs - len(judged_ids)) > 0) and attempts < max_judge_attempts:
        if not judge_base_url:
            raise click.ClickException(
                "Need --judge-base-url (or JUDGE_URL env) to auto-run the judge for missing models."
            )
        attempts += 1
        if attempts == 1:
            if rerun:
                click.echo("\nRunning translation comparisons with --rerun (will re-judge all pairs)...")
            else:
                click.echo("\nSince --no-auto-judge was not supplied, running translation comparisons now...")
        else:
            click.echo(f"\nRetrying judge run to backfill missing data (attempt {attempts}/{max_judge_attempts})...")
        analysis_path = run_auto_judge(
            pair_file,
            snapshot_path,
            task,
            judge_model,
            judge_profile,
            judge_base_url,
            judge_api_key_env,
            max_workers,
            concurrency_limit,
            skip_ids,
            gemini_judge,
        )
        present_models = collect_present_models(analysis_path)
        missing = sorted([safe for safe in manifest_models if safe not in present_models])
        missing_answers_ids = find_missing_answer_ids(analysis_path)
        judged_ids = collect_judged_ids(analysis_path)
        missing_rows = pair_ids - (judged_ids | set(missing_answers_ids))

        # Update skip_ids for next attempt (don't skip successfully judged items)
        skip_ids = judged_ids

        if missing:
            click.echo("Still missing models after judge run:")
            for safe in missing:
                click.echo(f"  - {manifest_models[safe]}")
        if missing_answers_ids:
            click.echo(f"Still {len(missing_answers_ids)} comparisons without <answer>; will retry if attempts remain.")
        if missing_rows:
            click.echo(f"Still {len(missing_rows)} comparisons missing entirely; will retry if attempts remain.")

    if (missing or missing_answers_ids or missing_rows) and not auto_judge:
        click.echo(
            "\nSome data are missing. Re-run with --auto-judge "
            "or execute translation_comparer_any_model.py manually.",
            err=True,
        )
    elif (missing or missing_answers_ids or missing_rows) and auto_judge and attempts >= max_judge_attempts:
        click.echo(
            f"\nReached max judge attempts ({max_judge_attempts}) but still have missing data."
            " You may re-run later or inspect API failures.",
            err=True,
        )

    if analysis_path and analysis_path.exists():
        click.echo(f"\nScoring judged comparisons from {analysis_path}")
        analyze_wins(
            analysis_path,
            snapshot_path,
            report_dir,
            manifest_models,
            manifest_payload,
            judge_model,
            pair_file,
            missing,
            task=task,
        )
    else:
        click.echo("\nNo judged base_set file found. Run translation_comparer_any_model.py --generate-base-set first.")


if __name__ == "__main__":
    main()
