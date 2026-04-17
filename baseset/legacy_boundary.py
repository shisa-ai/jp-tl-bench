import json
from pathlib import Path


LEGACY_JP_V1_BOUNDARY = "legacy_jp_v1"
LEGACY_JP_V1_SNAPSHOT = "v1.0"
SCHEMA_V2_SUFFIX = ".schema-v2"


def is_legacy_jp_v1_snapshot(snapshot_dir: Path | str) -> bool:
    return Path(snapshot_dir).name == LEGACY_JP_V1_SNAPSHOT


def schema_v2_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}{SCHEMA_V2_SUFFIX}{path.suffix}")


def legacy_candidate_paths(path: Path, snapshot_dir: Path | str) -> list[Path]:
    if not is_legacy_jp_v1_snapshot(snapshot_dir):
        return [path]
    versioned = schema_v2_path(path)
    if versioned == path:
        return [path]
    return [versioned, path]


def write_legacy_jp_v1_boundary_metadata(snapshot_dir: Path | str, judge_model: str | None = None) -> Path:
    snapshot_dir = Path(snapshot_dir)
    if not is_legacy_jp_v1_snapshot(snapshot_dir):
        raise ValueError(f"{snapshot_dir} is not the frozen JP v1.0 snapshot")

    sidecar_path = snapshot_dir / "legacy_jp_v1_boundary.json"
    payload = {
        "legacy_boundary": LEGACY_JP_V1_BOUNDARY,
        "artifact_policy": "additive_only",
        "pair_id_schema": "v1",
        "pair_fingerprint_policy": "schema_v2_exports_only",
        "grandfather_pre_fingerprint_judgments": True,
        "pair_file": "base_conversation_pairs.v1.0.jsonl",
        "judgments_file_pattern": "base_set.<judge>.jsonl",
        "schema_v2_judgments_pattern": "base_set.<judge>.schema-v2.jsonl",
        "score_report_pattern": "reports/base_set.<judge>_scores.json",
        "schema_v2_score_report_pattern": "reports/base_set.<judge>_scores.schema-v2.json",
    }
    rendered = json.dumps(payload, indent=2) + "\n"
    if sidecar_path.exists() and sidecar_path.read_text(encoding="utf-8") == rendered:
        return sidecar_path
    sidecar_path.write_text(rendered, encoding="utf-8")
    return sidecar_path


def report_metadata_path(report_path: Path) -> Path:
    return report_path.with_name(f"{report_path.stem}.metadata.json")


def write_snapshot_report_sidecar(
    report_path: Path | str,
    *,
    snapshot_dir: Path | str,
    manifest_payload: dict,
    task_config,
    judge_model: str,
    pair_file: Path | str | None = None,
    analysis_file: Path | str | None = None,
) -> Path:
    report_path = Path(report_path)
    snapshot_dir = Path(snapshot_dir)
    pair_file = Path(pair_file) if pair_file else None
    analysis_file = Path(analysis_file) if analysis_file else None
    sidecar_path = report_metadata_path(report_path)

    payload = {
        "snapshot_version": manifest_payload.get("snapshot_version", snapshot_dir.name),
        "task_id": manifest_payload.get("task_id", task_config.task_id),
        "task_type": manifest_payload.get("task_type", task_config.task_type),
        "task_version": manifest_payload.get("task_version", task_config.task_version),
        "task_config_digest": manifest_payload.get("task_config_digest", task_config.task_config_digest),
        "dataset_repo": manifest_payload.get("dataset_repo", task_config.dataset.repo),
        "dataset_config": manifest_payload.get("dataset_config", task_config.dataset.config),
        "dataset_split": manifest_payload.get("dataset_split", task_config.dataset.split),
        "dataset_revision": manifest_payload.get("dataset_revision", task_config.dataset.revision),
        "judge_model": judge_model,
        "report_file": report_path.name,
        "pair_file": pair_file.name if pair_file else None,
        "analysis_file": analysis_file.name if analysis_file else None,
        "manifest_file": manifest_payload.get("manifest_file"),
    }
    rendered = json.dumps(payload, indent=2) + "\n"
    if sidecar_path.exists() and sidecar_path.read_text(encoding="utf-8") == rendered:
        return sidecar_path
    sidecar_path.write_text(rendered, encoding="utf-8")
    return sidecar_path
