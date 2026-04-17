import json
from pathlib import Path
from typing import Any


def score_metadata_sidecar_path(score_file: str | Path) -> Path:
    score_file = Path(score_file)
    return score_file.with_name(f"{score_file.stem}.metadata.json")


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_first_jsonl_row(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                return json.loads(line)
    return {}


def load_score_with_sidecar(score_file: str | Path) -> Any:
    score_file = Path(score_file)
    payload = _read_json(score_file)
    if not isinstance(payload, dict):
        return payload

    sidecar_path = score_metadata_sidecar_path(score_file)
    if not sidecar_path.exists():
        return payload

    sidecar = _read_json(sidecar_path)
    if not isinstance(sidecar, dict):
        return payload

    merged = dict(payload)
    for key, value in sidecar.items():
        merged.setdefault(key, value)
    return merged


def write_score_metadata_sidecar(
    score_file: str | Path,
    *,
    test_model: str,
    task_config,
    judgments_file: str | Path,
    pairs_file: str | Path | None = None,
    workdir: str | Path = ".",
) -> Path:
    score_file = Path(score_file)
    judgments_file = Path(judgments_file)
    pairs_file = Path(pairs_file) if pairs_file else None
    workdir = Path(workdir)
    translation_file = workdir / "translations" / f"{test_model.replace('/', '__')}.jsonl"

    translation_row = _read_first_jsonl_row(translation_file)
    judgment_row = _read_first_jsonl_row(judgments_file)
    pair_row = _read_first_jsonl_row(pairs_file) if pairs_file else {}
    inferred_snapshot_version = None
    if len(score_file.parents) >= 3:
        inferred_snapshot_version = score_file.parents[2].name

    dataset_ref = translation_row.get("dataset_ref") or {
        "repo": task_config.dataset.repo,
        "config": task_config.dataset.config,
        "split": task_config.dataset.split,
        "revision": task_config.dataset.revision,
    }

    payload = {
        "task_id": translation_row.get("task_id", task_config.task_id),
        "task_type": translation_row.get("task_type", task_config.task_type),
        "task_version": translation_row.get("task_version", task_config.task_version),
        "task_config_digest": translation_row.get("task_config_digest", task_config.task_config_digest),
        "dataset_ref": dataset_ref,
        "generation_profile_id": translation_row.get("generation_profile_id"),
        "judge_profile_id": judgment_row.get("judge_profile_id"),
        "compare_prompt_profile_id": judgment_row.get("compare_prompt_profile_id"),
        "judge_parser_id": judgment_row.get("judge_parser_id"),
        "judge_contract_id": judgment_row.get("judge_contract_id"),
        "pair_id_schema": pair_row.get("pair_id_schema"),
        "snapshot_version": pair_row.get("snapshot_version") or inferred_snapshot_version,
        "translation_file": str(translation_file) if translation_file.exists() else None,
        "judgments_file": str(judgments_file),
        "pairs_file": str(pairs_file) if pairs_file else None,
        "score_file": str(score_file),
    }

    sidecar_path = score_metadata_sidecar_path(score_file)
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if sidecar_path.exists() and sidecar_path.read_text(encoding="utf-8") == rendered:
        return sidecar_path
    sidecar_path.write_text(rendered, encoding="utf-8")
    return sidecar_path
