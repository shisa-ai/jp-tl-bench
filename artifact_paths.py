from __future__ import annotations

from pathlib import Path


DEFAULT_JUDGE_PROFILE_ID = "default"


def safe_artifact_name(value: str) -> str:
    return value.replace("/", "__")


def judge_output_dirname(
    judge_model: str,
    judge_profile_id: str = DEFAULT_JUDGE_PROFILE_ID,
) -> str:
    safe_judge = safe_artifact_name(judge_model)
    safe_profile = safe_artifact_name(judge_profile_id or DEFAULT_JUDGE_PROFILE_ID)
    if safe_profile == DEFAULT_JUDGE_PROFILE_ID:
        return safe_judge
    return f"{safe_judge}.{safe_profile}"


def candidate_results_dir(
    snapshot_version: str,
    test_model: str,
    judge_model: str,
    judge_profile_id: str = DEFAULT_JUDGE_PROFILE_ID,
    *,
    root: str | Path = ".",
) -> Path:
    return (
        Path(root)
        / "results"
        / snapshot_version
        / safe_artifact_name(test_model)
        / judge_output_dirname(judge_model, judge_profile_id)
    )


def candidate_results_dir_candidates(
    snapshot_version: str,
    test_model: str,
    judge_model: str,
    judge_profile_id: str = DEFAULT_JUDGE_PROFILE_ID,
    *,
    root: str | Path = ".",
) -> list[Path]:
    preferred = candidate_results_dir(
        snapshot_version,
        test_model,
        judge_model,
        judge_profile_id,
        root=root,
    )
    legacy = candidate_results_dir(
        snapshot_version,
        test_model,
        judge_model,
        DEFAULT_JUDGE_PROFILE_ID,
        root=root,
    )
    candidates = [preferred]
    if legacy != preferred:
        candidates.append(legacy)
    return candidates


def preferred_result_file(
    snapshot_version: str,
    test_model: str,
    judge_model: str,
    filename: str,
    judge_profile_id: str = DEFAULT_JUDGE_PROFILE_ID,
    *,
    root: str | Path = ".",
) -> Path:
    return candidate_results_dir(
        snapshot_version,
        test_model,
        judge_model,
        judge_profile_id,
        root=root,
    ) / filename


def resolve_result_file_candidates(
    snapshot_version: str,
    test_model: str,
    judge_model: str,
    filename: str,
    judge_profile_id: str = DEFAULT_JUDGE_PROFILE_ID,
    *,
    root: str | Path = ".",
) -> list[Path]:
    return [
        candidate_dir / filename
        for candidate_dir in candidate_results_dir_candidates(
            snapshot_version,
            test_model,
            judge_model,
            judge_profile_id,
            root=root,
        )
    ]
