import json
from pathlib import Path

from artifact_metadata import load_score_with_sidecar, write_score_metadata_sidecar
from benchmark_tasks import load_task_config
from score_visualizer import extract_candidate_direction_rows


JP_TASK = "benchmark_tasks/translation_ja_en_bidirectional_v1.yaml"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_load_score_with_sidecar_preserves_legacy_slice_values(tmp_path):
    score_file = tmp_path / "scores.json"
    legacy_score = {
        "model": "fixture/model",
        "judge_model": "gemini-2.5-flash",
        "baseset_version": "v1.0",
        "en_ja": {
            "overall": {"difficulty": "all", "language": "english", "lt": 9.0, "wins": 9, "total": 10},
            "easy": {"difficulty": "easy", "language": "english", "lt": 8.5, "wins": 4, "total": 5},
            "hard": {"difficulty": "hard", "language": "english", "lt": 9.5, "wins": 5, "total": 5},
        },
        "ja_en": {
            "overall": {"difficulty": "all", "language": "japanese", "lt": 7.0, "wins": 7, "total": 10},
            "easy": {"difficulty": "easy", "language": "japanese", "lt": 6.5, "wins": 3, "total": 5},
            "hard": {"difficulty": "hard", "language": "japanese", "lt": 7.5, "wins": 4, "total": 5},
        },
    }
    score_file.write_text(json.dumps(legacy_score, indent=2), encoding="utf-8")

    task_config = load_task_config(JP_TASK)
    before = extract_candidate_direction_rows(
        load_score_with_sidecar(score_file),
        task_config=task_config,
        missing_ratio=0.0,
    )

    (tmp_path / "scores.metadata.json").write_text(
        json.dumps(
            {
                "task_id": "translation.ja-en",
                "task_version": "v1",
                "task_type": "translation",
                "task_config_digest": "sha256:fixture",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    loaded = load_score_with_sidecar(score_file)
    after = extract_candidate_direction_rows(
        loaded,
        task_config=task_config,
        missing_ratio=0.0,
    )

    assert before == after
    assert loaded["task_id"] == "translation.ja-en"
    assert loaded["task_version"] == "v1"


def test_write_score_metadata_sidecar_derives_result_metadata_from_artifacts(tmp_path):
    task_config = load_task_config(JP_TASK)
    translations_dir = tmp_path / "translations"
    results_dir = tmp_path / "results" / "v1.0" / "fixture__model" / "gemini-2.5-flash"
    translations_dir.mkdir(parents=True)
    results_dir.mkdir(parents=True)

    score_file = results_dir / "scores.json"
    judgments_file = results_dir / "judgments.jsonl"
    pairs_file = results_dir / "pairs.jsonl"
    translation_file = translations_dir / "fixture__model.jsonl"

    score_file.write_text(json.dumps({"model": "fixture/model", "baseset_version": "v1.0"}), encoding="utf-8")
    write_jsonl(
        translation_file,
        [
            {
                "item_id": "sample-1",
                "task_id": "translation.ja-en",
                "task_type": "translation",
                "task_version": "v1",
                "task_config_digest": "sha256:task",
                "generation_profile_id": "openai-chat/v1",
                "dataset_ref": {
                    "repo": "shisa-ai/bt_translation_set_global",
                    "config": "translation_ja_en_bidirectional_v1",
                    "split": "train",
                    "revision": "1b0d3bcb2ed1a611ad9130cc474007afae2b598c",
                    "resolved_revision": "1b0d3bcb2ed1a611ad9130cc474007afae2b598c",
                },
            }
        ],
    )
    write_jsonl(
        pairs_file,
        [
            {
                "id": "pair-1",
                "pair_id_schema": "v1",
            }
        ],
    )
    write_jsonl(
        judgments_file,
        [
            {
                "id": "pair-1",
                "judge_profile_id": "default",
                "compare_prompt_profile_id": "compare-default-v1",
                "judge_parser_id": "answer-parser/v1",
                "judge_contract_id": "gemini-2.5-flash::compare-default-v1::answer-parser/v1",
            }
        ],
    )

    sidecar_path = write_score_metadata_sidecar(
        score_file,
        test_model="fixture/model",
        task_config=task_config,
        judgments_file=judgments_file,
        pairs_file=pairs_file,
        workdir=tmp_path,
    )
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))

    assert payload["task_id"] == "translation.ja-en"
    assert payload["task_config_digest"] == "sha256:task"
    assert payload["generation_profile_id"] == "openai-chat/v1"
    assert payload["dataset_ref"]["config"] == "translation_ja_en_bidirectional_v1"
    assert payload["judge_profile_id"] == "default"
    assert payload["compare_prompt_profile_id"] == "compare-default-v1"
    assert payload["judge_contract_id"] == "gemini-2.5-flash::compare-default-v1::answer-parser/v1"
    assert payload["pair_id_schema"] == "v1"
    assert payload["snapshot_version"] == "v1.0"
