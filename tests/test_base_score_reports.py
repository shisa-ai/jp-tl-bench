import json
from pathlib import Path

from generate_base_scores import generate_base_scores


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_generate_base_scores_emits_task_defined_slice_labels_without_collisions(tmp_path):
    snapshot_dir = tmp_path / "baseset" / "v2.0"
    snapshot_dir.mkdir(parents=True)

    base_rows = [
        {
            "id": "zh-en-easy",
            "llm_a": "model_a",
            "llm_b": "model_b",
            "item_id": "item-1",
            "name": "item-1",
            "difficulty": "easy",
            "source_language": "zh",
            "target_language": "en",
            "analysis": "<answer>A</answer>",
        },
        {
            "id": "zh-en-hard",
            "llm_a": "model_a",
            "llm_b": "model_b",
            "item_id": "item-2",
            "name": "item-2",
            "difficulty": "hard",
            "source_language": "zh",
            "target_language": "en",
            "analysis": "<answer>A</answer>",
        },
        {
            "id": "en-zh-easy",
            "llm_a": "model_a",
            "llm_b": "model_b",
            "item_id": "item-3",
            "name": "item-3",
            "difficulty": "easy",
            "source_language": "en",
            "target_language": "zh",
            "analysis": "<answer>B</answer>",
        },
        {
            "id": "en-zh-hard",
            "llm_a": "model_a",
            "llm_b": "model_b",
            "item_id": "item-4",
            "name": "item-4",
            "difficulty": "hard",
            "source_language": "en",
            "target_language": "zh",
            "analysis": "<answer>B</answer>",
        },
    ]
    write_jsonl(snapshot_dir / "base_set.gemini-2.5-flash.jsonl", base_rows)

    assert generate_base_scores(
        "gemini-2.5-flash",
        str(snapshot_dir),
        task="benchmark_tasks/translation_zh_en_bidirectional_v1.yaml",
    )

    report_path = snapshot_dir / "reports" / "base_set.gemini-2.5-flash_scores.json"
    report_rows = json.loads(report_path.read_text(encoding="utf-8"))

    assert {
        "overall",
        "zh_en",
        "zh_en_easy",
        "zh_en_hard",
        "en_zh",
        "en_zh_easy",
        "en_zh_hard",
    } == {row["slice"] for row in report_rows}
    assert len({(row["model"], row["slice"]) for row in report_rows}) == len(report_rows)
    assert all(row["slice"] not in {"easy", "hard"} for row in report_rows)
