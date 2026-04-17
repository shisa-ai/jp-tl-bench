from pathlib import Path

from benchmark_tasks import load_task_config


REPO_ROOT = Path(__file__).resolve().parents[1]
JP_TASK_PATH = REPO_ROOT / "benchmark_tasks" / "translation_ja_en_bidirectional_v1.yaml"


def test_jp_task_normalizes_language_column_rows_without_legacy_english_field():
    task = load_task_config(JP_TASK_PATH)
    normalized = task.normalize_record(
        {
            "item_id": "sample_convo_15",
            "text": "Hello there.",
            "difficulty": "easy",
            "language": "english",
            "metadata": "NA",
        },
        require_source_text=True,
    )

    assert normalized["item_id"] == "sample_convo_15"
    assert normalized["name"] == "sample_convo_15"
    assert normalized["source_text"] == "Hello there."
    assert normalized["source_language"] == "en"
    assert normalized["target_language"] == "ja"
    assert normalized["english"] is True


def test_jp_task_supports_english_rows_and_skips_chinese_rows():
    task = load_task_config(JP_TASK_PATH)

    assert task.supports_record(
        {
            "item_id": "sample_convo_15",
            "text": "Hello there.",
            "difficulty": "easy",
            "language": "english",
            "metadata": "NA",
        }
    )
    assert not task.supports_record(
        {
            "item_id": "zh_01",
            "text": "你好，世界。",
            "difficulty": "hard",
            "language": "chinese",
            "metadata": "{}",
        }
    )
