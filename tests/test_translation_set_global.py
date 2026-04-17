import csv
import json
from pathlib import Path

import pytest
from click.testing import CliRunner

import dataset_tools.translation_set_global as translation_set_global
from dataset_tools.translation_set_global import (
    CN_SLOT_TO_FILE,
    build_chinese_source_manifest,
    build_translation_set_global,
    convert_cn_text_file_to_row,
    convert_legacy_row_to_global,
    validate_translation_set_global,
)
from scripts.validate_hf_dataset import main as validate_hf_dataset_cli


def test_convert_legacy_row_to_global_replaces_english_with_language_and_metadata():
    row = {
        "name": "sample_convo_15",
        "text": "Hello there.",
        "difficulty": "easy",
        "english": True,
    }

    converted = convert_legacy_row_to_global(row)

    assert converted == {
        "item_id": "sample_convo_15",
        "name": "sample_convo_15",
        "text": "Hello there.",
        "difficulty": "easy",
        "language": "en",
        "metadata": "NA",
    }


def test_convert_cn_text_file_to_row_extracts_body_and_metadata(tmp_path):
    source = tmp_path / "01_example.txt"
    source.write_text(
        "\n".join(
            [
                "标题：示例标题",
                "原始链接：https://example.com/original",
                "替换来源：https://example.com/replacement",
                "HTTP状态：200",
                "提取方式：cleaned",
                "",
                "第一段。",
                "第二段。",
            ]
        ),
        encoding="utf-8",
    )

    row = convert_cn_text_file_to_row(
        source,
        item_id="zh_01",
        difficulty="easy",
    )

    assert row["item_id"] == "zh_01"
    assert row["name"] == "zh_01"
    assert row["language"] == "zh"
    assert row["difficulty"] == "easy"
    assert row["text"] == "第一段。\n第二段。"
    metadata = json.loads(row["metadata"])
    assert metadata["title"] == "示例标题"
    assert metadata["original_url"] == "https://example.com/original"
    assert metadata["replacement_url"] == "https://example.com/replacement"
    assert metadata["source_file"] == "01_example.txt"


def test_build_translation_set_global_writes_both_task_configs(tmp_path):
    legacy_rows = [
        {"name": "en_item_1", "text": "Hello", "difficulty": "easy", "english": True},
        {"name": "ja_item_1", "text": "こんにちは", "difficulty": "hard", "english": False},
    ]

    cn_dir = tmp_path / "cn_texts"
    cn_dir.mkdir()
    (cn_dir / "01_cn.txt").write_text(
        "\n".join(
            [
                "标题：中文条目",
                "原始链接：https://example.com/cn",
                "HTTP状态：200",
                "提取方式：cleaned",
                "",
                "中文正文。",
            ]
        ),
        encoding="utf-8",
    )

    build_translation_set_global(
        output_root=tmp_path / "hf_datasets" / "bt_translation_set_global",
        legacy_rows=legacy_rows,
        cn_texts_dir=cn_dir,
        cn_slot_to_file={"zh_01": "01_cn.txt"},
        cn_difficulties={"zh_01": "hard"},
    )

    jp_file = tmp_path / "hf_datasets" / "bt_translation_set_global" / "data" / "translation_ja_en_bidirectional_v1" / "train.jsonl"
    zh_file = tmp_path / "hf_datasets" / "bt_translation_set_global" / "data" / "translation_zh_en_bidirectional_v1" / "train.jsonl"
    assert jp_file.exists()
    assert zh_file.exists()

    jp_rows = [json.loads(line) for line in jp_file.read_text(encoding="utf-8").splitlines()]
    zh_rows = [json.loads(line) for line in zh_file.read_text(encoding="utf-8").splitlines()]

    assert jp_rows == [
        {
            "item_id": "en_item_1",
            "name": "en_item_1",
            "text": "Hello",
            "difficulty": "easy",
            "language": "en",
            "metadata": "NA",
        },
        {
            "item_id": "ja_item_1",
            "name": "ja_item_1",
            "text": "こんにちは",
            "difficulty": "hard",
            "language": "ja",
            "metadata": "NA",
        },
    ]
    assert zh_rows == [
        {
            "item_id": "en_item_1",
            "name": "en_item_1",
            "text": "Hello",
            "difficulty": "easy",
            "language": "en",
            "metadata": "NA",
        },
        {
            "item_id": "zh_01",
            "name": "zh_01",
            "text": "中文正文。",
            "difficulty": "hard",
            "language": "zh",
            "metadata": json.dumps(
                {
                    "title": "中文条目",
                    "original_url": "https://example.com/cn",
                    "http_status": "200",
                    "extraction_method": "cleaned",
                    "source_file": "01_cn.txt",
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
        },
    ]


def test_build_translation_set_global_removes_legacy_default_train_jsonl(tmp_path):
    legacy_rows = [
        {"name": "en_item_1", "text": "Hello", "difficulty": "easy", "english": True},
        {"name": "ja_item_1", "text": "こんにちは", "difficulty": "hard", "english": False},
    ]
    cn_dir = tmp_path / "cn_texts"
    cn_dir.mkdir()
    (cn_dir / "01_cn.txt").write_text(
        "\n".join(
            [
                "标题：中文条目",
                "原始链接：https://example.com/cn",
                "HTTP状态：200",
                "提取方式：cleaned",
                "",
                "中文正文。",
            ]
        ),
        encoding="utf-8",
    )
    dataset_root = tmp_path / "hf_datasets" / "bt_translation_set_global"
    legacy_default_path = dataset_root / "data" / "train.jsonl"
    legacy_default_path.parent.mkdir(parents=True)
    legacy_default_path.write_text("{\"legacy\": true}\n", encoding="utf-8")

    build_translation_set_global(
        output_root=dataset_root,
        legacy_rows=legacy_rows,
        cn_texts_dir=cn_dir,
        cn_slot_to_file={"zh_01": "01_cn.txt"},
        cn_difficulties={"zh_01": "hard"},
    )

    assert not legacy_default_path.exists()


def test_build_chinese_source_manifest_includes_required_provenance_fields(tmp_path):
    cn_dir = tmp_path / "cn_texts"
    cn_dir.mkdir()
    (cn_dir / "01_cn.txt").write_text(
        "\n".join(
            [
                "标题：中文条目",
                "原始链接：https://example.com/cn",
                "HTTP状态：200",
                "提取方式：cleaned",
                "",
                "中文正文。",
            ]
        ),
        encoding="utf-8",
    )

    audit_path = tmp_path / "cn_texts_audit.csv"
    audit_path.write_text(
        "\n".join(
            [
                "file,status,source_family,audit_note",
                "01_cn.txt,pass,feature,clean extraction",
            ]
        ),
        encoding="utf-8",
    )
    qa_audit_path = tmp_path / "cn_benchmark_qa_audit.csv"
    qa_audit_path.write_text(
        "\n".join(
            [
                "file,source_quality,slot_fit,length_floor,judgeability,overall,notes",
                "01_cn.txt,pass,pass,pass,pass,pass,judgeable",
            ]
        ),
        encoding="utf-8",
    )
    checklist_path = tmp_path / "cn_benchmark_qa_checklist.md"
    checklist_path.write_text("Date: 2026-04-17\n", encoding="utf-8")
    links_path = tmp_path / "cn_links.txt"
    links_path.write_text("https://example.com/cn\n", encoding="utf-8")

    manifest_rows = build_chinese_source_manifest(
        cn_texts_dir=cn_dir,
        cn_slot_to_file={"zh_01": "01_cn.txt"},
        cn_difficulties={"zh_01": "easy"},
        cn_links_path=links_path,
        cn_texts_audit_path=audit_path,
        cn_benchmark_qa_audit_path=qa_audit_path,
        cn_benchmark_qa_checklist_path=checklist_path,
    )

    assert len(manifest_rows) == 1
    row = manifest_rows[0]
    assert row["item_id"] == "zh_01"
    assert row["source_text"] == "中文正文。"
    assert row["source_title"] == "中文条目"
    assert row["source_url"] == "https://example.com/cn"
    assert row["access_date"] == "2026-04-17"
    assert "clean extraction" in row["extraction_notes"]
    assert row["license_or_redistribution_status"]
    assert row["reviewer_signoff"]


def test_validate_translation_set_global_rejects_duplicate_item_ids(tmp_path):
    dataset_root = tmp_path / "hf_datasets" / "bt_translation_set_global"
    data_dir = dataset_root / "data" / "translation_ja_en_bidirectional_v1"
    data_dir.mkdir(parents=True)
    rows = [
        {
            "item_id": "dup",
            "name": "dup",
            "text": "a",
            "difficulty": "easy",
            "language": "en",
            "metadata": "NA",
        },
        {
            "item_id": "dup",
            "name": "dup",
            "text": "b",
            "difficulty": "hard",
            "language": "ja",
            "metadata": "NA",
        },
    ]
    (data_dir / "train.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate item_id"):
        validate_translation_set_global(dataset_root)


def test_validate_hf_dataset_cli_accepts_task_and_local_dataset_root(tmp_path, monkeypatch):
    legacy_rows = [
        {"name": "en_item_1", "text": "Hello", "difficulty": "easy", "english": True},
        {"name": "ja_item_1", "text": "こんにちは", "difficulty": "hard", "english": False},
    ]
    cn_dir = tmp_path / "cn_texts"
    cn_dir.mkdir()
    (cn_dir / "01_cn.txt").write_text(
        "\n".join(
            [
                "标题：中文条目",
                "原始链接：https://example.com/cn",
                "HTTP状态：200",
                "提取方式：cleaned",
                "",
                "中文正文。",
            ]
        ),
        encoding="utf-8",
    )
    dataset_root = tmp_path / "hf_datasets" / "bt_translation_set_global"
    build_translation_set_global(
        output_root=dataset_root,
        legacy_rows=legacy_rows,
        cn_texts_dir=cn_dir,
        cn_slot_to_file={"zh_01": "01_cn.txt"},
        cn_difficulties={"zh_01": "hard"},
    )
    audit_path = tmp_path / "cn_texts_audit.csv"
    audit_path.write_text(
        "\n".join(
            [
                "file,status,source_family,audit_note",
                "01_cn.txt,pass,feature,clean extraction",
            ]
        ),
        encoding="utf-8",
    )
    qa_audit_path = tmp_path / "cn_benchmark_qa_audit.csv"
    qa_audit_path.write_text(
        "\n".join(
            [
                "file,source_quality,slot_fit,length_floor,judgeability,overall,notes",
                "01_cn.txt,pass,pass,pass,pass,pass,judgeable",
            ]
        ),
        encoding="utf-8",
    )
    checklist_path = tmp_path / "cn_benchmark_qa_checklist.md"
    checklist_path.write_text("Date: 2026-04-17\n", encoding="utf-8")
    links_path = tmp_path / "cn_links.txt"
    links_path.write_text("https://example.com/cn\n", encoding="utf-8")
    manifest_path = tmp_path / "chinese_source_manifest.csv"
    manifest_rows = build_chinese_source_manifest(
        cn_texts_dir=cn_dir,
        cn_slot_to_file={"zh_01": "01_cn.txt"},
        cn_difficulties={"zh_01": "hard"},
        cn_links_path=links_path,
        cn_texts_audit_path=audit_path,
        cn_benchmark_qa_audit_path=qa_audit_path,
        cn_benchmark_qa_checklist_path=checklist_path,
    )
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=translation_set_global.CHINESE_SOURCE_MANIFEST_COLUMNS,
        )
        writer.writeheader()
        writer.writerows(manifest_rows)
    monkeypatch.setattr(
        translation_set_global,
        "EXPECTED_COUNTS",
        {
            "translation_ja_en_bidirectional_v1": {
                "rows": 2,
                "languages": {"en": 1, "ja": 1},
                "difficulties": {"easy": 1, "hard": 1},
            },
            "translation_zh_en_bidirectional_v1": {
                "rows": 2,
                "languages": {"en": 1, "zh": 1},
                "difficulties": {"easy": 1, "hard": 1},
            },
        },
    )

    runner = CliRunner()
    result = runner.invoke(
        validate_hf_dataset_cli,
        [
            "--task",
            "benchmark_tasks/translation_zh_en_bidirectional_v1.yaml",
            "--dataset-root",
            str(dataset_root),
            "--manifest-path",
            str(manifest_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "translation_zh_en_bidirectional_v1" in result.output


def test_cn_slot_mapping_constant_covers_all_expected_slots():
    assert "zh_01" in CN_SLOT_TO_FILE
    assert "zh_33" in CN_SLOT_TO_FILE
    for filename in CN_SLOT_TO_FILE.values():
        assert (Path("docs/cn_texts") / filename).exists(), filename
