from __future__ import annotations

import csv
import json
import os
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Iterable

from datasets import load_dataset

from benchmark_tasks import load_task_config


DATASET_REPO_NAME = "shisa-ai/bt_translation_set_global"
DATASET_CONFIGS = (
    "translation_ja_en_bidirectional_v1",
    "translation_zh_en_bidirectional_v1",
    "translation_zh_ja_bidirectional_v1",
)
REQUIRED_COLUMNS = ("item_id", "name", "text", "difficulty", "language", "metadata")
EXPECTED_COUNTS = {
    "translation_ja_en_bidirectional_v1": {
        "rows": 70,
        "languages": {"en": 34, "ja": 36},
        "difficulties": {"easy": 30, "hard": 40},
    },
    "translation_zh_en_bidirectional_v1": {
        "rows": 67,
        "languages": {"en": 34, "zh": 33},
        "difficulties": {"easy": 27, "hard": 40},
    },
    "translation_zh_ja_bidirectional_v1": {
        "rows": 69,
        "languages": {"ja": 36, "zh": 33},
        "difficulties": {"easy": 27, "hard": 42},
    },
}
HEADER_KEY_MAP = {
    "标题": "title",
    "原始链接": "original_url",
    "规范化链接": "normalized_url",
    "最终链接": "final_url",
    "替换来源": "replacement_url",
    "替代来源": "replacement_url",
    "HTTP状态": "http_status",
    "提取方式": "extraction_method",
}
INVENTORY_COLUMNS = (
    "item_id",
    "name",
    "source_language",
    "task_direction",
    "difficulty",
    "category",
    "description",
)
CHINESE_SOURCE_MANIFEST_COLUMNS = (
    "item_id",
    "file",
    "difficulty",
    "source_title",
    "source_url",
    "access_date",
    "extraction_notes",
    "license_or_redistribution_status",
    "reviewer_signoff",
    "source_family",
    "qa_overall",
    "source_text",
)
DEFAULT_LICENSE_OR_REDISTRIBUTION_STATUS = "private_dataset_only_manual_review_required"

CN_SLOT_TO_FILE = {
    "zh_01": "03_tvos.txt",
    "zh_02": "04__E5_9C_A8_E7_94_B5_E8_84_91_E6_88_96-mac-_E4_B8_8A_E4_B8_8B_.txt",
    "zh_03": "05_cold-calling-guide.txt",
    "zh_04": "22_overcoming-sales-objections-40-examples-strategies-and-rebut.txt",
    "zh_05": "11_uncategorized-298956_html.txt",
    "zh_06": "06_mzc002007knmh3g_html.txt",
    "zh_07": "27_hongmao_lantu_qixiazhuan.txt",
    "zh_08": "21_BLG026-ru-he-zhuan-xie-dui-hua-bao-kuo-ge-shi-li_html.txt",
    "zh_09": "23_page_htm.txt",
    "zh_10": "01_15078370_html.txt",
    "zh_11": "12_about.txt",
    "zh_12": "28_wo_yao_huahua_zhengqian.txt",
    "zh_13": "14_c_html.txt",
    "zh_14": "29_bali_island_feature.txt",
    "zh_15": "24_79258.txt",
    "zh_16": "15_c_html.txt",
    "zh_17": "13_onebook_php.txt",
    "zh_18": "30_women_de_qingchun_ch1.txt",
    "zh_19": "16_1039955910.txt",
    "zh_20": "17_1047229389.txt",
    "zh_21": "09_c418967-40115538_html.txt",
    "zh_22": "19_newsDetail_forward_2752725.txt",
    "zh_23": "25__E5_8D_B7007.txt",
    "zh_24": "26__E5_A0_B1_E4_BB_BB_E5_B0_91_E5_8D_BF_E6_9B_B8.txt",
    "zh_25": "07_c418958-32106952_html.txt",
    "zh_26": "31_siqinge_rile_profile.txt",
    "zh_27": "02_1012917.txt",
    "zh_28": "32_drone_flight_guide.txt",
    "zh_29": "20_newsDetail_forward_32975767.txt",
    "zh_30": "18_newsDetail_forward_2488152.txt",
    "zh_31": "08_c418925-32686461_html.txt",
    "zh_32": "10_c418925-40439145_html.txt",
    "zh_33": "33_xue_fan_profile.txt",
}
CN_SLOT_DIFFICULTY = {
    "zh_01": "easy",
    "zh_02": "easy",
    "zh_03": "easy",
    "zh_04": "easy",
    "zh_05": "easy",
    "zh_06": "easy",
    "zh_07": "easy",
    "zh_08": "easy",
    "zh_09": "easy",
    "zh_10": "easy",
    "zh_11": "easy",
    "zh_12": "easy",
    "zh_13": "hard",
    "zh_14": "hard",
    "zh_15": "hard",
    "zh_16": "hard",
    "zh_17": "hard",
    "zh_18": "hard",
    "zh_19": "hard",
    "zh_20": "hard",
    "zh_21": "hard",
    "zh_22": "hard",
    "zh_23": "hard",
    "zh_24": "hard",
    "zh_25": "hard",
    "zh_26": "hard",
    "zh_27": "hard",
    "zh_28": "hard",
    "zh_29": "hard",
    "zh_30": "hard",
    "zh_31": "hard",
    "zh_32": "hard",
    "zh_33": "hard",
}


def convert_legacy_row_to_global(row: dict, *, force_language: str | None = None) -> dict:
    if force_language is None:
        if row.get("english") not in (True, False):
            raise ValueError(f"Legacy row is missing boolean english flag: {row!r}")
        language = "en" if row["english"] else "ja"
    else:
        language = force_language
    item_id = row.get("item_id") or row.get("name")
    if not item_id:
        raise ValueError(f"Legacy row is missing name/item_id: {row!r}")
    return {
        "item_id": item_id,
        "name": item_id,
        "text": row["text"],
        "difficulty": row["difficulty"],
        "language": language,
        "metadata": "NA",
    }


def _legacy_rows_with_unique_item_ids(rows: Iterable[dict]) -> list[dict]:
    rows = [dict(row) for row in rows]
    name_counts = Counter(row.get("name") for row in rows)
    assigned_ids: set[str] = set()
    normalized_rows: list[dict] = []

    for row in rows:
        name = row.get("name")
        if not name:
            raise ValueError(f"Legacy row is missing name: {row!r}")
        if name_counts[name] == 1:
            item_id = name
        else:
            item_id = f"{name}__{row['difficulty']}"
            suffix = 2
            while item_id in assigned_ids:
                item_id = f"{name}__{row['difficulty']}__{suffix}"
                suffix += 1
        assigned_ids.add(item_id)
        row["item_id"] = item_id
        normalized_rows.append(row)

    return normalized_rows


def _parse_cn_headers(path: Path) -> tuple[dict, str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    metadata: dict[str, str] = {}
    body_start = 0
    for idx, line in enumerate(lines):
        if not line.strip():
            body_start = idx + 1
            break
        if "：" not in line:
            body_start = idx
            break
        key, value = line.split("：", 1)
        mapped = HEADER_KEY_MAP.get(key.strip())
        if mapped:
            metadata[mapped] = value.strip()
    else:
        body_start = len(lines)

    body = "\n".join(lines[body_start:]).strip()
    metadata["source_file"] = path.name
    return metadata, body


def convert_cn_text_file_to_row(path: Path, *, item_id: str, difficulty: str) -> dict:
    metadata, body = _parse_cn_headers(path)
    if not body:
        raise ValueError(f"Chinese source file has empty body: {path}")
    return {
        "item_id": item_id,
        "name": item_id,
        "text": body,
        "difficulty": difficulty,
        "language": "zh",
        "metadata": json.dumps(metadata, ensure_ascii=False, sort_keys=True),
    }


def load_legacy_translation_test_rows() -> list[dict]:
    dataset = load_dataset("shisa-ai/bt_translation_test", split="train")
    return [dict(row) for row in dataset]


def build_ja_rows(legacy_rows: Iterable[dict]) -> list[dict]:
    rows = [
        convert_legacy_row_to_global(row)
        for row in _legacy_rows_with_unique_item_ids(legacy_rows)
    ]
    return sorted(rows, key=lambda row: row["item_id"])


def build_zh_rows(
    legacy_rows: Iterable[dict],
    *,
    cn_texts_dir: Path,
    cn_slot_to_file: dict[str, str] | None = None,
    cn_difficulties: dict[str, str] | None = None,
) -> list[dict]:
    cn_slot_to_file = cn_slot_to_file or CN_SLOT_TO_FILE
    cn_difficulties = cn_difficulties or CN_SLOT_DIFFICULTY

    en_rows = [
        convert_legacy_row_to_global(row, force_language="en")
        for row in _legacy_rows_with_unique_item_ids(legacy_rows)
        if row.get("english") is True
    ]
    zh_rows = [
        convert_cn_text_file_to_row(
            cn_texts_dir / cn_slot_to_file[item_id],
            item_id=item_id,
            difficulty=cn_difficulties[item_id],
        )
        for item_id in sorted(cn_slot_to_file)
    ]
    return sorted(en_rows + zh_rows, key=lambda row: row["item_id"])


def build_zh_ja_rows(ja_rows: Iterable[dict], zh_rows: Iterable[dict]) -> list[dict]:
    rows = [
        dict(row)
        for row in [*ja_rows, *zh_rows]
        if row["language"] in {"ja", "zh"}
    ]
    return sorted(rows, key=lambda row: row["item_id"])


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def write_csv_rows(path: Path, rows: list[dict], columns: tuple[str, ...]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _read_csv_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _dataset_card_text() -> str:
    return """---
configs:
- config_name: translation_ja_en_bidirectional_v1
  data_files:
  - split: train
    path: data/translation_ja_en_bidirectional_v1/train.jsonl
- config_name: translation_zh_en_bidirectional_v1
  data_files:
  - split: train
    path: data/translation_zh_en_bidirectional_v1/train.jsonl
- config_name: translation_zh_ja_bidirectional_v1
  data_files:
  - split: train
    path: data/translation_zh_ja_bidirectional_v1/train.jsonl
---

# bt_translation_set_global

Private benchmark dataset export for the task configs in this repository.

## Configs

- `translation_ja_en_bidirectional_v1`: migrated version of `shisa-ai/bt_translation_test`
- `translation_zh_en_bidirectional_v1`: shared English-source rows plus the curated Chinese-source benchmark set
- `translation_zh_ja_bidirectional_v1`: existing Japanese-source rows plus the curated Chinese-source benchmark set

## Row Schema

Each row uses the same lightweight contract:

- `item_id`: stable item identifier
- `name`: display name
- `text`: source text
- `difficulty`: `easy` or `hard`
- `language`: source-language code
- `metadata`: `NA` for legacy rows or a JSON string for curated Chinese rows
"""


def build_dataset_card_text(
    *,
    manifest_rows: list[dict],
    inventory_rows: list[dict],
) -> str:
    direction_counts = Counter(row["task_direction"] for row in inventory_rows)
    category_counts = Counter(row["category"] for row in inventory_rows if row["category"])
    top_categories = ", ".join(
        f"{category} ({count})"
        for category, count in category_counts.most_common(6)
    )
    source_families = Counter(row["source_family"] for row in manifest_rows if row["source_family"])
    top_source_families = ", ".join(
        f"{family} ({count})"
        for family, count in source_families.most_common(6)
    )

    return f"""---
configs:
- config_name: translation_ja_en_bidirectional_v1
  data_files:
  - split: train
    path: data/translation_ja_en_bidirectional_v1/train.jsonl
- config_name: translation_zh_en_bidirectional_v1
  data_files:
  - split: train
    path: data/translation_zh_en_bidirectional_v1/train.jsonl
- config_name: translation_zh_ja_bidirectional_v1
  data_files:
  - split: train
    path: data/translation_zh_ja_bidirectional_v1/train.jsonl
---

# bt_translation_set_global

Private benchmark dataset export for the task configs in this repository.

## Configs

- `translation_ja_en_bidirectional_v1`: immutable JP v1 export with `{direction_counts.get("EN->JA", 0) + direction_counts.get("JA->EN", 0)}` rows
- `translation_zh_en_bidirectional_v1`: bidirectional ZH/EN export with `{len(manifest_rows)}` curated Chinese-source rows and shared English-source rows
- `translation_zh_ja_bidirectional_v1`: bidirectional ZH/JA export reusing the curated Chinese-source rows and existing Japanese-source rows

## Source Provenance

- Chinese source manifest rows: `{len(manifest_rows)}`
- Chinese source families: {top_source_families or "see docs/chinese_source_manifest.csv"}
- Inventory category highlights: {top_categories or "see docs/translation_set_inventory.csv"}

## Row Schema

Each row uses the same lightweight contract:

- `item_id`: stable item identifier
- `name`: display name
- `text`: source text
- `difficulty`: `easy` or `hard`
- `language`: source-language code
- `metadata`: `NA` for legacy rows or a JSON string for curated Chinese rows
"""


def write_dataset_card(
    output_root: Path,
    *,
    manifest_rows: list[dict],
    inventory_rows: list[dict],
) -> Path:
    readme_path = output_root / "README.md"
    readme_path.parent.mkdir(parents=True, exist_ok=True)
    readme_path.write_text(
        build_dataset_card_text(
            manifest_rows=manifest_rows,
            inventory_rows=inventory_rows,
        ),
        encoding="utf-8",
    )
    return readme_path


def build_translation_set_global(
    *,
    output_root: Path,
    legacy_rows: list[dict] | None = None,
    cn_texts_dir: Path | None = None,
    cn_slot_to_file: dict[str, str] | None = None,
    cn_difficulties: dict[str, str] | None = None,
) -> dict[str, Path]:
    legacy_rows = legacy_rows or load_legacy_translation_test_rows()
    cn_texts_dir = cn_texts_dir or Path("docs/cn_texts")

    ja_rows = build_ja_rows(legacy_rows)
    zh_rows = build_zh_rows(
        legacy_rows,
        cn_texts_dir=cn_texts_dir,
        cn_slot_to_file=cn_slot_to_file,
        cn_difficulties=cn_difficulties,
    )
    zh_ja_rows = build_zh_ja_rows(ja_rows, zh_rows)

    readme_path = output_root / "README.md"
    readme_path.parent.mkdir(parents=True, exist_ok=True)
    readme_path.write_text(_dataset_card_text(), encoding="utf-8")
    legacy_default_path = output_root / "data" / "train.jsonl"
    if legacy_default_path.exists():
        legacy_default_path.unlink()

    ja_path = output_root / "data" / "translation_ja_en_bidirectional_v1" / "train.jsonl"
    zh_path = output_root / "data" / "translation_zh_en_bidirectional_v1" / "train.jsonl"
    zh_ja_path = output_root / "data" / "translation_zh_ja_bidirectional_v1" / "train.jsonl"
    _write_jsonl(ja_path, ja_rows)
    _write_jsonl(zh_path, zh_rows)
    _write_jsonl(zh_ja_path, zh_ja_rows)

    return {
        "readme": readme_path,
        "translation_ja_en_bidirectional_v1": ja_path,
        "translation_zh_en_bidirectional_v1": zh_path,
        "translation_zh_ja_bidirectional_v1": zh_ja_path,
    }


def _load_access_date(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"^Date:\s*(\d{4}-\d{2}-\d{2})$", text, re.MULTILINE)
    if not match:
        raise ValueError(f"Could not find 'Date: YYYY-MM-DD' in {path}")
    return match.group(1)


def _read_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_rows_by_file(path: Path) -> dict[str, dict]:
    rows = _read_csv_rows(path)
    return {row["file"]: row for row in rows}


def build_chinese_source_manifest(
    *,
    cn_texts_dir: Path | None = None,
    cn_slot_to_file: dict[str, str] | None = None,
    cn_difficulties: dict[str, str] | None = None,
    cn_links_path: Path | None = None,
    cn_texts_audit_path: Path | None = None,
    cn_benchmark_qa_audit_path: Path | None = None,
    cn_benchmark_qa_checklist_path: Path | None = None,
) -> list[dict]:
    cn_texts_dir = cn_texts_dir or Path("docs/cn_texts")
    cn_slot_to_file = cn_slot_to_file or CN_SLOT_TO_FILE
    cn_difficulties = cn_difficulties or CN_SLOT_DIFFICULTY
    cn_links_path = cn_links_path or Path("docs/cn_links.txt")
    cn_texts_audit_path = cn_texts_audit_path or Path("docs/cn_texts_audit.csv")
    cn_benchmark_qa_audit_path = cn_benchmark_qa_audit_path or Path("docs/cn_benchmark_qa_audit.csv")
    cn_benchmark_qa_checklist_path = (
        cn_benchmark_qa_checklist_path or Path("docs/cn_benchmark_qa_checklist.md")
    )

    access_date = _load_access_date(cn_benchmark_qa_checklist_path)
    links = _read_lines(cn_links_path)
    ordered_item_ids = sorted(cn_slot_to_file)
    if len(links) < len(ordered_item_ids):
        raise ValueError(
            f"Expected at least {len(ordered_item_ids)} source links in {cn_links_path}, found {len(links)}"
        )
    link_by_item_id = {
        item_id: links[index]
        for index, item_id in enumerate(ordered_item_ids)
    }
    text_audit = _read_rows_by_file(cn_texts_audit_path)
    qa_audit = _read_rows_by_file(cn_benchmark_qa_audit_path)

    manifest_rows: list[dict] = []
    for item_id in ordered_item_ids:
        filename = cn_slot_to_file[item_id]
        metadata, body = _parse_cn_headers(cn_texts_dir / filename)
        source_audit = text_audit.get(filename, {})
        qa_row = qa_audit.get(filename, {})
        source_url = (
            metadata.get("replacement_url")
            or metadata.get("original_url")
            or metadata.get("normalized_url")
            or metadata.get("final_url")
            or link_by_item_id[item_id]
        )
        extraction_parts = [
            metadata.get("extraction_method"),
            source_audit.get("audit_note"),
            qa_row.get("notes"),
        ]
        extraction_notes = " | ".join(part for part in extraction_parts if part)
        qa_overall = qa_row.get("overall", "")
        reviewer_signoff = f"source_audit={source_audit.get('status', 'unknown')}; qa={qa_overall or 'unknown'}; reviewed={access_date}"

        manifest_rows.append(
            {
                "item_id": item_id,
                "file": filename,
                "difficulty": cn_difficulties[item_id],
                "source_title": metadata.get("title", item_id),
                "source_url": source_url,
                "access_date": access_date,
                "extraction_notes": extraction_notes,
                "license_or_redistribution_status": DEFAULT_LICENSE_OR_REDISTRIBUTION_STATUS,
                "reviewer_signoff": reviewer_signoff,
                "source_family": source_audit.get("source_family", ""),
                "qa_overall": qa_overall,
                "source_text": body,
            }
        )

    return manifest_rows


def build_translation_set_inventory(
    *,
    legacy_rows: list[dict] | None = None,
    template_path: Path | None = None,
) -> list[dict]:
    legacy_rows = legacy_rows or load_legacy_translation_test_rows()
    template_path = template_path or Path("docs/translation_set_inventory.csv")

    template_rows = _read_csv_rows(template_path)
    template_by_key = {
        (
            row["name"],
            row["source_language"],
            row["task_direction"],
            row["difficulty"],
        ): row
        for row in template_rows
    }

    inventory_rows: list[dict] = []
    for row in _legacy_rows_with_unique_item_ids(legacy_rows):
        source_language = "English" if row["english"] else "Japanese"
        task_direction = "EN->JA" if row["english"] else "JA->EN"
        key = (row["name"], source_language, task_direction, row["difficulty"])
        template = template_by_key.get(key, {})
        inventory_rows.append(
            {
                "item_id": row["item_id"],
                "name": row["name"],
                "source_language": source_language,
                "task_direction": task_direction,
                "difficulty": row["difficulty"],
                "category": template.get("category", ""),
                "description": template.get("description", ""),
            }
        )

    return sorted(inventory_rows, key=lambda row: row["item_id"])


def _load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _validate_rows(rows: list[dict], *, config_name: str) -> None:
    seen_ids: set[str] = set()
    language_counts: Counter[str] = Counter()
    difficulty_counts: Counter[str] = Counter()
    expected = EXPECTED_COUNTS[config_name]
    sorted_item_ids = sorted(row["item_id"] for row in rows)

    if [row["item_id"] for row in rows] != sorted_item_ids:
        raise ValueError(f"{config_name}: rows must be deterministically sorted by item_id")

    for row in rows:
        missing = [column for column in REQUIRED_COLUMNS if column not in row]
        if missing:
            raise ValueError(f"{config_name}: missing required columns {missing} in row {row!r}")
        if "english" in row:
            raise ValueError(f"{config_name}: unexpected legacy english column in row {row!r}")
        if row["item_id"] in seen_ids:
            raise ValueError(f"{config_name}: duplicate item_id '{row['item_id']}'")
        seen_ids.add(row["item_id"])
        language_counts[row["language"]] += 1
        difficulty_counts[row["difficulty"]] += 1
        if row["difficulty"] not in {"easy", "hard"}:
            raise ValueError(f"{config_name}: invalid difficulty '{row['difficulty']}'")
        if row["language"] not in expected["languages"]:
            raise ValueError(f"{config_name}: invalid language '{row['language']}' in row {row!r}")
        if not isinstance(row["metadata"], str):
            raise ValueError(f"{config_name}: metadata must be a string for row {row['item_id']}")
        if not row["text"].strip():
            raise ValueError(f"{config_name}: empty text for row {row['item_id']}")
        if row["name"] != row["item_id"]:
            raise ValueError(
                f"{config_name}: expected name to match item_id for row {row['item_id']}"
            )
        if row["metadata"] != "NA":
            metadata = json.loads(row["metadata"])
            if row["language"] == "zh":
                required_metadata = {"title", "source_file", "extraction_method"}
                missing_metadata = sorted(required_metadata - metadata.keys())
                if missing_metadata:
                    raise ValueError(
                        f"{config_name}: missing metadata keys {missing_metadata} for row {row['item_id']}"
                    )
                if not any(
                    metadata.get(key)
                    for key in ("original_url", "replacement_url", "normalized_url", "final_url")
                ):
                    raise ValueError(
                        f"{config_name}: expected at least one source URL in metadata for row {row['item_id']}"
                    )
        elif row["language"] == "zh":
            raise ValueError(f"{config_name}: Chinese row {row['item_id']} must not use 'NA' metadata")

    if len(rows) != expected["rows"]:
        raise ValueError(
            f"{config_name}: expected {expected['rows']} rows, found {len(rows)}"
        )
    if dict(language_counts) != expected["languages"]:
        raise ValueError(
            f"{config_name}: expected language counts {expected['languages']}, found {dict(language_counts)}"
        )
    if dict(difficulty_counts) != expected["difficulties"]:
        raise ValueError(
            f"{config_name}: expected difficulty counts {expected['difficulties']}, found {dict(difficulty_counts)}"
        )


def _validate_manifest_rows(rows: list[dict], *, dataset_rows: list[dict], config_name: str) -> None:
    if config_name not in {"translation_zh_en_bidirectional_v1", "translation_zh_ja_bidirectional_v1"}:
        return

    manifest_by_id = {row["item_id"]: row for row in rows}
    zh_dataset_rows = [row for row in dataset_rows if row["language"] == "zh"]
    if set(manifest_by_id) != {row["item_id"] for row in zh_dataset_rows}:
        raise ValueError(f"{config_name}: Chinese source manifest item_ids do not match dataset rows")

    for row in rows:
        missing = [column for column in CHINESE_SOURCE_MANIFEST_COLUMNS if not row.get(column)]
        if missing:
            raise ValueError(f"{config_name}: manifest row {row.get('item_id', 'unknown')} missing {missing}")
        dataset_row = next(dataset_row for dataset_row in zh_dataset_rows if dataset_row["item_id"] == row["item_id"])
        if row["source_text"] != dataset_row["text"]:
            raise ValueError(f"{config_name}: manifest source_text mismatch for {row['item_id']}")
        if row["difficulty"] != dataset_row["difficulty"]:
            raise ValueError(f"{config_name}: manifest difficulty mismatch for {row['item_id']}")


def _validate_round_trip_pair_generation(rows: list[dict], *, task: str | os.PathLike[str] | None) -> int:
    import generate_shootout_data

    task_config = load_task_config(task)
    with tempfile.TemporaryDirectory() as tempdir:
        temp_root = Path(tempdir)
        translations_dir = temp_root / "translations"
        snapshot_root = temp_root / "baseset" / "v1.0"
        base_translations_dir = snapshot_root / "translations"
        translations_dir.mkdir(parents=True)
        base_translations_dir.mkdir(parents=True)

        candidate_rows = []
        anchor_rows = []
        for row in rows:
            normalized = task_config.normalize_record(row, require_source_text=True)
            base_payload = {
                **normalized,
                "source_text": normalized["source_text"],
                "status": "ok",
                "temperature": 0.1,
                "top_p": 0.85,
                "low_context": False,
                "ultra_low_context": False,
            }
            candidate_rows.append(
                {
                    **base_payload,
                    "translation": f"candidate translation for {normalized['item_id']}",
                    "generation_config": {"model": "candidate"},
                }
            )
            anchor_rows.append(
                {
                    **base_payload,
                    "translation": f"anchor translation for {normalized['item_id']}",
                    "generation_config": {"model": "anchor"},
                }
            )

        candidate_path = translations_dir / "candidate.jsonl"
        anchor_path = base_translations_dir / "anchor.jsonl"
        _write_jsonl(candidate_path, candidate_rows)
        _write_jsonl(anchor_path, anchor_rows)

        original_cwd = Path.cwd()
        original_snapshot_dir = generate_shootout_data.BASESET_SNAPSHOT_DIR
        try:
            os.chdir(temp_root)
            generate_shootout_data.BASESET_SNAPSHOT_DIR = str(snapshot_root)
            output_path = temp_root / "pairs.jsonl"
            generate_shootout_data.generate_translation_pairs(
                test_model_file="candidate.jsonl",
                force=True,
                output_path=output_path,
                task=str(task_config.path),
            )
        finally:
            os.chdir(original_cwd)
            generate_shootout_data.BASESET_SNAPSHOT_DIR = original_snapshot_dir

        pair_rows = _load_jsonl(temp_root / "pairs.jsonl")
        if len(pair_rows) != len(rows):
            raise ValueError(
                f"{task_config.dataset.config}: expected {len(rows)} generated pairs, found {len(pair_rows)}"
            )
        return len(pair_rows)


def validate_translation_set_global(
    output_root: Path,
    *,
    task: str | os.PathLike[str] | None = None,
    manifest_path: Path | None = None,
) -> dict[str, int]:
    config_names = [load_task_config(task).dataset.config] if task else list(DATASET_CONFIGS)
    manifest_rows = _read_csv_rows(manifest_path) if manifest_path and manifest_path.exists() else None
    results: dict[str, int] = {}

    for config_name in config_names:
        path = output_root / "data" / config_name / "train.jsonl"
        if not path.exists():
            raise ValueError(f"Missing dataset file for {config_name}: {path}")
        rows = _load_jsonl(path)
        _validate_rows(rows, config_name=config_name)
        if manifest_rows is not None:
            _validate_manifest_rows(manifest_rows, dataset_rows=rows, config_name=config_name)
        active_task = task
        if active_task is None:
            active_task = Path("benchmark_tasks") / f"{config_name}.yaml"
        results[config_name] = len(rows)
        results[f"{config_name}:pair_round_trip"] = _validate_round_trip_pair_generation(
            rows,
            task=active_task,
        )

    return results
