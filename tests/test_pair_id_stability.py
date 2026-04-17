import hashlib
import json
from pathlib import Path

from click.testing import CliRunner

import generate_shootout_data
from generate_shootout_data import (
    PAIR_ID_SCHEMA_V1,
    compute_pair_fingerprint,
    compute_pair_id_v1,
    generate_translation_pairs,
)
from translation_comparer_any_model import existing_judgment_matches_pair, main as compare_cli


REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_RESULTS_DIR = REPO_ROOT / "results" / "v1.0" / "shisa-ai__chotto-e4b-20260408" / "gemini-2.5-flash"


def load_real_pair_and_judgment():
    judgments_by_id = {}
    with (REAL_RESULTS_DIR / "judgments.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            judgments_by_id[row["id"]] = row

    with (REAL_RESULTS_DIR / "pairs.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            judgment = judgments_by_id.get(row["id"])
            if judgment:
                return row, judgment

    raise AssertionError("Expected at least one matching pair/judgment row in the real v1.0 sample data")


def test_pair_id_v1_contract_and_fingerprint_track_content_identity():
    assert compute_pair_id_v1("foo.jsonl", "bar.jsonl", "amazon_1") == hashlib.md5(
        b"foo.jsonl_bar.jsonl_amazon_1"
    ).hexdigest()

    pair = {
        "id": compute_pair_id_v1("foo.jsonl", "bar.jsonl", "amazon_1"),
        "pair_id_schema": PAIR_ID_SCHEMA_V1,
        "llm_a": "foo",
        "llm_b": "bar",
        "name": "amazon_1",
        "english": False,
        "difficulty": "easy",
        "formatted_data": "## Name: amazon_1\n\n## Source Text:\nこんにちは\n\n## Translation A\nhello\n\n## Translation B\nhi\n\n---\n",
        "llm_a_generation_config": {"model": "foo", "temperature": 0.1},
        "llm_b_generation_config": {"model": "bar", "temperature": 0.2},
    }

    fingerprint = compute_pair_fingerprint(pair)
    changed_pair = dict(pair)
    changed_pair["formatted_data"] = pair["formatted_data"].replace("hello", "hey")

    assert fingerprint
    assert fingerprint != compute_pair_fingerprint(changed_pair)


def test_pair_fingerprint_v1_ignores_future_additive_pair_metadata():
    pair = {
        "id": compute_pair_id_v1("foo.jsonl", "bar.jsonl", "amazon_1"),
        "pair_id_schema": PAIR_ID_SCHEMA_V1,
        "llm_a": "foo",
        "llm_b": "bar",
        "name": "amazon_1",
        "english": False,
        "difficulty": "easy",
        "formatted_data": "## Name: amazon_1\n\n## Source Text:\nこんにちは\n\n## Translation A\nhello\n\n## Translation B\nhi\n\n---\n",
        "llm_a_generation_config": {"model": "foo", "temperature": 0.1},
        "llm_b_generation_config": {"model": "bar", "temperature": 0.2},
    }

    fingerprint = compute_pair_fingerprint(pair)
    future_pair = {
        **pair,
        "task_id": "translation_zh_en_bidirectional_v1",
        "task_version": "v1",
        "source_language": "zh",
        "target_language": "en",
        "slice_tags": ["hard", "literary"],
    }

    assert compute_pair_fingerprint(future_pair) == fingerprint


def test_legacy_v1_judgments_without_fingerprint_are_grandfathered_for_reuse():
    pair, judgment = load_real_pair_and_judgment()

    regenerated_pair = dict(pair)
    regenerated_pair["pair_id_schema"] = PAIR_ID_SCHEMA_V1
    regenerated_pair["pair_fingerprint"] = compute_pair_fingerprint(regenerated_pair)

    assert judgment.get("pair_fingerprint") is None
    assert existing_judgment_matches_pair(judgment, regenerated_pair)


def test_reuse_requires_matching_fingerprint_for_new_judgments():
    pair, _ = load_real_pair_and_judgment()

    regenerated_pair = dict(pair)
    regenerated_pair["pair_id_schema"] = PAIR_ID_SCHEMA_V1
    regenerated_pair["pair_fingerprint"] = compute_pair_fingerprint(regenerated_pair)

    matching_judgment = {
        "id": regenerated_pair["id"],
        "pair_id_schema": PAIR_ID_SCHEMA_V1,
        "pair_fingerprint": regenerated_pair["pair_fingerprint"],
    }
    mismatched_judgment = dict(matching_judgment)
    mismatched_judgment["pair_fingerprint"] = "deadbeef"

    assert existing_judgment_matches_pair(matching_judgment, regenerated_pair)
    assert not existing_judgment_matches_pair(mismatched_judgment, regenerated_pair)


def test_generate_base_pair_ids_are_stable_when_listdir_order_changes(tmp_path, monkeypatch):
    snapshot_dir = tmp_path / "baseset" / "v1.0"
    translations_dir = snapshot_dir / "translations"
    translations_dir.mkdir(parents=True)

    row = {
        "name": "amazon_1",
        "source_text": "hello",
        "translation": "こんにちは",
        "english": True,
        "difficulty": "easy",
        "status": "ok",
    }
    for filename in ("b_model.jsonl", "a_model.jsonl"):
        (translations_dir / filename).write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    monkeypatch.setattr(generate_shootout_data, "BASESET_SNAPSHOT_DIR", str(snapshot_dir))

    output_a = tmp_path / "pairs_a.jsonl"
    output_b = tmp_path / "pairs_b.jsonl"
    real_listdir = generate_shootout_data.os.listdir

    monkeypatch.setattr(
        generate_shootout_data.os,
        "listdir",
        lambda path: ["b_model.jsonl", "a_model.jsonl"] if Path(path) == translations_dir else real_listdir(path),
    )
    generate_translation_pairs(force=True, output_path=output_a)

    monkeypatch.setattr(
        generate_shootout_data.os,
        "listdir",
        lambda path: ["a_model.jsonl", "b_model.jsonl"] if Path(path) == translations_dir else real_listdir(path),
    )
    generate_translation_pairs(force=True, output_path=output_b)

    rows_a = [json.loads(line) for line in output_a.read_text(encoding="utf-8").splitlines()]
    rows_b = [json.loads(line) for line in output_b.read_text(encoding="utf-8").splitlines()]

    assert rows_a == rows_b
    assert rows_a[0]["llm_a"] == "a_model"
    assert rows_a[0]["llm_b"] == "b_model"
    assert rows_a[0]["pair_id_schema"] == PAIR_ID_SCHEMA_V1


def test_comparer_reuses_legacy_base_set_judgment_without_api_call(tmp_path):
    snapshot_dir = tmp_path / "v1.0"
    snapshot_dir.mkdir()

    pair_id = compute_pair_id_v1("candidate.jsonl", "anchor.jsonl", "amazon_1")
    pair = {
        "id": pair_id,
        "llm_a": "candidate",
        "llm_b": "anchor",
        "name": "amazon_1",
        "english": True,
        "difficulty": "easy",
        "formatted_data": "## Name: amazon_1\n\n## Source Text:\nhello\n\n## Translation A\nこんにちは\n\n## Translation B\nやあ\n\n---\n",
    }
    pair_file = tmp_path / "pairs.jsonl"
    pair_file.write_text(json.dumps(pair, ensure_ascii=False) + "\n", encoding="utf-8")

    legacy_judgment = {
        **pair,
        "analysis": "<answer>A</answer>",
        "judge_model": "gemini-2.5-flash",
        "judge_temperature": 0,
        "judge_generation_config": {"model": "gemini-2.5-flash"},
    }
    legacy_output = snapshot_dir / "base_set.gemini-2.5-flash.jsonl"
    legacy_output.write_text(json.dumps(legacy_judgment, ensure_ascii=False) + "\n", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        compare_cli,
        [
            "--base-url",
            "http://unused",
            "--judge-model",
            "gemini-2.5-flash",
            "--generate-base-set",
            "--pairs-file",
            str(pair_file),
        ],
        env={"BASESET_SNAPSHOT_DIR": str(snapshot_dir)},
    )

    assert result.exit_code == 0, result.output
    schema_v2_output = snapshot_dir / "base_set.gemini-2.5-flash.schema-v2.jsonl"
    assert schema_v2_output.exists()

    written_rows = [json.loads(line) for line in schema_v2_output.read_text(encoding="utf-8").splitlines()]
    assert len(written_rows) == 1
    assert written_rows[0]["id"] == pair_id
    assert written_rows[0]["pair_id_schema"] == PAIR_ID_SCHEMA_V1
    pair_view = {
        key: written_rows[0][key]
        for key in ("id", "pair_id_schema", "llm_a", "llm_b", "name", "english", "difficulty", "formatted_data")
    }
    assert written_rows[0]["pair_fingerprint"] == compute_pair_fingerprint(pair_view)
