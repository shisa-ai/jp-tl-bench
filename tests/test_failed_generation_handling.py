import json
from pathlib import Path

import pytest

import generate_shootout_data
from generate_translation_data import Translator
from translation_comparer_any_model import validate_pair_record

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "task1"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_json_fixture(name: str) -> dict:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def test_translator_parse_marks_successful_outputs_as_ok():
    translator = Translator(
        model_name="fixture-model",
        base_url="https://example.com/v1",
        api_key="test-key",
    )

    parsed = translator.parse(
        {
            "name": "fixture-item",
            "text": "hello",
            "difficulty": "easy",
            "english": True,
        },
        "<translation>こんにちは</translation>",
        "prompt body",
        {"temperature": 0.1},
    )

    assert parsed["status"] == "ok"
    assert parsed["translation"] == "こんにちは"


def test_translate_item_serializes_failures_with_machine_readable_status(monkeypatch):
    translator = Translator(
        model_name="fixture-model",
        base_url="https://example.com/v1",
        api_key="test-key",
    )

    def raise_api_error(**_kwargs):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr("generate_translation_data.time.sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("generate_translation_data.random.uniform", lambda *_args, **_kwargs: 0.0)
    monkeypatch.setattr(translator, "get_prompt", lambda _item: "prompt body")
    monkeypatch.setattr(translator.client.chat.completions, "create", raise_api_error)

    failed = translator.translate_item(
        {
            "name": "fixture-item",
            "text": "hello",
            "difficulty": "easy",
            "english": True,
        }
    )

    assert failed["status"] == "failed"
    assert failed["translation"].startswith("[TRANSLATION FAILED:")
    assert failed["generation_config"]["error"].startswith("API error:")


def test_generate_translation_pairs_rejects_failed_generation_rows(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    snapshot_dir = tmp_path / "baseset" / "v1.0"
    monkeypatch.setattr(generate_shootout_data, "BASESET_SNAPSHOT_DIR", str(snapshot_dir))
    failed_row = load_json_fixture("translation_row_failed.json")
    ok_row = load_json_fixture("translation_row_ok.json")

    write_jsonl(
        tmp_path / "translations" / "candidate__model.jsonl",
        [failed_row],
    )
    write_jsonl(
        snapshot_dir / "translations" / "base__model.jsonl",
        [ok_row],
    )

    with pytest.raises(ValueError, match="failed generation"):
        generate_shootout_data.generate_translation_pairs(
            test_model_file="candidate__model.jsonl",
            output_path=str(tmp_path / "pairs.jsonl"),
        )


def test_validate_pair_record_rejects_failed_generation_placeholders():
    failed_row = load_json_fixture("translation_row_failed.json")
    with pytest.raises(ValueError, match="failed generation placeholder"):
        validate_pair_record(
            {
                "id": "pair-1",
                "formatted_data": (
                    f"## Name: {failed_row['name']}\n\n"
                    "## Source Text:\n"
                    f"{failed_row['source_text']}\n\n"
                    "## Translation A\n"
                    f"{failed_row['translation']}\n\n"
                    "## Translation B\n"
                    "こんにちは\n\n"
                    "---\n"
                ),
            }
        )
