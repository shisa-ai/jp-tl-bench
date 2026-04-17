import json
from pathlib import Path

import pytest

from baseset.generate_set import write_legacy_jp_v1_boundary_metadata
from score_visualizer import load_base_anchor_scores


REPO_ROOT = Path(__file__).resolve().parents[1]
LEGACY_PAIR_FILE = REPO_ROOT / "baseset" / "v1.0" / "base_conversation_pairs.v1.0.jsonl"
LEGACY_REPORT_FILE = REPO_ROOT / "baseset" / "v1.0" / "reports" / "base_set.gemini-2.5-flash_scores.json"


def test_load_base_anchor_scores_reads_frozen_v1_report_shape():
    scores = load_base_anchor_scores("v1.0", "gemini-2.5-flash")

    assert "gemini-2.5-flash" in scores
    anchor = scores["gemini-2.5-flash"]
    assert {
        "overall",
        "en_ja",
        "en_ja_easy",
        "en_ja_hard",
        "ja_en",
        "ja_en_easy",
        "ja_en_hard",
    } <= set(anchor)
    assert anchor["overall"]["matches"] == 1273
    assert anchor["overall"]["lt"] == pytest.approx(9.892173610141887)


def test_load_base_anchor_scores_prefers_schema_v2_report_when_present(tmp_path, monkeypatch):
    report_dir = tmp_path / "baseset" / "v1.0" / "reports"
    report_dir.mkdir(parents=True)

    legacy_report = report_dir / "base_set.gemini-2.5-flash_scores.json"
    schema_v2_report = report_dir / "base_set.gemini-2.5-flash_scores.schema-v2.json"

    legacy_report.write_text(
        json.dumps([{"model": "gemini-2.5-flash", "slice": "overall", "wins": 1, "matches": 2, "LT": 1.0}]),
        encoding="utf-8",
    )
    schema_v2_report.write_text(
        json.dumps([{"model": "gemini-2.5-flash", "slice": "overall", "wins": 9, "matches": 10, "LT": 9.0}]),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    scores = load_base_anchor_scores("v1.0", "gemini-2.5-flash")

    assert scores["gemini-2.5-flash"]["overall"]["matches"] == 10
    assert scores["gemini-2.5-flash"]["overall"]["lt"] == pytest.approx(9.0)


def test_legacy_boundary_sidecar_is_additive_only(tmp_path):
    snapshot_dir = tmp_path / "v1.0"
    report_dir = snapshot_dir / "reports"
    report_dir.mkdir(parents=True)

    pair_copy = snapshot_dir / LEGACY_PAIR_FILE.name
    report_copy = report_dir / LEGACY_REPORT_FILE.name

    original_pair_bytes = LEGACY_PAIR_FILE.read_bytes()
    original_report_bytes = LEGACY_REPORT_FILE.read_bytes()

    pair_copy.write_bytes(original_pair_bytes)
    report_copy.write_bytes(original_report_bytes)

    sidecar_path = write_legacy_jp_v1_boundary_metadata(snapshot_dir, "gemini-2.5-flash")
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))

    assert pair_copy.read_bytes() == original_pair_bytes
    assert report_copy.read_bytes() == original_report_bytes
    assert sidecar_path not in {pair_copy, report_copy}
    assert payload["legacy_boundary"] == "legacy_jp_v1"
    assert payload["artifact_policy"] == "additive_only"
    assert payload["pair_id_schema"] == "v1"
    assert payload["grandfather_pre_fingerprint_judgments"] is True
    assert payload["pair_file"] == LEGACY_PAIR_FILE.name
    assert payload["judgments_file_pattern"] == "base_set.<judge>.jsonl"
    assert payload["schema_v2_judgments_pattern"] == "base_set.<judge>.schema-v2.jsonl"
    assert payload["score_report_pattern"] == "reports/base_set.<judge>_scores.json"
    assert payload["schema_v2_score_report_pattern"] == "reports/base_set.<judge>_scores.schema-v2.json"
