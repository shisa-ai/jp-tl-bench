import json
from pathlib import Path

import pytest

from choix_analyzer import build_score_summary


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "task1"
SCORING_FIXTURE_DIR = FIXTURE_DIR / "scoring"


def test_build_score_summary_preserves_legacy_slice_values_and_pair_accounting():
    expected = json.loads((SCORING_FIXTURE_DIR / "expected_scores.json").read_text(encoding="utf-8"))

    summary = build_score_summary(
        test_model="shisa-ai/chotto-e4b-20260408",
        judge_model="gemini-2.5-flash",
        base_version="v1.0-sample",
        base_file=SCORING_FIXTURE_DIR / "base_set.gemini-2.5-flash.jsonl",
        judgments_file=SCORING_FIXTURE_DIR / "judgments.jsonl",
        pairs_file=SCORING_FIXTURE_DIR / "pairs.jsonl",
    )

    assert summary["model"] == expected["model"]
    assert summary["judge_model"] == expected["judge_model"]
    assert summary["baseset_version"] == expected["baseset_version"]
    assert summary["expected_pairs"] == expected["expected_pairs"]
    assert summary["judged_pairs"] == expected["judged_pairs"]
    assert summary["missing_pairs"] == expected["missing_pairs"]
    assert summary["base_comparisons"] == expected["base_comparisons"]

    for direction in ("en_ja", "ja_en"):
        for slice_name, expected_slice in expected[direction].items():
            actual_slice = summary[direction][slice_name]
            assert actual_slice["difficulty"] == expected_slice["difficulty"]
            assert actual_slice["language"] == expected_slice["language"]
            assert actual_slice["wins"] == expected_slice["wins"]
            assert actual_slice["total"] == expected_slice["total"]
            assert actual_slice["win_rate"] == pytest.approx(expected_slice["win_rate"], abs=1e-12)
            assert actual_slice["lt"] == pytest.approx(expected_slice["lt"], abs=1e-9)


def test_build_score_summary_reports_missing_pairs_when_pairs_file_is_longer(tmp_path):
    expanded_pairs_file = tmp_path / "pairs.jsonl"
    expanded_pairs_file.write_text(
        (SCORING_FIXTURE_DIR / "pairs.jsonl").read_text(encoding="utf-8")
        + (
            '{"id":"extra-pair","llm_a":"shisa-ai__chotto-e4b-20260408","llm_b":"Rakuten__RakutenAI-2.0-mini-instruct",'
            '"name":"sample_convo_15","english":true,"difficulty":"easy","llm_a_generation_config":{"model":"shisa-ai/chotto-e4b-20260408","temperature":0.1,"top_p":0.85,"max_tokens":4096},"llm_b_generation_config":null}\n'
        ),
        encoding="utf-8",
    )

    summary = build_score_summary(
        test_model="shisa-ai/chotto-e4b-20260408",
        judge_model="gemini-2.5-flash",
        base_version="v1.0-sample",
        base_file=SCORING_FIXTURE_DIR / "base_set.gemini-2.5-flash.jsonl",
        judgments_file=SCORING_FIXTURE_DIR / "judgments.jsonl",
        pairs_file=expanded_pairs_file,
    )

    assert summary["expected_pairs"] == 9
    assert summary["judged_pairs"] == 8
    assert summary["missing_pairs"] == 1
