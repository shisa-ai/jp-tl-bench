import json

from benchmark_tasks import load_task_config
from score_visualizer import extract_candidate_direction_rows, load_base_anchor_scores


def test_load_base_anchor_scores_maps_legacy_ambiguous_rows_to_task_slice_namespace(tmp_path, monkeypatch):
    report_dir = tmp_path / "baseset" / "v1.1" / "reports"
    report_dir.mkdir(parents=True)
    report_path = report_dir / "base_set.gemini-2.5-flash_scores.json"
    report_path.write_text(
        json.dumps(
            [
                {"model": "anchor-a", "slice": "overall", "language": "all", "difficulty": "all", "wins": 5, "matches": 8, "LT": 5.0},
                {"model": "anchor-a", "slice": "overall", "language": "english", "difficulty": "all", "wins": 3, "matches": 4, "LT": 3.0},
                {"model": "anchor-a", "slice": "easy", "language": "english", "difficulty": "easy", "wins": 2, "matches": 2, "LT": 2.0},
                {"model": "anchor-a", "slice": "hard", "language": "english", "difficulty": "hard", "wins": 1, "matches": 2, "LT": 1.0},
                {"model": "anchor-a", "slice": "overall", "language": "japanese", "difficulty": "all", "wins": 2, "matches": 4, "LT": 2.5},
                {"model": "anchor-a", "slice": "easy", "language": "japanese", "difficulty": "easy", "wins": 1, "matches": 2, "LT": 1.5},
                {"model": "anchor-a", "slice": "hard", "language": "japanese", "difficulty": "hard", "wins": 1, "matches": 2, "LT": 1.0},
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    scores = load_base_anchor_scores(
        "v1.1",
        "gemini-2.5-flash",
        task="benchmark_tasks/translation_ja_en_bidirectional_v1.yaml",
    )

    anchor = scores["anchor-a"]
    assert {
        "overall",
        "en_ja",
        "en_ja_easy",
        "en_ja_hard",
        "ja_en",
        "ja_en_easy",
        "ja_en_hard",
    } <= set(anchor)
    assert anchor["en_ja_easy"]["lt"] == 2.0
    assert anchor["ja_en_easy"]["lt"] == 1.5


def test_extract_candidate_direction_rows_uses_task_defined_direction_keys():
    task_config = load_task_config("benchmark_tasks/translation_zh_en_bidirectional_v1.yaml")
    summary = {
        "model": "candidate-a",
        "zh_en": {
            "overall": {"lt": 6.0, "wins": 6, "total": 8},
            "easy": {"lt": 6.5, "wins": 3, "total": 4},
            "hard": {"lt": 5.5, "wins": 3, "total": 4},
        },
        "en_zh": {
            "overall": {"lt": 4.0, "wins": 2, "total": 8},
            "easy": {"lt": 4.5, "wins": 1, "total": 4},
            "hard": {"lt": 3.5, "wins": 1, "total": 4},
        },
    }

    rows = extract_candidate_direction_rows(summary, task_config=task_config, missing_ratio=0.0)

    assert set(rows) == {"zh_en", "en_zh"}
    assert rows["zh_en"]["overall_lt"] == 6.0
    assert rows["zh_en"]["total_matches"] == 8
    assert rows["en_zh"]["hard_lt"] == 3.5
