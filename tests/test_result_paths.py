import json
from pathlib import Path

from click.testing import CliRunner

import choix_analyzer
import generate_shootout_data
import translation_comparer_any_model
from artifact_paths import (
    candidate_results_dir,
    candidate_results_dir_candidates,
    judge_output_dirname,
)
from choix_analyzer import main as choix_analyzer_cli
from translation_comparer_any_model import TranslationComparer, main as compare_cli


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_result_paths_scope_non_default_judge_profiles():
    assert judge_output_dirname("google/gemini-2.5-flash", "default") == "google__gemini-2.5-flash"
    assert judge_output_dirname("google/gemini-2.5-flash", "cn_judge") == "google__gemini-2.5-flash.cn_judge"
    assert generate_shootout_data.default_pairs_path(
        "candidate/model",
        judge_model="google/gemini-2.5-flash",
        judge_profile_id="cn_judge",
    ).endswith(
        "results/v1.0/candidate__model/google__gemini-2.5-flash.cn_judge/pairs.jsonl"
    )


def test_result_path_candidates_include_legacy_fallback_for_non_default_profile():
    candidates = candidate_results_dir_candidates(
        "v-profile",
        "candidate/model",
        "google/gemini-2.5-flash",
        "cn_judge",
        root="/tmp/root",
    )
    assert candidates == [
        Path("/tmp/root/results/v-profile/candidate__model/google__gemini-2.5-flash.cn_judge"),
        Path("/tmp/root/results/v-profile/candidate__model/google__gemini-2.5-flash"),
    ]


def test_compare_cli_uses_profile_scoped_default_results_dir(tmp_path, monkeypatch):
    pair = {
        "id": "pair-1",
        "llm_a": "candidate__model",
        "llm_b": "anchor__model",
        "formatted_data": "## Name: zh_01\n\n## Source Text:\n你好。\n\n## Translation A\nHello.\n\n## Translation B\nHi.\n\n---\n",
        "item_id": "zh_01",
        "name": "zh_01",
        "task_id": "translation.zh-en",
        "task_type": "translation",
        "task_version": "v1",
        "source_language": "zh",
        "target_language": "en",
        "difficulty": "easy",
        "pair_fingerprint": "stub",
    }
    snapshot_dir = tmp_path / "baseset" / "v-profile"
    snapshot_dir.mkdir(parents=True)
    result_dir = candidate_results_dir(
        "v-profile",
        "candidate/model",
        "google/gemini-2.5-flash",
        "cn_judge",
        root=tmp_path,
    )
    _write_jsonl(result_dir / "pairs.jsonl", [pair])

    def fake_call(self, dataset, max_workers):
        row = dict(dataset[0])
        return [
            self.parse(
                row,
                "<translation_analysis>ok</translation_analysis><evaluation_summary>ok</evaluation_summary><answer>A</answer>",
                {"temperature": 0.0},
                self.model_name,
            )
        ]

    monkeypatch.setattr(TranslationComparer, "__call__", fake_call)
    monkeypatch.setattr(translation_comparer_any_model, "__file__", str(tmp_path / "translation_comparer_any_model.py"))

    runner = CliRunner()
    result = runner.invoke(
        compare_cli,
        [
            "--task",
            "benchmark_tasks/translation_zh_en_bidirectional_v1.yaml",
            "--judge-profile",
            "cn_judge",
            "--base-url",
            "http://unused",
            "--judge-model",
            "google/gemini-2.5-flash",
            "--test-model",
            "candidate/model",
        ],
        env={
            "OPENAI_API_KEY": "test-key",
            "BASESET_SNAPSHOT_DIR": str(snapshot_dir),
        },
    )

    assert result.exit_code == 0, result.output
    assert (result_dir / "judgments.jsonl").exists()


def test_choix_analyzer_uses_profile_scoped_default_results_dir(tmp_path, monkeypatch):
    snapshot_dir = tmp_path / "baseset" / "v-profile"
    snapshot_dir.mkdir(parents=True)
    (snapshot_dir / "base_set.google__gemini-2.5-flash.jsonl").write_text("", encoding="utf-8")
    result_dir = candidate_results_dir(
        "v-profile",
        "candidate/model",
        "google/gemini-2.5-flash",
        "cn_judge",
        root=tmp_path,
    )
    _write_jsonl(result_dir / "judgments.jsonl", [{"id": "pair-1"}])
    _write_jsonl(result_dir / "pairs.jsonl", [{"id": "pair-1"}])

    captured = {}

    def fake_build_score_summary(**kwargs):
        captured["judgments_file"] = Path(kwargs["judgments_file"])
        captured["pairs_file"] = Path(kwargs["pairs_file"])
        return {"model": kwargs["test_model"], "baseset_version": kwargs["base_version"]}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(choix_analyzer, "build_score_summary", fake_build_score_summary)
    monkeypatch.setattr(
        choix_analyzer,
        "write_score_metadata_sidecar",
        lambda *args, **kwargs: result_dir / "scores.metadata.json",
    )

    runner = CliRunner()
    result = runner.invoke(
        choix_analyzer_cli,
        [
            "--task",
            "benchmark_tasks/translation_zh_en_bidirectional_v1.yaml",
            "--judge-profile",
            "cn_judge",
            "--test-model",
            "candidate/model",
            "--judge-model",
            "google/gemini-2.5-flash",
            "--baseset-version",
            "v-profile",
        ],
        env={"BASESET_SNAPSHOT_DIR": str(snapshot_dir)},
    )

    assert result.exit_code == 0, result.output
    assert captured["judgments_file"].resolve() == (result_dir / "judgments.jsonl").resolve()
    assert captured["pairs_file"].resolve() == (result_dir / "pairs.jsonl").resolve()
    assert (result_dir / "scores.json").exists()
