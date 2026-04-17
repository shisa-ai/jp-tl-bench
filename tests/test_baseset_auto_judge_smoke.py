import json
import importlib.util
from pathlib import Path

import click
import pytest

import base_set_manager
from baseset import generate_set
from baseset.legacy_boundary import write_snapshot_report_sidecar
from benchmark_tasks import load_task_config


def load_backfill_module():
    module_path = Path(__file__).resolve().parents[1] / "baseset" / "backfill-1.0-judgements.py"
    spec = importlib.util.spec_from_file_location("baseset_backfill", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_generate_set_resolve_analysis_file_requires_exact_judge_match(tmp_path, monkeypatch):
    snapshot_dir = tmp_path / "baseset" / "v2.0"
    snapshot_dir.mkdir(parents=True)
    base_sets_dir = tmp_path / "base_sets"
    base_sets_dir.mkdir()
    (base_sets_dir / "base_set.some-other-judge.jsonl").write_text("", encoding="utf-8")

    monkeypatch.setattr(generate_set, "REPO_ROOT", tmp_path)

    assert generate_set.resolve_analysis_file("", snapshot_dir, "gemini-2.5-flash") is None


def test_generate_set_run_auto_judge_requires_snapshot_local_output(tmp_path, monkeypatch):
    snapshot_dir = tmp_path / "baseset" / "v2.0"
    snapshot_dir.mkdir(parents=True)
    pair_file = snapshot_dir / "pairs.jsonl"
    pair_file.write_text("", encoding="utf-8")

    base_sets_dir = tmp_path / "base_sets"
    base_sets_dir.mkdir()
    (base_sets_dir / "base_set.gemini-2.5-flash.jsonl").write_text("", encoding="utf-8")

    monkeypatch.setattr(generate_set, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(generate_set.subprocess, "run", lambda *args, **kwargs: None)

    with pytest.raises(click.ClickException, match="snapshot"):
        generate_set.run_auto_judge(
            pair_file=pair_file,
            snapshot_dir=snapshot_dir,
            task=None,
            judge_model="gemini-2.5-flash",
            judge_profile="default",
            base_url="https://example.invalid",
            api_key_env="DUMMY_KEY",
            max_workers=1,
            concurrency_limit=1,
        )


def test_generate_set_run_auto_judge_forwards_task_and_judge_profile(tmp_path, monkeypatch):
    snapshot_dir = tmp_path / "baseset" / "v2.0"
    snapshot_dir.mkdir(parents=True)
    pair_file = snapshot_dir / "pairs.jsonl"
    pair_file.write_text("", encoding="utf-8")
    judged_file = snapshot_dir / "base_set.gemini-2.5-flash.jsonl"
    judged_file.write_text("", encoding="utf-8")

    captured = {}

    def fake_run(cmd, cwd, check, env):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        captured["env"] = env

    monkeypatch.setattr(generate_set.subprocess, "run", fake_run)

    result = generate_set.run_auto_judge(
        pair_file=pair_file,
        snapshot_dir=snapshot_dir,
        task="benchmark_tasks/translation_zh_en_bidirectional_v1.yaml",
        judge_model="gemini-2.5-flash",
        judge_profile="cn_judge",
        base_url="https://example.invalid",
        api_key_env="DUMMY_KEY",
        max_workers=1,
        concurrency_limit=1,
    )

    assert result == judged_file
    assert "--task" in captured["cmd"]
    assert "benchmark_tasks/translation_zh_en_bidirectional_v1.yaml" in captured["cmd"]
    assert "--judge-profile" in captured["cmd"]
    assert "cn_judge" in captured["cmd"]
    assert captured["env"]["BASESET_SNAPSHOT_DIR"] == str(snapshot_dir)


def test_base_set_manager_accepts_raw_judge_ids(tmp_path, monkeypatch):
    snapshot_dir = tmp_path / "baseset" / "v2.0"
    snapshot_dir.mkdir(parents=True)
    expected = snapshot_dir / "base_set.openai__gpt-4o.jsonl"
    expected.write_text("", encoding="utf-8")

    monkeypatch.setenv("BASESET_SNAPSHOT_DIR", str(snapshot_dir))

    assert base_set_manager.find_base_set_file("openai/gpt-4o") == str(expected)


def test_backfill_script_is_explicitly_deprecated():
    module = load_backfill_module()

    with pytest.raises(SystemExit, match="deprecated"):
        module.main()


def test_write_snapshot_report_sidecar_includes_task_and_dataset_metadata(tmp_path):
    snapshot_dir = tmp_path / "baseset" / "v2.0"
    report_dir = snapshot_dir / "reports"
    report_dir.mkdir(parents=True)
    report_path = report_dir / "base_set.gemini-2.5-flash_scores.json"
    report_path.write_text("[]", encoding="utf-8")
    pair_file = snapshot_dir / "base_conversation_pairs.jsonl"
    analysis_file = snapshot_dir / "base_set.gemini-2.5-flash.jsonl"
    pair_file.write_text("", encoding="utf-8")
    analysis_file.write_text("", encoding="utf-8")
    task_config = load_task_config("benchmark_tasks/translation_zh_en_bidirectional_v1.yaml")

    sidecar_path = write_snapshot_report_sidecar(
        report_path,
        snapshot_dir=snapshot_dir,
        manifest_payload={"snapshot_version": "v2.0"},
        task_config=task_config,
        judge_model="gemini-2.5-flash",
        pair_file=pair_file,
        analysis_file=analysis_file,
    )

    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert payload["snapshot_version"] == "v2.0"
    assert payload["task_id"] == "translation.zh-en"
    assert payload["task_version"] == "v1"
    assert payload["dataset_repo"] == "shisa-ai/bt_translation_set_global"
    assert payload["dataset_config"] == "translation_zh_en_bidirectional_v1"
    assert payload["dataset_split"] == "train"
    assert payload["dataset_revision"] == "1b0d3bcb2ed1a611ad9130cc474007afae2b598c"
    assert payload["judge_model"] == "gemini-2.5-flash"
    assert payload["report_file"] == report_path.name
    assert payload["pair_file"] == pair_file.name
    assert payload["analysis_file"] == analysis_file.name
