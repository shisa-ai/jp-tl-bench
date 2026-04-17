import json
from pathlib import Path

from click.testing import CliRunner
import yaml

from benchmark_tasks import (
    load_judge_profile,
    load_task_config,
    resolve_dataset_ref,
    resolve_compare_prompt_path,
)
from choix_analyzer import build_score_summary
from generate_shootout_data import generate_translation_pairs
from generate_translation_data import Translator, main as generate_translation_data_cli


REPO_ROOT = Path(__file__).resolve().parents[1]
JP_TASK_PATH = REPO_ROOT / "benchmark_tasks" / "translation_ja_en_bidirectional_v1.yaml"
ZH_TASK_PATH = REPO_ROOT / "benchmark_tasks" / "translation_zh_en_bidirectional_v1.yaml"
DEFAULT_JUDGE_PROFILE_PATH = REPO_ROOT / "judge_profiles" / "default.yaml"
CN_JUDGE_PROFILE_PATH = REPO_ROOT / "judge_profiles" / "cn_judge.yaml"


def test_load_task_and_judge_config_resolve_compare_prompt_profiles():
    zh_task = load_task_config(ZH_TASK_PATH)
    default_judge = load_judge_profile(DEFAULT_JUDGE_PROFILE_PATH)
    cn_judge = load_judge_profile(CN_JUDGE_PROFILE_PATH)

    assert zh_task.dataset.repo == "shisa-ai/bt_translation_set_global"
    assert [direction.key for direction in zh_task.directions] == ["zh_en", "en_zh"]
    assert zh_task.get_prompt_path("zh", "en").name == "translate_prompt_from_chinese.txt"
    assert (
        zh_task.get_prompt_path("en", "zh", low_context=True).name
        == "translate_prompt_from_english_to_chinese_low_context.txt"
    )
    assert resolve_compare_prompt_path(zh_task, default_judge).name == "default.txt"
    assert resolve_compare_prompt_path(zh_task, cn_judge).name == "cn.txt"


def test_judge_profile_resolves_request_overrides_by_model_name():
    default_judge = load_judge_profile(DEFAULT_JUDGE_PROFILE_PATH)

    assert default_judge.resolve_request_settings("google/gemini-2.5-flash") == {
        "temperature": 0.0,
        "reasoning_effort": "low",
        "thinking_budget": 0,
    }
    assert default_judge.resolve_request_settings("google/gemini-2.5-pro") == {
        "temperature": 0.0,
        "reasoning_effort": "low",
        "thinking_budget": 128,
    }
    assert default_judge.resolve_request_settings("openai/gpt-5-mini") == {
        "temperature": None,
    }


def test_jp_task_normalizes_legacy_rows_to_explicit_language_fields():
    task = load_task_config(JP_TASK_PATH)
    normalized = task.normalize_record(
        {
            "name": "sample_convo_15",
            "text": "Hello there.",
            "difficulty": "easy",
            "english": True,
        },
        require_source_text=True,
    )

    assert normalized["item_id"] == "sample_convo_15"
    assert normalized["name"] == "sample_convo_15"
    assert normalized["source_text"] == "Hello there."
    assert normalized["source_language"] == "en"
    assert normalized["target_language"] == "ja"
    assert normalized["english"] is True


def test_zh_task_normalizes_to_item_id_backed_name_without_legacy_english():
    task = load_task_config(ZH_TASK_PATH)
    normalized = task.normalize_record(
        {
            "item_id": "zh_01",
            "name": "non_canonical_alias",
            "source_text": "你好，世界。",
            "difficulty": "hard",
            "source_language": "zh",
            "target_language": "en",
        },
        require_source_text=True,
    )

    assert normalized["item_id"] == "zh_01"
    assert normalized["name"] == "zh_01"
    assert normalized["source_language"] == "zh"
    assert normalized["target_language"] == "en"
    assert "english" not in normalized


def test_task_config_digest_is_stable_for_equivalent_yaml_and_changes_with_content(
    tmp_path,
):
    original_task = load_task_config(JP_TASK_PATH)
    original_payload = yaml.safe_load(JP_TASK_PATH.read_text(encoding="utf-8"))

    same_path = tmp_path / "same.yaml"
    reordered_path = tmp_path / "reordered.yaml"
    changed_path = tmp_path / "changed.yaml"

    same_path.write_text(JP_TASK_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    reordered_path.write_text(
        yaml.safe_dump(original_payload, allow_unicode=True, sort_keys=True),
        encoding="utf-8",
    )
    changed_path.write_text(
        yaml.safe_dump(
            {**original_payload, "task_version": "v2"},
            allow_unicode=True,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    same_task = load_task_config(same_path)
    reordered_task = load_task_config(reordered_path)
    changed_task = load_task_config(changed_path)

    assert same_task.task_config_digest == original_task.task_config_digest
    assert reordered_task.task_config_digest == original_task.task_config_digest
    assert changed_task.task_config_digest != original_task.task_config_digest


def test_generate_translation_data_cli_loads_private_task_dataset_with_hf_token(
    monkeypatch, tmp_path
):
    task = load_task_config(JP_TASK_PATH)
    captured = {}

    def fake_load_dataset(repo, name, split=None, revision=None, token=None):
        captured["dataset"] = {
            "repo": repo,
            "name": name,
            "split": split,
            "revision": revision,
            "token": token,
        }
        return [
            {
                "name": "sample_convo_15",
                "text": "Hello there.",
                "difficulty": "easy",
                "english": True,
            }
        ]

    class DummyTranslator:
        def __init__(
            self,
            model_name,
            base_url,
            api_key,
            low_context,
            ultra_low_context,
            concurrency_limit,
            max_tokens,
            task_config,
            dataset_ref,
        ):
            captured["translator"] = {
                "model_name": model_name,
                "api_key": api_key,
                "task_id": task_config.task_id,
                "dataset_ref": dataset_ref,
            }
            self.failed_items = []
            self.total_input_tokens = 0
            self.total_output_tokens = 0

        def __call__(self, dataset, max_workers):
            item = dataset[0]
            captured["normalized_item"] = item
            return [
                {
                    **item,
                    "status": "ok",
                    "full_response": "<translation>こんにちは</translation>",
                    "translation": "こんにちは",
                    "prompt": "prompt",
                    "temperature": 0.1,
                    "top_p": 0.85,
                    "frequency_penalty": None,
                    "reasoning_effort": None,
                    "low_context": False,
                    "ultra_low_context": False,
                    "generation_config": {"model": "example/model"},
                }
            ]

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("generate_translation_data.load_dataset", fake_load_dataset)
    monkeypatch.setattr("generate_translation_data.Translator", DummyTranslator)
    monkeypatch.setattr(
        "generate_translation_data.resolve_dataset_ref",
        lambda task_config, token=None: {
            "repo": task_config.dataset.repo,
            "config": task_config.dataset.config,
            "split": task_config.dataset.split,
            "revision": task_config.dataset.revision,
            "resolved_revision": "1234567890abcdef1234567890abcdef12345678",
        },
    )
    monkeypatch.setenv("HF_TOKEN", "hf_private_token")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    runner = CliRunner()
    result = runner.invoke(
        generate_translation_data_cli,
        [
            "--task",
            str(JP_TASK_PATH),
            "--base-url",
            "http://unused",
            "--test-model",
            "example/model",
            "--max-workers",
            "1",
            "--concurrency-limit",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["dataset"] == {
        "repo": "shisa-ai/bt_translation_set_global",
        "name": "translation_ja_en_bidirectional_v1",
        "split": "train",
        "revision": task.dataset.revision,
        "token": "hf_private_token",
    }
    assert captured["translator"]["task_id"] == "translation.ja-en"
    assert captured["translator"]["dataset_ref"]["resolved_revision"] == "1234567890abcdef1234567890abcdef12345678"
    assert captured["normalized_item"]["item_id"] == "sample_convo_15"
    assert captured["normalized_item"]["source_language"] == "en"
    assert captured["normalized_item"]["target_language"] == "ja"

    output_file = tmp_path / "translations" / "example__model.jsonl"
    written = json.loads(output_file.read_text(encoding="utf-8").splitlines()[0])
    assert written["task_id"] == "translation.ja-en"
    assert written["item_id"] == "sample_convo_15"
    assert written["source_language"] == "en"
    assert written["target_language"] == "ja"


def test_translator_outputs_include_task_config_digest():
    task = load_task_config(JP_TASK_PATH)
    translator = Translator(
        model_name="fixture-model",
        base_url="https://example.com/v1",
        api_key="test-key",
        task_config=task,
        dataset_ref=task.dataset.to_dict(),
    )

    parsed = translator.parse(
        {
            "name": "sample_convo_15",
            "text": "Hello there.",
            "difficulty": "easy",
            "english": True,
        },
        "<translation>こんにちは</translation>",
        "prompt body",
        {"temperature": 0.1},
    )

    assert parsed["task_config_digest"] == task.task_config_digest


def test_resolve_dataset_ref_records_requested_and_resolved_revision(monkeypatch, tmp_path):
    task_payload = yaml.safe_load(JP_TASK_PATH.read_text(encoding="utf-8"))
    task_payload["dataset"]["revision"] = "jp-v1"
    temp_task_path = tmp_path / "jp_task_resolution_test.yaml"
    temp_task_path.write_text(
        yaml.safe_dump(task_payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    task = load_task_config(temp_task_path)

    class DummyInfo:
        sha = "abcdef1234567890abcdef1234567890abcdef12"

    class DummyApi:
        def dataset_info(self, repo_id, *, revision=None, token=None, **kwargs):
            assert repo_id == "shisa-ai/bt_translation_set_global"
            assert revision == "jp-v1"
            assert token == "hf_private_token"
            return DummyInfo()

    monkeypatch.setattr("benchmark_tasks.HfApi", lambda: DummyApi())

    resolved = resolve_dataset_ref(task, token="hf_private_token")

    assert resolved == {
        "repo": "shisa-ai/bt_translation_set_global",
        "config": "translation_ja_en_bidirectional_v1",
        "split": "train",
        "revision": "jp-v1",
        "resolved_revision": "abcdef1234567890abcdef1234567890abcdef12",
    }


def test_generate_shootout_pairs_match_on_item_id_and_emit_explicit_task_fields(
    tmp_path, monkeypatch
):
    snapshot_dir = tmp_path / "baseset" / "v1.0"
    base_translations_dir = snapshot_dir / "translations"
    translations_dir = tmp_path / "translations"
    base_translations_dir.mkdir(parents=True)
    translations_dir.mkdir()

    candidate_row = {
        "item_id": "zh_01",
        "name": "candidate_alias",
        "source_text": "你好，世界。",
        "translation": "hello world",
        "difficulty": "easy",
        "source_language": "zh",
        "target_language": "en",
        "task_id": "translation.zh-en",
        "task_version": "v1",
        "status": "ok",
    }
    anchor_row = {
        "item_id": "zh_01",
        "name": "anchor_alias",
        "source_text": "你好，世界。",
        "translation": "hello, world",
        "difficulty": "easy",
        "source_language": "zh",
        "target_language": "en",
        "task_id": "translation.zh-en",
        "task_version": "v1",
        "status": "ok",
    }

    (translations_dir / "candidate.jsonl").write_text(
        json.dumps(candidate_row, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (base_translations_dir / "anchor.jsonl").write_text(
        json.dumps(anchor_row, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("generate_shootout_data.BASESET_SNAPSHOT_DIR", str(snapshot_dir))

    output_file = tmp_path / "pairs.jsonl"
    generate_translation_pairs(
        test_model_file="candidate.jsonl",
        force=True,
        output_path=output_file,
        task=str(ZH_TASK_PATH),
    )

    pair = json.loads(output_file.read_text(encoding="utf-8").splitlines()[0])
    assert pair["item_id"] == "zh_01"
    assert pair["name"] == "zh_01"
    assert pair["task_id"] == "translation.zh-en"
    assert pair["task_version"] == "v1"
    assert pair["source_language"] == "zh"
    assert pair["target_language"] == "en"
    assert "english" not in pair


def test_build_score_summary_uses_task_defined_direction_keys_for_explicit_language_rows(
    tmp_path,
):
    base_file = tmp_path / "base.jsonl"
    judgments_file = tmp_path / "judgments.jsonl"

    base_rows = [
        {
            "id": "base-en-zh",
            "llm_a": "anchor",
            "llm_b": "other",
            "name": "en_zh_01",
            "difficulty": "easy",
            "source_language": "en",
            "target_language": "zh",
            "analysis": "<answer>A</answer>",
        },
        {
            "id": "base-zh-en",
            "llm_a": "other",
            "llm_b": "anchor",
            "name": "zh_en_01",
            "difficulty": "hard",
            "source_language": "zh",
            "target_language": "en",
            "analysis": "<answer>B</answer>",
        },
    ]
    judgment_rows = [
        {
            "id": "judge-en-zh",
            "llm_a": "candidate",
            "llm_b": "anchor",
            "name": "en_zh_02",
            "difficulty": "easy",
            "source_language": "en",
            "target_language": "zh",
            "analysis": "<answer>A</answer>",
        },
        {
            "id": "judge-zh-en",
            "llm_a": "anchor",
            "llm_b": "candidate",
            "name": "zh_en_02",
            "difficulty": "hard",
            "source_language": "zh",
            "target_language": "en",
            "analysis": "<answer>B</answer>",
        },
    ]

    base_file.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in base_rows) + "\n",
        encoding="utf-8",
    )
    judgments_file.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in judgment_rows) + "\n",
        encoding="utf-8",
    )

    summary = build_score_summary(
        test_model="candidate",
        judge_model="gemini-2.5-flash",
        base_version="v1.0-zh",
        base_file=base_file,
        judgments_file=judgments_file,
        task=str(ZH_TASK_PATH),
    )

    assert summary["zh_en"]["overall"]["language"] == "zh_en"
    assert summary["en_zh"]["overall"]["language"] == "en_zh"
