import json
from pathlib import Path

from click.testing import CliRunner
import pytest
import yaml

from benchmark_tasks import load_task_config
from generate_translation_data import Translator, main as generate_translation_data_cli


REPO_ROOT = Path(__file__).resolve().parents[1]
JP_TASK_PATH = REPO_ROOT / "benchmark_tasks" / "translation_ja_en_bidirectional_v1.yaml"
ZH_TASK_PATH = REPO_ROOT / "benchmark_tasks" / "translation_zh_en_bidirectional_v1.yaml"


def _configured_translation_prompt_paths(task_path):
    task_payload = yaml.safe_load(task_path.read_text(encoding="utf-8"))
    return [
        prompt_path
        for direction in task_payload["directions"]
        for prompt_path in direction["translation_prompts"].values()
    ]


ZH_PROMPT_PATHS = sorted(_configured_translation_prompt_paths(ZH_TASK_PATH))
JP_PROMPT_PATHS = sorted(_configured_translation_prompt_paths(JP_TASK_PATH))
JP_PROMPT_PATHS_WITH_TRANSLATION = [
    prompt_path
    for prompt_path in JP_PROMPT_PATHS
    if not prompt_path.endswith("ultra_low_context.txt")
]
JP_DEFAULT_PROMPT_PATHS = sorted(
    {
        "prompts/translate_prompt_from_english.txt",
        "prompts/translate_prompt_from_japanese.txt",
    }
)


def test_translator_supports_generic_direction_prompt_with_language_placeholders(tmp_path):
    task_payload = yaml.safe_load(ZH_TASK_PATH.read_text(encoding="utf-8"))
    for direction in task_payload["directions"]:
        direction["translation_prompts"] = {
            "default": "prompts/translate/default.txt",
            "low_context": "prompts/translate/default.txt",
            "ultra_low_context": "prompts/translate/default.txt",
        }

    temp_task_path = tmp_path / "generic_prompt_task.yaml"
    temp_task_path.write_text(
        yaml.safe_dump(task_payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    translator = Translator(
        model_name="fixture-model",
        base_url="https://example.com/v1",
        api_key="test-key",
        task_config=load_task_config(temp_task_path),
    )

    prompt_text, prompt_path = translator.get_prompt(
        {
            "item_id": "zh_01",
            "source_text": "你好，世界。",
            "difficulty": "easy",
            "source_language": "zh",
            "target_language": "en",
        }
    )

    assert Path(prompt_path).name == "default.txt"
    assert "Chinese" in prompt_text
    assert "English" in prompt_text
    assert "{{src_lang}}" not in prompt_text
    assert "{{tgt_lang}}" not in prompt_text


def test_generate_translation_data_cli_fails_fast_when_api_key_is_missing(monkeypatch, tmp_path):
    captured = {"translator_called": False}

    def fake_load_dataset(repo, name, split=None, revision=None, token=None):
        return [
            {
                "name": "sample_convo_15",
                "text": "Hello there.",
                "difficulty": "easy",
                "english": True,
            }
        ]

    def fake_resolve_dataset_ref(task_config, token=None):
        return {
            "repo": task_config.dataset.repo,
            "config": task_config.dataset.config,
            "split": task_config.dataset.split,
            "revision": task_config.dataset.revision,
            "resolved_revision": task_config.dataset.revision,
        }

    class ExplodingTranslator:
        def __init__(self, *args, **kwargs):
            captured["translator_called"] = True
            raise AssertionError("Translator should not be constructed when API key is missing")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("generate_translation_data.load_dataset", fake_load_dataset)
    monkeypatch.setattr("generate_translation_data.resolve_dataset_ref", fake_resolve_dataset_ref)
    monkeypatch.setattr("generate_translation_data.Translator", ExplodingTranslator)
    monkeypatch.setenv("HF_TOKEN", "hf_private_token")
    monkeypatch.delenv("MISSING_TRANSLATOR_KEY", raising=False)

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
            "--api-key-env",
            "MISSING_TRANSLATOR_KEY",
            "--max-workers",
            "1",
            "--concurrency-limit",
            "1",
        ],
    )

    assert result.exit_code != 0
    assert "MISSING_TRANSLATOR_KEY" in result.output
    assert not captured["translator_called"]


def test_translator_outputs_generation_profile_id():
    task = load_task_config(JP_TASK_PATH)
    translator = Translator(
        model_name="fixture-model",
        base_url="https://example.com/v1",
        api_key="test-key",
        task_config=task,
        dataset_ref={
            **task.dataset.to_dict(),
            "resolved_revision": task.dataset.revision,
        },
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
        "prompts/translate_prompt_from_english.txt",
        {"temperature": 0.1, "top_p": 0.85},
    )

    assert parsed["generation_profile_id"] == "default"
    assert parsed["dataset_ref"]["resolved_revision"] == task.dataset.revision


@pytest.mark.parametrize("prompt_path", ZH_PROMPT_PATHS)
def test_chinese_translation_prompts_use_think_tags(prompt_path):
    prompt_text = (REPO_ROOT / prompt_path).read_text(encoding="utf-8")

    assert "<think>" in prompt_text
    assert "<translation_analysis>" not in prompt_text
    assert "<translation>" in prompt_text


@pytest.mark.parametrize("prompt_path", JP_PROMPT_PATHS)
def test_japanese_translation_prompts_do_not_use_think_tags(prompt_path):
    prompt_text = (REPO_ROOT / prompt_path).read_text(encoding="utf-8")

    assert "<think>" not in prompt_text


@pytest.mark.parametrize("prompt_path", JP_PROMPT_PATHS_WITH_TRANSLATION)
def test_japanese_translation_prompts_use_translation_tags(prompt_path):
    prompt_text = (REPO_ROOT / prompt_path).read_text(encoding="utf-8")

    assert "<translation>" in prompt_text


@pytest.mark.parametrize("prompt_path", JP_DEFAULT_PROMPT_PATHS)
def test_japanese_default_translation_prompts_keep_legacy_translation_analysis_tags(
    prompt_path,
):
    prompt_text = (REPO_ROOT / prompt_path).read_text(encoding="utf-8")

    assert "<translation_analysis>" in prompt_text
    assert "<think>" not in prompt_text
    assert "<translation>" in prompt_text
