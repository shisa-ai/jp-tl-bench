import json
from pathlib import Path

import pytest
from click.testing import CliRunner

import generate_shootout_data
import translation_comparer_any_model
from generate_shootout_data import generate_translation_pairs
from pair_contract import compute_pair_fingerprint, compute_pair_id_v1
from translation_comparer_any_model import (
    OpenAIJudgeAdapter,
    SkylarkResponsesJudgeAdapter,
    TranslationComparer,
    existing_judgment_matches_pair,
    main as compare_cli,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
ZH_TASK_PATH = REPO_ROOT / "benchmark_tasks" / "translation_zh_en_bidirectional_v1.yaml"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_generate_translation_pairs_rejects_mismatched_task_fields(tmp_path, monkeypatch):
    snapshot_dir = tmp_path / "baseset" / "v1.0"
    translations_dir = tmp_path / "translations"
    base_dir = snapshot_dir / "translations"
    translations_dir.mkdir(parents=True)
    base_dir.mkdir(parents=True)

    candidate_rows = [
        {
            "item_id": "zh_01",
            "name": "zh_01",
            "task_id": "translation.zh-en",
            "task_type": "translation",
            "task_version": "v1",
            "source_text": "你好，世界。",
            "difficulty": "easy",
            "source_language": "zh",
            "target_language": "en",
            "status": "ok",
            "translation": "Hello, world.",
        }
    ]
    anchor_rows = [
        {
            "item_id": "zh_01",
            "name": "zh_01",
            "task_id": "translation.zh-en",
            "task_type": "translation",
            "task_version": "v1",
            "source_text": "不同的原文。",
            "difficulty": "easy",
            "source_language": "zh",
            "target_language": "en",
            "status": "ok",
            "translation": "Different source text.",
        }
    ]

    _write_jsonl(translations_dir / "candidate.jsonl", candidate_rows)
    _write_jsonl(base_dir / "anchor.jsonl", anchor_rows)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(generate_shootout_data, "BASESET_SNAPSHOT_DIR", str(snapshot_dir))

    with pytest.raises(ValueError, match="source_text"):
        generate_translation_pairs(
            test_model_file="candidate.jsonl",
            force=True,
            output_path=tmp_path / "pairs.jsonl",
            task=str(ZH_TASK_PATH),
        )


def test_generate_translation_pairs_preserve_task_defined_slice_tags(tmp_path, monkeypatch):
    snapshot_dir = tmp_path / "baseset" / "v1.0"
    translations_dir = tmp_path / "translations"
    base_dir = snapshot_dir / "translations"
    translations_dir.mkdir(parents=True)
    base_dir.mkdir(parents=True)

    row = {
        "item_id": "zh_01",
        "name": "zh_01",
        "task_id": "translation.zh-en",
        "task_type": "translation",
        "task_version": "v1",
        "source_text": "你好，世界。",
        "difficulty": "easy",
        "source_language": "zh",
        "target_language": "en",
        "category": "support",
        "tags": ["official", "easy"],
        "status": "ok",
        "translation": "Hello, world.",
    }

    _write_jsonl(translations_dir / "candidate.jsonl", [row])
    _write_jsonl(base_dir / "anchor.jsonl", [row])

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(generate_shootout_data, "BASESET_SNAPSHOT_DIR", str(snapshot_dir))

    output_path = tmp_path / "pairs.jsonl"
    generate_translation_pairs(
        test_model_file="candidate.jsonl",
        force=True,
        output_path=output_path,
        task=str(ZH_TASK_PATH),
    )

    pair = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
    assert pair["category"] == "support"
    assert pair["tags"] == ["official", "easy"]
    assert "<source_text>\n你好，世界。\n</source_text>" in pair["formatted_data"]
    assert "<translation_a>\nHello, world.\n</translation_a>" in pair["formatted_data"]
    assert "<translation_b>\nHello, world.\n</translation_b>" in pair["formatted_data"]


def test_compare_cli_uses_task_judge_profile_prompt_and_writes_judge_contract(
    tmp_path, monkeypatch
):
    pair = {
        "id": compute_pair_id_v1("candidate.jsonl", "anchor.jsonl", "zh_01"),
        "llm_a": "candidate",
        "llm_b": "anchor",
        "formatted_data": "## Name: zh_01\n\n## Source Text:\n你好，世界。\n\n## Translation A\nHello, world.\n\n## Translation B\nHi, world.\n\n---\n",
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
    pair_file = tmp_path / "pairs.jsonl"
    pair_file.write_text(json.dumps(pair, ensure_ascii=False) + "\n", encoding="utf-8")

    captured = {}

    def fake_call(self, dataset, max_workers):
        captured["prompt_path"] = self.prompt_path
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

    runner = CliRunner()
    result = runner.invoke(
        compare_cli,
        [
            "--task",
            str(ZH_TASK_PATH),
            "--judge-profile",
            "cn_judge",
            "--base-url",
            "http://unused",
            "--judge-model",
            "google/gemini-2.5-flash",
            "--generate-base-set",
            "--pairs-file",
            str(pair_file),
        ],
        env={
            "GEMINI_API_KEY": "test-key",
            "BASESET_SNAPSHOT_DIR": str(tmp_path / "v1.0"),
        },
    )

    assert result.exit_code == 0, result.output
    assert captured["prompt_path"].name == "cn.txt"

    output_path = (
        tmp_path
        / "v1.0"
        / "base_set.google__gemini-2.5-flash.schema-v2.jsonl"
    )
    written = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
    assert written["compare_prompt_profile_id"] == "compare-cn-v1"
    assert written["judge_profile_id"] == "cn_judge"
    assert (
        written["judge_contract_id"]
        == "google/gemini-2.5-flash::compare-cn-v1::answer-parser/v1"
    )


def test_reuse_requires_matching_judge_contract():
    pair = {
        "id": "pair-1",
        "pair_id_schema": "v1",
        "pair_fingerprint": compute_pair_fingerprint(
            {
                "id": "pair-1",
                "llm_a": "candidate",
                "llm_b": "anchor",
                "name": "zh_01",
                "difficulty": "easy",
                "formatted_data": "formatted",
            }
        ),
        "judge_contract_id": "google/gemini-2.5-flash::compare-cn-v1::answer-parser/v1",
    }
    reused = {
        "id": "pair-1",
        "pair_id_schema": "v1",
        "pair_fingerprint": pair["pair_fingerprint"],
        "judge_contract_id": "google/gemini-2.5-flash::compare-default-v1::answer-parser/v1",
    }

    assert not existing_judgment_matches_pair(reused, pair)


def test_openai_judge_adapter_uses_profile_request_settings():
    captured = {}

    class DummyCompletion:
        class Choice:
            class Message:
                content = "<answer>A</answer>"

            message = Message()

        choices = [Choice()]
        usage = None

    class DummyClient:
        class Chat:
            class Completions:
                @staticmethod
                def create(**kwargs):
                    captured.update(kwargs)
                    return DummyCompletion()

            completions = Completions()

        chat = Chat()

    adapter = OpenAIJudgeAdapter(
        "openai/gpt-5-mini",
        request_settings={"temperature": None},
    )
    response = adapter.request(DummyClient(), "judge prompt")

    assert captured == {
        "messages": [{"role": "user", "content": "judge prompt"}],
        "model": "openai/gpt-5-mini",
    }
    assert response.generation_config == {"model": "openai/gpt-5-mini"}


def test_openai_judge_adapter_omits_gemini_native_only_request_settings():
    captured = {}

    class DummyCompletion:
        class Choice:
            class Message:
                content = "<answer>A</answer>"

            message = Message()

        choices = [Choice()]
        usage = None

    class DummyClient:
        class Chat:
            class Completions:
                @staticmethod
                def create(**kwargs):
                    captured.update(kwargs)
                    return DummyCompletion()

            completions = Completions()

        chat = Chat()

    adapter = OpenAIJudgeAdapter(
        "gemini-2.5-flash",
        request_settings={
            "temperature": 0.0,
            "thinking_budget": 128,
            "reasoning_effort": "low",
        },
    )
    response = adapter.request(DummyClient(), "judge prompt")

    assert captured["temperature"] == 0.0
    assert captured["reasoning_effort"] == "low"
    assert "thinking_budget" not in captured
    assert response.generation_config == {
        "model": "gemini-2.5-flash",
        "temperature": 0.0,
        "reasoning_effort": "low",
    }


def test_skylark_responses_adapter_posts_responses_payload(monkeypatch):
    captured = {}

    class DummyResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        @staticmethod
        def read():
            return json.dumps(
                {
                    "output": [
                        {
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": "<answer>A</answer>",
                                }
                            ]
                        }
                    ],
                    "usage": {"input_tokens": 11, "output_tokens": 3},
                }
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return DummyResponse()

    monkeypatch.setattr(
        translation_comparer_any_model.urllib.request,
        "urlopen",
        fake_urlopen,
    )

    adapter = SkylarkResponsesJudgeAdapter(
        "seed-2-0-pro-260328",
        "https://ark.ap-southeast.bytepluses.com/api/v3/responses",
        "test-key",
        request_settings={
            "temperature": 0.0,
            "max_tokens": 4096,
            "thinking": {"type": "disabled"},
            "thinking_budget": 128,
        },
    )
    response = adapter.request(None, "judge prompt")

    assert captured["url"] == "https://ark.ap-southeast.bytepluses.com/api/v3/responses"
    assert captured["timeout"] == 120
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["headers"]["Ark-beta-mcp"] == "true"
    assert captured["payload"] == {
        "model": "seed-2-0-pro-260328",
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "judge prompt"}],
            }
        ],
        "temperature": 0.0,
        "max_output_tokens": 4096,
        "thinking": {"type": "disabled"},
    }
    assert response.text == "<answer>A</answer>"
    assert response.input_tokens == 11
    assert response.output_tokens == 3
    assert response.generation_config == {
        "model": "seed-2-0-pro-260328",
        "temperature": 0.0,
        "max_output_tokens": 4096,
        "thinking": {"type": "disabled"},
    }
