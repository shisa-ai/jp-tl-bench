import json
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError

from benchmark_tasks import load_task_config
from generate_translation_data import Translator


REPO_ROOT = Path(__file__).resolve().parents[1]
JP_TASK_PATH = REPO_ROOT / "benchmark_tasks" / "translation_ja_en_bidirectional_v1.yaml"


class FakeHTTPResponse:
    def __init__(self, payload: dict):
        self._payload = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def build_chat_completion(text: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20),
    )


def build_item() -> dict:
    return {
        "name": "sample_convo_15",
        "text": "Hello there.",
        "difficulty": "easy",
        "english": True,
    }


def build_translator() -> Translator:
    task = load_task_config(JP_TASK_PATH)
    return Translator(
        model_name="fixture-model",
        base_url="http://127.0.0.1:8016/v1",
        api_key="test-key",
        task_config=task,
        dataset_ref=task.dataset.to_dict(),
        max_tokens=8000,
    )


def test_translate_item_clamps_max_tokens_from_vllm_tokenize(monkeypatch):
    translator = build_translator()
    requests = []

    def fake_urlopen(request, timeout=0):
        body = json.loads(request.data.decode("utf-8"))
        assert request.full_url == "http://127.0.0.1:8016/tokenize"
        assert body == {
            "model": "fixture-model",
            "messages": [{"role": "user", "content": "prompt body"}],
        }
        return FakeHTTPResponse({"count": 8385, "max_model_len": 16384, "tokens": []})

    def fake_create(**params):
        requests.append(params)
        return build_chat_completion("<translation>こんにちは</translation>")

    monkeypatch.setattr("generate_translation_data.time.sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("generate_translation_data.random.uniform", lambda *_args, **_kwargs: 0.0)
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr(translator, "get_prompt", lambda _item: ("prompt body", "prompt/path.txt"))
    monkeypatch.setattr(translator.client.chat.completions, "create", fake_create)

    translated = translator.translate_item(build_item())

    assert requests[0]["max_tokens"] == 7999
    assert translated["generation_config"]["max_tokens"] == 7999


def test_translate_item_falls_back_to_configured_cap_when_tokenize_is_unavailable(monkeypatch):
    translator = build_translator()
    requests = []

    def fake_urlopen(request, timeout=0):
        raise HTTPError(request.full_url, 404, "missing", hdrs=None, fp=None)

    def fake_create(**params):
        requests.append(params)
        return build_chat_completion("<translation>こんにちは</translation>")

    monkeypatch.setattr("generate_translation_data.time.sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("generate_translation_data.random.uniform", lambda *_args, **_kwargs: 0.0)
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr(translator, "get_prompt", lambda _item: ("prompt body", "prompt/path.txt"))
    monkeypatch.setattr(translator.client.chat.completions, "create", fake_create)

    translated = translator.translate_item(build_item())

    assert requests[0]["max_tokens"] == 8000
    assert translated["generation_config"]["max_tokens"] == 8000


def test_translate_item_fails_fast_when_prompt_consumes_full_context(monkeypatch):
    translator = build_translator()
    create_calls = []

    def fake_urlopen(request, timeout=0):
        return FakeHTTPResponse({"count": 16384, "max_model_len": 16384, "tokens": []})

    def fake_create(**params):
        create_calls.append(params)
        return build_chat_completion("<translation>こんにちは</translation>")

    monkeypatch.setattr("generate_translation_data.time.sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("generate_translation_data.random.uniform", lambda *_args, **_kwargs: 0.0)
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr(translator, "get_prompt", lambda _item: ("prompt body", "prompt/path.txt"))
    monkeypatch.setattr(translator.client.chat.completions, "create", fake_create)

    failed = translator.translate_item(build_item())

    assert create_calls == []
    assert failed["status"] == "failed"
    assert "No completion budget" in failed["generation_config"]["error"]
    assert failed["generation_config"]["max_tokens"] == 0
