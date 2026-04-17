import importlib
import sys
import types
from types import MethodType


def load_translator_class():
    dotenv_was_present = "dotenv" in sys.modules
    original_dotenv = sys.modules.get("dotenv")
    if not dotenv_was_present:
        dotenv_stub = types.ModuleType("dotenv")
        dotenv_stub.load_dotenv = lambda *args, **kwargs: None
        sys.modules["dotenv"] = dotenv_stub

    try:
        return importlib.import_module("generate_translation_data").Translator
    finally:
        if not dotenv_was_present:
            sys.modules.pop("dotenv", None)
        else:
            sys.modules["dotenv"] = original_dotenv


def build_translator():
    translator_class = load_translator_class()
    translator = translator_class(
        model_name="fixture-model",
        base_url="https://example.com/v1",
        api_key="test-key",
    )
    translator.build_output_base = MethodType(
        lambda _self, *_args, **_kwargs: {},
        translator,
    )
    return translator


def base_item():
    return {
        "name": "fixture-item",
        "text": "hello",
        "difficulty": "easy",
        "english": True,
    }


def test_parse_prefers_translation_tag_over_think_block():
    translator = build_translator()
    parsed = translator.parse(
        base_item(),
        "<think>draft reasoning</think><translation>你好</translation>",
        "prompt body",
        "prompt template",
        {"temperature": 0.1},
    )
    assert parsed["translation"] == "你好"


def test_parse_prefers_translation_tag_over_translation_analysis_block():
    translator = build_translator()
    parsed = translator.parse(
        base_item(),
        "<translation_analysis>legacy reasoning</translation_analysis><translation>Hello</translation>",
        "prompt body",
        "prompt template",
        {"temperature": 0.1},
    )
    assert parsed["translation"] == "Hello"


def test_parse_fallback_strips_think_blocks_when_translation_tag_is_missing():
    translator = build_translator()
    parsed = translator.parse(
        base_item(),
        "<think>private chain of thought</think>\nFinal answer only",
        "prompt body",
        "prompt template",
        {"temperature": 0.1},
    )
    assert parsed["translation"].strip() == "Final answer only"
    assert "<think>" not in parsed["translation"]


def test_parse_fallback_strips_translation_analysis_when_translation_tag_is_missing():
    translator = build_translator()
    parsed = translator.parse(
        base_item(),
        "<translation_analysis>legacy reasoning</translation_analysis>\nVisible translation",
        "prompt body",
        "prompt template",
        {"temperature": 0.1},
    )
    assert parsed["translation"].strip() == "Visible translation"
    assert "<translation_analysis>" not in parsed["translation"]


def test_parse_ignores_stray_think_tags_when_translation_exists():
    translator = build_translator()
    parsed = translator.parse(
        base_item(),
        "<think>scratchpad</think>\n<translation>Bonjour</translation>\n<think>extra</think>",
        "prompt body",
        "prompt template",
        {"temperature": 0.1},
    )
    assert parsed["translation"] == "Bonjour"


def test_parse_fallback_prefers_suffix_after_last_reasoning_block():
    translator = build_translator()
    parsed = translator.parse(
        base_item(),
        "Preface that should not survive\n<think>private reasoning</think>\nFinal answer only",
        "prompt body",
        "prompt template",
        {"temperature": 0.1},
    )
    assert parsed["translation"] == "Final answer only"


def test_parse_fallback_discards_reasoning_only_think_response():
    translator = build_translator()
    parsed = translator.parse(
        base_item(),
        "<think>private chain of thought only</think>",
        "prompt body",
        "prompt template",
        {"temperature": 0.1},
    )
    assert parsed["translation"] == ""


def test_parse_fallback_discards_reasoning_only_translation_analysis_response():
    translator = build_translator()
    parsed = translator.parse(
        base_item(),
        "<translation_analysis>legacy reasoning only</translation_analysis>",
        "prompt body",
        "prompt template",
        {"temperature": 0.1},
    )
    assert parsed["translation"] == ""


def test_parse_prefers_last_translation_tag():
    translator = build_translator()
    parsed = translator.parse(
        base_item(),
        "<translation>draft</translation>\n<think>scratchpad</think>\n<translation>final answer</translation>",
        "prompt body",
        "prompt template",
        {"temperature": 0.1},
    )
    assert parsed["translation"] == "final answer"
