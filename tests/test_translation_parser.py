import sys
import types
from types import MethodType

if "dotenv" not in sys.modules:
    dotenv_stub = types.ModuleType("dotenv")
    dotenv_stub.load_dotenv = lambda *args, **kwargs: None
    sys.modules["dotenv"] = dotenv_stub

from generate_translation_data import Translator


def build_translator():
    translator = Translator(
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
