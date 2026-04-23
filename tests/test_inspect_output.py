import importlib.util
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path


def load_inspect_output_module():
    module_path = Path(__file__).resolve().parents[1] / "inspect-output"
    loader = SourceFileLoader("inspect_output", str(module_path))
    spec = importlib.util.spec_from_loader("inspect_output", loader)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_comparison_win_detection_uses_selected_model_side():
    inspect_output = load_inspect_output_module()

    target_as_a_loss = inspect_output.ComparisonMetadata(
        index=0,
        line_num=0,
        name="row-a",
        english=False,
        difficulty="easy",
        comparison_id="a",
        llm_a="target__model",
        llm_b="anchor__model",
        answer="B",
    )
    target_as_b_loss = inspect_output.ComparisonMetadata(
        index=1,
        line_num=1,
        name="row-b",
        english=False,
        difficulty="easy",
        comparison_id="b",
        llm_a="anchor__model",
        llm_b="target__model",
        answer="A",
    )
    target_as_b_win = inspect_output.ComparisonMetadata(
        index=2,
        line_num=2,
        name="row-c",
        english=False,
        difficulty="easy",
        comparison_id="c",
        llm_a="anchor__model",
        llm_b="target__model",
        answer="B",
    )

    assert not target_as_a_loss.is_model_win("target__model")
    assert not target_as_b_loss.is_model_win("target__model")
    assert target_as_b_win.is_model_win("target__model")


def test_extract_answer_accepts_whitespace_in_answer_tag():
    inspect_output = load_inspect_output_module()

    assert inspect_output.extract_answer({"analysis": "<answer> A </answer>"}) == "A"
    assert inspect_output.extract_answer({"analysis": "<answer>\nB\n</answer>"}) == "B"


def test_parse_formatted_data_supports_xml_translation_tags():
    inspect_output = load_inspect_output_module()

    formatted_data = (
        "<item>\n"
        "<name>\nrow\n</name>\n\n"
        "<source_text>\n中文原文。\n</source_text>\n\n"
        "<translation_a>\nJapanese A.\n</translation_a>\n\n"
        "<translation_b>\nJapanese B.\n</translation_b>\n"
        "</item>\n"
    )

    assert inspect_output.parse_formatted_data(formatted_data) == (
        "中文原文。",
        "Japanese A.",
        "Japanese B.",
    )


def test_parse_formatted_data_keeps_markdown_fallback():
    inspect_output = load_inspect_output_module()

    formatted_data = (
        "## Name: row\n\n"
        "## Source Text:\n中文原文。\n\n"
        "## Translation A\nJapanese A.\n\n"
        "## Translation B\nJapanese B.\n\n"
        "---\n"
    )

    assert inspect_output.parse_formatted_data(formatted_data) == (
        "中文原文。",
        "Japanese A.",
        "Japanese B.",
    )
