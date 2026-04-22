import copy
import json
from pathlib import Path
import re

import pytest

from translation_comparer_any_model import swap_translation_pair_sides


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "task1" / "swap_pair.json"


def load_swap_pair_fixture():
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_swap_translation_pair_sides_keeps_side_metadata_and_labels_in_sync(newline):
    pair = load_swap_pair_fixture()
    if newline == "\r\n":
        pair["formatted_data"] = pair["formatted_data"].replace("\n", "\r\n")

    swapped = swap_translation_pair_sides(copy.deepcopy(pair))

    assert swapped["llm_a"] == pair["llm_b"]
    assert swapped["llm_b"] == pair["llm_a"]
    assert swapped["llm_a_low_context"] == pair["llm_b_low_context"]
    assert swapped["llm_a_ultra_low_context"] == pair["llm_b_ultra_low_context"]
    assert swapped["llm_a_temperature"] == pair["llm_b_temperature"]
    assert swapped["llm_a_generation_config"] == pair["llm_b_generation_config"]
    assert swapped["llm_b_low_context"] == pair["llm_a_low_context"]
    assert swapped["llm_b_ultra_low_context"] == pair["llm_a_ultra_low_context"]
    assert swapped["llm_b_temperature"] == pair["llm_a_temperature"]
    assert swapped["llm_b_generation_config"] == pair["llm_a_generation_config"]
    assert re.search(r"## Translation A\s+\*\*Example 1:\*\*", swapped["formatted_data"])
    assert re.search(r"## Translation B\s+Example 1:", swapped["formatted_data"])


def test_swap_translation_pair_sides_preserves_internal_markdown_rules():
    pair = load_swap_pair_fixture()
    pair["formatted_data"] = (
        "## Name: markdown_rule\n\n"
        "## Source Text:\nこんにちは\n\n"
        "## Translation A\n"
        "Plain translation.\n\n"
        "## Translation B\n"
        "### 译文\n"
        "---\n"
        "Translated body after a Markdown rule.\n"
        "---\n"
        "Translator note after another rule.\n\n"
        "---\n"
    )

    swapped = swap_translation_pair_sides(copy.deepcopy(pair))

    assert re.search(
        r"## Translation A\s+### 译文\s+---\s+Translated body after a Markdown rule\.\s+---\s+Translator note",
        swapped["formatted_data"],
    )
    assert re.search(r"## Translation B\s+Plain translation\.", swapped["formatted_data"])


def test_swap_translation_pair_sides_handles_tagged_sections_with_markdown_content():
    pair = load_swap_pair_fixture()
    pair["formatted_data"] = (
        "<item>\n"
        "<name>\nmarkdown_rule\n</name>\n\n"
        "<source_text>\nこんにちは\n</source_text>\n\n"
        "<translation_a>\n"
        "Plain translation.\n"
        "</translation_a>\n\n"
        "<translation_b>\n"
        "### 译文\n"
        "---\n"
        "Translated body after a Markdown rule.\n"
        "---\n"
        "Translator note after another rule.\n"
        "</translation_b>\n"
        "</item>\n"
    )

    swapped = swap_translation_pair_sides(copy.deepcopy(pair))

    assert re.search(
        r"<translation_a>\s+### 译文\s+---\s+Translated body after a Markdown rule\.\s+---\s+Translator note",
        swapped["formatted_data"],
    )
    assert re.search(r"<translation_b>\s+Plain translation\.\s+</translation_b>", swapped["formatted_data"])
