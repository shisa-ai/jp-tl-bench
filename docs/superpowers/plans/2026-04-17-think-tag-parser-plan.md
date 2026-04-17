# Think-Tag Prompt And Parser Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Support `<think>` reasoning tags in Chinese translation prompts while keeping Japanese prompts unchanged, and make translation parsing treat `<translation>` as the only canonical output.

**Architecture:** Keep the prompt contract backward-compatible by changing only the Chinese prompt wording and centralizing parser cleanup in `generate_translation_data.py`. Add dedicated parser tests first, then update the parser helpers, then update Chinese prompt fixtures and prompt-resolution tests so JP behavior stays unchanged.

**Tech Stack:** Python 3.12, Click CLI entrypoints, pytest, YAML task configs, plain-text prompt templates

---

## File Structure

- `generate_translation_data.py`
  - Owns translation request configuration and response parsing.
  - Add small parser helpers for stripping reasoning-tag blocks and choosing fallback text.
- `tests/test_translation_parser.py`
  - New focused regression tests for legacy JP and new CN parser behavior.
- `tests/test_generation_prompt_selection.py`
  - Extend prompt-selection coverage to assert the CN prompt contract changed and JP did not.
- `prompts/translate_prompt_from_chinese.txt`
  - Switch visible reasoning guidance from `<translation_analysis>` to `<think>`.
- `prompts/translate_prompt_from_chinese_low_context.txt`
  - Same contract change for the low-context CN prompt.
- `prompts/translate_prompt_from_chinese_ultra_low_context.txt`
  - Same contract change for the ultra-low-context CN prompt.
- `prompts/translate_prompt_from_english_to_chinese.txt`
  - Same contract change for EN->ZH default prompt.
- `prompts/translate_prompt_from_english_to_chinese_low_context.txt`
  - Same contract change for EN->ZH low-context prompt.
- `prompts/translate_prompt_from_english_to_chinese_ultra_low_context.txt`
  - Same contract change for EN->ZH ultra-low-context prompt.

### Task 1: Add Parser Regression Tests

**Files:**
- Create: `tests/test_translation_parser.py`
- Test: `tests/test_translation_parser.py`

- [x] **Step 1: Write the failing parser regression tests**

```python
from generate_translation_data import Translator


def build_translator():
    return Translator(
        model_name="fixture-model",
        base_url="https://example.com/v1",
        api_key="test-key",
    )


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
        "prompts/translate_prompt_from_chinese.txt",
        {"temperature": 0.1},
    )
    assert parsed["translation"] == "你好"


def test_parse_prefers_translation_tag_over_translation_analysis_block():
    translator = build_translator()
    parsed = translator.parse(
        base_item(),
        "<translation_analysis>legacy reasoning</translation_analysis><translation>Hello</translation>",
        "prompt body",
        "prompts/translate_prompt_from_japanese.txt",
        {"temperature": 0.1},
    )
    assert parsed["translation"] == "Hello"


def test_parse_fallback_strips_think_blocks_when_translation_tag_is_missing():
    translator = build_translator()
    parsed = translator.parse(
        base_item(),
        "<think>private chain of thought</think>\nFinal answer only",
        "prompt body",
        "prompts/translate_prompt_from_chinese.txt",
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
        "prompts/translate_prompt_from_japanese.txt",
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
        "prompts/translate_prompt_from_chinese.txt",
        {"temperature": 0.1},
    )
    assert parsed["translation"] == "Bonjour"
```

- [x] **Step 2: Run the new test file to verify it fails**

Run:

```bash
mamba run -n shisa-jp-tl-bench pytest tests/test_translation_parser.py -q
```

Expected:

```text
FAIL at least 2 tests because fallback parsing still keeps reasoning-tag content in translation
```

- [x] **Step 3: Commit the failing test scaffold**

```bash
git add tests/test_translation_parser.py
git commit -m "test: add translation parser tag regressions"
```

### Task 2: Refactor Translation Parsing

**Files:**
- Modify: `generate_translation_data.py`
- Test: `tests/test_translation_parser.py`

- [x] **Step 1: Add small parser helpers above `Translator.parse`**

```python
REASONING_BLOCK_PATTERNS = (
    re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<translation_analysis>.*?</translation_analysis>", re.DOTALL | re.IGNORECASE),
)


def strip_reasoning_blocks(text: str) -> str:
    cleaned = text
    for pattern in REASONING_BLOCK_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    return cleaned


def fallback_translation_text(response: str) -> str:
    cleaned = strip_reasoning_blocks(response)
    cleaned = cleaned.strip()
    if cleaned:
        return cleaned
    return response.strip()
```

- [x] **Step 2: Rewrite `Translator.parse` to use canonical `<translation>` extraction first**

```python
matches = re.findall(r"<translation>(.*?)</translation>", response, re.DOTALL | re.IGNORECASE)
if matches:
    translation = matches[-1].strip()
else:
    print(f"Error: No translation tags found in response for input: {input_data['name']}")
    translation = fallback_translation_text(response)
```

- [x] **Step 3: Preserve the rest of the parsed artifact shape unchanged**

```python
return {
    **self.build_output_base(input_data, prompt_text, prompt_path),
    "status": "ok",
    "generation_profile_id": generation_config.get("profile_id", "default"),
    "full_response": response,
    "translation": translation,
    "temperature": generation_config.get("temperature"),
    "top_p": generation_config.get("top_p"),
    "frequency_penalty": generation_config.get("frequency_penalty"),
    "reasoning_effort": generation_config.get("reasoning_effort"),
    "generation_config": generation_config,
}
```

- [x] **Step 4: Run the focused parser tests**

Run:

```bash
mamba run -n shisa-jp-tl-bench pytest tests/test_translation_parser.py -q
```

Expected:

```text
5 passed
```

- [x] **Step 5: Run the existing translation parsing regressions**

Run:

```bash
mamba run -n shisa-jp-tl-bench pytest \
  tests/test_failed_generation_handling.py \
  tests/test_generation_prompt_selection.py \
  tests/test_task_config.py -q
```

Expected:

```text
all selected tests pass with no changes to artifact shape assertions
```

- [x] **Step 6: Commit the parser refactor**

```bash
git add generate_translation_data.py tests/test_translation_parser.py
git commit -m "feat: strip reasoning tags from translation fallback"
```

### Task 3: Update Chinese Prompt Contracts

**Files:**
- Modify: `prompts/translate_prompt_from_chinese.txt`
- Modify: `prompts/translate_prompt_from_chinese_low_context.txt`
- Modify: `prompts/translate_prompt_from_chinese_ultra_low_context.txt`
- Modify: `prompts/translate_prompt_from_english_to_chinese.txt`
- Modify: `prompts/translate_prompt_from_english_to_chinese_low_context.txt`
- Modify: `prompts/translate_prompt_from_english_to_chinese_ultra_low_context.txt`
- Modify: `tests/test_generation_prompt_selection.py`
- Test: `tests/test_generation_prompt_selection.py`

- [x] **Step 1: Change the Chinese prompt wording from `<translation_analysis>` to `<think>`**

Example replacement for each Chinese prompt file:

```text
Before giving the final answer, reason in <think> tags about:
- wording choices that affect tone or register
- culturally specific references or implied context
- structural changes needed for natural {{tgt_lang}}

Provide the final translation inside <translation> tags.
```

- [x] **Step 2: Add prompt-contract regression tests**

```python
def test_chinese_translation_prompts_use_think_tags():
    zh_prompt = (REPO_ROOT / "prompts" / "translate_prompt_from_chinese.txt").read_text(encoding="utf-8")
    en_to_zh_prompt = (REPO_ROOT / "prompts" / "translate_prompt_from_english_to_chinese.txt").read_text(encoding="utf-8")

    assert "<think>" in zh_prompt
    assert "<translation_analysis>" not in zh_prompt
    assert "<think>" in en_to_zh_prompt
    assert "<translation_analysis>" not in en_to_zh_prompt


def test_japanese_translation_prompt_keeps_legacy_translation_analysis_tags():
    jp_prompt = (REPO_ROOT / "prompts" / "translate_prompt_from_japanese.txt").read_text(encoding="utf-8")

    assert "<translation_analysis>" in jp_prompt
    assert "<think>" not in jp_prompt
```

- [x] **Step 3: Run the prompt-selection tests**

Run:

```bash
mamba run -n shisa-jp-tl-bench pytest tests/test_generation_prompt_selection.py -q
```

Expected:

```text
all prompt-selection tests pass, including the new contract assertions
```

- [x] **Step 4: Commit the prompt updates**

```bash
git add \
  prompts/translate_prompt_from_chinese.txt \
  prompts/translate_prompt_from_chinese_low_context.txt \
  prompts/translate_prompt_from_chinese_ultra_low_context.txt \
  prompts/translate_prompt_from_english_to_chinese.txt \
  prompts/translate_prompt_from_english_to_chinese_low_context.txt \
  prompts/translate_prompt_from_english_to_chinese_ultra_low_context.txt \
  tests/test_generation_prompt_selection.py
git commit -m "feat: adopt think tags for chinese translation prompts"
```

### Task 4: Final Verification

**Files:**
- Modify: none
- Test: `tests/test_translation_parser.py`
- Test: `tests/test_generation_prompt_selection.py`
- Test: `tests/test_failed_generation_handling.py`
- Test: `tests/test_task_config.py`

- [x] **Step 1: Run the narrow final verification suite**

Run:

```bash
mamba run -n shisa-jp-tl-bench pytest \
  tests/test_translation_parser.py \
  tests/test_generation_prompt_selection.py \
  tests/test_failed_generation_handling.py \
  tests/test_task_config.py -q
```

Expected:

```text
all selected tests pass
```

- [x] **Step 2: Re-read the changed prompt files for accidental JP churn**

Run:

```bash
rg -n "<think>|<translation_analysis>" \
  prompts/translate_prompt_from_chinese*.txt \
  prompts/translate_prompt_from_english_to_chinese*.txt \
  prompts/translate_prompt_from_japanese*.txt
```

Expected:

```text
Chinese prompt files show <think>; Japanese prompt files still show <translation_analysis>
```

- [x] **Step 3: Review the diff before handoff**

Run:

```bash
git diff --stat HEAD~3..HEAD
```

Expected:

```text
only parser code, parser tests, generation prompt tests, and Chinese prompt templates changed
```

- [x] **Step 4: Commit any final cleanup if needed**

```bash
git status -sb
```

Expected:

```text
no unexpected tracked-file edits remain beyond the planned changes
```

## Self-Review

### Spec coverage

- CN prompt contract change: covered in Task 3.
- JP prompt compatibility: covered in Task 3 and Task 4.
- Parser ignores `<think>` and `<translation_analysis>`: covered in Task 2.
- Fallback stays enabled: covered in Task 1 and Task 2.
- Regression coverage for old/new prompt contracts: covered in Task 1 and Task 3.

### Placeholder scan

- No placeholder markers remain.
- All tasks name exact files and concrete commands.
- Code-changing steps include concrete code blocks.

### Type consistency

- Parser helpers are named `strip_reasoning_blocks` and `fallback_translation_text` consistently across tasks.
- Tests consistently call `Translator.parse(...)` and assert on the `translation` field.
