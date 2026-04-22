#!/usr/bin/env python

import os
import json
import random
import time
import urllib.error
import urllib.request
from pathlib import Path
from datasets import Dataset
from openai import OpenAI
import click
import concurrent.futures
import threading
from dataclasses import dataclass
from tqdm import tqdm
from dotenv import load_dotenv
from artifact_paths import candidate_results_dir, resolve_result_file_candidates
from baseset.legacy_boundary import (
    is_legacy_jp_v1_snapshot,
    schema_v2_path,
)
from benchmark_tasks import (
    load_judge_profile,
    load_task_config,
    resolve_compare_prompt_path,
)
from pair_contract import (
    PAIR_ID_SCHEMA_V1,
    compute_pair_fingerprint,
    ensure_pair_contract_metadata,
)

load_dotenv()

# Try to import google.genai for native Gemini support
try:
    import google.genai as genai
    from google.genai import types as genai_types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

FAILED_TRANSLATION_PREFIX = "[TRANSLATION FAILED:"
REUSE_MATCH_FIELDS = (
    "task_id",
    "task_type",
    "task_version",
    "source_language",
    "target_language",
    "difficulty",
    "snapshot_version",
    "judge_profile_id",
    "compare_prompt_profile_id",
    "judge_parser_id",
    "judge_contract_id",
)


def build_judge_contract_id(judge_model: str, compare_prompt_profile_id: str, parser_id: str) -> str:
    return f"{judge_model}::{compare_prompt_profile_id}::{parser_id}"


def annotate_pair_for_judging(
    pair: dict,
    *,
    judge_model: str,
    judge_profile_id: str,
    compare_prompt_profile_id: str,
    parser_id: str,
    snapshot_version: str,
) -> dict:
    annotated = dict(pair)
    annotated["snapshot_version"] = annotated.get("snapshot_version", snapshot_version)
    annotated["judge_profile_id"] = judge_profile_id
    annotated["compare_prompt_profile_id"] = compare_prompt_profile_id
    annotated["judge_parser_id"] = parser_id
    annotated["judge_contract_id"] = build_judge_contract_id(
        judge_model,
        compare_prompt_profile_id,
        parser_id,
    )
    return annotated


def swap_formatted_translation_sections(formatted_data: str) -> str:
    """Keep A/B headings stable while swapping the translation bodies under them."""
    lines = formatted_data.splitlines(keepends=True)

    def find_line(label: str, start: int = 0) -> int:
        for index in range(start, len(lines)):
            if lines[index].strip() == label:
                return index
        raise ValueError(f"Missing '{label}' section in formatted_data")

    a_header_idx = find_line("## Translation A")
    b_header_idx = find_line("## Translation B", start=a_header_idx + 1)
    end_idx = find_line("---", start=b_header_idx + 1)

    prefix = lines[: a_header_idx + 1]
    section_a = lines[a_header_idx + 1 : b_header_idx]
    b_header = lines[b_header_idx : b_header_idx + 1]
    section_b = lines[b_header_idx + 1 : end_idx]
    suffix = lines[end_idx:]

    return "".join(prefix + section_b + b_header + section_a + suffix)


def validate_pair_record(pair: dict) -> None:
    """Reject pair records that embed unresolved failed generations."""
    formatted_data = pair.get("formatted_data", "")
    if isinstance(formatted_data, str) and FAILED_TRANSLATION_PREFIX in formatted_data:
        raise ValueError(
            f"Cannot judge pair {pair.get('id', 'unknown')} because it contains a failed generation placeholder"
        )


def existing_judgment_matches_pair(existing: dict, pair: dict) -> bool:
    if not existing or existing.get("id") != pair.get("id"):
        return False

    pair_fingerprint = pair.get("pair_fingerprint")
    existing_fingerprint = existing.get("pair_fingerprint")
    for key in REUSE_MATCH_FIELDS:
        pair_value = pair.get(key)
        existing_value = existing.get(key)
        if pair_value and existing_value and pair_value != existing_value:
            return False

    if pair_fingerprint and existing_fingerprint:
        return pair_fingerprint == existing_fingerprint

    if (
        pair.get("pair_id_schema") == PAIR_ID_SCHEMA_V1
        and existing.get("pair_id_schema") in (None, PAIR_ID_SCHEMA_V1)
        and not existing_fingerprint
    ):
        return True

    return not pair_fingerprint and not existing_fingerprint


def extract_pair_payload(record: dict) -> dict:
    pair_payload = {
        "id": record.get("id"),
        "pair_id_schema": record.get("pair_id_schema") or PAIR_ID_SCHEMA_V1,
        "llm_a": record.get("llm_a"),
        "llm_b": record.get("llm_b"),
        "formatted_data": record.get("formatted_data"),
        "name": record.get("name"),
        "english": record.get("english"),
        "difficulty": record.get("difficulty"),
    }
    for key, value in record.items():
        if key.startswith("llm_a_") or key.startswith("llm_b_"):
            pair_payload[key] = value
    return pair_payload


def normalize_reused_judgment(existing: dict, pair: dict) -> dict:
    merged = dict(existing)
    if pair.get("pair_id_schema") and not merged.get("pair_id_schema"):
        merged["pair_id_schema"] = pair["pair_id_schema"]
    if not merged.get("pair_fingerprint"):
        merged["pair_fingerprint"] = compute_pair_fingerprint(extract_pair_payload(merged))
    for key in REUSE_MATCH_FIELDS:
        if pair.get(key) and not merged.get(key):
            merged[key] = pair[key]
    if pair.get("item_id") and not merged.get("item_id"):
        merged["item_id"] = pair["item_id"]
    return merged


def swap_translation_pair_sides(pair: dict) -> dict:
    """Swap pair sides, keeping metadata and displayed labels aligned."""
    pair["llm_a"], pair["llm_b"] = pair["llm_b"], pair["llm_a"]

    suffixes = {
        key[len("llm_a_") :]
        for key in pair
        if key.startswith("llm_a_")
    } | {
        key[len("llm_b_") :]
        for key in pair
        if key.startswith("llm_b_")
    }
    for suffix in suffixes:
        a_key = f"llm_a_{suffix}"
        b_key = f"llm_b_{suffix}"
        a_value = pair.get(a_key)
        b_value = pair.get(b_key)
        if a_key in pair or b_key in pair:
            pair[a_key] = b_value
            pair[b_key] = a_value

    pair["formatted_data"] = swap_formatted_translation_sections(pair["formatted_data"])
    return pair


def should_swap_pair(pair: dict) -> bool:
    pair_id = pair.get("id", "")
    if not pair_id:
        return False
    try:
        pair_value = int(pair_id, 16)
    except ValueError:
        pair_value = sum(ord(char) for char in pair_id)
    return pair_value % 2 == 1


@dataclass
class JudgeResponse:
    text: str
    generation_config: dict
    input_tokens: int = 0
    output_tokens: int = 0


class OpenAIJudgeAdapter:
    OPENAI_UNSUPPORTED_REQUEST_SETTINGS = {"thinking_budget"}

    def __init__(self, model_name: str, request_settings: dict | None = None):
        self.model_name = model_name
        self.request_settings = dict(request_settings or {})

    def request(self, client: OpenAI, prompt_text: str) -> JudgeResponse:
        call_params = {
            "messages": [{"role": "user", "content": prompt_text}],
            "model": self.model_name,
        }
        for key, value in self.request_settings.items():
            if value is not None and key not in self.OPENAI_UNSUPPORTED_REQUEST_SETTINGS:
                call_params[key] = value

        judge_generation_config = call_params.copy()
        judge_generation_config.pop("messages", None)

        chat_completion = client.chat.completions.create(**call_params)
        if not chat_completion.choices or chat_completion.choices[0].message.content is None:
            raise ValueError("Empty response from API")

        input_tokens = 0
        output_tokens = 0
        if hasattr(chat_completion, "usage") and chat_completion.usage:
            input_tokens = chat_completion.usage.prompt_tokens
            output_tokens = chat_completion.usage.completion_tokens

        return JudgeResponse(
            text=chat_completion.choices[0].message.content,
            generation_config=judge_generation_config,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )


class SkylarkResponsesJudgeAdapter:
    RESPONSE_UNSUPPORTED_REQUEST_SETTINGS = {"thinking_budget", "reasoning_effort"}

    def __init__(
        self,
        model_name: str,
        endpoint_url: str,
        api_key: str,
        request_settings: dict | None = None,
    ):
        self.model_name = model_name
        self.endpoint_url = endpoint_url.rstrip("/") or endpoint_url
        self.api_key = api_key
        self.request_settings = dict(request_settings or {})

    def request(self, client, prompt_text: str) -> JudgeResponse:
        payload = {
            "model": self.model_name,
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt_text}],
                }
            ],
        }
        for key, value in self.request_settings.items():
            if value is None or key in self.RESPONSE_UNSUPPORTED_REQUEST_SETTINGS:
                continue
            if key == "max_tokens":
                payload["max_output_tokens"] = value
            else:
                payload[key] = value

        judge_generation_config = {
            key: value for key, value in payload.items() if key != "input"
        }

        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint_url,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "ark-beta-mcp": "true",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                response_body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ValueError(f"Skylark API HTTP {exc.code}: {detail}") from exc

        response_json = self._parse_response_body(
            response_body,
            stream=bool(payload.get("stream")),
        )
        response_text = self._extract_response_text(response_json)
        if not response_text:
            raise ValueError("Empty response from Skylark API")

        usage = response_json.get("usage") or {}
        return JudgeResponse(
            text=response_text,
            generation_config=judge_generation_config,
            input_tokens=usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0,
            output_tokens=usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0,
        )

    @staticmethod
    def _parse_response_body(response_body: str, stream: bool = False) -> dict:
        if not stream:
            return json.loads(response_body)

        text_parts = []
        final_response = {}
        for line in response_body.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            data = line.removeprefix("data:").strip()
            if not data or data == "[DONE]":
                continue
            event = json.loads(data)
            if isinstance(event.get("delta"), str):
                text_parts.append(event["delta"])
            if event.get("type") in {"response.completed", "response.done"}:
                final_response = event.get("response") or final_response

        if final_response:
            if text_parts and "output_text" not in final_response:
                final_response["output_text"] = "".join(text_parts)
            return final_response
        return {"output_text": "".join(text_parts)}

    @staticmethod
    def _extract_response_text(response_json: dict) -> str:
        output_text = response_json.get("output_text")
        if isinstance(output_text, str):
            return output_text

        text_parts = []
        for output_item in response_json.get("output", []) or []:
            for content_item in output_item.get("content", []) or []:
                text = content_item.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
        return "".join(text_parts)


class GeminiJudgeAdapter:
    def __init__(self, model_name: str, safety_settings: list, request_settings: dict | None = None):
        self.model_name = model_name
        self.safety_settings = safety_settings
        self.request_settings = dict(request_settings or {})

    def request(self, client, prompt_text: str) -> JudgeResponse:
        thinking_budget = self.request_settings.get("thinking_budget")
        thinking_config = None
        if thinking_budget is not None:
            thinking_config = genai_types.ThinkingConfig(thinking_budget=thinking_budget)

        gen_config = genai_types.GenerateContentConfig(
            temperature=self.request_settings.get("temperature", 0.0),
            safety_settings=self.safety_settings,
            thinking_config=thinking_config,
        )

        judge_generation_config = {
            "temperature": self.request_settings.get("temperature", 0.0),
            "model": self.model_name,
        }
        for key in ("reasoning_effort",):
            if key in self.request_settings:
                judge_generation_config[key] = self.request_settings[key]
        if thinking_config is not None:
            judge_generation_config["thinking_budget"] = thinking_budget

        response = client.models.generate_content(
            model=self.model_name,
            contents=prompt_text,
            config=gen_config,
        )

        if not response.text:
            raise ValueError("Empty response from API")

        input_tokens = 0
        output_tokens = 0
        if hasattr(response, "usage_metadata"):
            input_tokens = getattr(response.usage_metadata, "prompt_token_count", 0)
            output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0)

        return JudgeResponse(
            text=response.text,
            generation_config=judge_generation_config,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )


class TranslationComparer:
    """Compares two translations and analyzes their differences."""

    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: str,
        concurrency_limit: int,
        prompt_path: Path,
        use_gemini: bool = False,
        use_skylark: bool = False,
        judge_request_settings: dict | None = None,
    ):
        self.model_name = model_name
        self.use_gemini = use_gemini
        self.use_skylark = use_skylark
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.failed_items = []
        self.semaphore = threading.BoundedSemaphore(concurrency_limit)
        self.prompt_path = prompt_path
        self.judge_request_settings = dict(judge_request_settings or {})

        if use_gemini and use_skylark:
            raise ValueError("Only one native judge transport can be selected")

        if use_gemini:
            if not GEMINI_AVAILABLE:
                raise ImportError("google-genai is required for native Gemini support. Install with: pip install google-genai")
            self.client = genai.Client(api_key=api_key)
            # Set up safety settings to bypass safety filters
            self.safety_settings = [
                genai_types.SafetySetting(
                    category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"
                ),
                genai_types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"
                ),
                genai_types.SafetySetting(
                    category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"
                ),
                genai_types.SafetySetting(
                    category="HARM_CATEGORY_HARASSMENT", threshold="OFF"
                ),
            ]
            self.adapter = GeminiJudgeAdapter(
                model_name,
                self.safety_settings,
                request_settings=self.judge_request_settings,
            )
        elif use_skylark:
            self.client = None
            self.adapter = SkylarkResponsesJudgeAdapter(
                model_name,
                base_url,
                api_key,
                request_settings=self.judge_request_settings,
            )
        else:
            self.client = OpenAI(
                base_url=base_url,
                api_key=api_key,
                timeout=120.0,
                max_retries=0,
            )
            self.adapter = OpenAIJudgeAdapter(
                model_name,
                request_settings=self.judge_request_settings,
            )

    def prompt(self, input_data: dict) -> str:
        """Generate a prompt for comparison using the template from prompts/compare_prompt.txt and the translation data."""
        with self.prompt_path.open("r", encoding="utf-8") as f:
            prompt_template = f.read()
        return prompt_template.replace("{{formatted_data}}", input_data["formatted_data"])

    def parse(self, input_data: dict, response: str, judge_generation_config: dict, judge_model: str) -> dict:
        """Parse the model response along with the input data into the desired output format."""
        output = {
            "item_id": input_data.get("item_id"),
            "name": input_data["name"],
            "difficulty": input_data["difficulty"],
            "task_id": input_data.get("task_id"),
            "task_type": input_data.get("task_type"),
            "task_version": input_data.get("task_version"),
            "source_language": input_data.get("source_language"),
            "target_language": input_data.get("target_language"),
            "snapshot_version": input_data.get("snapshot_version"),
            "id": input_data["id"],
            "pair_id_schema": input_data.get("pair_id_schema"),
            "pair_fingerprint": input_data.get("pair_fingerprint"),
            "judge_profile_id": input_data.get("judge_profile_id"),
            "compare_prompt_profile_id": input_data.get("compare_prompt_profile_id"),
            "judge_parser_id": input_data.get("judge_parser_id"),
            "judge_contract_id": input_data.get("judge_contract_id"),
            "llm_a": input_data["llm_a"],
            "llm_b": input_data["llm_b"],
            "formatted_data": input_data["formatted_data"],
            "analysis": response,
            "judge_model": judge_model,
            "judge_temperature": judge_generation_config.get("temperature"),
            "judge_generation_config": judge_generation_config,
            "llm_a_low_context": input_data.get("llm_a_low_context", False),
            "llm_a_ultra_low_context": input_data.get("llm_a_ultra_low_context", False),
            "llm_a_temperature": input_data.get("llm_a_temperature"),
            "llm_a_generation_config": input_data.get("llm_a_generation_config"),
            "llm_b_low_context": input_data.get("llm_b_low_context", False),
            "llm_b_ultra_low_context": input_data.get("llm_b_ultra_low_context", False),
            "llm_b_temperature": input_data.get("llm_b_temperature"),
            "llm_b_generation_config": input_data.get("llm_b_generation_config"),
        }
        if "english" in input_data:
            output["english"] = input_data["english"]
        for key in ("category", "tags", "slice_tags"):
            if key in input_data:
                output[key] = input_data[key]
        return output

    def compare_item(self, item: dict) -> dict:
        """Compares a single item with retry logic and concurrency control."""
        # Add jitter to spread out requests
        time.sleep(random.uniform(0.1, 0.5))

        with self.semaphore:
            prompt_text = self.prompt(item)

            max_retries = 5
            base_delay = 1

            for attempt in range(max_retries + 1):
                try:
                    judge_response = self.adapter.request(self.client, prompt_text)
                    self.total_input_tokens += judge_response.input_tokens
                    self.total_output_tokens += judge_response.output_tokens
                    parsed_result = self.parse(
                        item,
                        judge_response.text,
                        judge_response.generation_config,
                        self.model_name,
                    )
                    return parsed_result

                except Exception as e:
                    error_msg = f"API error: {type(e).__name__}: {str(e)}"
                    if attempt == max_retries:
                        # Track failed item
                        failed_item = {
                            "id": item.get("id", "unknown"),
                            "name": item.get("name", "unknown"),
                            "error": error_msg,
                            "attempts": max_retries + 1
                        }
                        self.failed_items.append(failed_item)
                        print(f"Failed to process item {item.get('name', 'unknown')} after {max_retries + 1} attempts: {error_msg}")
                        return None  # Return None for failed items
                    delay = base_delay * (2 ** attempt)
                    print(f"Attempt {attempt + 1} failed for {item.get('name', 'unknown')}: {error_msg}. Retrying in {delay}s...")
                    time.sleep(delay)

    def __call__(self, dataset: Dataset, max_workers: int) -> list:
        """Process the dataset in parallel and return a list of comparison results."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(
                tqdm(executor.map(self.compare_item, dataset), total=len(dataset))
            )
        return results


@click.command()
@click.option('--task', envvar='TASK_CONFIG', help='Task config path or name under benchmark_tasks/.')
@click.option(
    '--judge-profile',
    envvar='JUDGE_PROFILE',
    default='default',
    show_default=True,
    help='Judge profile path or name under judge_profiles/.',
)
@click.option(
    '--base-url',
    '-u',
    default=os.getenv("JUDGE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"),
    show_default=True,
    help='Base URL for the judge API endpoint (ignored if --gemini-judge is set).',
)
@click.option(
    '--judge-model',
    '-j',
    default="gemini-2.5-flash",
    show_default=True,
    help='Model name to use for judging the translations (mirrors run_translation_bench.sh defaults).',
)
@click.option('--test-model', '-t', help='Model name to test against base models')
@click.option('--generate-base-set', is_flag=True, help='Generate base set comparisons instead of testing a specific model')
@click.option('--max-workers', default=40, help='Number of worker threads for comparison.')
@click.option('--concurrency-limit', default=40, help='Max number of concurrent API requests.')
@click.option(
    '--api-key-env',
    default=os.getenv("JUDGE_API_KEY_ENV", "GEMINI_API_KEY"),
    show_default=True,
    help='Env var name that holds the judge API key.',
)
@click.option(
    '--pairs-file',
    type=click.Path(),
    help='Optional path to pair JSONL. Overrides the default snapshot-local base-set pair file or results-local candidate pair file.',
)
@click.option('--skip-ids', help='Comma-separated list of IDs to skip (already processed). Deprecated - use --skip-ids-file instead.')
@click.option('--skip-ids-file', type=click.Path(exists=True), help='Path to file containing IDs to skip (one per line)')
@click.option(
    '--gemini-judge/--no-gemini-judge',
    default=False,
    show_default=True,
    help='Use native Gemini API instead of OpenAI-compatible endpoint. Bypasses safety filtering and may avoid API errors.',
)
@click.option(
    '--skylark-judge/--no-skylark-judge',
    default=False,
    show_default=True,
    help='Use BytePlus Ark Responses API for Skylark/Seed judges instead of OpenAI-compatible chat completions.',
)
@click.option('--rejudge', is_flag=True, help='Ignore existing judgments and redo all pairs.')
def main(task, judge_profile, base_url, judge_model, test_model, generate_base_set, max_workers, concurrency_limit, api_key_env, pairs_file, skip_ids, skip_ids_file, gemini_judge, skylark_judge, rejudge):
    """Judge pairwise task outputs with a configured LLM judge.

    Reads pair records from JSONL, asks the configured judge to choose a
    winner, and writes the resulting judgment JSONL.
    """
    script_dir = Path(__file__).resolve().parent
    task_config = load_task_config(task)
    judge_profile_config = load_judge_profile(judge_profile)

    if not test_model and not generate_base_set:
        raise click.UsageError(
            "Either --test-model or --generate-base-set must be specified"
        )
    if test_model and generate_base_set:
        raise click.UsageError(
            "Cannot specify both --test-model and --generate-base-set"
        )
    if gemini_judge and skylark_judge:
        raise click.UsageError(
            "Cannot specify both --gemini-judge and --skylark-judge"
        )

    # Validate that prompt file exists
    prompt_path = resolve_compare_prompt_path(task_config, judge_profile_config)
    if not prompt_path.exists():
        raise SystemExit(f"Error: Missing compare prompt file: {prompt_path}. See README for setup.")

    snapshot_dir = Path(os.getenv("BASESET_SNAPSHOT_DIR", "baseset/v1.0"))
    base_version = snapshot_dir.name

    # Read translation pairs
    if pairs_file:
        input_file = Path(pairs_file)
        if not input_file.is_absolute():
            input_file = (Path.cwd() / input_file).resolve()
    else:
        if generate_base_set:
            canonical_snapshot_pairs = snapshot_dir / f"base_conversation_pairs.{base_version}.jsonl"
            generic_snapshot_pairs = snapshot_dir / "base_conversation_pairs.jsonl"
            if canonical_snapshot_pairs.exists():
                input_file = canonical_snapshot_pairs
            else:
                input_file = generic_snapshot_pairs
        else:
            pair_candidates = resolve_result_file_candidates(
                base_version,
                test_model,
                judge_model,
                "pairs.jsonl",
                judge_profile_id=judge_profile_config.profile_id,
                root=script_dir,
            )
            input_file = next((path for path in pair_candidates if path.exists()), pair_candidates[0])
    if not input_file.exists():
        raise SystemExit(f"Error: Input file not found: {input_file}. Please run generate_shootout_data.py first.")

    translation_pairs = []
    with input_file.open("r", encoding="utf-8") as f:
        for line in f:
            pair = ensure_pair_contract_metadata(json.loads(line))
            validate_pair_record(pair)
            translation_pairs.append(pair)

    # Validate that test model exists in the data when not generating base set
    if not generate_base_set:
        llm_a_models = set()
        for pair in translation_pairs:
            if "llm_a" in pair:
                llm_a_models.add(pair["llm_a"])
        
        # Convert test model name to safe format for comparison
        safe_test_model_name = test_model.replace("/", "__")
        if safe_test_model_name not in llm_a_models:
            raise ValueError(f"Model '{safe_test_model_name}' not found in llm_a position in {input_file}. Available llm_a models: {sorted(llm_a_models)}")

    # Deterministically swap half of pairs to account for position bias without depending on file order.
    for pair in translation_pairs:
        if should_swap_pair(pair):
            swap_translation_pair_sides(pair)
        pair["pair_fingerprint"] = compute_pair_fingerprint(pair)
        pair.update(
            annotate_pair_for_judging(
                pair,
                judge_model=judge_model,
                judge_profile_id=judge_profile_config.profile_id,
                compare_prompt_profile_id=judge_profile_config.compare_prompt_profile,
                parser_id=judge_profile_config.parser_id,
                snapshot_version=base_version,
            )
        )

    # Filter out pairs with IDs in skip_ids or skip_ids_file
    skip_id_set = set()
    if skip_ids_file:
        with open(skip_ids_file, 'r') as f:
            skip_id_set = {line.strip() for line in f if line.strip()}
        print(f"Loaded {len(skip_id_set):,} skip IDs from {skip_ids_file}")
    elif skip_ids:
        skip_id_set = set(skip_ids.split(','))
        print(f"Loaded {len(skip_id_set):,} skip IDs from command line")

    if skip_id_set:
        original_count = len(translation_pairs)
        translation_pairs = [p for p in translation_pairs if p.get('id') not in skip_id_set]
        filtered_count = original_count - len(translation_pairs)
        if filtered_count > 0:
            print(f"Skipping {filtered_count:,} already-processed items (out of {original_count:,} total)")

    # Default output path (per model/judge)
    legacy_output_path = None
    if generate_base_set:
        safe_judge_model = judge_model.replace("/", "__")
        output_dir = script_dir / snapshot_dir
        legacy_output_path = output_dir / f"base_set.{safe_judge_model}.jsonl"
        if is_legacy_jp_v1_snapshot(snapshot_dir):
            output_path = schema_v2_path(legacy_output_path)
        else:
            output_path = legacy_output_path
    else:
        output_dir = candidate_results_dir(
            base_version,
            test_model,
            judge_model,
            judge_profile_id=judge_profile_config.profile_id,
            root=script_dir,
        )
        output_path = output_dir / "judgments.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load existing judgments if reusing
    existing_by_id = {}
    existing_order = []
    reuse_sources = [output_path]
    if generate_base_set and legacy_output_path is not None and legacy_output_path != output_path:
        reuse_sources = [legacy_output_path, output_path]
    elif not generate_base_set:
        reuse_sources = resolve_result_file_candidates(
            base_version,
            test_model,
            judge_model,
            "judgments.jsonl",
            judge_profile_id=judge_profile_config.profile_id,
            root=script_dir,
        )

    if not rejudge:
        for reuse_source in reuse_sources:
            if not reuse_source.exists():
                continue
            with reuse_source.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    pid = item.get("id")
                    if not pid:
                        continue
                    if pid not in existing_by_id:
                        existing_order.append(pid)
                    existing_by_id[pid] = item
        if existing_by_id:
            print(f"Loaded {len(existing_by_id):,} reusable judgments from {len([p for p in reuse_sources if p.exists()]):,} source file(s)")

    # Determine pending pairs (not yet judged or forced rejudge)
    pending_pairs = []
    for pair in translation_pairs:
        pid = pair.get("id")
        existing = existing_by_id.get(pid)
        if not rejudge and existing_judgment_matches_pair(existing, pair):
            existing_by_id[pid] = normalize_reused_judgment(existing, pair)
            continue
        pending_pairs.append(pair)

    successful_results = []
    results = []
    failed_items = []
    total_input_tokens = 0
    total_output_tokens = 0
    if pending_pairs:
        # Create dataset and shuffle it
        translations = Dataset.from_list(pending_pairs)
        translations = translations.shuffle(seed=42)  # Set seed for reproducibility

        api_key = os.getenv(api_key_env)
        if not api_key:
            raise SystemExit(f"Error: Missing API key. Set {api_key_env} in your environment.")

        if gemini_judge:
            print(f"Using native Gemini API for judging with model: {judge_model}")
            if not GEMINI_AVAILABLE:
                raise click.UsageError(
                    "google-genai is required for --gemini-judge. Install with: pip install google-genai"
                )
        elif skylark_judge:
            print(f"Using BytePlus Ark Responses API at {base_url} with model: {judge_model}")
        else:
            print(f"Using OpenAI-compatible API at {base_url} with model: {judge_model}")

        comparer = TranslationComparer(
            model_name=judge_model,
            base_url=base_url,
            api_key=api_key,
            concurrency_limit=concurrency_limit,
            prompt_path=prompt_path,
            use_gemini=gemini_judge,
            use_skylark=skylark_judge,
            judge_request_settings=judge_profile_config.resolve_request_settings(judge_model),
        )

        results = comparer(translations, max_workers=max_workers)

        # Filter out None results from failed items
        successful_results = [item for item in results if item is not None]
        failed_items = comparer.failed_items
        total_input_tokens = comparer.total_input_tokens
        total_output_tokens = comparer.total_output_tokens
    else:
        print("All pairs satisfied by existing judgments; no judge requests needed.")

    if existing_by_id:
        print(f"Merging {len(successful_results):,} new results with existing file: {output_path}")
        print(f"  - Existing unique results: {len(existing_by_id):,}")
        print(f"  - New results: {len(successful_results):,}")

    # Combine existing and new results, preferring new judgments when IDs overlap
    merged = dict(existing_by_id)
    for item in successful_results:
        item_id = item.get("id")
        if not item_id:
            continue
        if item_id not in existing_by_id:
            existing_order.append(item_id)
        merged[item_id] = item

    if merged:
        # Preserve existing order, then append any new IDs not previously seen
        ordered_ids = existing_order + [i for i in merged.keys() if i not in existing_order]
        all_results = [merged[i] for i in ordered_ids]
    else:
        all_results = successful_results

    print(f"  - Total merged results: {len(all_results):,}")

    with open(output_path, "w", encoding="utf-8") as f:
        for item in all_results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    expected_count = len(set(p.get("id") for p in translation_pairs))
    judged_count = len(all_results)

    # Print summary
    print(f"\nProcessing Summary:")
    print(f"Pairs after skips: {expected_count}")
    print(f"Newly attempted: {len(results)}")
    print(f"Successful new: {len(successful_results)}")
    print(f"Failed new: {len(failed_items)}")
    print(f"Total judged entries now: {judged_count}")
    
    if failed_items:
        print(f"\nFailed items:")
        for failed in failed_items:
            print(f"  - {failed['name']} (ID: {failed['id']}): {failed['error']}")

    print(f"\nToken Usage Summary:")
    print(f"Input tokens: {total_input_tokens:,}")
    print(f"Output tokens: {total_output_tokens:,}")
    print(f"Total tokens: {total_input_tokens + total_output_tokens:,}")

if __name__ == "__main__":
    main()
