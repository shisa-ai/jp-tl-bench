#!/usr/bin/env python

import os
import json
import random
import time
from pathlib import Path
from datasets import Dataset
from openai import OpenAI
import click
import concurrent.futures
import threading
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# Try to import google.genai for native Gemini support
try:
    import google.genai as genai
    from google.genai import types as genai_types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


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
        request_delay_seconds: float = 0.0,
        request_timeout_seconds: float = 180.0,
    ):
        self.model_name = model_name
        self.use_gemini = use_gemini
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.failed_items = []
        self.semaphore = threading.BoundedSemaphore(concurrency_limit)
        self.rate_limit_lock = threading.Lock()
        self.last_request_at = 0.0
        self.request_delay_seconds = request_delay_seconds
        self.prompt_path = prompt_path
        self.request_timeout_seconds = request_timeout_seconds
        self.abort_event = threading.Event()
        self.abort_reason = None

        if use_gemini:
            if not GEMINI_AVAILABLE:
                raise ImportError("google-genai is required for native Gemini support. Install with: pip install google-genai")
            http_options = None
            if request_timeout_seconds > 0:
                http_options = genai_types.HttpOptions(timeout=int(request_timeout_seconds * 1000))
            self.client = genai.Client(api_key=api_key, http_options=http_options)
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
        else:
            timeout = request_timeout_seconds if request_timeout_seconds > 0 else None
            self.client = OpenAI(
                base_url=base_url,
                api_key=api_key,
                timeout=timeout,
            )

    def pace_request(self) -> None:
        """Apply a global minimum delay between judge API calls."""
        if self.request_delay_seconds <= 0:
            return

        with self.rate_limit_lock:
            now = time.monotonic()
            wait_seconds = self.request_delay_seconds - (now - self.last_request_at)
            if wait_seconds > 0:
                time.sleep(wait_seconds)
            self.last_request_at = time.monotonic()

    def abort_for_daily_quota(self, error_msg: str) -> bool:
        """Stop the batch when the judge model's daily request cap is exhausted."""
        if "generate_requests_per_model_per_day" not in error_msg:
            return False
        self.abort_reason = (
            "Gemini judge daily request quota exhausted. "
            "Checkpointed judgments were written; rerun after quota reset to resume."
        )
        self.abort_event.set()
        print(f"Aborting judge run: {self.abort_reason}")
        return True

    def prompt(self, input_data: dict) -> str:
        """Generate a prompt for comparison using the template from prompts/compare_prompt.txt and the translation data."""
        with self.prompt_path.open("r", encoding="utf-8") as f:
            prompt_template = f.read()
        return prompt_template.replace("{{formatted_data}}", input_data["formatted_data"])

    def parse(self, input_data: dict, response: str, judge_generation_config: dict, judge_model: str) -> dict:
        """Parse the model response along with the input data into the desired output format."""
        return {
            "name": input_data["name"],
            "english": input_data["english"],
            "difficulty": input_data["difficulty"],
            "id": input_data["id"],
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

    def compare_item_gemini(self, item: dict) -> dict:
        """Compares a single item using native Gemini API with retry logic and concurrency control."""
        if self.abort_event.is_set():
            return None

        # Add jitter to spread out requests
        time.sleep(random.uniform(0.1, 0.5))

        with self.semaphore:
            if self.abort_event.is_set():
                return None
            prompt_text = self.prompt(item)

            max_retries = 5
            base_delay = 1

            for attempt in range(max_retries + 1):
                if self.abort_event.is_set():
                    return None
                try:
                    # Set up thinking config for Gemini 2.5 models
                    # thinking_budget is in tokens: 0 = disabled (flash), 128 = low (pro), 512 = medium, higher = more thinking
                    thinking_config = None
                    thinking_budget = 0
                    if 'gemini-2.5-pro' in self.model_name:
                        thinking_budget = 128
                        thinking_config = genai_types.ThinkingConfig(thinking_budget=thinking_budget)
                    elif 'gemini-2.5' in self.model_name:
                        # Flash or other 2.5 models - disable thinking
                        thinking_budget = 0
                        thinking_config = genai_types.ThinkingConfig(thinking_budget=thinking_budget)

                    # Set up generation config
                    gen_config = genai_types.GenerateContentConfig(
                        temperature=0.0,
                        safety_settings=self.safety_settings,
                        thinking_config=thinking_config,
                    )

                    # Create judge generation config for saving
                    judge_generation_config = {
                        "temperature": 0.0,
                        "model": self.model_name,
                    }
                    if thinking_config:
                        judge_generation_config['thinking_budget'] = thinking_budget

                    # Generate content
                    self.pace_request()
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=prompt_text,
                        config=gen_config,
                    )

                    if not response.text:
                        raise ValueError("Empty response from API")

                    # Track token usage
                    if hasattr(response, 'usage_metadata'):
                        self.total_input_tokens += getattr(response.usage_metadata, 'prompt_token_count', 0)
                        self.total_output_tokens += getattr(response.usage_metadata, 'candidates_token_count', 0)

                    parsed_result = self.parse(item, response.text, judge_generation_config, self.model_name)
                    return parsed_result

                except Exception as e:
                    error_msg = f"API error: {type(e).__name__}: {str(e)}"
                    if self.abort_for_daily_quota(error_msg):
                        return None
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

    def compare_item(self, item: dict) -> dict:
        """Compares a single item with retry logic and concurrency control."""
        if self.use_gemini:
            return self.compare_item_gemini(item)

        if self.abort_event.is_set():
            return None

        # Add jitter to spread out requests
        time.sleep(random.uniform(0.1, 0.5))

        with self.semaphore:
            if self.abort_event.is_set():
                return None
            prompt_text = self.prompt(item)

            max_retries = 5
            base_delay = 1

            for attempt in range(max_retries + 1):
                if self.abort_event.is_set():
                    return None
                try:
                    temp = 0
                    lower_name = self.model_name.lower()
                    if "gpt-5" in lower_name:
                        temp = None  # gpt-5 disallows explicit temperature; use model default
                    call_params = {
                        "messages": [{"role": "user", "content": prompt_text}],
                        "model": self.model_name,
                    }
                    if temp is not None:
                        call_params["temperature"] = temp

                    if 'gemini-2.5' in self.model_name:
                        call_params['reasoning_effort'] = 'low'

                    # Create judge generation config for saving
                    judge_generation_config = call_params.copy()
                    judge_generation_config.pop("messages", None)  # Remove messages from config

                    self.pace_request()
                    chat_completion = self.client.chat.completions.create(**call_params)
                    if not chat_completion.choices or chat_completion.choices[0].message.content is None:
                        raise ValueError("Empty response from API")

                    response = chat_completion.choices[0].message.content

                    # Track token usage
                    if hasattr(chat_completion, 'usage') and chat_completion.usage:
                        self.total_input_tokens += chat_completion.usage.prompt_tokens
                        self.total_output_tokens += chat_completion.usage.completion_tokens

                    parsed_result = self.parse(item, response, judge_generation_config, self.model_name)
                    return parsed_result

                except Exception as e:
                    error_msg = f"API error: {type(e).__name__}: {str(e)}"
                    if self.abort_for_daily_quota(error_msg):
                        return None
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

    def __call__(self, dataset: Dataset, max_workers: int, on_result=None) -> list:
        """Process the dataset in parallel and return a list of comparison results."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.compare_item, item) for item in dataset]
            results = []
            with tqdm(total=len(futures)) as progress:
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    results.append(result)
                    if result is not None and on_result is not None:
                        on_result(result)
                    progress.update(1)
        return results


@click.command()
@click.option(
    '--base-url',
    '-u',
    default=os.getenv("JUDGE_URL", "https://api.openai.com/v1"),
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
    '--request-delay-seconds',
    default=0.0,
    type=float,
    show_default=True,
    help='Minimum delay between judge API calls. Use 13 for Gemini free-tier 5 RPM.',
)
@click.option(
    '--request-timeout-seconds',
    default=180.0,
    type=float,
    show_default=True,
    help='Per-request API timeout. Prevents stalled judge calls from blocking the whole run.',
)
@click.option(
    '--api-key-env',
    default=os.getenv("JUDGE_API_KEY_ENV", "OPENAI_API_KEY"),
    show_default=True,
    help='Env var name that holds the judge API key. Use GEMINI_API_KEY when --gemini-judge.',
)
@click.option(
    '--pairs-file',
    type=click.Path(),
    help='Optional path to translation pairs JSONL. Overrides default base_conversation_pairs.jsonl/latest_conversation_pairs.jsonl.',
)
@click.option('--skip-ids', help='Comma-separated list of IDs to skip (already processed). Deprecated - use --skip-ids-file instead.')
@click.option('--skip-ids-file', type=click.Path(exists=True), help='Path to file containing IDs to skip (one per line)')
@click.option(
    '--gemini-judge/--no-gemini-judge',
    default=False,
    show_default=True,
    help='Use native Gemini API instead of OpenAI-compatible endpoint. Bypasses safety filtering and may avoid API errors.',
)
@click.option('--rejudge', is_flag=True, help='Ignore existing judgments and redo all pairs.')
def main(base_url, judge_model, test_model, generate_base_set, max_workers, concurrency_limit, request_delay_seconds, request_timeout_seconds, api_key_env, pairs_file, skip_ids, skip_ids_file, gemini_judge, rejudge):
    """Compare translations between different models using a third LLM as analyzer.

    Reads the translation pairs from the JSONL file, creates a dataset,
    and uses an LLM to analyze the differences. Saves the analysis results
    to a new JSONL file.
    """
    script_dir = Path(__file__).resolve().parent

    if not test_model and not generate_base_set:
        raise click.UsageError(
            "Either --test-model-name or --generate-base-set must be specified"
        )
    if test_model and generate_base_set:
        raise click.UsageError(
            "Cannot specify both --test-model-name and --generate-base-set"
        )

    # Validate that prompt file exists
    prompt_path = script_dir / "prompts" / "compare_prompt.txt"
    if not prompt_path.exists():
        raise SystemExit("Error: Missing prompts/compare_prompt.txt. See README for setup.")

    snapshot_dir = Path(os.getenv("BASESET_SNAPSHOT_DIR", "baseset/v1.0"))
    base_version = snapshot_dir.name

    # Read translation pairs
    if pairs_file:
        input_file = Path(pairs_file)
        if not input_file.is_absolute():
            input_file = (Path.cwd() / input_file).resolve()
    else:
        if generate_base_set:
            input_file = script_dir / "base_conversation_pairs.jsonl"
        else:
            safe_test_model = test_model.replace("/", "__")
            safe_judge = judge_model.replace("/", "__")
            input_file = script_dir / "results" / base_version / safe_test_model / safe_judge / "pairs.jsonl"
    if not input_file.exists():
        raise SystemExit(f"Error: Input file not found: {input_file}. Please run generate_shootout_data.py first.")

    translation_pairs = []
    with input_file.open("r", encoding="utf-8") as f:
        for line in f:
            translation_pairs.append(json.loads(line))

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

    # Randomize positions to account for position bias
    random.seed(42)  # Set seed for reproducibility
    for pair in translation_pairs:
        if random.random() < 0.5:
            # Swap positions
            pair["llm_a"], pair["llm_b"] = pair["llm_b"], pair["llm_a"]
            # Update formatted_data to reflect the swap
            lines = pair["formatted_data"].split('\n')
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped == "## Translation A":
                    lines[i] = line.replace("## Translation A", "## Translation B")
                elif stripped == "## Translation B":
                    lines[i] = line.replace("## Translation B", "## Translation A")
            pair["formatted_data"] = '\n'.join(lines)

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
    if generate_base_set:
        safe_judge_model = judge_model.replace("/", "__")
        output_dir = script_dir / snapshot_dir
        output_path = output_dir / f"base_set.{safe_judge_model}.jsonl"
    else:
        safe_test_model = test_model.replace("/", "__")
        safe_judge_model = judge_model.replace("/", "__")
        output_dir = script_dir / "results" / base_version / safe_test_model / safe_judge_model
        output_path = output_dir / "judgments.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load existing judgments if reusing
    existing_by_id = {}
    existing_order = []
    if output_path.exists() and not rejudge:
        with output_path.open("r", encoding="utf-8") as f:
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
            print(f"Reusing {len(existing_by_id):,} existing judgments from {output_path}")

    # Determine pending pairs (not yet judged or forced rejudge)
    pending_pairs = []
    for pair in translation_pairs:
        pid = pair.get("id")
        if not rejudge and pid in existing_by_id:
            continue
        pending_pairs.append(pair)

    if rejudge:
        output_path.write_text("", encoding="utf-8")

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
    else:
        print(f"Using OpenAI-compatible API at {base_url} with model: {judge_model}")

    comparer = TranslationComparer(
        model_name=judge_model,
        base_url=base_url,
        api_key=api_key,
        concurrency_limit=concurrency_limit,
        prompt_path=prompt_path,
        use_gemini=gemini_judge,
        request_delay_seconds=request_delay_seconds,
        request_timeout_seconds=request_timeout_seconds,
    )

    checkpoint_lock = threading.Lock()
    checkpoint_seen_ids = set(existing_by_id.keys())

    def write_checkpoint(item: dict) -> None:
        item_id = item.get("id")
        if not item_id:
            return
        with checkpoint_lock:
            if item_id in checkpoint_seen_ids:
                return
            checkpoint_seen_ids.add(item_id)
            with output_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    results = comparer(translations, max_workers=max_workers, on_result=write_checkpoint)

    if comparer.abort_reason:
        raise SystemExit(comparer.abort_reason)

    # Filter out None results from failed items
    successful_results = [item for item in results if item is not None]

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
    print(f"Failed new: {len(comparer.failed_items)}")
    print(f"Total judged entries now: {judged_count}")
    
    if comparer.failed_items:
        print(f"\nFailed items:")
        for failed in comparer.failed_items:
            print(f"  - {failed['name']} (ID: {failed['id']}): {failed['error']}")

    print(f"\nToken Usage Summary:")
    print(f"Input tokens: {comparer.total_input_tokens:,}")
    print(f"Output tokens: {comparer.total_output_tokens:,}")
    print(f"Total tokens: {comparer.total_input_tokens + comparer.total_output_tokens:,}")

if __name__ == "__main__":
    main()
