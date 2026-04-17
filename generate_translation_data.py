import os
import json
import random
import time
from datasets import Dataset, load_dataset
from openai import OpenAI
import click
import re
from dotenv import load_dotenv
from tqdm import tqdm
import concurrent.futures
import threading
from dataclasses import dataclass
from typing import Optional

from benchmark_tasks import load_task_config, resolve_dataset_ref

load_dotenv()

FAILED_TRANSLATION_PREFIX = "[TRANSLATION FAILED:"

@dataclass
class GenerationAdapter:
    """Reusable model/provider generation configuration."""
    profile_id: str = "default"
    temperature: Optional[float] = 0.1
    top_p: Optional[float] = 0.85
    frequency_penalty: Optional[float] = None
    reasoning_effort: Optional[str] = None
    prompt_file: Optional[str] = None  # Override prompt file path (relative to project root)

    def build_request(self, *, model_name: str, prompt_text: str, max_tokens: int | None) -> dict:
        params = {
            "messages": [{"role": "user", "content": prompt_text}],
            "model": model_name,
        }
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.top_p is not None:
            params["top_p"] = self.top_p
        if self.frequency_penalty is not None:
            params["frequency_penalty"] = self.frequency_penalty
        if self.reasoning_effort is not None:
            params["reasoning_effort"] = self.reasoning_effort
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        return params


def resolve_generation_adapter(model_name: str) -> GenerationAdapter:
    """Map model quirks to a reusable generation adapter."""
    model_lower = model_name.lower()

    if "claude-opus-4-1" in model_lower:
        return GenerationAdapter(
            profile_id="claude-opus-4-1",
            temperature=0.2,
            top_p=None,
        )

    if "gpt-5-mini" in model_lower or "gpt-5-nano" in model_lower:
        return GenerationAdapter(
            profile_id="gpt-5-lite",
            temperature=None,
            top_p=None,
        )

    if "gpt-5" in model_lower and "gpt-5-chat-latest" not in model_lower:
        return GenerationAdapter(
            profile_id="gpt-5",
            reasoning_effort="minimal",
            temperature=0.1,
            top_p=0.85,
        )

    if "gemini-2.5-pro" in model_lower:
        return GenerationAdapter(
            profile_id="gemini-2.5-pro",
            reasoning_effort="low",
        )

    if "gemini-2.5" in model_lower:
        return GenerationAdapter(
            profile_id="gemini-2.5",
            reasoning_effort="low",
        )

    if "cat-translate" in model_lower:
        return GenerationAdapter(
            profile_id="cat-translate",
            prompt_file="prompts/translate_prompt_simple.txt",
        )

    return GenerationAdapter()

class Translator:
    """Translates text using a specified model."""

    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: str,
        task_config=None,
        dataset_ref: Optional[dict] = None,
        low_context: bool = False,
        ultra_low_context: bool = False,
        concurrency_limit: int = 5,
        max_tokens: int = 8192,
    ):
        self.model_name = model_name
        self.task_config = task_config or load_task_config()
        self.dataset_ref = dataset_ref or self.task_config.dataset.to_dict()
        self.low_context = low_context
        self.ultra_low_context = ultra_low_context
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.failed_items = []
        self.semaphore = threading.BoundedSemaphore(concurrency_limit)
        self.max_tokens = max_tokens

    def get_generation_adapter(self) -> GenerationAdapter:
        return resolve_generation_adapter(self.model_name)

    def get_prompt_path(self, source_language: str, target_language: str) -> str:
        """Determine the task-configured prompt file for the active direction and context setting."""
        return str(
            self.task_config.get_prompt_path(
                source_language,
                target_language,
                low_context=self.low_context,
                ultra_low_context=self.ultra_low_context,
            )
        )

    def get_prompt(self, input_data: dict) -> tuple[str, str]:
        """Generate a prompt for translation using the appropriate template based on input language."""
        normalized = self.task_config.normalize_record(input_data, require_source_text=True)
        generation_adapter = self.get_generation_adapter()
        if generation_adapter.prompt_file:
            prompt_path = generation_adapter.prompt_file
        else:
            prompt_path = self.get_prompt_path(
                normalized["source_language"],
                normalized["target_language"],
            )

        if not os.path.exists(prompt_path):
            raise SystemExit(f"Error: Missing prompt file: {prompt_path}. See README for setup.")
        
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_template = f.read()
        src_lang = self.task_config.get_language_name(normalized["source_language"])
        tgt_lang = self.task_config.get_language_name(normalized["target_language"])
        prompt_text = (prompt_template
            .replace("{{text}}", normalized["source_text"])
            .replace("{{src_lang}}", src_lang)
            .replace("{{tgt_lang}}", tgt_lang)
        )
        return prompt_text, prompt_path

    def build_output_base(self, input_data: dict, prompt_text: str, prompt_path: str) -> dict:
        normalized = self.task_config.normalize_record(input_data, require_source_text=True)
        output = {
            "item_id": normalized["item_id"],
            "name": normalized["name"],
            "task_id": normalized["task_id"],
            "task_type": normalized["task_type"],
            "task_version": normalized["task_version"],
            "source_text": normalized["source_text"],
            "difficulty": normalized["difficulty"],
            "source_language": normalized["source_language"],
            "target_language": normalized["target_language"],
            "dataset_ref": self.dataset_ref,
            "task_config_digest": self.task_config.task_config_digest,
            "model": self.model_name,
            "generation_profile_id": "default",
            "prompt_profile": self.task_config.get_prompt_variant(
                low_context=self.low_context,
                ultra_low_context=self.ultra_low_context,
            ),
            "prompt_template": prompt_path,
            "prompt": prompt_text,
            "low_context": self.low_context,
            "ultra_low_context": self.ultra_low_context,
        }
        if "english" in normalized:
            output["english"] = normalized["english"]
        return output

    def parse(
        self,
        input_data: dict,
        response: str,
        prompt_text: str,
        prompt_path: str | dict,
        generation_config: Optional[dict] = None,
    ) -> dict:
        """Parse the model response along with the input data into the desired output format."""
        if generation_config is None:
            generation_config = prompt_path
            prompt_path = "<in-memory>"
        matches = re.findall(r'<translation>(.*?)</translation>', response, re.DOTALL)
        if not matches:
            print(f"Error: No translation tags found in response for input: {input_data['name']}")
            analysis_split = response.split('</translation_analysis>')
            if len(analysis_split) > 1:
                print(f"Found </translation_analysis> tag, using text after it for input: {input_data['name']}")
                filtered_response = analysis_split[1]
                filtered_matches = re.findall(r'<translation>(.*?)</translation>', filtered_response, re.DOTALL)
                translation = filtered_matches[-1] if filtered_matches else filtered_response
            else:
                print(f"No </translation_analysis> tag found, using full response for input: {input_data['name']}")
                translation = response
        else:
            translation = matches[-1]

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
            "generation_config": generation_config
        }

    def translate_item(self, item: dict) -> dict:
        """Translates a single item with retry logic and concurrency control."""
        # Add jitter to spread out requests
        time.sleep(random.uniform(0.1, 0.5))

        with self.semaphore:
            prompt_result = self.get_prompt(item)
            if isinstance(prompt_result, tuple):
                prompt_text, prompt_path = prompt_result
            else:
                prompt_text = prompt_result
                prompt_path = self.get_prompt_path(
                    self.task_config.normalize_record(item)["source_language"],
                    self.task_config.normalize_record(item)["target_language"],
                )

            max_retries = 5
            base_delay = 1

            for attempt in range(max_retries + 1):
                try:
                    # Get model-specific configuration
                    generation_adapter = self.get_generation_adapter()
                    params = generation_adapter.build_request(
                        model_name=self.model_name,
                        prompt_text=prompt_text,
                        max_tokens=self.max_tokens,
                    )

                    # Create generation config for saving
                    generation_config = params.copy()
                    generation_config.pop("messages", None)  # Remove messages from config
                    generation_config["profile_id"] = generation_adapter.profile_id

                    chat_completion = self.client.chat.completions.create(**params)
                    if not chat_completion.choices or chat_completion.choices[0].message.content is None:
                        raise ValueError("Empty response from API")

                    response = chat_completion.choices[0].message.content

                    # Track token usage
                    if hasattr(chat_completion, 'usage') and chat_completion.usage:
                        self.total_input_tokens += chat_completion.usage.prompt_tokens
                        self.total_output_tokens += chat_completion.usage.completion_tokens

                    parsed_result = self.parse(item, response, prompt_text, prompt_path, generation_config)
                    return parsed_result

                except Exception as e:
                    error_msg = f"API error: {type(e).__name__}: {str(e)}"
                    if attempt == max_retries:
                        # Track failed item
                        failed_item = {
                            "name": item.get("name", "unknown"),
                            "error": error_msg,
                            "attempts": max_retries + 1,
                        }
                        self.failed_items.append(failed_item)
                        print(
                            f"Failed to process item {item.get('name', 'unknown')} after {max_retries + 1} attempts: {error_msg}"
                        )
                        # Return a placeholder result so downstream scripts
                        # always see the full set of items.
                        placeholder_translation = f"{FAILED_TRANSLATION_PREFIX} {error_msg}]"
                        return {
                            **self.build_output_base(item, prompt_text, prompt_path),
                            "status": "failed",
                            "generation_profile_id": self.get_generation_adapter().profile_id,
                            "full_response": "",
                            "translation": placeholder_translation,
                            "temperature": None,
                            "top_p": None,
                            "frequency_penalty": None,
                            "reasoning_effort": None,
                            "generation_config": {
                                "error": error_msg,
                                "profile_id": self.get_generation_adapter().profile_id,
                                "temperature": None,
                                "top_p": None,
                                "frequency_penalty": None,
                                "reasoning_effort": None,
                                "max_tokens": self.max_tokens,
                            },
                        }
                    delay = base_delay * (2 ** attempt)
                    print(
                        f"Attempt {attempt + 1} failed for {item.get('name', 'unknown')}: {error_msg}. Retrying in {delay}s..."
                    )
                    time.sleep(delay)

    def __call__(self, dataset: Dataset, max_workers: int) -> list:
        """Process the dataset in parallel and return a list of translation results."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm(executor.map(self.translate_item, dataset), total=len(dataset)))
        return results


@click.command()
@click.option('--task', envvar='TASK_CONFIG', help='Task config path or name under benchmark_tasks/.')
@click.option('--base-url', '-u', required=True, help='Base URL for the API endpoint')
@click.option('--test-model', '-t', required=True, help='Model name to use for translation')
@click.option('--low-context', is_flag=True, help='Use low context prompts')
@click.option('--ultra-low-context', is_flag=True, help='Use ultra low context prompts (4096 tokens)')
@click.option('--max-workers', default=5, help='Number of worker threads for translation.')
@click.option('--concurrency-limit', default=5, help='Max number of concurrent API requests.')
@click.option('--api-key-env', default='OPENAI_API_KEY', help='Env var name that holds the API key')
@click.option('--max-tokens', default=8192, show_default=True, help='Maximum completion tokens per translation.')
def main(task, base_url, test_model, low_context, ultra_low_context, max_workers, concurrency_limit, api_key_env, max_tokens):
    """Translate text using the specified model.

    Loads the translation task set from the configured Hugging Face dataset,
    translates the source texts, and saves the results to a JSONL file.
    """
    os.makedirs("translations", exist_ok=True)
    task_config = load_task_config(task)
    hf_token = None
    if task_config.dataset.hf_token_env:
        hf_token = os.getenv(task_config.dataset.hf_token_env)
        if not hf_token:
            raise SystemExit(
                f"Error: Missing Hugging Face token in env var {task_config.dataset.hf_token_env} "
                f"for private dataset {task_config.dataset.repo}"
            )

    dataset = load_dataset(
        task_config.dataset.repo,
        task_config.dataset.config,
        split=task_config.dataset.split,
        revision=task_config.dataset.revision,
        token=hf_token,
    )
    normalized_dataset = [
        task_config.normalize_record(dict(item), require_source_text=True)
        for item in dataset
        if task_config.supports_record(dict(item))
    ]
    try:
        dataset_ref = resolve_dataset_ref(task_config, token=hf_token)
    except Exception as exc:
        raise SystemExit(
            f"Error: Could not resolve an immutable dataset revision for "
            f"{task_config.dataset.repo}@{task_config.dataset.revision}: {exc}"
        ) from exc

    api_key = os.getenv(api_key_env)
    if not api_key:
        raise SystemExit(f"Error: Missing API key. Set {api_key_env} in your environment.")
    translator = Translator(
        model_name=test_model,
        base_url=base_url,
        api_key=api_key,
        task_config=task_config,
        dataset_ref=dataset_ref,
        low_context=low_context,
        ultra_low_context=ultra_low_context,
        concurrency_limit=concurrency_limit,
        max_tokens=max_tokens,
    )

    results = translator(normalized_dataset, max_workers=max_workers)
    
    safe_model_name = test_model.replace("/", "__")
    output_path = os.path.join("translations", f"{safe_model_name}.jsonl")
    
    with open(output_path, "w", encoding="utf-8") as f:
        for item in results:
            try:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
            except json.JSONDecodeError as e:
                print(f"Error encoding JSON for item {item.get('name', 'unknown')}: {e}")
                continue

    # Print summary
    print(f"\nProcessing Summary:")
    print(f"Total items: {len(results)}")
    print(f"Successful: {len(results) - len(translator.failed_items)}")
    print(f"Failed: {len(translator.failed_items)}")
    
    if translator.failed_items:
        print(f"\nFailed items:")
        for failed in translator.failed_items:
            print(f"  - {failed['name']}: {failed['error']}")

    print(f"\nToken Usage Summary:")
    print(f"Input tokens: {translator.total_input_tokens:,}")
    print(f"Output tokens: {translator.total_output_tokens:,}")
    print(f"Total tokens: {translator.total_input_tokens + translator.total_output_tokens:,}")


if __name__ == "__main__":
    main()
