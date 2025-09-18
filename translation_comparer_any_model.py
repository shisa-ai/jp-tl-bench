import os
import json
import random
import time
from datasets import Dataset
from openai import OpenAI
import click
import concurrent.futures
import threading
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()


class TranslationComparer:
    """Compares two translations and analyzes their differences."""

    def __init__(self, model_name: str, base_url: str, api_key: str, concurrency_limit: int):
        self.model_name = model_name
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.failed_items = []
        self.semaphore = threading.BoundedSemaphore(concurrency_limit)

    def prompt(self, input_data: dict) -> str:
        """Generate a prompt for comparison using the template from prompts/compare_prompt.txt and the translation data."""
        with open("prompts/compare_prompt.txt", "r", encoding="utf-8") as f:
            prompt_template = f.read()
        return prompt_template.replace("{{formatted_data}}", input_data["formatted_data"])

    def parse(self, input_data: dict, response: str) -> dict:
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
        }

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
                    call_params = {
                        "messages": [{"role": "user", "content": prompt_text}],
                        "model": self.model_name,
                        "temperature": 0,
                    }
                    
                    if 'gemini-2.5' in self.model_name:
                        call_params['reasoning_effort'] = 'low'
                    
                    chat_completion = self.client.chat.completions.create(**call_params)
                    if not chat_completion.choices or chat_completion.choices[0].message.content is None:
                        raise ValueError("Empty response from API")
                    
                    response = chat_completion.choices[0].message.content
                    
                    # Track token usage
                    if hasattr(chat_completion, 'usage') and chat_completion.usage:
                        self.total_input_tokens += chat_completion.usage.prompt_tokens
                        self.total_output_tokens += chat_completion.usage.completion_tokens
                    
                    parsed_result = self.parse(item, response)
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
@click.option('--base-url', '-u', required=True, help='Base URL for the API endpoint')
@click.option('--judge-model', '-j', required=True, help='Model name to use for judging the translations')
@click.option('--test-model', '-t', help='Model name to test against base models')
@click.option('--generate-base-set', is_flag=True, help='Generate base set comparisons instead of testing a specific model')
@click.option('--max-workers', default=40, help='Number of worker threads for comparison.')
@click.option('--concurrency-limit', default=40, help='Max number of concurrent API requests.')
@click.option('--api-key-env', default='OPENAI_API_KEY', help='Env var name that holds the API key')
def main(base_url, judge_model, test_model, generate_base_set, max_workers, concurrency_limit, api_key_env):
    """Compare translations between different models using a third LLM as analyzer.

    Reads the translation pairs from the JSONL file, creates a dataset,
    and uses an LLM to analyze the differences. Saves the analysis results
    to a new JSONL file.
    """

    if not test_model and not generate_base_set:
        raise click.UsageError(
            "Either --test-model-name or --generate-base-set must be specified"
        )
    if test_model and generate_base_set:
        raise click.UsageError(
            "Cannot specify both --test-model-name and --generate-base-set"
        )

    # Validate that prompt file exists
    if not os.path.exists("prompts/compare_prompt.txt"):
        raise SystemExit("Error: Missing prompts/compare_prompt.txt. See README for setup.")

    # Create output directory if it doesn't exist
    os.makedirs("scores", exist_ok=True)

    # Read translation pairs
    input_file = (
        "base_conversation_pairs.jsonl"
        if generate_base_set
        else "latest_conversation_pairs.jsonl"
    )
    if not os.path.exists(input_file):
        raise SystemExit(f"Error: Input file not found: {input_file}. Please run generate_shootout_data.py first.")

    translation_pairs = []
    with open(input_file, "r", encoding="utf-8") as f:
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

    # Create dataset and shuffle it
    translations = Dataset.from_list(translation_pairs)
    translations = translations.shuffle(seed=42)  # Set seed for reproducibility

    api_key = os.getenv(api_key_env)

    comparer = TranslationComparer(
        model_name=judge_model, base_url=base_url, api_key=api_key, concurrency_limit=concurrency_limit
    )

    results = comparer(translations, max_workers=max_workers)

    # Save analysis results
    if generate_base_set:
        safe_judge_model = judge_model.replace("/", "__")
        output_path = os.path.join(
            "base_sets", f"base_set.{safe_judge_model}.jsonl"
        )
    else:
        safe_test_model = test_model.replace("/", "__")
        safe_judge_model = judge_model.replace("/", "__")
        output_path = os.path.join(
            "scores", f"{safe_test_model}.{safe_judge_model}.jsonl"
        )

    # Filter out None results from failed items
    successful_results = [item for item in results if item is not None]
    
    with open(output_path, "w", encoding="utf-8") as f:
        for item in successful_results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Print summary
    print(f"\nProcessing Summary:")
    print(f"Total items: {len(results)}")
    print(f"Successful: {len(successful_results)}")
    print(f"Failed: {len(comparer.failed_items)}")
    
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
