import os
import json
from datasets import Dataset
from openai import OpenAI
import click
import concurrent.futures
from tqdm import tqdm


class TranslationComparer:
    """Compares two translations and analyzes their differences."""

    def __init__(self, model_name: str, base_url: str, api_key: str):
        self.model_name = model_name
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )

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
        """Compares a single item."""
        prompt_text = self.prompt(item)
        chat_completion = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt_text}],
            model=self.model_name,
        )
        response = chat_completion.choices[0].message.content
        parsed_result = self.parse(item, response)
        return parsed_result

    def __call__(self, dataset: Dataset, max_workers: int) -> list:
        """Process the dataset in parallel and return a list of comparison results."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(
                tqdm(executor.map(self.compare_item, dataset), total=len(dataset))
            )
        return results


@click.command()
@click.option('--base-url', '-u', required=True, help='Base URL for the API endpoint')
@click.option('--judge-model-name', '-j', required=True, help='Model name to use for judging the translations')
@click.option('--test-model-name', '-t', help='Model name to test against base models')
@click.option('--generate-base-set', is_flag=True, help='Generate base set comparisons instead of testing a specific model')
@click.option('--max-workers', default=10, help='Number of worker threads for comparison.')
def main(base_url, judge_model_name, test_model_name, generate_base_set, max_workers):
    """Compare translations between different models using a third LLM as analyzer.

    Reads the translation pairs from the JSONL file, creates a dataset,
    and uses an LLM to analyze the differences. Saves the analysis results
    to a new JSONL file.
    """

    if not test_model_name and not generate_base_set:
        raise click.UsageError(
            "Either --test-model-name or --generate-base-set must be specified"
        )
    if test_model_name and generate_base_set:
        raise click.UsageError(
            "Cannot specify both --test-model-name and --generate-base-set"
        )

    # Create output directory if it doesn't exist
    os.makedirs("analysis", exist_ok=True)

    # Read translation pairs
    input_file = (
        "base_conversation_pairs.jsonl"
        if generate_base_set
        else "latest_conversation_pairs.jsonl"
    )
    translation_pairs = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            translation_pairs.append(json.loads(line))

    # Create dataset and shuffle it
    translations = Dataset.from_list(translation_pairs)
    translations = translations.shuffle(seed=42)  # Set seed for reproducibility

    api_key = None
    if "generativelanguage.googleapis.com" in base_url:
        api_key = os.environ.get("GEMINI_API_KEY")
    elif "openai.com" in base_url:
        api_key = os.environ.get("OPENAI_API_KEY")

    comparer = TranslationComparer(
        model_name=judge_model_name, base_url=base_url, api_key=api_key
    )

    results = comparer(translations, max_workers=max_workers)

    # Save analysis results
    if generate_base_set:
        safe_judge_model_name = judge_model_name.replace("/", "__")
        output_path = os.path.join(
            "analysis", f"base_set.{safe_judge_model_name}.jsonl"
        )
    else:
        safe_test_model_name = test_model_name.replace("/", "__")
        safe_judge_model_name = judge_model_name.replace("/", "__")
        output_path = os.path.join(
            "analysis", f"{safe_test_model_name}.{safe_judge_model_name}.jsonl"
        )

    with open(output_path, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
