import os
import json
from datasets import Dataset
from bespokelabs import curator
import click


class TranslationComparer(curator.LLM):
    """Compares two translations and analyzes their differences."""

    def prompt(self, input: dict) -> str:
        """Generate a prompt for comparison using the template from prompts/compare_prompt.txt and the translation data."""
        with open("prompts/compare_prompt.txt", "r", encoding="utf-8") as f:
            prompt_template = f.read()
        return prompt_template.replace("{{formatted_data}}", input["formatted_data"])

    def parse(self, input: dict, response: str) -> dict:
        """Parse the model response along with the input data into the desired output format."""
        return {
            "name" : input["name"],
            "english": input["english"],
            "difficulty" : input["difficulty"],
            "id": input["id"],
            "llm_a": input["llm_a"],
            "llm_b": input["llm_b"],
            "formatted_data": input["formatted_data"],
            "analysis": response
        }


@click.command()
@click.option('--base-url', '-u', required=True, help='Base URL for the API endpoint')
@click.option('--judge-model-name', '-j', required=True, help='Model name to use for judging the translations')
@click.option('--test-model-name', '-t', help='Model name to test against base models')
@click.option('--generate-base-set', is_flag=True, help='Generate base set comparisons instead of testing a specific model')
def main(base_url, judge_model_name, test_model_name, generate_base_set):
    """Compare translations between different models using a third LLM as analyzer.

    Reads the translation pairs from the JSONL file, creates a dataset,
    and uses an LLM to analyze the differences. Saves the analysis results
    to a new JSONL file.
    """

    if not test_model_name and not generate_base_set:
        raise click.UsageError("Either --test-model-name or --generate-base-set must be specified")
    if test_model_name and generate_base_set:
        raise click.UsageError("Cannot specify both --test-model-name and --generate-base-set")

    # Create output directory if it doesn't exist
    os.makedirs("analysis", exist_ok=True)
    
    # Read translation pairs
    input_file = "base_conversation_pairs.jsonl" if generate_base_set else "latest_conversation_pairs.jsonl"
    translation_pairs = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            translation_pairs.append(json.loads(line))
    
    # Create dataset and shuffle it
    translations = Dataset.from_list(translation_pairs)
    translations = translations.shuffle(seed=42)  # Set seed for reproducibility

    backend = "litellm"
    backend_params = {"base_url": base_url,
                    "max_requests_per_minute": 256,
                    "max_tokens_per_minute": 50000000,
                    "max_concurrent_requests": 128,
                    "request_timeout": 120}

    comparer = TranslationComparer(
        model_name="hosted_vllm/"+ judge_model_name,
        backend=backend,  
        backend_params=backend_params,
    )

    results = comparer(translations)
    
    # Save analysis results
    if generate_base_set:
        safe_judge_model_name = judge_model_name.replace("/", "__")
        output_path = os.path.join("analysis", f"base_set.{safe_judge_model_name}.jsonl")
    else:
        safe_test_model_name = test_model_name.replace("/", "__")
        safe_judge_model_name = judge_model_name.replace("/", "__")
        output_path = os.path.join("analysis", f"{safe_test_model_name}.{safe_judge_model_name}.jsonl")
    
    with open(output_path, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
