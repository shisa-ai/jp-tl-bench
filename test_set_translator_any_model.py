import os
import json
from datasets import Dataset, load_dataset
from bespokelabs import curator
import click
import re


class Translator(curator.LLM):
    """Translates text using a specified model."""

    def __init__(self, model_name: str, backend: str, backend_params: dict, low_context: bool = False):
        super().__init__(model_name=model_name, backend=backend, backend_params=backend_params)
        self.low_context = low_context

    def get_prompt_path(self, english_in_input: bool) -> str:
        """Determine which prompt file to use based on input language and context setting."""
        base_name = "translate_prompt_from_english" if english_in_input else "translate_prompt_from_japanese"
        if self.low_context:
            base_name += "_low_context"
        return f"prompts/{base_name}.txt"

    def prompt(self, input: dict) -> str:
        """Generate a prompt for translation using the appropriate template based on input language."""
        english_in_input = input.get("english_in_input", True)  # Default to English if not specified
        prompt_path = self.get_prompt_path(english_in_input)
        
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_template = f.read()
        return prompt_template.replace("{{text}}", input["text"])

    def parse(self, input: dict, response: str) -> dict:
        """Parse the model response along with the input data into the desired output format."""
        # Find all matches of text between translation tags
        matches = re.findall(r'<translation>(.*?)</translation>', response, re.DOTALL)
        # Use the last match if found, otherwise use "none" and print error
        if not matches:
            print(f"Error: No translation tags found in response for input: {input['name']}")
            translation = "none"
        else:
            translation = matches[-1]
        
        # Get the prompt that was used
        prompt_text = self.prompt(input)
        
        return {
            "name": input["name"],
            "source_text": input["text"],
            "difficulty" : input["difficulty"],
            "english" : input["english"],
            "full_response": response,
            "translation": translation,
            "prompt": prompt_text  # Add the prompt to the output
        }


@click.command()
@click.option('--base-url', '-u', required=True, help='Base URL for the API endpoint')
@click.option('--test-model-name', '-t', required=True, help='Model name to use for translation')
@click.option('--low-context', is_flag=True, help='Use low context prompts')
def main(base_url, test_model_name, low_context):
    """Translate text using the specified model.

    Loads the translation test set from shisa-ai/bt_translation_test,
    translates the source texts, and saves the results to a JSONL file.
    """
    # Create output directory if it doesn't exist
    os.makedirs("translations", exist_ok=True)
    
    # Load the dataset
    dataset = load_dataset("shisa-ai/bt_translation_test")["train"]


    backend = "litellm"
    backend_params = {"base_url": base_url,
                    "max_requests_per_minute": 128,
                    "max_tokens_per_minute": 10000000}

    translator = Translator(
        model_name="hosted_vllm/" + test_model_name,
        backend=backend,
        backend_params=backend_params,
        low_context=low_context
    )

    results = translator(dataset)
    
    # Save translation results
    safe_model_name = test_model_name.replace("/", "__")
    output_path = os.path.join("translations", f"{safe_model_name}.jsonl")
    
    with open(output_path, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
