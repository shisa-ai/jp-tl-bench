import os
import json
from datasets import Dataset, load_dataset
from bespokelabs import curator
import click
import re


class Translator(curator.LLM):
    """Translates text using a specified model."""

    def __init__(self, model_name: str, backend: str, backend_params: dict, low_context: bool = False, ultra_low_context: bool = False):
        super().__init__(model_name=model_name, backend=backend, backend_params=backend_params)
        self.low_context = low_context
        self.ultra_low_context = ultra_low_context

    def get_prompt_path(self, english: bool) -> str:
        """Determine which prompt file to use based on input language and context setting."""
        base_name = "translate_prompt_from_english" if english else "translate_prompt_from_japanese"
        if self.ultra_low_context:
            base_name += "_ultra_low_context"
        elif self.low_context:
            base_name += "_low_context"
        return f"prompts/{base_name}.txt"

    def prompt(self, input: dict) -> str:
        """Generate a prompt for translation using the appropriate template based on input language."""
        english = input.get("english", True)  # Default to English if not specified
        prompt_path = self.get_prompt_path(english)
        
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_template = f.read()
        return prompt_template.replace("{{text}}", input["text"])

    def parse(self, input: dict, response: str) -> dict:
        """Parse the model response along with the input data into the desired output format."""
        # Find all matches of text between translation tags
        matches = re.findall(r'<translation>(.*?)</translation>', response, re.DOTALL)
        # Use the last match if found, otherwise check for analysis tag
        if not matches:
            print(f"Error: No translation tags found in response for input: {input['name']}")
            analysis_split = response.split('</translation_analysis>')
            if len(analysis_split) > 1:
                print(f"Found </translation_analysis> tag, using text after it for input: {input['name']}")
                filtered_response = analysis_split[1]
                # Try to find translation tags in the filtered response
                filtered_matches = re.findall(r'<translation>(.*?)</translation>', filtered_response, re.DOTALL)
                if filtered_matches:
                    translation = filtered_matches[-1]
                else:
                    translation = filtered_response
            else:
                print(f"No </translation_analysis> tag found, using full response for input: {input['name']}")
                translation = response
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
@click.option('--ultra-low-context', is_flag=True, help='Use ultra low context prompts (4096 tokens)')
@click.option('--commercial-model', is_flag=True, help='Use commercial model rate limits')
def main(base_url, test_model_name, low_context, ultra_low_context, commercial_model):
    """Translate text using the specified model.

    Loads the translation test set from shisa-ai/bt_translation_test,
    translates the source texts, and saves the results to a JSONL file.
    """
    # Create output directory if it doesn't exist
    os.makedirs("translations", exist_ok=True)
    
    # Load the dataset
    dataset = load_dataset("shisa-ai/bt_translation_test")["train"]

    backend = "litellm"
    if commercial_model:
        backend_params = {
            "max_requests_per_minute": 12,
            "max_concurrent_requests": 12
        }
        translator = Translator(
            model_name=test_model_name,
            backend=backend,
            backend_params=backend_params,
            low_context=low_context,
            ultra_low_context=ultra_low_context
        )
    else:
        backend_params = {
            "base_url": base_url,
            "max_requests_per_minute": 128,
            "max_tokens_per_minute": 10000000,
            "request_timeout" : 120 #Sometimes models will go a little nuts on harder tasks, this timeout prevents that.
        }
        translator = Translator(
            model_name="hosted_vllm/" + test_model_name,
            backend=backend,
            backend_params=backend_params,
            low_context=low_context,
            ultra_low_context=ultra_low_context
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
