import os
import json
from datasets import Dataset, load_dataset
from openai import OpenAI
import click
import re
from dotenv import load_dotenv
from tqdm import tqdm
import concurrent.futures

load_dotenv()

class Translator:
    """Translates text using a specified model."""

    def __init__(self, model_name: str, base_url: str, api_key: str, low_context: bool = False, ultra_low_context: bool = False):
        self.model_name = model_name
        self.low_context = low_context
        self.ultra_low_context = ultra_low_context
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )

    def get_prompt_path(self, english: bool) -> str:
        """Determine which prompt file to use based on input language and context setting."""
        base_name = "translate_prompt_from_english" if english else "translate_prompt_from_japanese"
        if self.ultra_low_context:
            base_name += "_ultra_low_context"
        elif self.low_context:
            base_name += "_low_context"
        return f"prompts/{base_name}.txt"

    def get_prompt(self, input_data: dict) -> str:
        """Generate a prompt for translation using the appropriate template based on input language."""
        english = input_data.get("english", True)
        prompt_path = self.get_prompt_path(english)

        if not os.path.exists(prompt_path):
            raise SystemExit(f"Error: Missing prompt file: {prompt_path}. See README for setup.")
        
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_template = f.read()
        return prompt_template.replace("{{text}}", input_data["text"])

    def parse(self, input_data: dict, response: str, prompt_text: str) -> dict:
        """Parse the model response along with the input data into the desired output format."""
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
            "name": input_data["name"],
            "source_text": input_data["text"],
            "difficulty" : input_data["difficulty"],
            "english" : input_data["english"],
            "full_response": response,
            "translation": translation,
            "prompt": prompt_text
        }

    def translate_item(self, item: dict) -> dict:
        """Translates a single item."""
        prompt_text = self.get_prompt(item)
        params = {
            "messages": [{"role": "user", "content": prompt_text}],
            "model": self.model_name,
            "temperature": 0.1,
            "top_p": 0.85,
            "frequency_penalty": 0.25
        }
        if "gemini-2.5" in self.model_name:
            params["reasoning_effort"] = "low"
        
        chat_completion = self.client.chat.completions.create(**params)
        response = chat_completion.choices[0].message.content
        parsed_result = self.parse(item, response, prompt_text)
        return parsed_result

    def __call__(self, dataset: Dataset, max_workers: int) -> list:
        """Process the dataset in parallel and return a list of translation results."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm(executor.map(self.translate_item, dataset), total=len(dataset)))
        return results


@click.command()
@click.option('--base-url', '-u', required=True, help='Base URL for the API endpoint')
@click.option('--test-model-name', '-t', required=True, help='Model name to use for translation')
@click.option('--low-context', is_flag=True, help='Use low context prompts')
@click.option('--ultra-low-context', is_flag=True, help='Use ultra low context prompts (4096 tokens)')
@click.option('--max-workers', default=5, help='Number of worker threads for translation.')
@click.option('--api-key-env', default='OPENAI_API_KEY', help='Env var name that holds the API key')
def main(base_url, test_model_name, low_context, ultra_low_context, max_workers, api_key_env):
    """Translate text using the specified model.

    Loads the translation test set from shisa-ai/bt_translation_test,
    translates the source texts, and saves the results to a JSONL file.
    """
    os.makedirs("translations", exist_ok=True)
    
    dataset = load_dataset("shisa-ai/bt_translation_test")["train"]

    api_key = os.getenv(api_key_env)
    translator = Translator(
        model_name=test_model_name,
        base_url=base_url,
        api_key=api_key,
        low_context=low_context,
        ultra_low_context=ultra_low_context
    )

    results = translator(dataset, max_workers=max_workers)
    
    safe_model_name = test_model_name.replace("/", "__")
    output_path = os.path.join("translations", f"{safe_model_name}.jsonl")
    
    with open(output_path, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()