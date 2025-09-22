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

load_dotenv()

@dataclass
class ModelConfig:
    """Configuration for model-specific parameters."""
    temperature: Optional[float] = 0.1
    top_p: Optional[float] = 0.85
    frequency_penalty: Optional[float] = None
    reasoning_effort: Optional[str] = None

class Translator:
    """Translates text using a specified model."""

    def __init__(self, model_name: str, base_url: str, api_key: str, low_context: bool = False, ultra_low_context: bool = False, concurrency_limit: int = 5):
        self.model_name = model_name
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

    def get_model_config(self) -> ModelConfig:
        """Get model-specific configuration based on model name."""
        model_lower = self.model_name.lower()
        
        # Claude Opus 4.1 - no top_p
        if "claude-opus-4-1" in model_lower:
            return ModelConfig(
                temperature=0.2,
                top_p=None
            )
        
        # GPT-5 mini/nano - no temperature/top_p 
        if "gpt-5-mini" in model_lower or "gpt-5-nano" in model_lower:
            return ModelConfig(
                temperature=None,
                top_p=None
            )
        
        # Gemini 2.5 Pro - reasoning effort required
        if "gemini-2.5-pro" in model_lower:
            return ModelConfig(
                reasoning_effort="low"
            )
        
        # GPT-5 (not chat-latest) - minimal reasoning
        if "gpt-5" in model_lower and "gpt-5-chat-latest" not in model_lower:
            return ModelConfig(
                reasoning_effort="minimal"
            )
        
        # Legacy Gemini 2.5 support (keeping existing behavior)
        if "gemini-2.5" in model_lower:
            return ModelConfig(
                reasoning_effort="low"
            )
        
        # Default configuration
        return ModelConfig()

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

    def parse(self, input_data: dict, response: str, prompt_text: str, generation_config: dict) -> dict:
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
            "prompt": prompt_text,
            "temperature": generation_config.get("temperature"),
            "top_p": generation_config.get("top_p"),
            "frequency_penalty": generation_config.get("frequency_penalty"),
            "reasoning_effort": generation_config.get("reasoning_effort"),
            "low_context": self.low_context,
            "ultra_low_context": self.ultra_low_context,
            "generation_config": generation_config
        }

    def translate_item(self, item: dict) -> dict:
        """Translates a single item with retry logic and concurrency control."""
        # Add jitter to spread out requests
        time.sleep(random.uniform(0.1, 0.5))

        with self.semaphore:
            prompt_text = self.get_prompt(item)
            
            max_retries = 5
            base_delay = 1
            
            for attempt in range(max_retries + 1):
                try:
                    # Get model-specific configuration
                    model_config = self.get_model_config()
                    
                    # Build parameters based on model config
                    params = {
                        "messages": [{"role": "user", "content": prompt_text}],
                        "model": self.model_name,
                    }
                    
                    # Add parameters only if they are not None
                    if model_config.temperature is not None:
                        params["temperature"] = model_config.temperature
                    if model_config.top_p is not None:
                        params["top_p"] = model_config.top_p
                    if model_config.frequency_penalty is not None:
                        params["frequency_penalty"] = model_config.frequency_penalty
                    if model_config.reasoning_effort is not None:
                        params["reasoning_effort"] = model_config.reasoning_effort

                    # Create generation config for saving
                    generation_config = params.copy()
                    generation_config.pop("messages", None)  # Remove messages from config

                    chat_completion = self.client.chat.completions.create(**params)
                    if not chat_completion.choices or chat_completion.choices[0].message.content is None:
                        raise ValueError("Empty response from API")

                    response = chat_completion.choices[0].message.content

                    # Track token usage
                    if hasattr(chat_completion, 'usage') and chat_completion.usage:
                        self.total_input_tokens += chat_completion.usage.prompt_tokens
                        self.total_output_tokens += chat_completion.usage.completion_tokens

                    parsed_result = self.parse(item, response, prompt_text, generation_config)
                    return parsed_result
                    
                except Exception as e:
                    error_msg = f"API error: {type(e).__name__}: {str(e)}"
                    if attempt == max_retries:
                        # Track failed item
                        failed_item = {
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
        """Process the dataset in parallel and return a list of translation results."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm(executor.map(self.translate_item, dataset), total=len(dataset)))
        return results


@click.command()
@click.option('--base-url', '-u', required=True, help='Base URL for the API endpoint')
@click.option('--test-model', '-t', required=True, help='Model name to use for translation')
@click.option('--low-context', is_flag=True, help='Use low context prompts')
@click.option('--ultra-low-context', is_flag=True, help='Use ultra low context prompts (4096 tokens)')
@click.option('--max-workers', default=5, help='Number of worker threads for translation.')
@click.option('--concurrency-limit', default=5, help='Max number of concurrent API requests.')
@click.option('--api-key-env', default='OPENAI_API_KEY', help='Env var name that holds the API key')
def main(base_url, test_model, low_context, ultra_low_context, max_workers, concurrency_limit, api_key_env):
    """Translate text using the specified model.

    Loads the translation test set from shisa-ai/bt_translation_test,
    translates the source texts, and saves the results to a JSONL file.
    """
    os.makedirs("translations", exist_ok=True)
    
    dataset = load_dataset("shisa-ai/bt_translation_test")["train"]

    api_key = os.getenv(api_key_env)
    translator = Translator(
        model_name=test_model,
        base_url=base_url,
        api_key=api_key,
        low_context=low_context,
        ultra_low_context=ultra_low_context,
        concurrency_limit=concurrency_limit
    )

    results = translator(dataset, max_workers=max_workers)
    
    safe_model_name = test_model.replace("/", "__")
    output_path = os.path.join("translations", f"{safe_model_name}.jsonl")
    
    # Filter out None results from failed items
    successful_results = [item for item in results if item is not None]
    
    with open(output_path, "w", encoding="utf-8") as f:
        for item in successful_results:
            try:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
            except json.JSONDecodeError as e:
                print(f"Error encoding JSON for item {item.get('name', 'unknown')}: {e}")
                continue

    # Print summary
    print(f"\nProcessing Summary:")
    print(f"Total items: {len(results)}")
    print(f"Successful: {len(successful_results)}")
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