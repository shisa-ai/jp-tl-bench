# Shisa Translation Bench

This project provides a suite of tools to benchmark the translation capabilities of Large Language Models (LLMs) between English and Japanese. It uses a judge model to compare and rank the translation quality of a target model against a set of base models.

## Installation

To set up the environment, first create and activate a conda/mamba environment:

```bash
mamba create -n shisa-jp-tl-bench python=3.10
mamba activate shisa-jp-tl-bench
```

Then, install the required packages:

```bash
pip install -r requirements.txt
```

## Configuration

API keys are managed using a `.env` file. Create a `.env` file in the root of the project directory.

**Example `.env` file:**

```
OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxx"
GEMINI_API_KEY="xxxxxxxxxxxxxxxxxxxxxxxxxxxx"
MY_CUSTOM_API_KEY="xxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

The benchmark script uses environment variables to determine which API key to use for the model being tested and for the judge model.

## How to Run the Benchmark

The `run_translation_bench.sh` script is the main entry point for running the benchmark. It requires `MODEL` and `OPENAI_URL` to be set.

### Basic Example
This example runs the benchmark for a local model, using the default judge model (`gemini-2.5-flash`).

```bash
MODEL="my-local-model/my-model-7b-instruct" \
OPENAI_URL="http://localhost:8000/v1" \
./run_translation_bench.sh
```

### Specifying a Custom Judge Model

This example uses a different model and API endpoint for the judge.

```bash
MODEL="my-local-model/my-model-7b-instruct" \
OPENAI_URL="http://localhost:8000/v1" \
JUDGE_MODEL="Nexusflow/Athene-V2-Chat" \
JUDGE_URL="http://athene-v2/v1" \
./run_translation_bench.sh
```

### Using Custom API Keys

If your API keys are stored in environment variables with different names, you can specify them using `MODEL_API_KEY_ENV` and `JUDGE_API_KEY_ENV`.

```bash
MODEL="my-local-model/my-model-7b-instruct" \
OPENAI_URL="http://localhost:8000/v1" \
MODEL_API_KEY_ENV="MY_CUSTOM_API_KEY" \

JUDGE_MODEL="google/gemini-pro" \
JUDGE_URL="https://generativelanguage.googleapis.com/v1beta/openai/" \
JUDGE_API_KEY_ENV="GEMINI_API_KEY" \
./run_translation_bench.sh
```

### Environment Variables Breakdown

-   `MODEL`: (Required) The name of the model you are testing.
-   `OPENAI_URL`: (Required) The API base URL for your test model.
-   `JUDGE_MODEL`: The name of the judge model. Defaults to `gemini-2.5-flash`.
-   `JUDGE_URL`: The API base URL for the judge model. Defaults to the Google Generative Language API.
-   `MODEL_API_KEY_ENV`: The name of the environment variable holding the API key for your test model. Defaults to `OPENAI_API_KEY`.
-   `JUDGE_API_KEY_ENV`: The name of the environment variable holding the API key for the judge model. Defaults to `GEMINI_API_KEY`.
-   `LOW_CONTEXT` / `ULTRA_LOW_CONTEXT`: Set to `true` to use prompts with a smaller context window. Disabled by default.

## Workflow

1.  **Translate**: The script prompts the target model to translate a predefined set of ~70 English and Japanese text samples.
2.  **Generate Pairs**: The new translations are paired up with existing translations from the base models (`base_translations/`) to create comparison pairs.
3.  **Judge**: The judge model evaluates each pair and picks a winner. The analysis for each comparison is saved to the `analysis/` directory.
4.  **Rank**: The script analyzes the win/loss data using a Bradley-Terry model to calculate scores and generate rankings. 

## Output Files

-   **Raw Analysis**: Individual JSONL files containing the judge's reasoning for each comparison are saved in `analysis/`.
-   **Scores**: The final rankings and scores are saved to `scores/<model_name>_tl_bench_scores.jsonl`.
-   **Answers**: A copy of the raw analysis file is also saved to `scores/<model_name>_tl_bench_answers.jsonl` for archival purposes.

## Utilities

### Cleaning Analysis Files

Over time, analysis files may accumulate invalid entries (e.g., malformed JSON, incorrect answer formats). You can clean these files using the `utils/clean_analysis_file.py` script. This script will read an analysis file, remove any invalid lines, and overwrite the original file with the cleaned version.

**Usage:**

```bash
python utils/clean_analysis_file.py path/to/your/analysis_file.jsonl
```

**Example:**

```bash
python utils/clean_analysis_file.py analysis/base_set.gemini-2.5-flash.jsonl
```
