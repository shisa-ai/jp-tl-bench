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
-   `BASESET_SNAPSHOT_DIR`: Snapshot directory containing frozen anchor translations and base-set artifacts used for comparisons. Defaults to `baseset/v1.0`.

## Workflow

1.  **Translate**: The script prompts the target model to translate a predefined set of ~70 English and Japanese text samples.
2.  **Generate Pairs**: The new translations are paired up with existing translations from the base models in the frozen snapshot (by default `baseset/v1.0`) to create comparison pairs. `generate_shootout_data.py` reads the anchor translations directly from `BASESET_SNAPSHOT_DIR/translations`, so every run is evaluated against the same anchor set.
3.  **Judge**: The judge model evaluates each pair and picks a winner. The analysis for each comparison is saved to the `analysis/` directory.
4.  **Rank**: The script analyzes the win/loss data using a Bradley-Terry model to calculate scores and generate rankings. 

## Output Files

-   **Raw Analysis**: Individual JSONL files containing the judge's reasoning for each comparison are saved in `analysis/`.
-   **Scores**: The final rankings and scores are saved to `scores/<model_name>_tl_bench_scores.jsonl`.
-   **Answers**: A copy of the raw analysis file is also saved to `scores/<model_name>_tl_bench_answers.jsonl` for archival purposes.

## Utilities

### Generate Translation Data with vLLM

The `utils/generate_translation_data_with_vllm.sh` script provides a convenient way to generate translation data using a local vLLM server. It automatically starts a vLLM server with the specified model, waits for it to be ready, runs the translation generation, and then cleans up by shutting down the server.

**Usage:**

```bash
./utils/generate_translation_data_with_vllm.sh <model_name>
```

**Example:**

```bash
./utils/generate_translation_data_with_vllm.sh shisa-ai/035-rakuten-2.0-mini-1.5b-v2new-dpo405b
```

Make sure the script is executable:

```bash
chmod +x utils/generate_translation_data_with_vllm.sh
```

This script is useful when you want to generate translation data for a single model without running the full benchmark pipeline.

### Generate Translation Data Directly

You can also run the translation generation script directly using `generate_translation_data.py`. This gives you more control over the process and allows you to use different API endpoints and models.

**Usage:**

```bash
python generate_translation_data.py --base-url <api_url> --test-model <model_name> [OPTIONS]
```

**Required Options:**
- `--base-url`: The API endpoint URL for your model
- `--test-model`: The name of the model to use for translation

**Optional Parameters:**
- `--api-key-env`: Name of the environment variable containing the API key (defaults to `OPENAI_API_KEY`)
- `--low-context`: Use prompts optimized for smaller context windows
- `--ultra-low-context`: Use prompts optimized for very small context windows (4096 tokens)
- `--max-workers`: Number of worker threads for translation (default: 5)
- `--concurrency-limit`: Maximum number of concurrent API requests (default: 5)

**Example:**

```bash
python generate_translation_data.py --base-url https://generativelanguage.googleapis.com/v1beta/openai/ --test-model gemini-2.5-pro --api-key-env GEMINI_API_KEY
```

This will generate translation data using Google's Gemini model and save the results to `translations/<model_name>.jsonl`.

### Viewing Results with the TUI

After running benchmarks, you can interactively browse results using the Text User Interface (TUI) viewer. The viewer uses lazy loading for fast performance even with large datasets and supports two viewing modes.

**Usage:**

```bash
python view_tl_bench_tui.py [OPTIONS]
```

**Options:**
- `--scores-dir`: Path to scores directory (default: `./scores`)
- `--baseset-dir`: Path to baseset directory (default: `./baseset`)
- `--translations-dir`: Path to translations directory (default: `./translations`)

**Example:**

```bash
python view_tl_bench_tui.py
```

**Viewing Modes:**
1. **Comparisons Mode** - View pairwise comparisons (A vs B) with judge analysis
   - Source text and both translations side-by-side
   - Winner highlighted with judge reasoning
   - Filter by wins/losses
2. **Translations Mode** - View individual model translation outputs
   - Source text → Translation for each prompt
   - Generation settings (temperature, etc.)
   - Browse all 70 translation prompts per model

**Features:**
- **View Mode Selector**: Switch between Comparisons and Translations views
- **Category Selector**: Switch between Test Models, Base Set v1.0, and Base Set v0.9
- **Fast Lazy Loading**: Loads metadata instantly (~0.3s), full details on-demand
- **Model Browser**: Browse all models with LT/EN scores and win rates
- **Navigation**: Step through items one at a time with n/p keys

**Keyboard Shortcuts:**
  - `q`: Quit
  - `v`: Toggle between Comparisons/Translations view
  - `n`/`p`: Next/Previous item
  - `w`: Toggle wins-only filter (Comparisons mode)
  - `l`: Toggle losses-only filter (Comparisons mode)
  - Arrow keys/Tab: Navigate between sections

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
