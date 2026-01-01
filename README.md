# JP-TL-Bench

**JP-TL-Bench** is an anchored, pairwise LLM-judged benchmark for Japanese ↔ English translation quality.

We have created a set of Easy and Hard translations for each direction (JA->EN and EN->JA).

Scoring is done by an LLM-as-a-Judge against a carefully designed fixed (versioned) "Base Set" of reference model translations designed to allow reliable scoring of models based on the win/loss outcomes.

- Blog: [https://shisa.ai/posts/jp-tl-bench/](https://shisa.ai/posts/jp-tl-bench/)
- Paper: [JP-TL-Bench: Anchored Pairwise LLM Evaluation for Bidirectional Japanese-English Translation](docs/paper.pdf)

## Eval Summary

- **Task:** ~70 prompts spanning English→Japanese and Japanese→English.
- **Evaluation:** Pairwise A/B comparisons (test model vs anchors) judged by a configurable judge model.
- **Scoring:** Win rate (WR%) plus Bradley–Terry-derived scores (EN, LT) for rankings.
- **Reproducibility:** Comparisons are anchored to a versioned base set snapshot (`BASESET_SNAPSHOT_DIR`, default `baseset/v1.0`).

## Base Set v1.0

The v1.0 base set is a frozen snapshot of 20 anchor models used to keep scores comparable across runs. We selected anchors to cover a roughly even spread of win rates (from very strong to very weak baselines), which helps stabilize relative scoring for new models.

The WR% and LT below are computed within the v1.0 anchor round-robin using the default judge (`gemini-2.5-flash`). Values will change if you re-judge the base set with a different judge.

Terminology: WR% is the overall win rate in the anchor round-robin; LT/EN are 0–10 rescalings of the Bradley–Terry fit used by the tooling (`choix_analyzer.py`).

| # | Model | Source | WR% | LT |
| --- | --- | --- | --- | --- |
| 1 | gemini-2.5-pro | `base_translations/gemini-2.5-pro.jsonl` | 96.15 | 9.94 |
| 2 | gemini-2.5-flash | `base_translations/gemini-2.5-flash.jsonl` | 92.92 | 9.89 |
| 3 | Qwen/Qwen3-30B-A3B-Instruct-2507 | `translations/Qwen__Qwen3-30B-A3B-Instruct-2507.jsonl` | 84.33 | 9.63 |
| 4 | shisa-ai/shisa-v2-llama3.1-405b | `translations/shisa-ai__shisa-v2-llama3.1-405b.jsonl` | 81.45 | 9.49 |
| 5 | openai/gpt-4o | `base_translations/openai__gpt-4o.jsonl` | 76.02 | 9.12 |
| 6 | shisa-ai/shisa-v2-unphi4-14b | `translations/shisa-ai__shisa-v2-unphi4-14b.jsonl` | 72.81 | 8.83 |
| 7 | tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5 | `base_translations/tokyotech-llm__Llama-3.1-Swallow-8B-Instruct-v0.5.jsonl` | 62.16 | 7.45 |
| 8 | nvidia/NVIDIA-Nemotron-Nano-12B-v2 | `translations/nvidia__NVIDIA-Nemotron-Nano-12B-v2.jsonl` | 59.91 | 7.07 |
| 9 | meta-llama/Llama-3.3-70B-Instruct | `base_translations/meta-llama__Llama-3.3-70B-Instruct.jsonl` | 58.05 | 6.74 |
| 10 | microsoft/phi-4 | `base_translations/microsoft__phi-4.jsonl` | 49.80 | 5.13 |
| 11 | cyberagent/Mistral-Nemo-Japanese-Instruct-2408 | `base_translations/cyberagent__Mistral-Nemo-Japanese-Instruct-2408.jsonl` | 47.52 | 4.69 |
| 12 | Qwen/Qwen3-4B | `base_translations/Qwen__Qwen3-4B.jsonl` | 44.68 | 4.11 |
| 13 | LiquidAI/LFM2-2.6B | `translations/LiquidAI__LFM2-2.6B.jsonl` | 43.83 | 3.92 |
| 14 | meta-llama/Llama-3.1-8B-Instruct | `translations/meta-llama__Llama-3.1-8B-Instruct.jsonl` | 38.84 | 2.95 |
| 15 | microsoft/Phi-4-mini-instruct | `translations/microsoft__Phi-4-mini-instruct.jsonl` | 24.94 | 0.98 |
| 16 | augmxnt/shisa-7b-v1 | `base_translations/augmxnt__shisa-7b-v1.jsonl` | 21.36 | 0.68 |
| 17 | meta-llama/Llama-3.2-3B-Instruct | `translations/meta-llama__Llama-3.2-3B-Instruct.jsonl` | 19.18 | 0.54 |
| 18 | Rakuten/RakutenAI-2.0-mini-instruct | `translations/Rakuten__RakutenAI-2.0-mini-instruct.jsonl` | 14.20 | 0.29 |
| 19 | LiquidAI/LFM2-350M | `translations/LiquidAI__LFM2-350M.jsonl` | 8.75 | 0.13 |
| 20 | SakanaAI/TinySwallow-1.5B | `translations/SakanaAI__TinySwallow-1.5B.jsonl` | 2.52 | 0.03 |

To switch base sets, set `BASESET_SNAPSHOT_DIR` to another snapshot directory (for example `baseset/v0.9`). For details on how snapshots are built (and how to create new versions like `v2.0`), see [baseset/README.md](baseset/README.md).

## Installation

To set up the environment, first create and activate a conda/mamba environment:

```bash
mamba create -n jp-tl-bench python=3.12
mamba activate jp-tl-bench
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
MODEL="shisa-ai/shisa-v2.1-qwen3-8b" \
OPENAI_URL="http://localhost:8000/v1" \
./run_translation_bench.sh
```

### Specifying a Custom Judge Model

This example uses a different model and API endpoint for the judge.

```bash
MODEL=""shisa-ai/shisa-v2.1-qwen3-8b" \
OPENAI_URL="http://localhost:8000/v1" \
JUDGE_MODEL=""shisa-ai/shisa-v2-llama3.1-405b" \
JUDGE_URL="http://shisa-v2-405b/v1" \
./run_translation_bench.sh
```
Runtime depends on both your generation and judging concurrency/speeds, but we find most runs take between 15-30 minutes and the average cost for a run using `gemini-2.5-flash` as a judge is around $7 (USD). 

### Using Custom API Keys

If your API keys are stored in environment variables with different names, you can specify them using `MODEL_API_KEY_ENV` and `JUDGE_API_KEY_ENV`.

```bash
MODEL=""shisa-ai/shisa-v2.1-qwen3-8b" \
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

## Citation

If you use JP-TL-Bench in your research, you can cite this repo:

```bibtex
@misc{jp-tl-bench,
  title={JP-TL-Bench: Anchored Pairwise LLM Evaluation for Bidirectional Japanese-English Translation},
  author={Shisa AI},
  year={2025},
  howpublished={\url{https://github.com/shisa-ai/jp-tl-bench}}
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
