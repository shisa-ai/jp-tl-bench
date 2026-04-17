# JP-TL-Bench

**JP-TL-Bench** is an anchored, pairwise LLM-judged benchmark workspace for versioned translation tasks.

The original public benchmark is Japanese ↔ English, and the current task-config migration also supports bidirectional Chinese ↔ English.

Scoring is done by an LLM-as-a-Judge against a carefully designed fixed (versioned) "Base Set" of reference model translations designed to allow reliable scoring of models based on the win/loss outcomes.

- Blog: [https://shisa.ai/posts/jp-tl-bench/](https://shisa.ai/posts/jp-tl-bench/)
- Paper: [JP-TL-Bench: Anchored Pairwise LLM Evaluation for Bidirectional Japanese-English Translation](docs/paper.pdf)

## Eval Summary

- **Task:** JP v1.0 uses ~70 prompts spanning `en->ja` and `ja->en`; task configs define the active source/target directions for each run.
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

To set up the environment, create the project env explicitly:

```bash
mamba create -n shisa-jp-tl-bench python=3.12
```

Then install the required packages into that env:

```bash
mamba run -n shisa-jp-tl-bench pip install -r requirements.txt
```

## Configuration

API keys are managed using a `.env` file. Create a `.env` file in the root of the project directory.

**Example `.env` file:**

```
OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxx"
GEMINI_API_KEY="xxxxxxxxxxxxxxxxxxxxxxxxxxxx"
MY_CUSTOM_API_KEY="xxxxxxxxxxxxxxxxxxxxxxxxxxxx"
HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

The benchmark script uses environment variables to determine which API key to use for the model being tested and for the judge model.
Task configs can also declare a private Hugging Face dataset token env var; the current task configs use `HF_TOKEN`.

## How to Run the Benchmark

The `run_translation_bench.sh` script is the main entry point for running the benchmark. It now requires an explicit `TASK_CONFIG` as well as `MODEL` and `OPENAI_URL`. Internally it runs each Python step via `mamba run -n shisa-jp-tl-bench ...` (or `conda run -n ...` if `mamba` is unavailable), so the workflow does not depend on an already-activated shell.

### Basic Example
This example runs the JP task for a local model, using the default judge profile and judge model (`gemini-2.5-flash`).

```bash
TASK_CONFIG="translation_ja_en_bidirectional_v1" \
MODEL="shisa-ai/shisa-v2.1-qwen3-8b" \
OPENAI_URL="http://localhost:8000/v1" \
./run_translation_bench.sh
```

### Specifying a Custom Judge Model

This example uses a different model and API endpoint for the judge.

```bash
TASK_CONFIG="translation_ja_en_bidirectional_v1" \
MODEL="shisa-ai/shisa-v2.1-qwen3-8b" \
OPENAI_URL="http://localhost:8000/v1" \
JUDGE_MODEL="shisa-ai/shisa-v2-llama3.1-405b" \
JUDGE_URL="http://shisa-v2-405b/v1" \
./run_translation_bench.sh
```
Runtime depends on both your generation and judging concurrency/speeds, but we find most runs take between 15-30 minutes and the average cost for a run using `gemini-2.5-flash` as a judge is around $7 (USD). 

### Chinese Task Example

The first Chinese release should use the `cn_judge` profile so the compare prompt is `compare-cn-v1`:

```bash
TASK_CONFIG="translation_zh_en_bidirectional_v1" \
JUDGE_PROFILE="cn_judge" \
MODEL="gemini-2.5-flash" \
OPENAI_URL="https://generativelanguage.googleapis.com/v1beta/openai/" \
MODEL_API_KEY_ENV="GEMINI_API_KEY" \
JUDGE_MODEL="gemini-2.5-flash" \
JUDGE_URL="https://generativelanguage.googleapis.com/v1beta/openai/" \
JUDGE_API_KEY_ENV="GEMINI_API_KEY" \
./run_translation_bench.sh
```

### Using Custom API Keys

If your API keys are stored in environment variables with different names, you can specify them using `MODEL_API_KEY_ENV` and `JUDGE_API_KEY_ENV`.

```bash
TASK_CONFIG="translation_ja_en_bidirectional_v1" \
MODEL="shisa-ai/shisa-v2.1-qwen3-8b" \
OPENAI_URL="http://localhost:8000/v1" \
MODEL_API_KEY_ENV="MY_CUSTOM_API_KEY" \
JUDGE_MODEL="google/gemini-pro" \
JUDGE_URL="https://generativelanguage.googleapis.com/v1beta/openai/" \
JUDGE_API_KEY_ENV="GEMINI_API_KEY" \
./run_translation_bench.sh
```

### Environment Variables Breakdown

-   `TASK_CONFIG`: (Required) Task config path or name under `benchmark_tasks/`.
-   `MODEL`: (Required) The name of the model you are testing.
-   `OPENAI_URL`: (Required) The API base URL for your test model.
-   `JUDGE_MODEL`: The name of the judge model. Defaults to `gemini-2.5-flash`.
-   `JUDGE_URL`: The API base URL for the judge model. Defaults to the Google Generative Language API.
-   `JUDGE_PROFILE`: Judge profile path or name under `judge_profiles/`. Defaults to `default`; use `cn_judge` for the first Chinese release.
-   `MODEL_API_KEY_ENV`: The name of the environment variable holding the API key for your test model. Defaults to `OPENAI_API_KEY`.
-   `JUDGE_API_KEY_ENV`: The name of the environment variable holding the API key for the judge model. Defaults to `GEMINI_API_KEY`.
-   `LOW_CONTEXT` / `ULTRA_LOW_CONTEXT`: Set to `true` to use prompts with a smaller context window. Disabled by default.
-   `BASESET_SNAPSHOT_DIR`: Snapshot directory containing frozen anchor translations and base-set artifacts used for comparisons. Defaults to `baseset/v1.0`.

## Workflow

1.  **Translate**: The script loads the configured task items, then generates outputs for the active source-language and target-language directions.
2.  **Generate Pairs**: The new outputs are paired against the frozen anchor outputs in `BASESET_SNAPSHOT_DIR/translations/`, and the pair file is written to `results/<snapshot>/<model>/<judge_dir>/pairs.jsonl`.
3.  **Judge**: The judge model evaluates each pair and writes `results/<snapshot>/<model>/<judge_dir>/judgments.jsonl`.
4.  **Rank**: The analyzer fits the Bradley-Terry model and writes `results/<snapshot>/<model>/<judge_dir>/scores.json` plus `scores.metadata.json`.

## Output Files

-   **Model Outputs**: Generated task outputs are saved to `translations/<safe_model>.jsonl`.
-   **Pairs**: Comparison pairs are saved to `results/<snapshot>/<model>/<judge_dir>/pairs.jsonl`.
-   **Judgments**: Pairwise judge outputs are saved to `results/<snapshot>/<model>/<judge_dir>/judgments.jsonl`.
-   **Scores**: Canonical score summaries are saved to `results/<snapshot>/<model>/<judge_dir>/scores.json`, with additive provenance in `scores.metadata.json`.

`judge_dir` is the safe judge model name for the default profile, and `safe_judge.safe_profile` for non-default judge profiles such as `cn_judge`.

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
mamba run -n shisa-jp-tl-bench python generate_translation_data.py --task <task_config> --base-url <api_url> --test-model <model_name> [OPTIONS]
```

**Required Options:**
- `--base-url`: The API endpoint URL for your model
- `--test-model`: The name of the model to use for translation

**Optional Parameters:**
- `--api-key-env`: Name of the environment variable containing the API key (defaults to `OPENAI_API_KEY`)
- `--task`: Task config path or config name under `benchmark_tasks/` (defaults to `translation_ja_en_bidirectional_v1.yaml`)
- `--low-context`: Use prompts optimized for smaller context windows
- `--ultra-low-context`: Use prompts optimized for very small context windows (4096 tokens)
- `--max-workers`: Number of worker threads for translation (default: 5)
- `--concurrency-limit`: Maximum number of concurrent API requests (default: 5)

**Example:**

```bash
mamba run -n shisa-jp-tl-bench python generate_translation_data.py --task translation_ja_en_bidirectional_v1 --base-url https://generativelanguage.googleapis.com/v1beta/openai/ --test-model gemini-2.5-pro --api-key-env GEMINI_API_KEY
```

This will generate translation data using Google's Gemini model and save the results to `translations/<model_name>.jsonl`.

Task configs live under `benchmark_tasks/` and now control the dataset repo/config/revision, supported directions, translation prompt templates, and available compare-prompt profiles. The checked-in configs currently target the private dataset repo `shisa-ai/bt_translation_set_global`, so you will need `HF_TOKEN` set before generation runs can load the task items.
Generated translation artifacts now record both the configured dataset revision and the resolved immutable Hub commit SHA in `dataset_ref`, so downstream runs remain attributable even when the task config started from a temporary publish label.
Judge prompt selection is now config-driven as well: `translation_comparer_any_model.py` can load a task config plus a judge profile, and the resulting judgment rows carry `judge_profile_id`, `compare_prompt_profile_id`, and `judge_contract_id` for reuse safety.

For the JP legacy boundary, additive result sidecars, and the first Chinese release plan, see [docs/migration-jp-v1-to-task-config.md](docs/migration-jp-v1-to-task-config.md).

### Migrating Existing JP Result Files

Existing JP `scores.json` files stay unchanged. To attach task and dataset metadata beside them, write an additive sidecar:

```bash
mamba run -n shisa-jp-tl-bench python scripts/migrate_result_metadata.py \
  --task translation_ja_en_bidirectional_v1 \
  --scores-file results/v1.0/<model>/<judge_dir>/scores.json
```

### Exporting And Validating The HF Dataset

Task content now lives in the private Hugging Face dataset repo `shisa-ai/bt_translation_set_global`, exported as two configs:

- `translation_ja_en_bidirectional_v1`
- `translation_zh_en_bidirectional_v1`

Rebuild the local export plus provenance docs with:

```bash
mamba run -n shisa-jp-tl-bench python scripts/export_hf_dataset.py
```

Validate a specific task export before any publish with:

```bash
mamba run -n shisa-jp-tl-bench python scripts/validate_hf_dataset.py --task benchmark_tasks/translation_zh_en_bidirectional_v1.yaml
```

That validator checks required columns, deterministic ordering, `item_id` uniqueness, expected language/difficulty balance, Chinese-source provenance metadata, and a pair-generation round trip against the current task config.

The export script also refreshes:

- `docs/chinese_source_manifest.csv`
- `docs/translation_set_inventory.csv`
- `hf_datasets/bt_translation_set_global/README.md`

For the full publish/update workflow, including the post-publish revision lock step, see [docs/hf-dataset-publishing.md](docs/hf-dataset-publishing.md).

### Viewing Results with the TUI

After running benchmarks, you can interactively browse results using the Text User Interface (TUI) viewer. The viewer uses lazy loading for fast performance even with large datasets and supports two viewing modes.

**Usage:**

```bash
./inspect-output [OPTIONS]
```

**Options:**
- `--scores-dir`: Path to scores directory (default: `./scores`)
- `--baseset-dir`: Path to baseset directory (default: `./baseset`)
- `--translations-dir`: Path to translations directory (default: `./translations`)

**Example:**

```bash
./inspect-output
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
mamba run -n shisa-jp-tl-bench python utils/clean_analysis_file.py path/to/your/judgments.jsonl
```

**Example:**

```bash
mamba run -n shisa-jp-tl-bench python utils/clean_analysis_file.py results/v1.0/<model>/<judge_dir>/judgments.jsonl
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
