# Base Set v1.0 Snapshot

The `v1.0/` directory freezes the translation outputs for the 20 baseline models we plan to use as anchors for Shisa JP TL Bench v1.0. All files live in one place so that future runs can be reproduced even if the original folders change.

## Layout

- `v1.0/manifest.json` – ordered list of display names plus the repo-relative source path for each translation dump.
- `v1.0/translations/` – auto-populated JSONL copies produced by `prepare_v1_0.py`.
- `v1.0/base_conversation_pairs.v1.0.jsonl` and `v1.0/base_set.<judge>.jsonl` – generated pair file and judged base sets captured by the helper.
- `v1.0/reports/` – coverage summaries and scoring tables emitted by the helper script.

## prepare_v1_0.py

`python baseset/prepare_v1_0.py` executes the full offline pipeline for the v1.0 snapshot (a legacy `set-generate.py` shim is kept for compatibility):

1. Copy every manifest entry into `baseset/v1.0/translations/`, logging each `source -> destination`.
2. Produce the canonical pair file at `v1.0/base_conversation_pairs.v1.0.jsonl`.
3. Print/snapshot coverage of matches per model.
4. Locate (or generate) a judged `base_set.<judge>.jsonl` and display the LT table.

Key options:

- `--judge-base-url` / `JUDGE_URL` – default matches `run_translation_bench.sh` (`https://generativelanguage.googleapis.com/v1beta/openai/`). Needed when `--auto-judge` (default) calls `translation_comparer_any_model.py --generate-base-set` for missing comparisons. The script temporarily installs `v1.0/base_conversation_pairs.v1.0.jsonl` as `base_conversation_pairs.jsonl`, runs the comparer, then copies the resulting `base_sets/base_set.<judge>.jsonl` into `v1.0/` for archival.
- `--judge-model` – judge identifier passed through to the comparer (default `gemini-2.5-flash`).
- `--judge-api-key-env` / `JUDGE_API_KEY_ENV` – defaults to `GEMINI_API_KEY`, same as the bench script. `--max-workers`, `--concurrency-limit` tune comparer parallelism.
- `--analysis-file` – skip auto-generation and score a specific judged file (relative to repo root or this directory).
- `--no-auto-judge` – rebuild translations/pairs/reports without making API calls.

Example (auto-run judge):

```bash
python baseset/prepare_v1_0.py \
  --judge-model gemini-2.5-flash
```

Example (just rebuild artifacts, no network):

```bash
python baseset/prepare_v1_0.py --no-auto-judge
```

After the pair file is created you can still run `translation_comparer_any_model.py --generate-base-set ...` manually (pointing at the judge+URL of your choice) and then re-run `prepare_v1_0.py --analysis-file <that file>` to update the scoring table.

## Models Covered

The manifest lists the following 20 anchors (shown here so you can quickly verify every source path before cloning the snapshot onto a new machine), along with their v1.0 overall win rate and LT scores:

| # | Model | Source | WR% | LT |
| --- | --- | --- | --- | --- |
| 1 | gemini-2.5-pro | `base_translations/gemini-2.5-pro.jsonl` | 96.07 | 9.96 |
| 2 | gemini-2.5-flash | `base_translations/gemini-2.5-flash.jsonl` | 93.00 | 9.92 |
| 3 | shisa-ai/shisa-v2-llama3.1-405b | `translations/shisa-ai__shisa-v2-llama3.1-405b.jsonl` | 81.56 | 9.62 |
| 4 | Qwen/Qwen3-30B-A3B-Instruct-2507 | `translations/Qwen__Qwen3-30B-A3B-Instruct-2507.jsonl` | 84.04 | 9.71 |
| 5 | shisa-ai/shisa-v2-unphi4-14b | `translations/shisa-ai__shisa-v2-unphi4-14b.jsonl` | 74.21 | 9.20 |
| 6 | openai/gpt-4o | `base_translations/openai__gpt-4o.jsonl` | 76.16 | 9.33 |
| 7 | meta-llama/Llama-3.3-70B-Instruct | `base_translations/meta-llama__Llama-3.3-70B-Instruct.jsonl` | 59.40 | 7.42 |
| 8 | nvidia/NVIDIA-Nemotron-Nano-12B-v2 | `translations/nvidia__NVIDIA-Nemotron-Nano-12B-v2.jsonl` | 60.66 | 7.63 |
| 9 | tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5 | `base_translations/tokyotech-llm__Llama-3.1-Swallow-8B-Instruct-v0.5.jsonl` | 62.88 | 7.97 |
| 10 | microsoft/phi-4 | `base_translations/microsoft__phi-4.jsonl` | 52.29 | 6.07 |
| 11 | cyberagent/Mistral-Nemo-Japanese-Instruct-2408 | `base_translations/cyberagent__Mistral-Nemo-Japanese-Instruct-2408.jsonl` | 49.10 | 5.37 |
| 12 | meta-llama/Llama-3.1-8B-Instruct | `translations/meta-llama__Llama-3.1-8B-Instruct.jsonl` | 40.80 | 3.52 |
| 13 | Qwen/Qwen3-4B | `base_translations/Qwen__Qwen3-4B.jsonl` | 45.64 | 4.59 |
| 14 | microsoft/Phi-4-mini-instruct | `translations/microsoft__Phi-4-mini-instruct.jsonl` | 28.38 | 1.35 |
| 15 | LiquidAI/LFM2-1.2B | `translations/LiquidAI__LFM2-1.2B.jsonl` | 28.11 | 1.32 |
| 16 | meta-llama/Llama-3.2-3B-Instruct | `translations/meta-llama__Llama-3.2-3B-Instruct.jsonl` | 25.16 | 0.98 |
| 17 | Rakuten/RakutenAI-2.0-mini-instruct | `translations/Rakuten__RakutenAI-2.0-mini-instruct.jsonl` | 20.87 | 0.62 |
| 18 | LiquidAI/LFM2-350M | `translations/LiquidAI__LFM2-350M.jsonl` | 14.09 | 0.24 |
| 19 | SakanaAI/TinySwallow-1.5B | `translations/SakanaAI__TinySwallow-1.5B.jsonl` | 6.54 | 0.04 |
| 20 | augmxnt/shisa-7b-v1 | `base_translations/augmxnt__shisa-7b-v1.jsonl` | 0.71 | 0.00 |

These files plus the pairwise judgments produced by your judge (stored either in `base_sets/` or `baseset/v1.0/`) are sufficient to recreate LT/EN scores via `prepare_v1_0.py` (which reuses `choix_analyzer.py` under the hood).

## Scoring New Translation Runs Against v1.0

Follow this checklist any time you want to benchmark a fresh translation dump against the frozen anchors:

1. **Generate your translation outputs**  
   Run the usual bench pipeline (e.g., `MODEL=... OPENAI_URL=... ./run_translation_bench.sh`) so that `translations/<safe_model>.jsonl` is produced. You can also call `generate_translation_data.py` directly if you need extra flags like `--low-context`.

2. **Create comparison pairs vs the anchors**  
   ```bash
   python generate_shootout_data.py --test-model "$MODEL"
   ```
   The command above rebuilds `latest_conversation_pairs.jsonl`, pairing your model against each snapshot reference in `baseset/v1.0/translations/` (or whatever `BASESET_SNAPSHOT_DIR` points to).

4. **Judge the pairs and archive the outputs**  
   ```bash
   python translation_comparer_any_model.py \
     --base-url "$JUDGE_URL" \
     --judge-model "$JUDGE_MODEL" \
     --test-model "$MODEL" \
     --api-key-env "$JUDGE_API_KEY_ENV"
   ```
   The comparer writes `scores/<safe_model>.<safe_judge>.jsonl`. If you want to refresh the anchor judgments as well, copy `baseset/v1.0/base_set.<safe_judge>.jsonl` into `base_sets/` (or rerun `prepare_v1_0.py --auto-judge`).

5. **Compute LT/EN scores tied to v1.0**  
   Make sure `base_sets/base_set.<safe_judge>.jsonl` points to the v1.0 file (either the copy mentioned above or the one produced by `prepare_v1_0.py`). Then run:
   ```bash
   python choix_analyzer.py --test-model "$MODEL" --judge-model "$JUDGE_MODEL"
   ```
   The analyzer automatically loads the shared base set plus your comparison file and emits slice-level LT/EN scores in `scores/<safe_model>_tl_bench_scores.jsonl`.

That’s all that’s required to keep new evals comparable with the v1.0 anchor snapshot—no edits to the manifest are needed unless you intend to version the base set again.

## Generating New Base Sets

You can reuse the same tooling to build future base sets (for example, `baseset/v2.0/`), decoupled from the v1.0 preparation logic.

1. **Create a snapshot directory and manifest**  
   Make a new directory such as `baseset/v2.0/` and add a `manifest.json` with a `"models"` list following the v1.0 shape (`{"model": "...", "source": "..."}`), at minimum including the `model` field for each anchor.

2. **Populate `translations/` for the snapshot**  
   For every model in the manifest, generate and copy its translation dump into `baseset/v2.0/translations/<safe_name>.jsonl`, where `<safe_name>` is the model name with `/` replaced by `__` (e.g., `openai/gpt-4o` → `openai__gpt-4o.jsonl`).  
   For v1.0, this copy/sync step is captured in `prepare_v1_0.py`; you can follow that script as a template for future versions if you want a reproducible “from prior runs” snapshot.

3. **Run the generic base-set generator**  
   Once `baseset/v2.0/manifest.json` and `baseset/v2.0/translations/` are in place, run:
   ```bash
   python baseset/generate_set.py \
     --snapshot-dir baseset/v2.0 \
     --pair-filename base_conversation_pairs.v2.0.jsonl \
     --judge-model gemini-2.5-flash
   ```
   This writes the pair file at `baseset/v2.0/base_conversation_pairs.v2.0.jsonl`, a coverage report at `baseset/v2.0/reports/pair_coverage.json`, auto-generates (by default) a judged `baseset/v2.0/base_set.<judge>.jsonl`, and stores the slice-level LT/EN scores in `baseset/v2.0/reports/base_set.<judge>_scores.json`.

4. **Optional: re-score from an existing judged file**  
   If you already have a judged base-set file, you can skip auto-judging and just point the generator at it:
   ```bash
   python baseset/generate_set.py \
     --snapshot-dir baseset/v2.0 \
     --pair-filename base_conversation_pairs.v2.0.jsonl \
     --no-auto-judge \
     --analysis-file baseset/v2.0/base_set.<safe_judge>.jsonl
   ```
   This rebuilds reports while leaving the judged data untouched.
