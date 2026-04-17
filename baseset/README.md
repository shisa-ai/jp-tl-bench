# Base Set Snapshots

The `v1.0/` directory freezes the translation outputs for the 20 baseline models we plan to use as anchors for Shisa JP TL Bench v1.0. Newer snapshots should follow the same layout, but are now task-aware: a snapshot is tied to a task config plus a snapshot version, and its reports carry task/dataset provenance sidecars.

## Layout

- `v1.0/manifest.json` – ordered list of display names plus the repo-relative source path for each translation dump.
- `v1.0/translations/` – frozen JSONL copies for the snapshot. `prepare_v1_0.py` now reuses these in place instead of overwriting them.
- `v1.0/base_conversation_pairs.v1.0.jsonl` and `v1.0/base_set.<judge>.jsonl` – generated pair file and judged base sets captured by the helper.
- `v1.0/reports/` – frozen coverage summaries and scoring tables for the legacy boundary, plus additive sidecars for newer schema-aware reruns.
- `v1.0/legacy_jp_v1_boundary.json` – sidecar metadata declaring the frozen JP v1.0 compatibility contract.

The `v1.0/` snapshot is now additive-only. Existing JSONL files and canonical reports are treated as frozen legacy artifacts. If you rerun the helper after the pair-contract refactor, it reuses the frozen files when possible and writes any new exports beside them using `.schema-v2` filenames instead of mutating the originals.

## prepare_v1_0.py

`mamba run -n shisa-jp-tl-bench python baseset/prepare_v1_0.py` executes the full offline pipeline for the v1.0 snapshot:

1. Copy every manifest entry into `baseset/v1.0/translations/`, logging each `source -> destination`.
2. Produce the canonical pair file at `v1.0/base_conversation_pairs.v1.0.jsonl`.
3. Print/snapshot coverage of matches per model.
4. Locate (or generate) a judged `base_set.<judge>.jsonl` and display the LT table.

The helper also treats invalid or missing `<answer>` tags as gaps to backfill, reuses existing judgments via skip-IDs when auto-judging so nothing is re-evaluated, filters scoring to manifest models only, and writes a score-report metadata sidecar beside any refreshed report output.

Key options:

- `--judge-base-url` / `JUDGE_URL` – default matches `run_translation_bench.sh` (`https://generativelanguage.googleapis.com/v1beta/openai/`). Needed when `--auto-judge` (default) calls `translation_comparer_any_model.py --generate-base-set` for missing comparisons. Under the frozen v1.0 boundary, the helper reuses `v1.0/base_conversation_pairs.v1.0.jsonl` directly and any new judged/rerun outputs land beside the frozen files as `.schema-v2` exports.
- `--judge-model` – judge identifier passed through to the comparer (default `gemini-2.5-flash`).
- `--judge-api-key-env` / `JUDGE_API_KEY_ENV` – defaults to `GEMINI_API_KEY`, same as the bench script. `--max-workers`, `--concurrency-limit` tune comparer parallelism.
- `--analysis-file` – skip auto-generation and score a specific judged file. Judge resolution is exact-match-or-error; the helper no longer falls back to unrelated `base_set.*.jsonl` files.
- `--no-auto-judge` – rebuild translations/pairs/reports without making API calls.
- `--task` – override the task config used for scoring/report metadata. Leaving it unset on `prepare_v1_0.py` keeps the JP v1 default.

Example (auto-run judge):

```bash
mamba run -n shisa-jp-tl-bench python baseset/prepare_v1_0.py \
  --judge-model gemini-2.5-flash
```

Example (just rebuild artifacts, no network):

```bash
mamba run -n shisa-jp-tl-bench python baseset/prepare_v1_0.py --no-auto-judge
```

After the pair file is created you can still run `translation_comparer_any_model.py --generate-base-set ...` manually and then re-run `prepare_v1_0.py --analysis-file <that file>` to update the scoring table. On v1.0, new reruns are written as `.schema-v2` files so the frozen canonical artifacts stay unchanged. The older `baseset/backfill-1.0-judgements.py` path is deprecated and should not be used.

## Models Covered

The manifest lists the following 20 anchors (shown here so you can quickly verify every source path before cloning the snapshot onto a new machine), along with their v1.0 overall win rate and LT scores:

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

These files plus the pairwise judgments produced by your judge (stored either in `baseset/v1.0/` or as `.schema-v2` side exports) are sufficient to recreate LT/EN scores via `prepare_v1_0.py` (which reuses `choix_analyzer.py` under the hood).

## Scoring New Runs Against v1.0

Follow this checklist any time you want to benchmark a fresh task run against the frozen v1.0 anchors:

1. **Generate your task outputs**  
   Run the usual bench pipeline (e.g., `TASK_CONFIG=... MODEL=... OPENAI_URL=... ./run_translation_bench.sh`) so that `translations/<safe_model>.jsonl` is produced. You can also call `generate_translation_data.py` directly if you need extra flags like `--low-context`.

2. **Create comparison pairs vs the anchors**  
   ```bash
   mamba run -n shisa-jp-tl-bench python generate_shootout_data.py \
     --task translation_ja_en_bidirectional_v1 \
     --judge-profile default \
     --test-model "$MODEL" \
     --judge-model "$JUDGE_MODEL"
   ```
   The command above writes `results/v1.0/<safe_model>/<judge_dir>/pairs.jsonl`, pairing your model against each snapshot reference in `baseset/v1.0/translations/` (or whatever `BASESET_SNAPSHOT_DIR` points to).

3. **Judge the pairs and archive the outputs**  
   ```bash
   mamba run -n shisa-jp-tl-bench python translation_comparer_any_model.py \
     --task translation_ja_en_bidirectional_v1 \
     --judge-profile default \
     --base-url "$JUDGE_URL" \
     --judge-model "$JUDGE_MODEL" \
     --test-model "$MODEL" \
     --api-key-env "$JUDGE_API_KEY_ENV"
   ```
   The comparer writes `results/v1.0/<safe_model>/<judge_dir>/judgments.jsonl`. If you want to refresh the anchor judgments as well, rerun `prepare_v1_0.py --auto-judge`; on the frozen v1.0 snapshot that produces `.schema-v2` side exports instead of replacing `base_set.<safe_judge>.jsonl`.

4. **Compute LT/EN scores tied to v1.0**  
   Make sure the analyzer sees the desired v1.0 judged file (`base_set.<safe_judge>.jsonl` for the frozen legacy snapshot, or the `.schema-v2` side export from a newer rerun). Then run:
   ```bash
   mamba run -n shisa-jp-tl-bench python choix_analyzer.py \
     --task translation_ja_en_bidirectional_v1 \
     --judge-profile default \
     --test-model "$MODEL" \
     --judge-model "$JUDGE_MODEL"
   ```
   The analyzer automatically loads the shared base set plus your comparison file and emits slice-level LT/EN scores in `results/v1.0/<safe_model>/<judge_dir>/scores.json`, with additive provenance in `scores.metadata.json`.

   `judge_dir` is the safe judge model name for the default profile, and `safe_judge.safe_profile` for non-default judge profiles.

That’s all that’s required to keep new evals comparable with the v1.0 anchor snapshot—no edits to the manifest are needed unless you intend to version the base set again.

## Generating New Base Sets

You can reuse the same tooling to build future base sets (for example, `baseset/v2.0/`), decoupled from the v1.0 preparation logic.

1. **Create a snapshot directory and manifest**  
   Make a new directory such as `baseset/v2.0/` and add a `manifest.json` with a `"models"` list following the v1.0 shape (`{"model": "...", "source": "..."}`).
   For new snapshots, also record `task_id`, `task_type`, `task_version`, `dataset_repo`, `dataset_config`, `dataset_split`, and `dataset_revision`. If those fields are omitted, `generate_set.py --task ...` will backfill them from the task config and write them into the manifest.

2. **Populate `translations/` for the snapshot**  
   For every model in the manifest, generate and copy its translation dump into `baseset/v2.0/translations/<safe_name>.jsonl`, where `<safe_name>` is the model name with `/` replaced by `__` (e.g., `openai/gpt-4o` → `openai__gpt-4o.jsonl`).  
   For v1.0, this copy/sync step is captured in `prepare_v1_0.py`; you can follow that script as a template for future versions if you want a reproducible “from prior runs” snapshot.

3. **Run the generic base-set generator**  
   Once `baseset/v2.0/manifest.json` and `baseset/v2.0/translations/` are in place, run:
   ```bash
   mamba run -n shisa-jp-tl-bench python baseset/generate_set.py \
     --snapshot-dir baseset/v2.0 \
     --pair-filename base_conversation_pairs.v2.0.jsonl \
     --task benchmark_tasks/translation_zh_en_bidirectional_v1.yaml \
     --judge-model gemini-2.5-flash
   ```
   This writes the pair file at `baseset/v2.0/base_conversation_pairs.v2.0.jsonl`, a coverage report at `baseset/v2.0/reports/pair_coverage.json`, auto-generates a judged `baseset/v2.0/base_set.<judge>.jsonl`, and stores the slice-level LT/EN scores in `baseset/v2.0/reports/base_set.<judge>_scores.json`.
   The report now gets a sidecar at `baseset/v2.0/reports/base_set.<judge>_scores.metadata.json` carrying snapshot, task, dataset, judge, and artifact provenance.

4. **Optional: re-score from an existing judged file**  
   If you already have a judged base-set file, you can skip auto-judging and just point the generator at it:
   ```bash
   mamba run -n shisa-jp-tl-bench python baseset/generate_set.py \
     --snapshot-dir baseset/v2.0 \
     --pair-filename base_conversation_pairs.v2.0.jsonl \
     --no-auto-judge \
     --analysis-file baseset/v2.0/base_set.<safe_judge>.jsonl
   ```
   This rebuilds reports while leaving the judged data untouched. If the requested judge file is absent, the command errors instead of silently picking another `base_set.*.jsonl`.
