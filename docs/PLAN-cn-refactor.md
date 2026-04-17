# CN / Language-Agnostic Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor JP-TL-Bench into a translation-task-configured benchmark workspace that can evaluate multiple language pairs and multiple judges reproducibly, while preserving JP v1.0 comparability and adding a versioned path for the upcoming Chinese dataset on Hugging Face.

**Architecture:** Replace the current JP/EN- and Gemini-shaped pipeline with a config-first translation task model: dataset metadata defines source/target languages and slice labels, judge adapters define provider-specific API behavior, and canonical artifacts carry enough metadata to remain comparable across dataset version, snapshot version, prompt profile, and judge configuration. Keep the existing pairwise + anchored-scoring workflow, but move every hardcoded language and judge assumption behind explicit translation-task and judge configuration, with additive compatibility for the frozen JP v1.0 path.

**Tech Stack:** Python 3.12, Click, JSONL, Hugging Face Datasets/Hub, OpenAI-compatible APIs, optional native Gemini client, Bradley-Terry scoring via `choix`, repo-local snapshot artifacts under `baseset/`.

---

## Context

The current repo has already started a scoring-layout cleanup, but the core pipeline still assumes:

- a single dataset repo: `shisa-ai/bt_translation_test`
- a binary `english` field that really means “EN source vs JA source”
- fixed prompt file names tied to `from_english` / `from_japanese`
- fixed score buckets `en_ja` and `ja_en`
- a mostly Gemini-shaped judge path, with OpenAI-compatible behavior as the default

That is enough for JP v1.0 maintenance, but it is not enough for a Chinese expansion or any future language pair. The current state also hides a few correctness bugs that should be fixed before we trust new scores.

## Approaches Considered

### Option A: Ship a Chinese-only fork

- Pros: fastest path to a working CN benchmark
- Cons: duplicates today’s hardcoding, does not solve judge abstraction, makes future maintenance worse

### Option B: Config-first refactor over the current pipeline

- Pros: smallest change that still solves language/judge generalization; preserves existing scripts and artifacts where possible; easiest migration path for JP v1.0
- Cons: requires a careful contract pass across dataset schema, prompt selection, scoring slices, and docs

### Option C: Rewrite into a new framework/package

- Pros: clean slate
- Cons: highest risk, longest migration, unnecessary for the current need

**Recommendation:** Option B. Generalize the existing pipeline with explicit translation-task config, judge adapters, and versioned dataset/snapshot metadata. Do not do a fork, and do not do a framework rewrite.

This plan is intentionally scoped to `task_type: translation`. The extension seam for other pairwise task types can be left in the config layer, but non-translation task support is not part of this refactor.

## Decisions Recorded

- Hugging Face data moves to a new dataset repo instead of continuing to evolve `shisa-ai/bt_translation_test`.
- The Chinese benchmark is bidirectional. The task/config layer must explicitly specify supported source and target languages.
- Prompt selection is config-driven. We can keep a default compare prompt family, but task or judge configs must be able to opt into language-specific compare prompts when that improves judgment quality.

## Current Findings

### Must-fix correctness issues

1. `translation_comparer_any_model.py`
   The A/B position randomization swaps `llm_a` and `llm_b`, but it leaves side-specific metadata fields like `llm_a_generation_config` and `llm_b_generation_config` attached to the wrong side after the swap. This makes stored judgments internally inconsistent.

2. `generate_base_scores.py`
   The current script is broken against `LLMRanker.get_rankings()` because it passes `language="all"`, which triggers an indexing error instead of generating anchor reports.

3. `generate_base_scores.py` and `score_visualizer.py`
   Even if the crash were fixed, the current base-score generator emits ambiguous slice labels (`overall`, `easy`, `hard`) while the visualizer expects anchor slices like `en_ja`, `en_ja_easy`, `ja_en`, and `ja_en_hard`. Re-running the generator would silently break anchor loading in the visualizer.

4. `baseset/generate_set.py`, `baseset/prepare_v1_0.py`, and `translation_comparer_any_model.py`
   The auto-judge snapshot path still swaps repo-root `base_conversation_pairs.jsonl`, does not consistently use `--pairs-file`, and disagrees on where generated base-set outputs should land. This is a correctness and reproducibility bug, not just cleanup work.

### Structural risks

1. `generate_translation_data.py`
   Hard-loads `shisa-ai/bt_translation_test` with no repo/revision pinning beyond the default dataset state, which is not reproducible enough for benchmark inputs.

2. `generate_translation_data.py`, `generate_shootout_data.py`, `choix_analyzer.py`, `score_visualizer.py`, `baseset/generate_set.py`, `baseset/prepare_v1_0.py`
   These all encode the EN/JA task shape directly via `english`, `from_english`, `from_japanese`, `en_ja`, and `ja_en`.

3. `run_translation_bench.sh`
   Still documents and names the workflow as translation-specific, activates a fixed env internally, and does not expose task/dataset selection as a first-class input.

4. Repository docs and scripts
   The shell’s default env is not the intended project env; running plain `python` from the default shell fails on missing packages. The documented workflow should consistently prefer `mamba run -n shisa-jp-tl-bench ...` or a similarly explicit execution path.

## Target State

### Canonical benchmark concepts

- `task_type`: translation
- `task`: dataset + prompt contract + score slices + language metadata
- `snapshot`: frozen anchor outputs and judged anchor graph for one task version
- `judge`: explicit provider/model/prompt adapter combination
- `legacy_jp_v1`: additive compatibility boundary for frozen JP v1.0 assets; existing JSONL/report files stay byte-for-byte unchanged
- `artifact`: output file that carries `{task_type, task_id, task_version, snapshot_version, schema_version, dataset_ref, task_config_digest, generation_profile_id, compare_prompt_profile_id, judge_contract_id, pair_id_schema}`

### Canonical task record shape

Recommended task-item schema for Hugging Face and repo-local consumers:

```json
{
  "item_id": "zh_01",
  "name": "zh_01",
  "task_id": "translation.zh-en",
  "task_version": "v1",
  "source_language": "zh",
  "target_language": "en",
  "source_text": "...",
  "difficulty": "easy",
  "category": "consumer_help",
  "tags": ["support", "product"],
  "notes": null
}
```

Recommended generated-output schema:

```json
{
  "item_id": "zh_01",
  "name": "zh_01",
  "task_id": "translation.zh-en",
  "task_version": "v1",
  "task_type": "translation",
  "source_language": "zh",
  "target_language": "en",
  "difficulty": "easy",
  "status": "ok",
  "model": "example/model",
  "prompt_profile": "default",
  "prompt_template": "prompts/translate/default.txt",
  "generation_profile_id": "default-v1",
  "dataset_ref": {
    "repo": "shisa-ai/<new-benchmark-dataset>",
    "config": "translation_zh_en_bidirectional_v1",
    "split": "train",
    "revision": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
  },
  "task_config_digest": "sha256:...",
  "source_text": "...",
  "translation": "...",
  "full_response": "...",
  "generation_config": {
    "temperature": 0.1,
    "top_p": 0.85,
    "reasoning_effort": null
  }
}
```

Recommended canonical score summary shape:

```json
{
  "schema_version": "v2",
  "task_type": "translation",
  "task_id": "translation.zh-en",
  "task_version": "v1",
  "snapshot_version": "v1.0",
  "dataset_ref": {
    "repo": "shisa-ai/<new-benchmark-dataset>",
    "config": "translation_zh_en_bidirectional_v1",
    "split": "train",
    "revision": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
  },
  "task_config_digest": "sha256:...",
  "generation_profile_id": "default-v1",
  "compare_prompt_profile_id": "compare-cn-v1",
  "judge_contract_id": "google/gemini-2.5-flash::compare-cn-v1::answer-parser/v1",
  "pair_id_schema": "v2",
  "model": "example/model",
  "expected_pairs": 1340,
  "judged_pairs": 1340,
  "missing_pairs": 0,
  "slices": {
    "overall": {"lt": 9.4, "wins": 610, "total": 680, "win_rate": 0.897},
    "source=zh": {"lt": 9.4, "wins": 610, "total": 680, "win_rate": 0.897},
    "difficulty=easy": {"lt": 9.5, "wins": 280, "total": 300, "win_rate": 0.933}
  }
}
```

The exact slice-key syntax can change, but it must be task-driven instead of assuming only `en_ja` and `ja_en`.

Pair identity also needs an explicit compatibility contract:

- `pair_id_schema=v1` preserves the current JP v1.0 formula so existing judgments remain reusable.
- `pair_id_schema=v2` can use `item_id` plus model identifiers, but only after the migration layer is in place.
- Every pair and judgment artifact should also carry a `pair_fingerprint` or content hash derived from the item text, task metadata, and the two candidate outputs so reuse can be rejected when content changes under the same nominal ID.

## File Map

### Config and contracts

- Create: `benchmark_tasks/translation_ja_en_bidirectional_v1.yaml`
- Create: `benchmark_tasks/translation_zh_en_bidirectional_v1.yaml`
- Create: `judge_profiles/default.yaml`
- Create: `judge_profiles/cn_judge.yaml`
- Create: `benchmark_tasks/schema.md`
- Create: `prompts/translate/default.txt`
- Create: `prompts/compare/default.txt`
- Create: `prompts/compare/cn_judge.txt`
- Modify: `README.md`
- Modify: `baseset/README.md`
- Modify: `docs/translation_set_inventory.csv`

### Pipeline

- Modify: `generate_translation_data.py`
- Modify: `generate_shootout_data.py`
- Modify: `translation_comparer_any_model.py`
- Modify: `choix_analyzer.py`
- Modify: `score_visualizer.py`
- Modify: `generate_base_scores.py`
- Modify: `run_translation_bench.sh`
- Modify: `baseset/generate_set.py`
- Modify: `baseset/prepare_v1_0.py`
- Modify: `base_set_manager.py`

### Tests / verification

- Create: `tests/test_task_config.py`
- Create: `tests/test_pair_generation.py`
- Create: `tests/test_judge_metadata_swaps.py`
- Create: `tests/test_failed_generation_handling.py`
- Create: `tests/test_pair_id_stability.py`
- Create: `tests/test_legacy_jp_v1_boundary.py`
- Create: `tests/test_scoring_slices.py`
- Create: `tests/test_base_score_reports.py`
- Create: `tests/test_score_visualizer_loading.py`
- Create: `tests/test_baseset_auto_judge_smoke.py`

## Phases

### Task 1: Freeze The Current Contract And Add Regression Fixtures

**Files:**
- Modify: `generate_translation_data.py`
- Modify: `generate_shootout_data.py`
- Modify: `translation_comparer_any_model.py`
- Modify: `choix_analyzer.py`
- Create: `benchmark_tasks/schema.md`
- Create: `tests/test_judge_metadata_swaps.py`
- Create: `tests/test_failed_generation_handling.py`
- Create: `tests/test_scoring_slices.py`

- [x] Capture one tiny golden fixture set from the existing JP v1.0 workflow.
  Minimum content: one swap-eligible pair with distinct `llm_a_generation_config` vs `llm_b_generation_config`, one failed-generation row, one valid score summary, and one tiny snapshot fixture that can later exercise the auto-judge `--pairs-file` path without touching repo-root shared files.
- [x] Add a regression test for the comparer’s A/B swap logic that asserts model names, labels, and side-specific metadata move together.
- [x] Add a regression test showing that failed generations are not allowed to flow into pair generation and scoring as if they were valid translations.
- [x] Add a regression test for score extraction so later generalization does not silently change JP v1.0 slice values or pair accounting.
- [x] Record the current artifact schema in `benchmark_tasks/schema.md` before changing it.
- [x] Run: `mamba run -n shisa-jp-tl-bench python -m pytest tests/test_judge_metadata_swaps.py tests/test_failed_generation_handling.py tests/test_scoring_slices.py -q`

### Task 2: Freeze The Legacy JP v1 Boundary And Pair Identity Contract

**Files:**
- Modify: `generate_shootout_data.py`
- Modify: `translation_comparer_any_model.py`
- Modify: `score_visualizer.py`
- Modify: `baseset/generate_set.py`
- Modify: `baseset/prepare_v1_0.py`
- Create: `tests/test_pair_id_stability.py`
- Create: `tests/test_legacy_jp_v1_boundary.py`

- [x] Declare `legacy_jp_v1` as additive-only: existing JP v1.0 JSONL files and reports do not get rewritten in place.
- [x] Explicitly grandfather the frozen JP v1.0 judgments and reports in their pre-bug form; the swap fix and new metadata rules apply only to new judgments and schema-versioned reruns.
- [x] Define and document `pair_id_schema=v1` as the current JP-compatible formula.
- [x] Add `pair_fingerprint` and reuse validation rules so future judgments are keyed by both stable ID and content identity.
- [x] Create sidecar metadata or schema-versioned exports for JP v1.0 rather than mutating the frozen artifacts.
- [x] Add a regression test that regenerated JP v1 pair files still resolve existing judgments without rejudging.
- [x] Add a regression test that JP v1.0 base-set report loading still works with the frozen legacy files after the new compatibility layer is introduced.
- [x] Run: `mamba run -n shisa-jp-tl-bench pytest tests/test_pair_id_stability.py tests/test_legacy_jp_v1_boundary.py -q`

### Task 3: Introduce A Translation Task Config Layer

**Files:**
- Create: `benchmark_tasks/translation_ja_en_bidirectional_v1.yaml`
- Create: `benchmark_tasks/translation_zh_en_bidirectional_v1.yaml`
- Create: `judge_profiles/default.yaml`
- Create: `judge_profiles/cn_judge.yaml`
- Create: `benchmark_tasks/__init__.py`
- Create: `benchmark_tasks/schema.md`
- Modify: `generate_translation_data.py`
- Modify: `generate_shootout_data.py`
- Modify: `choix_analyzer.py`

- [x] Define a config system that includes task configs plus judge/profile configs. At minimum it must cover dataset repo/config/split, translation prompt templates, compare prompt profiles, supported language directions, source and target language codes, and scoring slice definitions.
- [x] Replace the boolean `english` assumption with explicit `source_language` and `target_language` fields throughout the pipeline.
- [x] Keep a compatibility shim so existing JP artifacts with `english` and legacy `name` still load during migration.
- [x] During the transition, write both `item_id` and legacy `name` for JP tasks and document the deterministic `item_id <-> name` mapping.
- [x] Make bidirectional tasks first-class: the first Chinese task config must explicitly cover both `zh -> en` and `en -> zh`.
- [x] Make compare-prompt selection configurable per task or judge profile so Chinese-facing judges can use a CN-oriented compare prompt without adding hardcoded language branches.
- [x] Make every script accept `--task` or `TASK_CONFIG` instead of inferring task shape from field names.
- [x] Run: `mamba run -n shisa-jp-tl-bench pytest tests/test_task_config.py -q`

### Task 4: Version The Hugging Face Dataset Properly

**Files:**
- Modify: `generate_translation_data.py`
- Modify: `README.md`
- Modify: `docs/translation_set_inventory.csv`
- Create: `docs/chinese_source_manifest.csv`
- Create: `docs/hf-dataset-publishing.md`
- Create: `scripts/export_hf_dataset.py`
- Create: `scripts/validate_hf_dataset.py`

- [x] Publish to a new dataset repo; do not keep extending `shisa-ai/bt_translation_test`.
- [x] Publish JP v1 as an immutable task config or dataset config instead of a mutable default split.
- [x] Add the new Chinese task items under a bidirectional config/version with stable `item_id` values and task metadata.
- [x] Build a pre-publish Chinese source manifest with actual `source_text`, source URL/title, access date, extraction notes, license or redistribution status, and reviewer sign-off.
- [x] Add a validation script that checks required columns, uniqueness of `item_id`, deterministic ordering, difficulty balance, source/target language consistency, provenance metadata, and round-trip compatibility with pair generation before any push.
- [x] Make `generate_translation_data.py` load `{repo, config, split, revision}` from task config so runs pin a known immutable HF revision.
- [x] Require every downstream artifact and snapshot manifest to record `dataset_repo`, `dataset_config`, `dataset_split`, and resolved Hub commit SHA.
- [x] Add a post-publish lock step so task configs never point at a symbolic ref like `main`.
- [x] Generate the HF dataset card/README from the publish manifest instead of maintaining it by hand.
- [x] Document the publish/update workflow, including who is allowed to mutate task content versus only add new versions.
- [x] Run: `mamba run -n shisa-jp-tl-bench python scripts/validate_hf_dataset.py --task benchmark_tasks/translation_zh_en_bidirectional_v1.yaml`

### Task 5: Generalize Translation Generation

**Files:**
- Modify: `generate_translation_data.py`
- Modify: `prompts/translate_prompt_from_english.txt`
- Modify: `prompts/translate_prompt_from_japanese.txt`
- Create: `prompts/translate/default.txt`
- Create: `tests/test_generation_prompt_selection.py`

- [x] Replace prompt-file selection by `from_english` / `from_japanese` with a task-configured prompt template and language placeholders.
- [x] Move provider/model quirks into a reusable generation adapter instead of scattering checks for Gemini, GPT-5, and `cat-translate` in the translator path.
- [x] Preserve generation metadata in a model-neutral way.
- [x] Replace placeholder failure translations with a machine-readable failure state; pair generation and scoring must refuse to consume unresolved failures.
- [x] Add a guard that fails early if the requested API-key env var is missing.
- [x] Keep a compatibility path for old JP prompts until downstream prompt contracts are migrated.
- [x] Run: `mamba run -n shisa-jp-tl-bench pytest tests/test_generation_prompt_selection.py -q`

### Task 6: Generalize Pair Generation And Judge Execution

**Files:**
- Modify: `generate_shootout_data.py`
- Modify: `translation_comparer_any_model.py`
- Create: `tests/test_pair_generation.py`
- Create: `tests/test_judge_metadata_swaps.py`

- [x] Update pair records to carry `item_id`, `task_id`, `task_version`, `source_language`, `target_language`, and any task-defined slice tags.
- [x] Keep the pair formatter translation-specific but direction-agnostic; do not widen this refactor into non-translation task formatting.
- [x] Require strict equality checks for all task-defining item fields (`item_id`/`name`, `source_text`, difficulty, source/target language, task tags) before a pair is emitted.
- [x] Fix the comparer’s swap bug so side-specific metadata is swapped together with labels and model IDs.
- [x] Introduce explicit judge adapters keyed by provider/model instead of scattering Gemini/OpenAI branching through the comparer.
- [x] Load the compare prompt through the task/judge config layer so the same pipeline can choose `prompts/compare/default.txt` or a language-specific compare prompt profile such as `prompts/compare/cn_judge.txt`.
- [x] Ensure reuse-by-default still works, but only when the existing output matches the same task, snapshot, judge contract, prompt version, pair fingerprint, and pair-ID schema.
- [x] Run: `mamba run -n shisa-jp-tl-bench pytest tests/test_pair_generation.py tests/test_judge_metadata_swaps.py -q`

### Task 7: Generalize Scoring And Visualization

**Files:**
- Modify: `choix_analyzer.py`
- Modify: `generate_base_scores.py`
- Modify: `score_visualizer.py`
- Create: `tests/test_base_score_reports.py`
- Create: `tests/test_score_visualizer_loading.py`

- [x] Replace hardcoded `en_ja` / `ja_en` output fields with task-defined slice keys.
- [x] Keep JP v1.0 display compatibility by providing a mapping layer in the visualizer for legacy score files.
- [x] Fix `generate_base_scores.py` so it uses one explicitly defined ranking API contract and emits stable, unambiguous anchor slice labels that match what the visualizer consumes.
- [x] Add a dedicated regression test for duplicate slice-label collisions so EN and JA anchor rows cannot silently overwrite each other in the visualizer.
- [x] Move slice/report writing into shared code used by `generate_base_scores.py`, `baseset/generate_set.py`, and `baseset/prepare_v1_0.py` so there is one canonical report contract.
- [x] Make the visualizer task-aware so it can render any translation direction names from config rather than using flag emojis and fixed EN/JA headings.
- [x] Validate that base-anchor report loading and candidate-score loading use the same slice namespace.
- [x] Run: `mamba run -n shisa-jp-tl-bench pytest tests/test_base_score_reports.py tests/test_score_visualizer_loading.py -q`

### Task 8: Make Snapshot Tooling Task-Aware

**Files:**
- Modify: `baseset/generate_set.py`
- Modify: `baseset/prepare_v1_0.py`
- Modify: `baseset/backfill-1.0-judgements.py`
- Modify: `base_set_manager.py`
- Modify: `baseset/README.md`

- [x] Remove JP-specific assumptions from snapshot pair generation and reporting so a snapshot is tied to `task_id` plus `snapshot_version`, not to the JP v1.0 directory layout alone.
- [x] Store task metadata and dataset provenance in snapshot manifests and base-set report sidecars.
- [x] Keep v1.0 behavior intact, but route all new snapshot creation through the generic path instead of expanding `prepare_v1_0.py`.
- [x] Fix the broken auto-judge path: stop swapping repo-root `base_conversation_pairs.jsonl`, pass `--pairs-file` through to the comparer, and unify on one output location.
- [x] Make judge-file resolution exact-match-or-error; if the requested judge file is absent, require `--analysis-file` instead of falling back to an unrelated `base_set.*.jsonl`.
- [x] Make base-set management commands accept raw judge IDs safely and derive safe filenames internally.
- [x] Either repair and cover `baseset/backfill-1.0-judgements.py` or deprecate/remove it so it is not a misleading recovery path.
- [x] Run: `mamba run -n shisa-jp-tl-bench pytest tests/test_baseset_auto_judge_smoke.py -q`
- [x] Run: `mamba run -n shisa-jp-tl-bench python baseset/generate_set.py --help`
- [x] Run: `mamba run -n shisa-jp-tl-bench python base_set_manager.py --help`

### Task 9: Migrate Existing JP Artifacts And Add The Chinese Task

**Files:**
- Modify: `README.md`
- Modify: `run_translation_bench.sh`
- Modify: `docs/chinese_pair_sourcing_template.csv`
- Modify: `docs/translation_set_inventory.csv`
- Create: `docs/migration-jp-v1-to-task-config.md`

- [x] Add additive sidecars or schema-versioned exports for existing JP result files and snapshot assets; do not rewrite the frozen JP v1.0 artifacts in place.
- [x] Add a compatibility test proving that old and migrated JP snapshot metadata score identically.
- [x] Build the first Chinese task inventory from `docs/chinese_pair_sourcing_template.csv`, then promote it into the publish manifest only after provenance, redistribution, and reviewer checks pass.
- [x] Create a Chinese snapshot plan: anchor model list, manifest, pair coverage target, judge choice, and expected cost.
- [x] Choose an explicit compare-prompt profile for the first Chinese release and record it in config and artifact metadata so CN-optimized judging is attributable and reproducible.
- [x] Document how Chinese scores will stay separate from JP scores at the task and snapshot levels.
- [x] Run a dry-run end-to-end generation + pair + score pass on a small Chinese subset before any full snapshot judging.

### Task 10: Clean Up The CLI And Docs

**Files:**
- Modify: `README.md`
- Modify: `run_translation_bench.sh`
- Modify: `baseset/README.md`
- Modify: `docs/PLAN-cn-refactor.md`

- [x] Rename CLI/docs concepts from “translation bench” and “english/japanese” where they are really “task”, “source language”, “target language”, and “judge”.
- [x] Standardize docs and scripts on explicit env execution, preferring `mamba run -n shisa-jp-tl-bench ...` for reproducibility.
- [x] Update examples so both JP and CN tasks are represented.
- [x] Remove stale help text that still mentions `latest_conversation_pairs.jsonl` and other pre-refactor paths.
- [x] Run: `mamba run -n shisa-jp-tl-bench python generate_translation_data.py --help`
- [x] Run: `mamba run -n shisa-jp-tl-bench python translation_comparer_any_model.py --help`
- [x] Run: `mamba run -n shisa-jp-tl-bench python choix_analyzer.py --help`

## Migration Strategy

1. Land the regression tests and bug fixes first.
2. Freeze the legacy JP v1 boundary and pair-identity contract before changing schemas.
3. Add the translation task config layer and compatibility shims.
4. Publish the versioned HF dataset contract before switching the generation script default.
5. Migrate JP readers through additive sidecars and schema-versioned exports, without changing the frozen base judgments or reports.
6. Bring up the Chinese task on the new contract.
7. Only then create or judge a Chinese base snapshot.

This order keeps the existing JP benchmark usable while the generalization work lands incrementally.

## Acceptance Criteria

- JP v1.0 legacy score files, frozen snapshot files, and reports remain byte-for-byte unchanged.
- A pinned JP v1.0 fixture reproduces identical `expected_pairs`, `judged_pairs`, `missing_pairs`, and slice LT/win-rate values after the refactor.
- New runs require an explicit task config and record task metadata in every artifact.
- Every generated artifact records immutable dataset provenance (`repo/config/split/revision`), task-config digest, generation profile, and judge contract.
- Judge reuse is scoped by task + snapshot + judge contract + prompt/parser version + pair fingerprint.
- The HF dataset publish path is versioned and validated before upload.
- A small bidirectional Chinese dry-run can execute end to end without any EN/JA-specific code path.
- The known comparer, failed-generation, base-score, and snapshot auto-judge bugs are covered by tests.
