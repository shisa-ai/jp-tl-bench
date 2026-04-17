# Legacy Artifact Schema Notes

This document records the current v1 translation-benchmark artifact shapes before the broader task-config refactor lands.

## Task Configs

Task configs now live under `benchmark_tasks/*.yaml`.

Each task config defines:

- `task_id`, `task_type`, `task_version`
- `dataset.repo`, `dataset.config`, `dataset.split`, `dataset.revision`
- `dataset.hf_token_env` for private Hugging Face repos
- `directions[*].key`
- `directions[*].source_language`
- `directions[*].target_language`
- `directions[*].translation_prompts.{default,low_context,ultra_low_context}`
- `compare_prompt_profile`
- `compare_prompt_profiles.*.prompt_path`
- `compatibility.name_alias_mode`
- `compatibility.emit_legacy_english`
- `scoring.direction_order`

Current checked-in task configs:

- `translation_ja_en_bidirectional_v1`
- `translation_zh_en_bidirectional_v1`

Compatibility rules:

- JP v1 keeps legacy `name` and `english` semantics for migration.
- ZH v1 uses `item_id` as the canonical `name` alias and does not emit legacy `english`.
- Scripts resolve `--task` or `TASK_CONFIG` against `benchmark_tasks/` and default to `translation_ja_en_bidirectional_v1.yaml`.
- `dataset.revision` is expected to be an immutable Hub commit SHA for released task configs. If a temporary branch/tag name is used during dataset publication, it must be resolved and replaced before the task config is treated as canonical.

## Judge Profiles

Judge profiles now live under `judge_profiles/*.yaml`.

Each judge profile currently defines:

- `judge_profile_id`
- `compare_prompt_profile`
- `parser_id`
- optional `request_defaults`
- optional `model_request_overrides[*].contains`
- optional `model_request_overrides[*].settings`

Current checked-in judge profiles:

- `default`
- `cn_judge`

The compare-prompt profile is resolved through the active task config so a task can expose multiple compare prompts without hardcoding language branches into the pipeline.
Request-time judge quirks such as `temperature`, `reasoning_effort`, and native Gemini `thinking_budget` now resolve through the judge profile instead of being hardcoded inside the adapter implementation.

## Generated Translation Rows

Current translation output rows carry:

- `item_id`
- `name`
- `task_id`
- `task_type`
- `task_version`
- `source_text`
- `difficulty`
- `english`
- `source_language`
- `target_language`
- `dataset_ref`
- `task_config_digest`
- `generation_profile_id`
- `model`
- `prompt_profile`
- `prompt_template`
- `status`
- `full_response`
- `translation`
- `prompt`
- `temperature`
- `top_p`
- `frequency_penalty`
- `reasoning_effort`
- `low_context`
- `ultra_low_context`
- `generation_config`

Compatibility notes:

- `english=true` means English source and Japanese target.
- `english=false` means Japanese source and English target.
- `source_language` and `target_language` are now the canonical direction fields.
- `item_id` is now the canonical item identity field.
- `status=ok` is the valid row state.
- `status=failed` is a machine-readable failure marker and must not be consumed by pair generation.
- `dataset_ref.revision` records the task config's requested dataset revision.
- `dataset_ref.resolved_revision` records the immutable resolved Hub commit SHA used for reproducibility checks and downstream attribution.
- `generation_profile_id` records the reusable generation-adapter profile that selected model-specific request quirks such as GPT-5 temperature handling or the compatibility `cat-translate` prompt override.

## Pair Rows

Current pair rows carry:

- `id`
- `pair_id_schema`
- `pair_fingerprint`
- `llm_a`
- `llm_b`
- `formatted_data`
- `item_id`
- `name`
- `task_id`
- `task_type`
- `task_version`
- `snapshot_version`
- `english`
- `source_language`
- `target_language`
- `difficulty`
- `category`
- `tags`
- `slice_tags`
- `llm_a_low_context`
- `llm_a_ultra_low_context`
- `llm_a_temperature`
- `llm_a_generation_config`
- `llm_b_low_context`
- `llm_b_ultra_low_context`
- `llm_b_temperature`
- `llm_b_generation_config`

Compatibility notes:

- `pair_id_schema=v1` is the current JP-compatible identity contract: `md5(f"{file_a}_{file_b}_{name}")`.
- `pair_fingerprint` currently uses an explicit v1 payload contract, not an open-ended "all fields" hash. The v1 payload includes `llm_a`, `llm_b`, `formatted_data`, `name`, `english`, `difficulty`, and any side-specific `llm_a_*` / `llm_b_*` metadata present on the pair row.
- Additive task metadata introduced later must not silently change v1 fingerprints. If a future field needs to participate in reuse identity, that requires an explicit fingerprint-contract revision instead of widening the v1 hash implicitly.
- Pair matching now prefers `item_id` and falls back to `name` for legacy artifacts.
- Pair generation now requires strict equality across all task-defining item fields before a pair is emitted, including source text, direction, and any task slice tags present on the rows.
- Frozen JP v1.0 pair files remain on disk in their legacy pre-fingerprint form; the comparer computes missing contract metadata on load, and any rerun exports use `.schema-v2` side files instead of mutating the frozen artifacts.
- `formatted_data` is translation-specific and uses fixed `Translation A` / `Translation B` headings.

## Judgment Rows

Current judgment rows carry:

- `item_id`
- `name`
- `english`
- `difficulty`
- `task_id`
- `task_type`
- `task_version`
- `source_language`
- `target_language`
- `id`
- `pair_id_schema`
- `pair_fingerprint`
- `judge_profile_id`
- `compare_prompt_profile_id`
- `judge_parser_id`
- `judge_contract_id`
- `snapshot_version`
- `llm_a`
- `llm_b`
- `formatted_data`
- `analysis`
- `judge_model`
- `judge_temperature`
- `judge_generation_config`
- `llm_a_low_context`
- `llm_a_ultra_low_context`
- `llm_a_temperature`
- `llm_a_generation_config`
- `llm_b_low_context`
- `llm_b_ultra_low_context`
- `llm_b_temperature`
- `llm_b_generation_config`

Compatibility notes:

- Side-specific metadata must move with the corresponding candidate when A/B positions are randomized.
- Explicit task and direction fields now travel with judgment rows so scoring can operate without relying on `english`.
- Judge rows now carry the resolved compare-prompt profile and parser contract so reuse can reject stale judgments produced under a different prompt/judge contract.
- Existing frozen JP v1.0 judgment files remain the legacy compatibility boundary and are grandfathered without `pair_fingerprint`.
- Schema-versioned reruns may normalize legacy reused rows by attaching `pair_id_schema` / `pair_fingerprint` in new `.schema-v2` outputs while leaving the frozen files untouched.

## Score Summary Rows

Current score summaries carry:

- `model`
- `judge_model`
- `baseset_version`
- `pairs_file`
- `judgments_file`
- `timestamp_utc`
- `expected_pairs`
- `judged_pairs`
- `missing_pairs`
- `base_comparisons`
- `en_ja`
- `ja_en`

Compatibility notes:

- `en_ja` and `ja_en` are legacy direction buckets and will need a compatibility layer during task generalization.
- Score summaries are now populated from task-config direction keys; JP v1 preserves `en_ja` / `ja_en`, while newer tasks can use keys like `zh_en` / `en_zh`.
- Each bucket currently contains `overall`, `easy`, and `hard` slices with `difficulty`, `language`, `lt`, `wins`, `total`, and `win_rate`.
