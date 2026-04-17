# JP v1 To Task Config Migration

This repo now treats `translation.ja-en` JP v1.0 as a frozen legacy boundary and uses task configs for all new work. The migration strategy is additive: existing JP files stay in place, and new metadata lives in sidecars or schema-versioned reruns.

## JP v1.0 Boundary

- Frozen snapshot artifacts under `baseset/v1.0/` stay byte-for-byte unchanged.
- Frozen candidate score files stay byte-for-byte unchanged.
- New metadata is written beside legacy files rather than into them.

Current additive metadata surfaces:

- `baseset/v1.0/legacy_jp_v1_boundary.json`
- `baseset/v1.0/reports/*.metadata.json` for refreshed score reports
- `results/<snapshot>/<model>/<judge>/scores.metadata.json` for migrated or newly generated candidate scores

The visualizer now loads `scores.metadata.json` automatically when it exists, but the score rows still come from the original `scores.json`. That preserves historical LT and win-rate values while making task/dataset provenance discoverable.

## Migrating Existing JP Result Files

Write an additive sidecar for an existing JP result file with:

```bash
mamba run -n shisa-jp-tl-bench python scripts/migrate_result_metadata.py \
  --task translation_ja_en_bidirectional_v1 \
  --scores-file results/v1.0/<model>/<judge>/scores.json
```

The helper derives metadata from:

- `translations/<safe_model>.jsonl`
- sibling `judgments.jsonl`
- sibling `pairs.jsonl`

It writes `scores.metadata.json` and leaves `scores.json` untouched.

## First Chinese Release Choice

The first Chinese release is explicitly defined as:

- Task config: `benchmark_tasks/translation_zh_en_bidirectional_v1.yaml`
- Judge profile: `judge_profiles/cn_judge.yaml`
- Compare prompt profile: `compare-cn-v1`
- Judge contract form: `<judge_model>::compare-cn-v1::answer-parser/v1`

That choice is now attributable in result metadata through:

- `judge_profile_id=cn_judge`
- `compare_prompt_profile_id=compare-cn-v1`
- `judge_contract_id=<judge>::compare-cn-v1::answer-parser/v1`

JP and CN scores remain separate by task metadata and snapshot version. A `translation.ja-en` run against `baseset/v1.0` is not comparable to a `translation.zh-en` run against a future Chinese snapshot, even if the judge model is the same.

## Chinese Dry-Run

On April 17, 2026, I ran a small end-to-end Chinese dry-run against a temporary local-export-backed task config:

- Dataset source: local `hf_datasets/bt_translation_set_global`
- Split: `train[:1]+train[34:35]`
- Covered directions: one `en -> zh` item (`batman1`) and one `zh -> en` item (`zh_01`)
- Anchor models: `gemini-2.0-flash`, `gemma-3-12b-it`
- Candidate model: `gemini-2.5-flash`
- Judge model: `gemini-2.5-flash`
- Judge profile: `cn_judge`

Observed outcome:

- Anchor round-robin judged pairs: `2/2`
- Candidate judged pairs: `4/4`
- Candidate score summary: `expected_pairs=4`, `judged_pairs=4`, `missing_pairs=0`
- Candidate result metadata recorded `task_id=translation.zh-en`, local dataset provenance, `judge_profile_id=cn_judge`, and `compare_prompt_profile_id=compare-cn-v1`

Operational note:

- The initial dry-run used explicit script-by-script commands because `HF_TOKEN` was unset and the temporary local-export-backed task config lived outside the checked-in task list.
- The current CLI now forwards `--task` and `--judge-profile` through the snapshot auto-judge path, and non-default judge profiles use profile-scoped result directories by default.

## Chinese Snapshot Plan

Proposed first real Chinese snapshot:

- Snapshot version: `baseset/zh-v1`
- Task config: `translation_zh_en_bidirectional_v1`
- Judge profile: `cn_judge`
- Judge model: `gemini-2.5-flash`
- Compare prompt profile: `compare-cn-v1`

Proposed anchor list:

1. `gemini-2.5-pro`
2. `gemini-2.5-flash`
3. `openai/gpt-4o`
4. `Qwen/Qwen3-30B-A3B-Instruct-2507`
5. `Qwen/Qwen3-4B`
6. `meta-llama/Llama-3.3-70B-Instruct`
7. `meta-llama/Llama-3.1-8B-Instruct`
8. `meta-llama/Llama-3.2-3B-Instruct`
9. `microsoft/phi-4`
10. `microsoft/Phi-4-mini-instruct`
11. `nvidia/NVIDIA-Nemotron-Nano-12B-v2`
12. `LiquidAI/LFM2-2.6B`

Required manifest fields:

- `snapshot_version`
- `task_id`
- `task_type`
- `task_version`
- `dataset_repo`
- `dataset_config`
- `dataset_split`
- `dataset_revision`
- `models`

Pair coverage target:

- Candidate-vs-anchor run: `12 anchors * 67 items = 804 judged pairs`
- Anchor round-robin: `C(12, 2) * 67 items = 4,422 judged pairs`

Expected cost:

- Candidate run with `gemini-2.5-flash` judge: roughly `$4-5` inferred from current JP v1 candidate-run cost scaling
- Anchor snapshot judging: roughly `$25-30` inferred from the same per-pair scaling
- Generation cost is separate and model-dependent

That plan keeps the first Chinese snapshot small enough to execute cheaply while still spanning a meaningful quality range.
