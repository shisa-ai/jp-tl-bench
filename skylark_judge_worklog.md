# Skylark Judge Worklog

## 2026-04-22

- Added a BytePlus Ark Responses transport for `translation_comparer_any_model.py`.
- Exposed it through `--skylark-judge` and `JUDGE_TRANSPORT=skylark` in `run_translation_bench.sh`.
- Documented the `seed-2-0-pro-260328` judge invocation with `BYTEPLUS_TOKEN` and the Ark Responses endpoint.
- Added adapter coverage for request construction, `max_tokens` to `max_output_tokens` mapping, and response text/token parsing.
- Generated the `zh-ja-v1.0` Skylark base set:
  - `baseset/zh-ja-v1.0/base_set.seed-2-0-pro-260328.jsonl`
  - 1,035 / 1,035 judgments succeeded; one malformed answer-tag row was rejudged and replaced.
  - Final answer-tag validation: 487 A, 548 B, 0 missing.
  - Generated score report at `baseset/zh-ja-v1.0/reports/base_set.seed-2-0-pro-260328_scores.json`.
- Generated the larger `zh-ja-chinese-models-v1.0` Skylark base set:
  - `baseset/zh-ja-chinese-models-v1.0/base_set.seed-2-0-pro-260328.jsonl`
  - Seeded 690 overlapping judgments from `zh-ja-v1.0`, then judged the remaining 9,867 pairs with concurrency 10.
  - 9,867 / 9,867 new judgments succeeded; final merged total is 10,557.
  - Final answer-tag validation: 4,740 A, 5,817 B, 0 missing.
  - Generated score report at `baseset/zh-ja-chinese-models-v1.0/reports/base_set.seed-2-0-pro-260328_scores.json`.
- Ran `shisa-ai/chotto-e4b-20260408` against the larger Skylark base set:
  - Generated 1,242 pairs in `results/zh-ja-chinese-models-v1.0/shisa-ai__chotto-e4b-20260408/seed-2-0-pro-260328.cn_judge/pairs.jsonl`.
  - 1,242 / 1,242 judgments succeeded with `seed-2-0-pro-260328`; answer-tag validation: 556 A, 686 B, 0 missing.
  - Generated scores at `results/zh-ja-chinese-models-v1.0/shisa-ai__chotto-e4b-20260408/seed-2-0-pro-260328.cn_judge/scores.json`.
- Extended `zh-ja-chinese-models-v1.0` with four BytePlus/ModelArk anchor models:
  - Added `seed-2-0-mini-260215`, `seed-2-0-lite-260228`, `seed-2-0-pro-260328`, and `deepseek-v3-2-251201`.
  - Dropped `glm-4-7-251222` from this base-set update after the generation run proved too slow.
  - Regenerated the full base pair file to 15,939 rows and judged only the 5,382-row BytePlus delta in chunks.
  - Delta judging succeeded 5,382 / 5,382 with 0 failures; final merged base set has 15,939 rows.
  - Final answer-tag validation for `base_set.seed-2-0-pro-260328.jsonl`: 6,766 A, 9,173 B, 0 missing.
  - Regenerated `baseset/zh-ja-chinese-models-v1.0/reports/base_set.seed-2-0-pro-260328_scores.json`.
- Reran `shisa-ai/chotto-e4b-20260408` against the expanded 22-anchor Skylark base set:
  - Regenerated 1,518 pairs in `results/zh-ja-chinese-models-v1.0/shisa-ai__chotto-e4b-20260408/seed-2-0-pro-260328.cn_judge/pairs.jsonl`.
  - Reused 1,242 judgments and judged 276 new pairs; 276 / 276 new judgments succeeded with 0 failures.
  - Final answer-tag validation: 639 A, 879 B, 0 missing.
  - Generated scores at `results/zh-ja-chinese-models-v1.0/shisa-ai__chotto-e4b-20260408/seed-2-0-pro-260328.cn_judge/scores.json`.
