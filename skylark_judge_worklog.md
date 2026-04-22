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
