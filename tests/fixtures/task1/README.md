# Task 1 Fixture Provenance

These fixtures are trimmed projections of tracked JP v1.0 artifacts. They keep the fields consumed by the Task 1 tests while avoiding copying large judgment analyses or full translation prompts into the test fixture set.

## Swap Fixture

- `swap_pair.json`
  Extracted from `results/v1.0/shisa-ai__chotto-e4b-20260408/gemini-2.5-flash/pairs.jsonl`
  Pair ID: `a3553cc5f4bcf5a532be903c1aaa6c9e`

## Translation Fixtures

- `translation_row_ok.json`
  Trimmed from `baseset/v1.0/translations/gemini-2.5-flash.jsonl`
  Item name: `simple_1`
- `translation_row_failed.json`
  Derived from the same `simple_1` translation row by replacing the translation text with the failure placeholder and adding `status=failed`

## Scoring Fixtures

- `scoring/base_set.gemini-2.5-flash.jsonl`
  Trimmed rows extracted from `baseset/v1.0/base_set.gemini-2.5-flash.jsonl`
  IDs:
  - `75dd71b664b85443802e10c75c333653`
  - `563d6b7ecde46074ba9dfe820c75718e`
  - `dcd18405ce30ba6858f8ad69530e66d0`
  - `6db727302cc159fa9b66f02155607e77`
- `scoring/judgments.jsonl`
  Trimmed rows extracted from `results/v1.0/shisa-ai__chotto-e4b-20260408/gemini-2.5-flash/judgments.jsonl`
  IDs:
  - `fbcca29407c3336b499ea94cbccaa8d0`
  - `346afb95c5bf0a60b4712709c7612820`
  - `3f6e391ff902f1cc079e761b862a1586`
  - `ab54c93ea87bd5facedc224626742ddb`
  - `0a6b5e5b67f4c099edfacc4c88824a53`
  - `e490f3281cae5e4b2be2781bb6793a56`
  - `b9b1eda6c88d707013f4c77e6eed543e`
  - `51c48c6e2966f7630dd4709e306d6888`
- `scoring/pairs.jsonl`
  Trimmed pair projections extracted from `results/v1.0/shisa-ai__chotto-e4b-20260408/gemini-2.5-flash/pairs.jsonl`
  Matching the same eight judgment IDs
- `scoring/expected_scores.json`
  Captured from running `build_score_summary()` over the extracted scoring sample in `mamba run -n shisa-jp-tl-bench`

## Tiny Snapshot Fixture

- `tiny_snapshot_pairs.jsonl`
  Extracted from `baseset/v1.0/base_conversation_pairs.v1.0.jsonl`
  Pair ID: `eb229a422afe582fac632930ec05b28e`
