# HF Dataset Publishing

This repo now treats `shisa-ai/bt_translation_set_global` as the canonical private Hugging Face dataset for translation benchmark task items.

## Rules

- Existing published task configs are immutable once benchmark runs depend on them.
- Content fixes for an existing released config should happen only before the revision is locked to a Hub commit SHA.
- Additive benchmark growth should create a new config or task version instead of mutating a released one in place.
- Chinese source provenance must be refreshed together with the dataset export.

## Prerequisites

- `HF_TOKEN` with write access to `shisa-ai/bt_translation_set_global`
- `mamba run -n shisa-jp-tl-bench ...` for all export and validation commands

## Local Export

Rebuild the dataset export, manifest, and inventory:

```bash
mamba run -n shisa-jp-tl-bench python scripts/export_hf_dataset.py
```

This writes:

- `hf_datasets/bt_translation_set_global/data/translation_ja_en_bidirectional_v1/train.jsonl`
- `hf_datasets/bt_translation_set_global/data/translation_zh_en_bidirectional_v1/train.jsonl`
- `hf_datasets/bt_translation_set_global/data/translation_zh_ja_bidirectional_v1/train.jsonl`
- `hf_datasets/bt_translation_set_global/README.md`
- `docs/chinese_source_manifest.csv`
- `docs/translation_set_inventory.csv`

## Validation

Validate each task config against the exported folder before any push:

```bash
mamba run -n shisa-jp-tl-bench python scripts/validate_hf_dataset.py --task benchmark_tasks/translation_ja_en_bidirectional_v1.yaml
mamba run -n shisa-jp-tl-bench python scripts/validate_hf_dataset.py --task benchmark_tasks/translation_zh_en_bidirectional_v1.yaml
mamba run -n shisa-jp-tl-bench python scripts/validate_hf_dataset.py --task benchmark_tasks/translation_zh_ja_bidirectional_v1.yaml
```

The validator checks:

- required columns and deterministic `item_id` ordering
- expected row counts plus source-language and difficulty balance
- Chinese-source provenance manifest coverage and text alignment
- round-trip compatibility with task-aware pair generation

## Publish

Upload the local export folder to the private dataset repo and delete the legacy single-config `data/train.jsonl` path:

```bash
python - <<'PY'
import os
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
current = api.dataset_info("shisa-ai/bt_translation_set_global")
commit = api.upload_folder(
    repo_id="shisa-ai/bt_translation_set_global",
    repo_type="dataset",
    folder_path="hf_datasets/bt_translation_set_global",
    parent_commit=current.sha,
    delete_patterns="data/train.jsonl",
    commit_message="Publish versioned translation task configs",
)
print(commit.oid)
PY
```

Using `parent_commit` prevents silently overwriting concurrent changes.

## Lock Step

After a successful publish, resolve the returned dataset commit SHA and replace `dataset.revision` in every released task config with that SHA. Do not leave canonical task configs pointed at branch names or temporary publish labels.

The current checked-in release is locked to:

- `ead55791383dd96468692a8883b88af865416845`

The released task configs now point to that commit.

## Ownership Boundary

- `docs/chinese_source_manifest.csv` and the `docs/cn_texts*` audit files govern source provenance and should only be updated by the dataset curator.
- `benchmark_tasks/*.yaml` controls benchmark consumption and should only be updated when a task version is intentionally introduced or locked.
- Benchmark runs should consume released task configs, not in-progress local dataset exports.
