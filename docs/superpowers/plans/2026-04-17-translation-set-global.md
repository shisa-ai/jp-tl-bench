# Translation Set Global Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local, ready-to-publish `shisa-ai/bt_translation_set_global` export that preserves the JP set, adds the new Chinese task set, and supports a `language` + `metadata` row contract.

**Architecture:** Add a small dataset-builder module that converts the existing JP Hugging Face dataset and the curated Chinese source files into config-specific JSONL exports under a repo-shaped directory. Update task normalization so rows with `language` can be consumed without the legacy `english` boolean, then validate the exported files with focused tests and a CLI checker.

**Tech Stack:** Python, Hugging Face `datasets`, JSONL exports, pytest

---

### Task 1: Add Red Tests For The New Row Contract

**Files:**
- Modify: `tests/test_task_config.py`
- Create: `tests/test_translation_set_global.py`

- [ ] Add a task-config normalization test for rows that have `language` instead of `english`.
- [ ] Add dataset-builder tests for JP row conversion, Chinese row conversion, and local export validation.
- [ ] Run the targeted tests and confirm they fail for the missing functionality.

### Task 2: Teach Task Normalization About `language`

**Files:**
- Modify: `benchmark_tasks/__init__.py`

- [ ] Update direction resolution so a row with `language` can be treated as `source_language`.
- [ ] Infer the target direction from task config when the source language uniquely matches one direction.
- [ ] Keep legacy `english` support intact for old rows.

### Task 3: Build The New Dataset Exporter

**Files:**
- Create: `dataset_tools/translation_set_global.py`
- Create: `scripts/build_translation_set_global.py`

- [ ] Convert `shisa-ai/bt_translation_test` JP rows into the new contract with `language` and `metadata`.
- [ ] Reuse the English-source subset for the ZH task config.
- [ ] Convert the curated Chinese text files into stable `zh_01` ... `zh_33` rows with extracted metadata.
- [ ] Write repo-shaped output under `hf_datasets/bt_translation_set_global/`.

### Task 4: Add Validation And Repo Metadata

**Files:**
- Create: `scripts/validate_hf_dataset.py`
- Create: `hf_datasets/bt_translation_set_global/README.md`

- [ ] Validate required columns, uniqueness, direction compatibility, and deterministic row counts.
- [ ] Add Hugging Face config metadata for `translation_ja_en_bidirectional_v1` and `translation_zh_en_bidirectional_v1`.

### Task 5: Verify End To End

**Files:**
- Verify: `hf_datasets/bt_translation_set_global/**/*`
- Verify: `tests/test_task_config.py`
- Verify: `tests/test_translation_set_global.py`

- [ ] Run targeted pytest for the new behavior.
- [ ] Run the dataset build CLI and the validation CLI.
- [ ] Confirm row counts and column shapes for both task configs.
