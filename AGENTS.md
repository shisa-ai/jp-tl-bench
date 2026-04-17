# JP-TL-Bench Agent Guide

This repository is an evaluation workspace for pairwise, model-judged language tasks. The current implementation is translation-focused, but agents should treat language pair, judge, prompt shape, and scoring details as configurable unless a task explicitly says otherwise.

## Summary

- Primary outcome: reproducible benchmark runs with versioned inputs, reusable expensive artifacts, and canonical result outputs.
- Run relevant commands inside the appropriate mamba environment for this workflow. If there is any ambiguity, prefer `mamba run -n <env> <command>` over assuming an interactive shell is already configured correctly.
- Treat repo-local documentation and current script help text as the source of truth for exact commands and paths. This guide is for working rules and invariants, not for freezing the implementation.

## Start Here

Read the subset that matches the task:

1. `README.md`
2. `baseset/README.md` if the task touches anchor snapshots, judged base sets, or scoring comparability
3. `PLAN-refactor.md` and `IMPLEMENTATION-refactor.md` if the task touches the ongoing generalization/refactor work
4. `prompts/` and the relevant script entrypoints if the task changes prompt contracts, parsing, or artifact layout

## Main Surfaces

These are the current workflow surfaces. Treat them as examples of the present pipeline, not permanent architecture.

| Path | Current role |
| --- | --- |
| `README.md` | Top-level usage, configuration, and workflow overview |
| `run_translation_bench.sh` | Current end-to-end orchestration example |
| `generate_translation_data.py` | Generates model outputs for the active task set |
| `generate_shootout_data.py` | Builds comparison pairs against the active snapshot/base set |
| `translation_comparer_any_model.py` | Produces or reuses pairwise judgments from the configured judge |
| `choix_analyzer.py` | Produces score/ranking artifacts from judged comparisons |
| `score_visualizer.py` | Displays leaderboard-style outputs from canonical score files |
| `baseset/` | Versioned snapshot inputs, reports, and helper tooling |
| `prompts/` | Prompt templates and output-shape expectations |

## Workflow Expectations

### Before Starting

- Run `git status -sb`.
- Identify which docs and scripts are actually relevant before editing.
- Confirm the appropriate mamba environment for the commands you plan to run.
- Check whether shared pipeline files are already being edited, especially `README.md`, `run_translation_bench.sh`, `baseset/README.md`, `prompts/*`, and the core pipeline scripts.

### During Work

- Keep changes scoped to one pipeline concern at a time: generation, pairing, judging, scoring, visualization, snapshotting, or documentation.
- Prefer neutral terms like `source language`, `target language`, `judge`, `snapshot`, `slice`, and `artifact` over hardcoded assumptions about one language pair.
- Keep judge-specific behavior isolated and explicit. Do not let one provider or one model silently define the generic path.
- For local generation via vLLM, prefer startup settings that target roughly `20` concurrent requests and about `12k` context length when the model, runtime, and VRAM allow it. Only fall back to lower concurrency or shorter context after confirming that model limits, vLLM validation, or environment constraints require it.
- Preserve reproducibility. A result should remain attributable to a specific snapshot/base-set version, test model, and judge configuration.
- Reuse expensive artifacts when possible. Do not casually force regeneration of judgments or other costly outputs unless the task requires it.
- If you change a prompt, parser, output schema, or artifact path, update the nearby docs and downstream consumers in the same change.
- Prefer additive, documented migrations over silent breakage of existing workflows.

### Before Claiming Done

- Run the narrowest relevant verification command in the appropriate mamba environment.
- Re-read the touched docs, script help text, or output paths if behavior or invocation changed.
- Review the diff for accidental hardcoded language-pair or judge assumptions.
- Leave unrelated local outputs and in-progress runs alone.

## Artifact Boundaries

- Treat versioned base snapshots under `baseset/` as reference artifacts, not casual scratch space.
- Treat generated directories such as `translations/`, `results/`, and similar run outputs as working artifacts unless a task explicitly promotes something into tracked documentation or snapshot data.
- Keep canonical result layouts stable enough that downstream analysis and visualization can discover them deterministically.
- Do not mix outputs from different snapshot versions or different judge configurations without making that distinction explicit in code, docs, and filenames.

## Prompt And Parsing Discipline

- Prompt files and response parsers are part of the benchmark contract.
- If a workflow depends on tagged output or a specific response shape, preserve that contract deliberately or version the change.
- When generalizing beyond the current task, avoid introducing prompt wording or parser logic that only works for one language direction unless that restriction is intentional and documented.

## Coordination Hygiene

- High-conflict files include `README.md`, `run_translation_bench.sh`, `baseset/README.md`, `prompts/*`, `generate_translation_data.py`, `generate_shootout_data.py`, `translation_comparer_any_model.py`, `choix_analyzer.py`, and `score_visualizer.py`.
- If two pieces of work need the same high-conflict file, coordinate before making broad edits.
- Do not overwrite or clean up another person's local benchmark outputs unless that is the explicit task.

## Meta

- Keep this file focused on stable working rules for the repository.
- When the pipeline changes, prefer updating the invariants and decision rules here rather than turning this document into an outdated command transcript.
- If a rule depends on a volatile implementation detail, document the principle and point readers to the current source-of-truth file for the exact command or path.
