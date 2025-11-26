Plan: Stable LT/WR leaderboard vs baseset/v1.0 (parallel-safe, cost-aware)

Objectives
- Show LT/WR that are anchored to baseset/v1.0 only, per judge, per test model.
- Avoid paying for re-judging pairs that already exist; only fill gaps unless an explicit --rejudge is given.
- Produce a leaderboard/visualizer that cannot be polluted by other runs.
- After fixes, re-score all existing translations against v1.0 with no wasted API calls.

Problems observed today
- score_visualizer.py ingests every *_tl_bench_scores.jsonl and overwrites rows per model with whichever file was read last. Each file contains a full BT fit (base anchors + the current test model), so base models get different LT/win rates per run. Result: mismatched LT vs win rate and unstable numbers.
- choix_analyzer.py saves rankings for all models in the fit, so anchor rows from different runs collide in the visualizer.
- translation_comparer_any_model.py can re-run full judgments even when most pairs are already judged.

Proposed workflow (v1.0 anchored, parallel-safe)
1) Translations (unchanged): translations/<safe_model>.jsonl.
2) Pairs (per-model, no shared temp): generate_shootout_data.py --test-model <model> writes results/v1.0/<safe_model>/<safe_judge>/pairs.jsonl (or pairs/ if judge-agnostic). Idempotent; keep it for reproducibility/parallel runs.
3) Judging (reuse by default, optional rejudge): translation_comparer_any_model.py reads pairs.jsonl, loads existing judgments.jsonl if present, skips already-judged IDs, and only hits the judge for missing ones. --rejudge forces a full refresh. Output: results/v1.0/<safe_model>/<safe_judge>/judgments.jsonl (merged).
4) Scoring (anchored, target-only canonical): choix_analyzer.py consumes base_sets/base_set.<safe_judge>.jsonl + merged judgments, emits results/v1.0/<safe_model>/<safe_judge>/scores.jsonl containing only the tested model’s EN→JA/JA→EN easy/hard/all slices with metadata {baseset_version, judge_model, total_pairs, missing_pairs, timestamp}. Optionally emit rankings_full.jsonl for debug, kept out of the leaderboard glob.
5) Leaderboard/visualizer: glob only canonical scores under results/v1.0/*/<safe_judge>/scores.jsonl, validate baseset_version==v1.0 and missing_pairs==0, no cross-run overwrites. Warn/skip mismatches.

Data layout/invariants to enforce
- Results root: results/v1.0/<safe_model>/<safe_judge>/.
- pairs.jsonl: pairs vs anchors only; safe to regenerate; parallel-safe name.
- judgments.jsonl: merged A/B answers; reused unless --rejudge.
- scores.jsonl: canonical target-only slices + metadata.
- rankings_full.jsonl (optional): full BT ranking for debug only.
- Visualizer reads only results/v1.0/*/<safe_judge>/scores.jsonl.

Implementation tasks (ordered)
1) generate_shootout_data.py: allow output path override; default to results/v1.0/<safe_model>/<safe_judge>/pairs.jsonl (judge optional).
2) translation_comparer_any_model.py: add --rejudge; reuse existing judgments to skip already-judged IDs; write merged to results/.../judgments.jsonl; validate coverage vs pairs.
3) choix_analyzer.py: accept baseset_version/judge args; emit canonical target-only scores.jsonl with metadata; optionally rankings_full.jsonl for debug.
4) score_visualizer.py: read canonical scores only; enforce baseset_version==v1.0 and missing_pairs==0; no overwrite collisions.
5) run_translation_bench.sh (and docs): wire new paths; note reuse-by-default and --rejudge.
6) Optional helper: coverage/report script.
7) Migration: stop globbing old *_tl_bench_scores.jsonl; re-score existing translations via the new workflow with reuse.

Re-run plan after fixes
- For each model in translations/: generate pairs vs baseset/v1.0, run comparer (reuse by default), run analyzer to emit canonical scores.
- Then run the updated visualizer to produce a clean leaderboard for v1.0.
