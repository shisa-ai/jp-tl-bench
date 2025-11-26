Implementation Punchlist (baseset/v1.0 anchored, parallel-safe)

Scope
- New per-model result layout: results/v1.0/<safe_model>/<safe_judge>/{pairs.jsonl, judgments.jsonl, scores.jsonl, rankings_full.jsonl?}
- Reuse existing judgments by default; only rejudge missing pairs unless --rejudge.
- Canonical scores are target-only slices with metadata; visualizer consumes only these.

Tasks
1) generate_shootout_data.py
   - Add output path flag; default to results/v1.0/<safe_model>/<safe_judge>/pairs.jsonl (judge optional).
   - Ensure idempotent regeneration; no global temp.

2) translation_comparer_any_model.py
   - Add --rejudge to force full redo.
   - Load existing judgments.jsonl if present; build map id->record; skip already-judged IDs when not rejudging.
   - Validate coverage vs pairs.jsonl; error/warn on missing pairs; only call judge for missing IDs.
   - Write merged judgments.jsonl to results/v1.0/<safe_model>/<safe_judge>/.

3) choix_analyzer.py
   - Accept baseset_version (from env/flag; default v1.0).
   - Output canonical scores.jsonl (target-only: EN→JA/JA→EN easy/hard/all) with metadata {baseset_version, judge_model, total_pairs, missing_pairs, timestamp, test_model}.
   - Optionally emit rankings_full.jsonl (full BT rankings) in same folder; not used by leaderboard.

4) score_visualizer.py
   - Read only results/v1.0/*/<safe_judge>/scores.jsonl (or a configurable root).
   - Validate baseset_version==v1.0 and missing_pairs==0; skip/warn otherwise.
   - No overwrite collisions—one row per scores file.

5) run_translation_bench.sh + docs
   - Wire new paths (pairs/judgments/scores).
   - Mention reuse-by-default and --rejudge to force.

6) Optional helper
   - coverage/report script to show expected vs judged counts for a model/judge.

7) Migration/re-run
   - Stop globbing old *_tl_bench_scores.jsonl in visualizer.
   - Re-score existing translations using new flow (reuse judgments).
