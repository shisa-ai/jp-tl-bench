### Finished tasks.
1. Add rich support for choix_analyzer.
2. Support multiple base sets.
3. Rewrite readme.md to reflect current usage.
4. Remove unneeded install script.
5. Fix hashing issue in generate_shootout_data.
6. Remove comments about prepending "base__" to model names in choix analyzer, we don't do that anymore.
7. Remove generate_base flag in generate_shootout_data.generate_translation_pairs, it's unused. (Generate base logic is handled at the main script level.)
8. Added bounded sempahor to translation_comparer_any_model.py
9. Added dotenv support (No more inferring API keys from base ur.), updated run_translation_bench.sh and readme to support new version.
10. Added better error messages if files were not found.
11. Remove hard-coded mention of athene in run_translation_bench.sh
12. Added a cleaner for analysis files in case of bad xml.
13. Added pipefail so that run_translation_bench.sh will error out if any command fails.
14. Standardized command line argument names across files.
15. Make it so we save analysis files in "scores" instead of "analysis", saving a redundant copy that was also a common point of failure.
16. Renamed "analysis" to "base_sets".
17. Removed command line option to set special results directory for scores. It was redundant and not needed.

### TODO
1. Allow easy adding to base set.
2. Change "low-context" and "ultra-low context" to using a "reasoning" flag.
3. Record temp and other settings in JSONL.
4. Certain items in translation comparerer are not getting compared, should look at why. (Filters?)
