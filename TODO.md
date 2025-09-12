### Finished tasks.
1. Add rich support for choix_analyzer.
2. Support multiple base sets.
3. Rewrite readme.md to reflect current usage.
4. Remove unneeded install script.
5. Fix hashing issue in generate_shootout_data.
6. Remove comments about prepending "base__" to model names in choix analyzer, we don't do that anymore.
7. Remove generate_base flag in generate_shootout_data.generate_translation_pairs, it's unused. (Generate base logic is handled at the main script level.)
8. Added bounded sempahor to translation_comparer_any_model.py
9. Added dotenv support (No more inferring from base name.), updated run_translation_bench.sh and readme to support new version.
10. Added better error messages if files were not found.
11. Remove hard-coded mention of athene in run_translation_bench.sh


### TODO
1. Allow easy adding to base set.
2. Change "low-context" and "ultra-low context" to using a "reasoning" flag.
3. Can we run the bench without like 30 environment variables?
4. Can we make the .sh understand if a script failed and error out?
5. Do we really need the bounded sempahor? 
6. Clear invalid answers from base set.

