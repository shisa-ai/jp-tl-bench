# Shisa-Translation-Bench Readme 

# Results Save Location
/scores directory

## How To Run Translation Bench

### With Default Judge (Athene-v2 at its alias)
MODEL="meta-llama/Llama-3.3-70B-Instruct" OPENAI_URL="http://llama33/v1" ./run_translation_bench.sh

### With Specified Judge (If you don't want to use Athene-v2)
MODEL="meta-llama/Llama-3.3-70B-Instruct" OPENAI_URL="http://llama33/v1" JUDGE_MODEL="Nexusflow/Athene-V2-Chat" JUDGE_URL="http://athenev2/v1" ./run_translation_bench.sh

### With Low Context (If you need lower Context Length)
LOW_CONTEXT="true" MODEL="meta-llama/Llama-3.3-70B-Instruct" OPENAI_URL="http://llama33/v1" ./run_translation_bench.sh

Breakdown:
MODEL = Model name in standard format. 
OPENAI_URL = URL used by the model. (Probably localhost:8000/v1 in most cases)
JUDGE_MODEL = Name of the judge model. Defaults to Athene-v2. 
JUDGE_URL = Judge API URL. We recommend using only Athene-v2, as other models have given inconsistent results. 
LOW_CONTEXT = Forces the model to use 8192 context window. Off by default. This will cause the model to use a simpler prompt. 

# What It Does
1. Has the model translate approximately 70 Japanese and English conversation pairs. 
2. Pairs each translation with every other translation in the folder and saves that to latest_translation_pairs.jsonl.
3. Shows these translation pairs to the judge AI for evaluation, who rates them across several criteria and picks a winner. Analyses are saved to the analysis folder.
4. Generates a ranking based on the analysis using choix_analyzer, and saves the results file to scores/scores.jsonl

# NOTES
1. The mamba environment used is "shisa-jp-tl-bench", which is created by install.sh and activated in the scripts.
