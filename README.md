# Shisa-RP-Bench Readme 

## Results save location: 
scores/scores.jsonl

## How To Run RP Bench

### With Default Model (Athene-v2 at its alias)
MODEL="meta-llama/Llama-3.3-70B-Instruct" OPENAI_URL="http://llama33/v1" ./run_translation_bench.sh

### With Specified Judge (If you don't want to use Athene-v2)

MODEL="meta-llama/Llama-3.3-70B-Instruct" OPENAI_URL="http://llama33/v1" JUDGE_MODEL="Nexusflow/Athene-V2-Chat" JUDGE_URL="http://athenev2/v1" ./run_translation_bench.sh

### With Low Context (If you need 8192 Context Length)
LOW_CONTEXT="true" MODEL="meta-llama/Llama-3.3-70B-Instruct" OPENAI_URL="http://llama33/v1" ./run_translation_bench.sh

Breakdown:
MODEL = Model name in standard format. 
OPENAI_URL = URL used by the model. (Probably localhost:8000/v1 in most cases, I assume)
JUDGE_NAME = Name of the judge model.  Defaults to Athene. 
JUDGE_URL = Judge API URL. I recommend using only Athene, as Tulu/Llama3.3 have given extremely spotty results. 
LOW_CONTEXT= Forces the model to use 8192 context window. Off by default. This will cause the model to use a simpler prompt. 



# What It Does
1. Has the model translate 20 Japanese and 20 English sample texts.. This uses TQDM but does run in parallel.
2. Pairs them off with every other translation in the folder and saves that to latest_translation_pairs.jsonl.
3. Shows those convos to the judge AI for evaluation, who rates them across a number of criteria and picks a winner. Analyses are saved to the analysis folder.
4. Generates a ranking based off of analysis using choix_analyzer, and saves the results file to scores/scores.jsonl


# NOTES
1. The mamba environment used is "shisa-translation-bench", which is activated in the sh.





