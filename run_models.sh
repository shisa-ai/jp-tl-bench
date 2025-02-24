#!/bin/bash

# Array of models to evaluate
models=(
    #"allenai/Llama-3.1-Tulu-3-8B"
    #"Aratako/calm3-22b-RP-v2"
    #"augmxnt/shisa-7b-v1"
    #"cyberagent/Mistral-Nemo-Japanese-Instruct-2408"
    #"karakuri-ai/karakuri-lm-8x7b-instruct-v0.1"
    #"mistralai/Mistral-Small-24B-Instruct-2501"
    #"Deepreneur/blue-lizard"
    #"elyza/ELYZA-japanese-Llama-2-7b-instruct"
    #"elyza/Llama-3-ELYZA-JP-8B"
    #"meta-llama/Llama-3.1-8B-Instruct"
    #"meta-llama/Llama-3.3-70B-Instruct"
    #"micosoft/phi-4"
    #"Nexusflow/Athene-V2-Chat"
    #"shisa-ai/shisa-v1-llama3-70b"
    #"shisa-ai/Llama-3.1-Tulu-3-405B-FP8-Dynamic"
    "tokyotech-llm/Swallow-7b-instruct-v0.1"
    #"tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3"
    #"tokyotech-llm/Llama-3.1-Swallow-70B-Instruct-v0.3"
    #"umiyuki/Llama-3-Umievo-itr014-Shizuko-8b"
    #"SakanaAI/TinySwallow-1.5B-Instruct"
    #"weblab-GENIAC/Tanuki-8B-dpo-v1.0"
)

# Array of models that need ultra-low context
ultra_low_context_models=(
    "augmxnt/shisa-7b-v1"
    "augmxnt/shisa-gamma-7b-v1"
    "elyza/ELYZA-japanese-Llama-2-7b-instruct"
    "Deepreneur/blue-lizard"

)

# Run each model
for model in "${models[@]}"; do
    echo "Starting benchmark for model: $model"
    echo "----------------------------------------"
    
    # Check if this model needs ultra-low context
    if [[ " ${ultra_low_context_models[@]} " =~ " ${model} " ]]; then
        echo "Using ultra-low context (4096) for this model"
        MODEL="$model" ULTRA_LOW_CONTEXT="true" srun generate_conversation_data.slurm --ultra-low-context
    else
        echo "Using standard context settings"
        MODEL="$model" LOW_CONTEXT="true" srun generate_conversation_data.slurm --low-context
    fi
    
    echo "Completed benchmark for model: $model"
    echo "----------------------------------------"
    sleep 10
done
