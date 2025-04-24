#!/bin/bash

# Array of models to evaluate
models=(
#	"/fsx2/outputs/ablation-151-finalsft-shisa-v2-mistral-nemo-12b"
#	"/fsx2/outputs/ablation-152-finalsft-shisa-v2-mistral-nemo-japanese-12b"
#	"/fsx2/outputs/ablation-155-a151.dpofinal-shisa-v2-mistral-nemo-12b/"
#	"/fsx2/outputs/ablation-156-a152.dpofinal-shisa-v2-mistral-nemo-japanese-12b/"
#	"/fsx2/outputs/ablation-169-a163.dpo.finaldpo.if50-shisa-v2-llama-3.1-8b/"
	#"mistralai/Mistral-Nemo-Instruct-2407"
	#"mistralai/Mistral-Small-3.1-24B-Instruct-2503"
	#"cyberagent/Mistral-Nemo-Japanese-Instruct-2408"
	# "/fsx2/outputs/ablation-153-finalsft-shisa-v2-unphi-4-14b/"
	# "/fsx2/outputs/ablation-167-finaldpo.3e7-shisa-v2-unphi-4-14b/"

# /fsx2/outputs/ablation-175-finalsft2-shisa-v2-llama-3.1-8b
# /fsx2/outputs/ablation-176-a175.finaldpo2-shisa-v2-llama-3.1-8b
# /fsx2/outputs/ablation-177-finalsft2-shisa-v2-mistral-nemo-12b
# /fsx2/outputs/ablation-178-finalsft2-shisa-v2-mistral-nemo-japanese-12b
# /fsx2/outputs/ablation-179-finalsft2-shisa-v2-unphi-4-14b
# /fsx2/outputs/ablation-180-finalsft2-shisa-v2-llama-3.3-70b
# /fsx2/outputs/ablation-182-a177.finaldpo2.6.5e7-shisa-v2-mistral-nemo-12b
# /fsx2/outputs/ablation-183-a177.finaldpo2.4.5e7-shisa-v2-mistral-nemo-12b
)

# Array of models that need ultra-low context
ultra_low_context_models=(
#    "augmxnt/shisa-7b-v1"
    "augmxnt/shisa-gamma-7b-v1"
#    "elyza/ELYZA-japanese-Llama-2-7b-instruct"
#    "Deepreneur/blue-lizard"
)

long_context_models=(
#    "tokyotech-llm/Llama-3.1-Swallow-70B-Instruct-v0.3"
#    "mistralai/Mistral-Small-24B-Instruct-2501"
#    "karakuri-ai/karakuri-lm-8x7b-instruct-v0.1"
#    "mistralai/Mistral-Small-24B-Instruct-2501"
#    "microsoft/phi-4"
#    "shisa-ai/shisa-v1-llama3-70b"
#    "tokyotech-llm/Swallow-7b-instruct-v0.1"
)
# Run each model
for model in "${models[@]}"; do
    echo "Starting benchmark for model: $model"
    echo "----------------------------------------"
    
    # Check if this model needs ultra-low context
    if [[ " ${ultra_low_context_models[@]} " =~ " ${model} " ]]; then
        echo "Using ultra-low context (4096) for this model"
        MODEL="$model" ULTRA_LOW_CONTEXT="true" srun generate_translation_data.slurm --ultra-low-context
    # Check if this model supports long context
    elif [[ " ${long_context_models[@]} " =~ " ${model} " ]]; then
        echo "Using standard context settings without low-context flag"
        MODEL="$model" srun ggenerate_translation_data.slurm
    else
        echo "Using standard context settings"
        MODEL="$model" LOW_CONTEXT="true" srun generate_translation_data.slurm --low-context
    fi
    
    echo "Completed benchmark for model: $model"
    echo "----------------------------------------"
    sleep 10
done
