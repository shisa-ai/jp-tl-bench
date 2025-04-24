#!/bin/bash

# Array of model names to test
MODELS=(
  "/fsx2/outputs/ablation-163-shisav2.if50-shisa-v2-llama-3.1-8b"
  "/fsx2/outputs/ablation-164-shisav2.if100-shisa-v2-llama-3.1-8b"
  "/fsx2/outputs/ablation-167-finaldpo.3e7-shisa-v2-unphi-4-14b"
  "/fsx2/outputs/ablation-168-a163.dpo.finaldpo.if25-shisa-v2-llama-3.1-8b"
  "/fsx2/outputs/ablation-170-a174.dpo.finaldpo.if50.pl25-shisa-v2-llama-3.1-8b"
  "/fsx2/outputs/ablation-171-a174.dpo.finaldpo.if50.pl50-shisa-v2-llama-3.1-8b"
  "/fsx2/outputs/ablation-172-a174.dpo.finaldpo.if50.pl100-shisa-v2-llama-3.1-8b"
  "/fsx2/outputs/ablation-174-shisav2.if50.tltweak-shisa-v2-llama-3.1-8b"
  "/fsx2/outputs/ablation-175-finalsft2-shisa-v2-llama-3.1-8b"
  "/fsx2/outputs/ablation-176-a175.finaldpo2-shisa-v2-llama-3.1-8b"
  "/fsx2/outputs/ablation-177-finalsft2-shisa-v2-mistral-nemo-12b"
  "/fsx2/outputs/ablation-178-finalsft2-shisa-v2-mistral-nemo-japanese-12b"
  "/fsx2/outputs/ablation-179-finalsft2-shisa-v2-unphi-4-14b"
  "/fsx2/outputs/ablation-180-finalsft2-shisa-v2-llama-3.3-70b"
  "/fsx2/outputs/ablation-182-a177.finaldpo2.6.5e7-shisa-v2-mistral-nemo-12b"
  "/fsx2/outputs/ablation-183-a177.finaldpo2.4.5e7-shisa-v2-mistral-nemo-12b"
  "/fsx2/outputs/ablation-184-a178.finaldpo2.6.5e7-shisa-v2-mistral-nemo-japanese-12b"
  "/fsx2/outputs/ablation-185-a178.finaldpo2.4.5e7-shisa-v2-mistral-nemo-japanese-12b"
)

# Loop through each model and run the translation bench
for model in "${MODELS[@]}"; do
  echo "Running translation bench for model: $model"
  
  full_model_path="$model"
  
  # Run the translation bench with the specified model
  MODEL="$full_model_path" JUDGE_MODEL="gemini/gemini-2.0-flash" CURATOR_CACHE_DIR=".cache" ./run_translation_bench.sh
  
  # Wait for 10 seconds before running the next model
  echo "Waiting 10 seconds before running the next model..."
  sleep 10
done

echo "All models have been processed." 
