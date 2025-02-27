#!/bin/bash

# Check for conda/mamba - prefer mamba
if command -v mamba &> /dev/null; then
    echo "Found mamba, using it..."
    CONDA_COMMAND="mamba"
elif command -v conda &> /dev/null; then
    echo "Found conda, using it..."
    CONDA_COMMAND="conda"
else
    echo "Error: Neither mamba nor conda found. Please install one of them."
    exit 1
fi

# Source the initialization
. "$(dirname $(which $CONDA_COMMAND))/../etc/profile.d/conda.sh"
. "$(dirname $(which $CONDA_COMMAND))/../etc/profile.d/mamba.sh" &> /dev/null


# Env setup/start
ENV_NAME="shisa-jp-tl-bench"
if $CONDA_COMMAND env list | grep -q "^$ENV_NAME "; then
    echo "Environment '$ENV_NAME' already exists, skipping creation."
else
    echo "Creating environment '$ENV_NAME'..."
    $CONDA_COMMAND create -n $ENV_NAME python=3.12 -y
fi
$CONDA_COMMAND activate $ENV_NAME

# Install
pip install uv
uv pip install -r requirements.txt
