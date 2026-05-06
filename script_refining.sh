#!/bin/bash

# Usage: ./run_full_pipe.sh <folder_containing_csvs>

TARGET_DIR="$1"
MODELS_DIR="./models"

for csv_file in "$TARGET_DIR"/*.csv; do
    [ -e "$csv_file" ] || continue
    filename=$(basename "$csv_file")

    # Extract info from filename
    [[ "$filename" == *"_mnist_"* ]] && DATASET="mnist" || DATASET="cifar10"
    [[ "$filename" == *"norm_inf"* ]] && NORM="inf" || NORM="2"

    body="${filename#new_experiment_}"
    if [ "$NORM" == "inf" ]; then
        body=$(echo "$body" | sed 's/_norm_inf.csv//')
    else
        body=$(echo "$body" | sed 's/_norm_2.csv//')
    fi

    MODEL_PATH="$MODELS_DIR/$body.pth"
    temp_arch="${body#vanilla_}"
    MODEL_ARCH="${temp_arch%_${DATASET}_*}"

    python evaluate_robustness.py \
        --csv "$csv_file" \
        --model "$MODEL_ARCH" \
        --model_path "$MODEL_PATH" \
        --dataset "$DATASET" \
        --norm "$NORM"
done