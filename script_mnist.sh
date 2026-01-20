#!/bin/bash

# 1. Define the base path
MODEL_BASE_PATH="/home/aws_install/robustess_project/Robust_Benchmark/models"

# 2. Define the list of models
MODELS=(
    # --- MLP Models ---
    "vanilla_MLP_MNIST_1_LIP_Bjork_mnist_tau_a250.0_T0.08_bs64_lr0.001_eps0.01_medium_1765204755_acc0.80.pth"
    "vanilla_MLP_MNIST_1_LIP_Bjork_mnist_tau_a250.0_T100.0_bs64_lr0.001_eps0.01_medium_1765205348_acc0.98.pth"
    "vanilla_MLP_MNIST_1_LIP_GNP_mnist_tau_a250.0_T0.09_bs64_lr0.001_eps0.01_medium_1765202951_acc0.82.pth"
    "vanilla_MLP_MNIST_1_LIP_GNP_mnist_tau_a250.0_T100.0_bs64_lr0.001_eps0.01_medium_1765206486_acc0.98.pth"
    "vanilla_MLP_MNIST_1_LIP_mnist_tau_a250.0_T0.1_bs64_lr0.001_eps0.01_medium_1765200291_acc0.80.pth"
    "vanilla_MLP_MNIST_1_LIP_mnist_tau_a250.0_T100.0_bs64_lr0.001_eps0.01_medium_1765206045_acc0.98.pth"

    # --- ConvSmall Models ---
    "vanilla_ConvSmall_MNIST_1_LIP_Bjork_mnist_tau_a250.0_T0.2_bs64_lr0.001_eps0.5_medium_1765204811_acc0.87.pth"
    "vanilla_ConvSmall_MNIST_1_LIP_Bjork_mnist_tau_a250.0_T100.0_bs64_lr0.001_eps0.5_medium_1765205989_acc0.99.pth"
    "vanilla_ConvSmall_MNIST_1_LIP_GNP_mnist_tau_a250.0_T0.4_bs64_lr0.001_eps0.5_medium_1765202236_acc0.91.pth"
    "vanilla_ConvSmall_MNIST_1_LIP_GNP_mnist_tau_a250.0_T100.0_bs64_lr0.001_eps0.5_medium_1765208172_acc0.99.pth"
    "vanilla_ConvSmall_MNIST_1_LIP_mnist_tau_a250.0_T0.4_bs64_lr0.001_eps0.5_medium_1765200992_acc0.86.pth"
    "vanilla_ConvSmall_MNIST_1_LIP_mnist_tau_a250.0_T100.0_bs64_lr0.001_eps0.5_medium_1765211561_acc0.99.pth"
)

echo "Starting batch processing l2..."
echo "---------------------------------------------------"

# 3. Loop through the array (L2)
for model_file in "${MODELS[@]}"; do

    FULL_MODEL_PATH="${MODEL_BASE_PATH}/${model_file}"

    # Check if file exists
    if [ ! -f "$FULL_MODEL_PATH" ]; then
        echo "❌ Error: File not found: $model_file"
        continue
    fi

    # Parse logic
    temp_name="${model_file#vanilla_}"
    arch_name="${temp_name%%_mnist*}"
    csv_name="./results2/new_experiment_${model_file%.pth}.csv"

    # Print a "working" message
    echo -n "Processing $arch_name ... "

    # FIX 1: Added 'if' before 'python'
    if python main_auto.py \
        --model_path "$FULL_MODEL_PATH" \
        --model "$arch_name" \
        --dataset 'mnist' \
        --output_csv "$csv_name" \
        --batch_size 4   > /dev/null 2>&1; then 
        # > /dev/null 2>&1
        echo "✅ Done. (Saved to $csv_name)"
    else
        echo "❌ Failed."
    fi

done

echo "---------------------------------------------------"
echo "All l2 experiments finished."


echo "Starting batch processing linf..."
echo "---------------------------------------------------"

# 4. Loop through the array (Linf)
for model_file in "${MODELS[@]}"; do

    FULL_MODEL_PATH="${MODEL_BASE_PATH}/${model_file}"

    if [ ! -f "$FULL_MODEL_PATH" ]; then
        echo "❌ Error: File not found: $model_file"
        continue
    fi

    temp_name="${model_file#vanilla_}"
    arch_name="${temp_name%%_mnist*}"
    csv_name="./results2/new_experiment_${model_file%.pth}.csv"

    echo -n "Processing $arch_name ... "

    # FIX 2: Fixed '---batch_size' to '--batch_size'
    if python main_auto.py \
        --model_path "$FULL_MODEL_PATH" \
        --model "$arch_name" \
        --dataset 'mnist' \
        --output_csv "$csv_name" \
        --norm 'inf' \
        --batch_size 4 > /dev/null 2>&1; then
        
        echo "✅ Done. (Saved to $csv_name)"
    else
        echo "❌ Failed."
    fi

done

echo "---------------------------------------------------"
echo "All linf experiments finished."