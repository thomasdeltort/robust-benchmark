#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 1. Define the base path
MODEL_BASE_PATH="models"

# 2. Define the list of vanilla ConvLarge models
MODELS=(
     "vanilla_ConvLarge_CIFAR10_1_LIP_Bjork_cifar10_tau_a250.0_T1.0_bs128_lr0.001_eps0.01_heavy_1765374676_acc0.75.pth"
#     "vanilla_ConvLarge_CIFAR10_1_LIP_Bjork_cifar10_tau_a250.0_T400.0_bs256_lr0.00025_eps0.01_heavy_1765292395_acc0.85.pth"
    # "vanilla_ConvLarge_CIFAR10_1_LIP_cifar10_tau_a250.0_T10.0_bs256_lr0.001_eps0.155_light_1766739715_acc0.75.pth"
    # "vanilla_ConvLarge_CIFAR10_1_LIP_cifar10_tau_a250.0_T200.0_bs256_lr0.001_eps0.01_heavy_1765859376_acc0.84.pth"
#     "vanilla_ConvLarge_CIFAR10_1_LIP_GNP_cifar10_tau_1.0_lr0.001_eps0.155_1766393872_acc0.75.pth"
#     "vanilla_ConvLarge_CIFAR10_1_LIP_GNP_cifar10_tau_15.0_lr0.001_eps0.155_1766411827_acc0.82.pth"
)

echo "Starting batch processing l2 (ConvLarge Vanilla)..."
echo "---------------------------------------------------"

# 3. Loop through the array
for model_file in "${MODELS[@]}"; do

    FULL_MODEL_PATH="${MODEL_BASE_PATH}/${model_file}"

    # Check if file exists
    if [ ! -f "$FULL_MODEL_PATH" ]; then
        echo "❌ Error: File not found: $model_file"
        continue
    fi

    # Parsing logic
    # 1. Remove "vanilla_" prefix
    temp_name="${model_file#vanilla_}"
    
    # 2. Extract architecture by cutting at "_cifar10"
    # Example: "ConvLarge_CIFAR10_1_LIP_Bjork_cifar10..." -> "ConvLarge_CIFAR10_1_LIP_Bjork"
    arch_name="${temp_name%%_cifar10*}"
    
    csv_name="./results/new_experiment_${model_file%.pth}.csv"

    echo "Processing $arch_name ..."

    # Run command (Output visible)
    if python main_auto.py \
        --model_path "$FULL_MODEL_PATH" \
        --model "$arch_name" \
        --high_tau True \
        --dataset 'cifar10' \
        --output_csv "$csv_name" \
        --start_step 1\
        --batch_size 1 ; then
        # > /dev/null 2>&1
        echo "✅ Done. (Saved to $csv_name)"
    else
        echo "❌ Failed."
    fi

done

echo "---------------------------------------------------"
echo "All l2 experiments finished."


# echo "Starting batch processing linf (ConvLarge Vanilla)..."
# echo "---------------------------------------------------"
#
# # 3. Loop through the array
# for model_file in "${MODELS[@]}"; do
#
#     FULL_MODEL_PATH="${MODEL_BASE_PATH}/${model_file}"
#
#     if [ ! -f "$FULL_MODEL_PATH" ]; then
#         echo "❌ Error: File not found: $model_file"
#         continue
#     fi
#
#     # Parsing logic
#     temp_name="${model_file#vanilla_}"
#     arch_name="${temp_name%%_cifar10*}"
#    
#     csv_name="./results/new_experiment_${model_file%.pth}.csv"
#
#     echo "Processing $arch_name ..."
#
#     # Run command (Output visible)
#     if python main_auto.py \
#         --model_path "$FULL_MODEL_PATH" \
#         --model "$arch_name" \
#         --dataset 'cifar10' \
#         --output_csv "$csv_name" \
#         --norm 'inf'\
#         --batch_size 1 > /dev/null 2>&1; then
#         # > /dev/null 2>&1
#         echo "✅ Done. (Saved to $csv_name)"
#     else
#         echo "❌ Failed."
#     fi
#
# done
#
# echo "---------------------------------------------------"
# echo "All linf experiments finished."