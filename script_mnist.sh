#!/bin/bash

# 1. Define the base path
MODEL_BASE_PATH="models"

# 2. Define the list of CNNA models
MODELS=(
    "vanilla_CNNA_CIFAR10_1_LIP_cifar10_tau_a250.0_T100.0_bs256_lr0.001_eps0.155_light_1766942854_acc0.73.pth"
    
    # "vanilla_CNNA_CIFAR10_1_LIP_Bjork_cifar10_tau_a250.0_T0.5_bs256_lr0.001_eps0.155_light_1767607351_acc0.62.pth"

    # "vanilla_CNNA_CIFAR10_1_LIP_Bjork_cifar10_tau_a250.0_T25.0_bs128_lr0.002_eps0.01_medium_1765221236_acc0.75.pth"
    # "vanilla_CNNA_CIFAR10_1_LIP_GNP_cifar10_tau_a250.0_T0.5_bs256_lr0.001_eps0.155_light-_1766953221_acc0.62.pth"
    # "vanilla_CNNA_CIFAR10_1_LIP_cifar10_tau_a250.0_T1.5_bs256_lr0.001_eps0.155_light_1766940990_acc0.63.pth"
    # "vanilla_CNNA_CIFAR10_1_LIP_GNP_cifar10_tau_a250.0_T6.0_bs256_lr0.001_eps0.155_light-_1767608609_acc0.69.pth"


)

echo "Starting batch processing l2 (CNNA CIFAR10)..."
echo "---------------------------------------------------"

# 3. Loop through the array
for model_file in "${MODELS[@]}"; do

    FULL_MODEL_PATH="${MODEL_BASE_PATH}/${model_file}"

    # Check if file exists
    if [ ! -f "$FULL_MODEL_PATH" ]; then
        echo "❌ Error: File not found: $model_file"
        continue
    fi

    # --- FIX IS HERE ---
    # 1. Remove "vanilla_" prefix if it exists
    temp_name="${model_file#vanilla_}"
    
    # 2. Extract architecture by removing everything starting from "_cifar10" (lowercase)
    # This transforms: "CNNA_CIFAR10_1_LIP_Bjork_cifar10_..." 
    # Into: "CNNA_CIFAR10_1_LIP_Bjork" (which is a valid key in your model_zoo)
    arch_name="${temp_name%%_cifar10*}"
    
    csv_name="./results_fixed_bs/new_experiment_${model_file%.pth}.csv"

    echo "Processing $arch_name ..."

    # Run the command with output visible
    if python main_auto.py \
        --model_path "$FULL_MODEL_PATH" \
        --model "$arch_name" \
        --dataset 'cifar10' \
        --high_tau True \
        --output_csv "$csv_name" \
        --batch_size 2 ; then

        # > /dev/null 2>&1 
        echo "✅ Done. (Saved to $csv_name)"
    else
        echo "❌ Failed."
    fi

done

echo "---------------------------------------------------"
echo "All l2 experiments finished."

# echo "Starting batch processing l2 (CNNA CIFAR10)..."
# echo "---------------------------------------------------"

# # 3. Loop through the array
# for model_file in "${MODELS[@]}"; do

#     FULL_MODEL_PATH="${MODEL_BASE_PATH}/${model_file}"

#     # Check if file exists
#     if [ ! -f "$FULL_MODEL_PATH" ]; then
#         echo "❌ Error: File not found: $model_file"
#         continue
#     fi

#     # --- FIX IS HERE ---
#     # 1. Remove "vanilla_" prefix if it exists
#     temp_name="${model_file#vanilla_}"
    
#     # 2. Extract architecture by removing everything starting from "_cifar10" (lowercase)
#     # This transforms: "CNNA_CIFAR10_1_LIP_Bjork_cifar10_..." 
#     # Into: "CNNA_CIFAR10_1_LIP_Bjork" (which is a valid key in your model_zoo)
#     arch_name="${temp_name%%_cifar10*}"
    
#     csv_name="./results_augmentedlr_custom/new_experiment_${model_file%.pth}.csv"

#     echo "Processing $arch_name ..."

#     # Run the command with output visible
#     if python main_auto.py \
#         --model_path "$FULL_MODEL_PATH" \
#         --model "$arch_name" \
#         --dataset 'cifar10' \
#         --output_csv "$csv_name" \
#         --batch_size 2 ; then
#         # > /dev/null 2>&1 
#         echo "✅ Done. (Saved to $csv_name)"
#     else
#         echo "❌ Failed."
#     fi

# done

# echo "---------------------------------------------------"
# echo "All l2 experiments finished."




# echo "Starting batch processing linf (CNNA CIFAR10)..."
# echo "---------------------------------------------------"

# # 3. Loop through the array
# for model_file in "${MODELS[@]}"; do

#     FULL_MODEL_PATH="${MODEL_BASE_PATH}/${model_file}"

#     if [ ! -f "$FULL_MODEL_PATH" ]; then
#         echo "❌ Error: File not found: $model_file"
#         continue
#     fi

#     # Same parsing logic fix here
#     temp_name="${model_file#vanilla_}"
#     arch_name="${temp_name%%_cifar10*}"
    
#     csv_name="./results_newmodel_sdp_fix/new_experiment_${model_file%.pth}.csv"

#     echo "Processing $arch_name ..."

#     if python main_auto.py \
#         --model_path "$FULL_MODEL_PATH" \
#         --model "$arch_name" \
#         --dataset 'cifar10' \
#         --output_csv "$csv_name" \
#         --norm 'inf'\
#         --batch_size 2 > /dev/null 2>&1; then
        
#         echo "✅ Done. (Saved to $csv_name)"
#     else
#         echo "❌ Failed."
#     fi

# done

# echo "---------------------------------------------------"
# echo "All linf experiments finished."