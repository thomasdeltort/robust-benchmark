#!/bin/bash

# ==============================================================================
# MNIST Robustness Experiment Runner (Specific Checkpoints)
# ==============================================================================
# This script runs evaluation experiments on the specified vanilla model 
# checkpoints using L2 and Linf perturbations with various epsilon values.
# ==============================================================================

# 1. Define the Specific Vanilla Models (Checkpoints)
# ------------------------------------------------------------------------------
# These are the specific vanilla_ files provided in the list.
# Ensure these files exist in the directory specified by MODEL_DIR (below).
MODELS=(
    # --- MLP Models ---
    "vanilla_MLP_MNIST_1_LIP_Bjork_mnist_tau_a250.0_T0.06_bs64_lr0.0001_eps0.5_medium_1764685552_acc0.77.pth"
    "vanilla_MLP_MNIST_1_LIP_Bjork_mnist_tau_a250.0_T100.0_bs64_lr0.0001_eps0.5_medium_1764686681_acc0.96.pth"
    "vanilla_MLP_MNIST_1_LIP_GNP_mnist_tau_a250.0_T0.05_bs64_lr0.0001_eps0.5_medium_1764683996_acc0.77.pth"
    "vanilla_MLP_MNIST_1_LIP_mnist_tau_a250.0_T0.1_bs64_lr0.0001_eps0.5_medium_1764682272_acc0.80.pth"
    "vanilla_MLP_MNIST_1_LIP_mnist_tau_a250.0_T100.0_bs64_lr0.0001_eps0.5_medium_1764687420_acc0.97.pth"

    # --- ConvSmall Models ---
    "vanilla_ConvSmall_MNIST_1_LIP_Bjork_mnist_tau_a250.0_T0.2_bs64_lr0.0001_eps0.5_medium_1764757369_acc0.85.pth"
    "vanilla_ConvSmall_MNIST_1_LIP_Bjork_mnist_tau_a250.0_T0.3_bs64_lr0.0001_eps0.5_medium_1764768444_acc0.87.pth"
    "vanilla_ConvSmall_MNIST_1_LIP_Bjork_mnist_tau_a250.0_T100.0_bs64_lr0.0001_eps0.5_medium_1764769286_acc0.99.pth"
    "vanilla_ConvSmall_MNIST_1_LIP_GNP_mnist_tau_a250.0_T0.4_bs64_lr0.0001_eps0.5_medium_1764754178_acc0.86.pth"
    "vanilla_ConvSmall_MNIST_1_LIP_GNP_mnist_tau_a250.0_T100.0_bs64_lr0.0001_eps0.5_medium_1764770225_acc0.98.pth"
    "vanilla_ConvSmall_MNIST_1_LIP_mnist_tau_a250.0_T100.0_bs64_lr0.0001_eps0.5_medium_1764688998_acc0.99.pth"
    "vanilla_ConvSmall_MNIST_1_LIP_mnist_tau_a250.0_T1.0_bs64_lr0.0001_eps0.5_medium_1764690322_acc0.89.pth"
)

# Directory where models are stored. Change to "." if they are in the current dir.
MODEL_DIR="./train_models/models"

# 2. Define Epsilon Ranges for MNIST
# ------------------------------------------------------------------------------
# L_inf epsilons: typically small (0.05 to 0.5)
# L_2 epsilons: typically larger (0.1 to 7.0)
# L_inf: 0.005 to 1.0 (Dense steps)
EPSILONS_LINF=(
    0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10 \
    0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.20 \
    0.21 0.22 0.23 0.24 0.25 0.26 0.27 0.28 0.29 0.30 \
    0.31 0.32 0.33 0.34 0.35 0.36 0.37 0.38 0.39 0.40
)

# L_2: 40 values from 0.05 to 2.00 (Step 0.05) - Kept <= 2.0
EPSILONS_L2=(
    0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 \
    0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 1.00 \
    1.05 1.10 1.15 1.20 1.25 1.30 1.35 1.40 1.45 1.50 \
    1.55 1.60 1.65 1.70 1.75 1.80 1.85 1.90 1.95 2.00
)

# 3. Experiment Loop
# ------------------------------------------------------------------------------

echo "Starting Experiments on specific vanilla models..."
echo "Total models to process: ${#MODELS[@]}"
echo "----------------------------------------------------------------"

for model_file in "${MODELS[@]}"; do
    # 1. Create a clean folder name (remove .pth extension)
    model_name_clean=$(basename "$model_file" .pth)

    # 2. Extract the Python Function Name for --model
    # Step A: Remove the "vanilla_" prefix
    temp_name="${model_file#vanilla_}"
    # Step B: Remove the suffix starting at "_mnist" (the dataset/param section)
    # This leaves us with strings like "ConvSmall_MNIST_1_LIP_Bjork" or "MLP_MNIST_1_LIP"
    model_func_name="${temp_name%%_mnist*}"
    
    echo "Processing Checkpoint: $model_file"
    echo " > Detected Architecture: $model_func_name"

    # --- Run L_inf Experiments ---
    echo "  > Starting L_inf perturbations..."
    for eps in "${EPSILONS_LINF[@]}"; do
        echo "    Model: $model_name_clean | Norm: Linf | Epsilon: $eps"
        
        python main.py \
            --model "$model_func_name" \
            --dataset mnist \
            --model_path "${MODEL_DIR}/${model_file}" \
            --norm "inf" \
            --epsilon "$eps" \
            --output_csv "./results/${model_name_clean}_linf_${eps}.csv"
            
        if [ $? -ne 0 ]; then
            echo "    [ERROR] Experiment failed for $model_name_clean (Linf, eps=$eps)"
        fi
    done

    # --- Run L_2 Experiments ---
    echo "  > Starting L_2 perturbations..."
    for eps in "${EPSILONS_L2[@]}"; do
        echo "    Model: $model_name_clean | Norm: L2   | Epsilon: $eps"
        
        python main.py \
            --model "$model_func_name" \
            --dataset mnist \
            --model_path "${MODEL_DIR}/${model_file}" \
            --epsilon "$eps" \
            --save_dir "./results/${model_name_clean}_l2_${eps}.csv"

        if [ $? -ne 0 ]; then
            echo "    [ERROR] Experiment failed for $model_name_clean (L2, eps=$eps)"
        fi
    done

    echo "----------------------------------------------------------------"
done

echo "All experiments completed."