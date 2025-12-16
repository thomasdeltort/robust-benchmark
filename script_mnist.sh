#!/bin/bash

# ==============================================================================
# MNIST Robustness Experiment Runner (High Granularity)
# ==============================================================================
# This script runs evaluation experiments on the specified vanilla model 
# checkpoints using very dense L2 and Linf perturbation steps.
# ==============================================================================

# 1. Define the Specific Vanilla Models (Checkpoints)
# ------------------------------------------------------------------------------
# Only MNIST models are included here to match the --dataset mnist flag.
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

# Directory where models are stored.
MODEL_DIR="./models"

# 2. Define High-Granularity Epsilon Ranges
# ------------------------------------------------------------------------------
# We use `seq` to generate dense lists of floats.
# Format: seq START STEP END

# L_inf: 0.01 to 0.40 with step 0.002
# Example: 0.010, 0.012, 0.014 ... 0.400 (~196 values)
EPSILONS_LINF=($(seq -f "%.3f" 0.005  0.002 0.40))
# 0.005
# L_2: 0.05 to 2.00 with step 0.01
# Example: 0.05, 0.06, 0.07 ... 2.00 (~196 values)
EPSILONS_L2=($(seq -f "%.2f" 0.005  0.01 2.00))
# 0.005
# 3. Experiment Loop
# ------------------------------------------------------------------------------

echo "Starting High-Granularity Experiments..."
echo "Total models: ${#MODELS[@]}"
echo "L_inf density: ${#EPSILONS_LINF[@]} epsilons per model"
echo "L_2 density:   ${#EPSILONS_L2[@]} epsilons per model"
echo "----------------------------------------------------------------"

for model_file in "${MODELS[@]}"; do
    # 1. Clean filename
    model_name_clean=$(basename "$model_file" .pth)

    # 2. Extract Architecture
    temp_name="${model_file#vanilla_}"
    model_func_name="${temp_name%%_mnist*}"
    
    echo "Processing Checkpoint: $model_file"
    echo " > Architecture: $model_func_name"

    # --- Run L_inf Experiments ---
    echo "  > Starting L_inf perturbations..."
    for eps in "${EPSILONS_LINF[@]}"; do
        # Only echo status every 10 steps to reduce terminal spam
        # Remove the 'if' condition if you want to see every line
        # if (( $(echo "$eps * 1000" | bc | cut -d. -f1) % 20 == 0 )); then
            echo "    Model: $model_name_clean | Norm: Linf | Epsilon: $eps"
        # fi
        
        python main.py \
            --model "$model_func_name" \
            --dataset mnist \
            --model_path "${MODEL_DIR}/${model_file}" \
            --norm "inf" \
            --epsilon "$eps" \
            --output_csv "./results/${model_name_clean}_linf_.csv" > /dev/null 2>&1
            
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
            --output_csv "./results/${model_name_clean}_l2_.csv" > /dev/null 2>&1

        if [ $? -ne 0 ]; then
            echo "    [ERROR] Experiment failed for $model_name_clean (L2, eps=$eps)"
        fi
    done

    echo "----------------------------------------------------------------"
done

echo "All experiments completed."