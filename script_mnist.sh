#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_BASE_PATH="./train_models/models_convlarge_retrained"
#MODEL_BASE_PATH="./train_models/retrained_models"
OUTPUT_DIR="./results_lirpa_retrain"
mkdir -p "$OUTPUT_DIR"

# ConvLarge Models Only
MODELS=(

#"vanilla_ConvLarge_CIFAR10_1_LIP_GNP_cifar10_tau_a250.0_T15.0_bs256_lr0.0001_1781373330_TestAcc0.77_TestCRA69.19.pth"
#"vanilla_ConvLarge_CIFAR10_1_LIP_GNP_cifar10_tau_a250.0_T15.0_bs256_lr0.0001_1781498785_TestAcc0.77_TestCRA69.06.pth"
#"vanilla_ConvLarge_CIFAR10_1_LIP_GNP_cifar10_tau_a250.0_T15.0_bs256_lr0.0001_1781499609_TestAcc0.76_TestCRA67.93.pth"
"vanilla_ConvLarge_CIFAR10_1_LIP_GNP_cifar10_tau_a250.0_T15.0_bs256_lr0.0001_1781500413_TestAcc0.76_TestCRA67.84.pth"
"vanilla_ConvLarge_CIFAR10_1_LIP_GNP_cifar10_tau_a250.0_T15.0_bs256_lr0.0001_1781501221_TestAcc0.77_TestCRA68.41.pth"


"vanilla_ConvLarge_CIFAR10_1_LIP_GNP_cifar10_tau_a250.0_T0.5_bs256_lr0.0001_1781169300_TestAcc0.63_TestCRA62.31.pth"
"vanilla_ConvLarge_CIFAR10_1_LIP_GNP_cifar10_tau_a250.0_T0.5_bs256_lr0.0001_1781455385_TestAcc0.63_TestCRA62.54.pth"
"vanilla_ConvLarge_CIFAR10_1_LIP_GNP_cifar10_tau_a250.0_T0.5_bs256_lr0.0001_1781456207_TestAcc0.63_TestCRA62.44.pth"
"vanilla_ConvLarge_CIFAR10_1_LIP_GNP_cifar10_tau_a250.0_T0.5_bs256_lr0.0001_1781457029_TestAcc0.64_TestCRA62.82.pth"
"vanilla_ConvLarge_CIFAR10_1_LIP_GNP_cifar10_tau_a250.0_T0.5_bs256_lr0.0001_1781457846_TestAcc0.64_TestCRA62.97.pth"

##########

# ATTENTION CHANGER LE REPO VOIR SCRIPT MNIST 2 !!!

##########


#"vanilla_ConvLarge_CIFAR10_1_LIP_cifar10_tau_a250.0_T2.0_bs256_lr0.0003_1781634791_TestAcc0.73_TestCRA70.92.pth"
#"vanilla_ConvLarge_CIFAR10_1_LIP_cifar10_tau_a250.0_T2.0_bs256_lr0.0003_1781822715_TestAcc0.67_TestCRA65.99.pth"
#"vanilla_ConvLarge_CIFAR10_1_LIP_cifar10_tau_a250.0_T2.0_bs256_lr0.0003_1781825085_TestAcc0.67_TestCRA65.22.pth"
#"vanilla_ConvLarge_CIFAR10_1_LIP_cifar10_tau_a250.0_T2.0_bs256_lr0.0003_1781827510_TestAcc0.67_TestCRA65.40.pth"
#"vanilla_ConvLarge_CIFAR10_1_LIP_cifar10_tau_a250.0_T2.0_bs256_lr0.0003_1781829864_TestAcc0.67_TestCRA65.30.pth"


)

echo "Starting ConvLarge batch processing..."
echo "---------------------------------------------------"

for model_file in "${MODELS[@]}"; do
    FULL_MODEL_PATH="${MODEL_BASE_PATH}/${model_file}"

    if [ ! -f "$FULL_MODEL_PATH" ]; then
        echo "❌ Error: File not found: $model_file"
        continue
    fi

    temp_name="${model_file#vanilla_}"
    arch_name="${temp_name%%_cifar10*}"
    
    csv_name="${OUTPUT_DIR}/new_experiment_${model_file%.pth}.csv"

    echo "Processing $arch_name ..."

    if python main_auto.py \
        --model_path "$FULL_MODEL_PATH" \
        --model "$arch_name" \
        --high_tau True \
        --dataset 'cifar10' \
        --output_csv "$csv_name" \
        --start_step 0 \
        --batch_size 1 \
        --epsilon_max 0.031372 \
        --num_points 1 \
        --split_index 1 \
        --solvers_config '{"aa": True, "cra": True, "cra_pi": False, "alphacrown": True, "heavy_certified": True, "hybrid": True}' ; then
        echo "✅ Done. (Saved to $csv_name)"
    else
        echo "❌ Failed."
    fi
        if python main_auto.py \
        --model_path "$FULL_MODEL_PATH" \
        --model "$arch_name" \
        --high_tau True \
        --dataset 'cifar10' \
        --output_csv "$csv_name" \
        --start_step 0 \
        --batch_size 1 \
        --epsilon_max 0.141176 \
        --num_points 1 \
        --split_index 1 \
        --solvers_config '{"aa": True, "cra": True, "cra_pi": False, "alphacrown": True, "heavy_certified": True, "hybrid": True}' ; then
        echo "✅ Done. (Saved to $csv_name)"
    else
        echo "❌ Failed."
    fi
done
echo "---------------------------------------------------"