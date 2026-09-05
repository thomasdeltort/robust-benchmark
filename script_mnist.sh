#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_BASE_PATH="./train_models/retrained_models"
#MODEL_BASE_PATH="./train_models/retrained_models"
OUTPUT_DIR="./results_lirpa_retrain_conventionnal_groupsort_debug"
mkdir -p "$OUTPUT_DIR"

# ConvLarge Models Only
MODELS=(

#"vanilla_ConvLarge_CIFAR10_1_LIP_GNP_cifar10_tau_a250.0_T15.0_bs256_lr0.0001_1781373330_TestAcc0.77_TestCRA69.19.pth"
#"vanilla_ConvLarge_CIFAR10_1_LIP_GNP_cifar10_tau_a250.0_T15.0_bs256_lr0.0001_1781498785_TestAcc0.77_TestCRA69.06.pth"
"vanilla_ConvLarge_CIFAR10_1_LIP_GNP_cifar10_tau_a250.0_T15.0_bs256_lr0.0001_1781499609_TestAcc0.76_TestCRA67.93.pth"
#"vanilla_ConvLarge_CIFAR10_1_LIP_GNP_cifar10_tau_a250.0_T15.0_bs256_lr0.0001_1781500413_TestAcc0.76_TestCRA67.84.pth"
#"vanilla_ConvLarge_CIFAR10_1_LIP_GNP_cifar10_tau_a250.0_T15.0_bs256_lr0.0001_1781501221_TestAcc0.77_TestCRA68.41.pth"
#
#
#"vanilla_ConvLarge_CIFAR10_1_LIP_GNP_cifar10_tau_a250.0_T0.5_bs256_lr0.0001_1781169300_TestAcc0.63_TestCRA62.31.pth"
"vanilla_ConvLarge_CIFAR10_1_LIP_GNP_cifar10_tau_a250.0_T0.5_bs256_lr0.0001_1781455385_TestAcc0.63_TestCRA62.54.pth"
#"vanilla_ConvLarge_CIFAR10_1_LIP_GNP_cifar10_tau_a250.0_T0.5_bs256_lr0.0001_1781456207_TestAcc0.63_TestCRA62.44.pth"
#"vanilla_ConvLarge_CIFAR10_1_LIP_GNP_cifar10_tau_a250.0_T0.5_bs256_lr0.0001_1781457029_TestAcc0.64_TestCRA62.82.pth"
#"vanilla_ConvLarge_CIFAR10_1_LIP_GNP_cifar10_tau_a250.0_T0.5_bs256_lr0.0001_1781457846_TestAcc0.64_TestCRA62.97.pth"

#    "vanilla_CNNA_CIFAR10_1_LIP_Bjork_cifar10_tau_a250.0_T25.0_bs128_lr0.002_1781186749_TestAcc0.73_TestCRA70.10.pth"
#    "vanilla_CNNA_CIFAR10_1_LIP_Bjork_cifar10_tau_a250.0_T25.0_bs128_lr0.002_1781190440_TestAcc0.73_TestCRA69.89.pth"
#    "vanilla_CNNA_CIFAR10_1_LIP_Bjork_cifar10_tau_a250.0_T25.0_bs128_lr0.002_1781194152_TestAcc0.72_TestCRA69.30.pth"
#    "vanilla_CNNA_CIFAR10_1_LIP_Bjork_cifar10_tau_a250.0_T25.0_bs128_lr0.002_1781197855_TestAcc0.72_TestCRA68.71.pth"
#    "vanilla_CNNA_CIFAR10_1_LIP_Bjork_cifar10_tau_a250.0_T25.0_bs128_lr0.002_eps0.01_medium_1765221236_acc0.75.pth"
    
#    "vanilla_CNNA_CIFAR10_1_LIP_Bjork_cifar10_tau_a250.0_T1.0_bs256_lr0.0003_1779880544_TestAcc0.64_TestCRA62.71.pth"
#    "vanilla_CNNA_CIFAR10_1_LIP_Bjork_cifar10_tau_a250.0_T1.0_bs256_lr0.0003_1779881747_TestAcc0.63_TestCRA61.78.pth"
#    "vanilla_CNNA_CIFAR10_1_LIP_Bjork_cifar10_tau_a250.0_T1.0_bs256_lr0.0003_1779882984_TestAcc0.63_TestCRA61.62.pth"
#    "vanilla_CNNA_CIFAR10_1_LIP_Bjork_cifar10_tau_a250.0_T1.0_bs256_lr0.0003_1779884232_TestAcc0.63_TestCRA61.44.pth"
#    
#    "vanilla_CNNA_CIFAR10_1_LIP_Bjork_cifar10_tau_a250.0_T30.0_bs256_lr0.0003_1779885447_TestAcc0.70_TestCRA48.46.pth"
#    "vanilla_CNNA_CIFAR10_1_LIP_Bjork_cifar10_tau_a250.0_T30.0_bs256_lr0.0003_1779886626_TestAcc0.71_TestCRA50.20.pth"
#    "vanilla_CNNA_CIFAR10_1_LIP_Bjork_cifar10_tau_a250.0_T30.0_bs256_lr0.0003_1779887846_TestAcc0.69_TestCRA47.57.pth"
#    "vanilla_CNNA_CIFAR10_1_LIP_Bjork_cifar10_tau_a250.0_T30.0_bs256_lr0.0003_1779889061_TestAcc0.70_TestCRA49.83.pth"

    "vanilla_CNNA_CIFAR10_1_LIP_Bjork_cifar10_tau_a250.0_T1.0_bs256_lr0.0003_1776931197_acc0.64.pth"
    "vanilla_CNNA_CIFAR10_1_LIP_Bjork_cifar10_tau_a250.0_T30.0_bs256_lr0.0003_1776944362_acc0.70.pth"
    "vanilla_CNNA_CIFAR10_1_LIP_GNP_cifar10_tau_a250.0_T0.5_bs256_lr0.001_eps0.155_light-_1766953221_acc0.62.pth"
    "vanilla_CNNA_CIFAR10_1_LIP_GNP_cifar10_tau_a250.0_T9.1414_bs256_lr0.0003_1776948490_acc0.60.pth"
    
#    "vanilla_CNNA_CIFAR10_1_LIP_GNP_cifar10_tau_a250.0_T0.5_bs256_lr0.001_1779890271_TestAcc0.60_TestCRA56.43.pth"
#    "vanilla_CNNA_CIFAR10_1_LIP_GNP_cifar10_tau_a250.0_T0.5_bs256_lr0.001_1779891515_TestAcc0.59_TestCRA56.03.pth"
#    "vanilla_CNNA_CIFAR10_1_LIP_GNP_cifar10_tau_a250.0_T0.5_bs256_lr0.001_1779892714_TestAcc0.60_TestCRA56.50.pth"
#    "vanilla_CNNA_CIFAR10_1_LIP_GNP_cifar10_tau_a250.0_T0.5_bs256_lr0.001_1779893940_TestAcc0.59_TestCRA56.00.pth"
#    
#    "vanilla_CNNA_CIFAR10_1_LIP_GNP_cifar10_tau_a250.0_T9.1414_bs256_lr0.0003_1779895166_TestAcc0.57_TestCRA49.60.pth"
#    "vanilla_CNNA_CIFAR10_1_LIP_GNP_cifar10_tau_a250.0_T9.1414_bs256_lr0.0003_1779896391_TestAcc0.59_TestCRA52.67.pth"
#    "vanilla_CNNA_CIFAR10_1_LIP_GNP_cifar10_tau_a250.0_T9.1414_bs256_lr0.0003_1779897596_TestAcc0.58_TestCRA50.64.pth"
#    "vanilla_CNNA_CIFAR10_1_LIP_GNP_cifar10_tau_a250.0_T9.1414_bs256_lr0.0003_1779898814_TestAcc0.57_TestCRA50.38.pth"
    
    ##########

# ATTENTION CHANGER LE REPO VOIR SCRIPT MNIST 2 !!!

##########
    
#    "vanilla_CNNA_CIFAR10_1_LIP_cifar10_tau_a250.0_T2.0_bs256_lr0.0003_1781625832_TestAcc0.65_TestCRA63.42.pth"
#    "vanilla_CNNA_CIFAR10_1_LIP_cifar10_tau_a250.0_T2.0_bs256_lr0.0003_1781781358_TestAcc0.65_TestCRA63.15.pth"
#    "vanilla_CNNA_CIFAR10_1_LIP_cifar10_tau_a250.0_T2.0_bs256_lr0.0003_1781782164_TestAcc0.66_TestCRA64.10.pth"
#    "vanilla_CNNA_CIFAR10_1_LIP_cifar10_tau_a250.0_T2.0_bs256_lr0.0003_1781782971_TestAcc0.66_TestCRA63.55.pth"
#    "vanilla_CNNA_CIFAR10_1_LIP_cifar10_tau_a250.0_T2.0_bs256_lr0.0003_1781783787_TestAcc0.65_TestCRA62.66.pth"
#    
#      
#    "vanilla_CNNA_CIFAR10_1_LIP_cifar10_tau_a250.0_T40.0_bs256_lr0.0003_1781629934_TestAcc0.69_TestCRA37.93.pth"
#    "vanilla_CNNA_CIFAR10_1_LIP_cifar10_tau_a250.0_T40.0_bs256_lr0.0003_1781787784_TestAcc0.70_TestCRA38.93.pth"
#    "vanilla_CNNA_CIFAR10_1_LIP_cifar10_tau_a250.0_T40.0_bs256_lr0.0003_1781788596_TestAcc0.69_TestCRA39.51.pth"
#    "vanilla_CNNA_CIFAR10_1_LIP_cifar10_tau_a250.0_T40.0_bs256_lr0.0003_1781789400_TestAcc0.71_TestCRA40.17.pth"
#    "vanilla_CNNA_CIFAR10_1_LIP_cifar10_tau_a250.0_T40.0_bs256_lr0.0003_1781790202_TestAcc0.70_TestCRA39.28.pth"



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
        --high_tau \
        --dataset 'cifar10' \
        --output_csv "$csv_name" \
        --start_step 0 \
        --batch_size 1 \
        --epsilon_max 0.031372 \
        --num_points 1 \
        --split_index 1 \
        --use_conventional_groupsort \
        --solvers_config '{"aa": True, "cra": True, "cra_pi": True, "alphacrown": True, "heavy_certified": False, "hybrid": True}' ; then
        echo "✅ Done. (Saved to $csv_name)"
    else
        echo "❌ Failed."
    fi
        if python main_auto.py \
        --model_path "$FULL_MODEL_PATH" \
        --model "$arch_name" \
        --high_tau \
        --dataset 'cifar10' \
        --output_csv "$csv_name" \
        --start_step 0 \
        --batch_size 1 \
        --epsilon_max 0.141176 \
        --num_points 1 \
        --split_index 1 \
        --use_conventional_groupsort \
        --solvers_config '{"aa": True, "cra": True, "cra_pi": True, "alphacrown": True, "heavy_certified": False, "hybrid": True}' ; then
        echo "✅ Done. (Saved to $csv_name)"
    else
        echo "❌ Failed."
    fi
done
echo "---------------------------------------------------"