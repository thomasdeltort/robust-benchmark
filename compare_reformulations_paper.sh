#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

CSV_FILE="comparison_reformulation.csv"

echo "Starting ALPHA-CROWN evaluation for all models..."
echo "Results will be appended to: $CSV_FILE"



# # =========================================================
# # 2. CONVLARGE (CIFAR-10)
# # =========================================================
# echo ""
# echo ">>> Evaluating ConvLarge Models <<<"

# echo "Running ConvLarge GNP (Low Tau T=0.5)..."
# python compare_reformulations_paper.py \
#     --model ConvLarge_CIFAR10_1_LIP_GNP \
#     --dataset cifar10 \
#     --model_path "models_paper/vanilla_ConvLarge_CIFAR10_1_LIP_GNP_cifar10_tau_a250.0_T0.5_bs256_lr0.0003_1776927248_acc0.65.pth" \
#     --csv_name $CSV_FILE

# echo "Running ConvLarge GNP (High Tau T=15.0)..."
# python compare_reformulations_paper.py \
#     --model ConvLarge_CIFAR10_1_LIP_GNP \
#     --dataset cifar10 \
#     --model_path "models_paper/vanilla_ConvLarge_CIFAR10_1_LIP_GNP_cifar10_tau_15.0_lr0.001_eps0.155_1766411827_acc0.82.pth" \
#     --csv_name $CSV_FILE

# echo "Running ConvLarge Björck (Low Tau T=0.6)..."
# python compare_reformulations_paper.py \
#     --model ConvLarge_CIFAR10_1_LIP_Bjork \
#     --dataset cifar10 \
#     --model_path "models_paper/vanilla_ConvLarge_CIFAR10_1_LIP_Bjork_cifar10_tau_a250.0_T0.6_bs256_lr0.0003_1777019976_acc0.67.pth" \
#     --csv_name $CSV_FILE

# echo "Running ConvLarge Björck (High Tau T=100.0)..."
# python compare_reformulations_paper.py \
#     --model ConvLarge_CIFAR10_1_LIP_Bjork \
#     --dataset cifar10 \
#     --model_path "models_paper/vanilla_ConvLarge_CIFAR10_1_LIP_Bjork_cifar10_tau_a250.0_T100.0_bs256_lr0.0003_1777061308_acc0.81.pth" \
#     --csv_name $CSV_FILE
# =========================================================
# 1. CNN-A (CIFAR-10)
# =========================================================
echo ""
echo ">>> Evaluating CNN-A Models <<<"

echo "Running CNN-A GNP (Low Tau T=0.5)..."
python compare_reformulations_paper.py \
    --model CNNA_CIFAR10_1_LIP_GNP \
    --dataset cifar10 \
    --model_path "models_paper/vanilla_CNNA_CIFAR10_1_LIP_GNP_cifar10_tau_a250.0_T0.5_bs256_lr0.001_eps0.155_light-_1766953221_acc0.62.pth" \
    --csv_name $CSV_FILE

echo "Running CNN-A GNP (High Tau T=9.14)..."
python compare_reformulations_paper.py \
    --model CNNA_CIFAR10_1_LIP_GNP \
    --dataset cifar10 \
    --model_path "models_paper/vanilla_CNNA_CIFAR10_1_LIP_GNP_cifar10_tau_a250.0_T9.1414_bs256_lr0.0003_1776948490_acc0.60.pth" \
    --csv_name $CSV_FILE

echo "Running CNN-A Björck (Low Tau T=1.0)..."
python compare_reformulations_paper.py \
    --model CNNA_CIFAR10_1_LIP_Bjork \
    --dataset cifar10 \
    --model_path "models_paper/vanilla_CNNA_CIFAR10_1_LIP_Bjork_cifar10_tau_a250.0_T1.0_bs256_lr0.0003_1776931197_acc0.64.pth" \
    --csv_name $CSV_FILE

echo "Running CNN-A Björck (High Tau T=30.0)..."
python compare_reformulations_paper.py \
    --model CNNA_CIFAR10_1_LIP_Bjork \
    --dataset cifar10 \
    --model_path "models_paper/vanilla_CNNA_CIFAR10_1_LIP_Bjork_cifar10_tau_a250.0_T30.0_bs256_lr0.0003_1776944362_acc0.70.pth" \
    --csv_name $CSV_FILE





# =========================================================
# 3. CONVSMALL (MNIST)
# =========================================================
echo ""
echo ">>> Evaluating ConvSmall Models <<<"

echo "Running ConvSmall GNP (Low Tau T=0.4)..."
python compare_reformulations_paper.py \
    --model ConvSmall_MNIST_1_LIP_GNP \
    --dataset mnist \
    --model_path "models_paper/vanilla_ConvSmall_MNIST_1_LIP_GNP_mnist_tau_a250.0_T0.4_bs64_lr0.001_eps0.5_medium_1765202236_acc0.91.pth" \
    --csv_name $CSV_FILE

echo "Running ConvSmall GNP (High Tau T=100.0)..."
python compare_reformulations_paper.py \
    --model ConvSmall_MNIST_1_LIP_GNP \
    --dataset mnist \
    --model_path "models_paper/vanilla_ConvSmall_MNIST_1_LIP_GNP_mnist_tau_a250.0_T100.0_bs64_lr0.001_eps0.5_medium_1765208172_acc0.99.pth" \
    --csv_name $CSV_FILE

echo "Running ConvSmall Björck (Low Tau T=0.2)..."
python compare_reformulations_paper.py \
    --model ConvSmall_MNIST_1_LIP_Bjork \
    --dataset mnist \
    --model_path "models_paper/vanilla_ConvSmall_MNIST_1_LIP_Bjork_mnist_tau_a250.0_T0.2_bs64_lr0.001_eps0.5_medium_1765204811_acc0.87.pth" \
    --csv_name $CSV_FILE

echo "Running ConvSmall Björck (High Tau T=100.0)..."
python compare_reformulations_paper.py \
    --model ConvSmall_MNIST_1_LIP_Bjork \
    --dataset mnist \
    --model_path "models_paper/vanilla_ConvSmall_MNIST_1_LIP_Bjork_mnist_tau_a250.0_T100.0_bs64_lr0.001_eps0.5_medium_1765205989_acc0.99.pth" \
    --csv_name $CSV_FILE


# =========================================================
# 4. MLP (MNIST)
# =========================================================
echo ""
echo ">>> Evaluating MLP Models <<<"

echo "Running MLP GNP (Low Tau T=0.5225)..."
python compare_reformulations_paper.py \
    --model MLP_MNIST_1_LIP_GNP \
    --dataset mnist \
    --model_path "models_paper/vanilla_MLP_MNIST_1_LIP_GNP_mnist_tau_a250.0_T0.5225_bs128_lr0.0003_1776869005_acc0.92.pth" \
    --csv_name $CSV_FILE

echo "Running MLP GNP (High Tau T=100.0)..."
python compare_reformulations_paper.py \
    --model MLP_MNIST_1_LIP_GNP \
    --dataset mnist \
    --model_path "models_paper/vanilla_MLP_MNIST_1_LIP_GNP_mnist_tau_a250.0_T100.0_bs64_lr0.001_eps0.01_medium_1765206486_acc0.98.pth" \
    --csv_name $CSV_FILE


echo "All 18 ALPHA-CROWN runs completed! Results safely stored in $CSV_FILE."
