#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

CSV_FILE="comparison_reformulation.csv"

echo "Results will be appended to: $CSV_FILE"

echo ""
echo ">>> Evaluating CNN-A Models <<<"

echo "Running CNN-A GNP (Low Tau T=0.5)..."
python compare_reformulations_paper.py \
    --model CNNA_CIFAR10_1_LIP_GNP \
    --dataset cifar10 \
    --model_path "models/vanilla_CNNA_CIFAR10_1_LIP_GNP_cifar10_tau_a250.0_T0.5_bs256_lr0.001_eps0.155_light-_1766953221_acc0.62.pth" \
    --csv_name $CSV_FILE

echo "Running ConvSmall GNP (Low Tau T=0.4)..."
python compare_reformulations_paper.py \
    --model ConvSmall_MNIST_1_LIP_GNP \
    --dataset mnist \
    --model_path "models_paper/vanilla_ConvSmall_MNIST_1_LIP_GNP_mnist_tau_a250.0_T0.4_bs64_lr0.001_eps0.5_medium_1765202236_acc0.91.pth" \
    --csv_name $CSV_FILE
