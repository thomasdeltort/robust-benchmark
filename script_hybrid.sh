#!/bin/bash

# Configuration
SPLIT_IDX=8
SOLVERS="{'alphacrown': False, 'heavy_certified': False, 'hybrid': True}"

echo "================================================================="
echo "Starting Comprehensive Evaluation | Split Index: $SPLIT_IDX"
echo "================================================================="

# --- SECTION 2: ResNet18 Models ---

# 2.1 ResNet18 Bjork (High T)
python main_auto.py \
  --model_path "models/vanilla_ResNet18_1_LIP_Bjork_cifar10_tau_a250.0_T100.0_bs256_lr0.0003_1775518222_acc0.80.pth" \
  --model "ResNet18_1_LIP_Bjork" \
  --dataset cifar10 --split_index $SPLIT_IDX --batch_size 1 \
  --solvers_config "$SOLVERS" --output_csv results/resnet_bjork_T100_split${SPLIT_IDX}.csv

# 2.2 ResNet18 Bjork (Low T)
python main_auto.py \
  --model_path "models/vanilla_ResNet18_1_LIP_Bjork_cifar10_tau_a250.0_T5.0_bs256_lr0.0003_1775647600_acc0.73.pth" \
  --model "ResNet18_1_LIP_Bjork" \
  --dataset cifar10 --split_index $SPLIT_IDX --batch_size 1 \
  --solvers_config "$SOLVERS" --output_csv results/resnet_bjork_T5_split${SPLIT_IDX}.csv

# 2.3 ResNet18 GNP (High T)
python main_auto.py \
  --model_path "/models/vanilla_ResNet18_1_LIP_GNP_cifar10_tau_a250.0_T15.0_bs256_lr0.0003_1775510806_acc0.85.pth" \
  --model "ResNet18_1_LIP_GNP" \
  --dataset cifar10 --split_index $SPLIT_IDX --batch_size 1 \
  --solvers_config "$SOLVERS" --output_csv results/resnet_gnp_T15_split${SPLIT_IDX}.csv

# 2.4 ResNet18 GNP (Low T)
python main_auto.py \
  --model_path "models/vanilla_ResNet18_1_LIP_GNP_cifar10_tau_a250.0_T1.0_bs256_lr0.0003_1775506701_acc0.73.pth" \
  --model "ResNet18_1_LIP_GNP" \
  --dataset cifar10 --split_index $SPLIT_IDX --batch_size 1 \
  --solvers_config "$SOLVERS" --output_csv results/resnet_gnp_T1_split${SPLIT_IDX}.csv

echo "================================================================="
echo "All Evaluations Complete! Results in 'results/'"
echo "================================================================="