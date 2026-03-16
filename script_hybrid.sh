#!/bin/bash

# Ensure the results directory exists
mkdir -p results

# Define the split index here so it's easy to change later!
SPLIT_IDX=3


# EXPERIENCE 1 low vs high Tau


# echo "================================================================="
# echo "Starting Evaluation 1/4: Bjork Model (High Temp) | Split: $SPLIT_IDX"
# echo "================================================================="
# python main_auto.py \
#   --model_path /home/aws_install/robust-benchmark/models/vanilla_ConvLarge_CIFAR10_1_LIP_Bjork_cifar10_tau_a250.0_T400.0_bs256_lr0.00025_eps0.01_heavy_1765292395_acc0.85.pth \
#   --model "ConvLarge_CIFAR10_1_LIP_Bjork" \
#   --dataset cifar10 \
#   --split_index $SPLIT_IDX \
#   --batch_size 1 \
#   --high_tau True \
#   --output_csv results_highlowtau/bjork_T400_split${SPLIT_IDX}_study.csv

#   echo "================================================================="
# echo "Starting Evaluation 1/4: Bjork Model (High Temp) | Split: $SPLIT_IDX"
# echo "================================================================="
# python main_auto.py \
#   --model_path /home/aws_install/robust-benchmark/models/vanilla_ConvLarge_CIFAR10_1_LIP_Bjork_cifar10_tau_a250.0_T400.0_bs256_lr0.00025_eps0.01_heavy_1765292395_acc0.85.pth \
#   --model "ConvLarge_CIFAR10_1_LIP_Bjork" \
#   --dataset cifar10 \
#   --split_index $SPLIT_IDX \
#   --batch_size 1 \
#   --output_csv results_highlowtau/bjork_T400_split${SPLIT_IDX}_study.csv


# Experience 2 conventional sdp crown


# echo "================================================================="
# echo "Starting Evaluation 1/4: Bjork Model (High Temp) | Split: $SPLIT_IDX"
# echo "================================================================="
# python main_auto.py \
#   --model_path /home/aws_install/robust-benchmark/models/vanilla_ConvLarge_CIFAR10_1_LIP_Bjork_cifar10_tau_a250.0_T400.0_bs256_lr0.00025_eps0.01_heavy_1765292395_acc0.85.pth \
#   --model "ConvLarge_CIFAR10_1_LIP_Bjork" \
#   --dataset cifar10 \
#   --split_index $SPLIT_IDX \
#   --batch_size 1 \
#   --output_csv results/bjork_T400_split${SPLIT_IDX}_study.csv

# echo ""
# echo "================================================================="
# echo "Starting Evaluation 2/4: Bjork Model (Low Temp) | Split: $SPLIT_IDX"
# echo "================================================================="
# python main_auto.py \
#   --model_path /home/aws_install/robust-benchmark/models/vanilla_ConvLarge_CIFAR10_1_LIP_Bjork_cifar10_tau_a250.0_T1.0_bs128_lr0.001_eps0.01_heavy_1765374676_acc0.75.pth \
#   --model "ConvLarge_CIFAR10_1_LIP_Bjork" \
#   --dataset cifar10 \
#   --split_index $SPLIT_IDX \
#   --batch_size 1 \
#   --output_csv results/bjork_T1_split${SPLIT_IDX}_study.csv

# echo ""
# echo "================================================================="
# echo "Starting Evaluation 3/4: GNP Model (High Tau) | Split: $SPLIT_IDX"
# echo "================================================================="
# python main_auto.py \
#   --model_path /home/aws_install/robust-benchmark/models/vanilla_ConvLarge_CIFAR10_1_LIP_GNP_cifar10_tau_15.0_lr0.001_eps0.155_1766411827_acc0.82.pth \
#   --model "ConvLarge_CIFAR10_1_LIP_GNP" \
#   --dataset cifar10 \
#   --split_index $SPLIT_IDX \
#   --batch_size 1 \
#   --output_csv results/gnp_tau15_split${SPLIT_IDX}_study.csv

# echo ""
# echo "================================================================="
# echo "Starting Evaluation 4/4: GNP Model (Low Tau) | Split: $SPLIT_IDX"
# echo "================================================================="
# python main_auto.py \
#   --model_path /home/aws_install/robust-benchmark/models/vanilla_ConvLarge_CIFAR10_1_LIP_GNP_cifar10_tau_1.0_lr0.001_eps0.155_1766393872_acc0.75.pth \
#   --model "ConvLarge_CIFAR10_1_LIP_GNP" \
#   --dataset cifar10 \
#   --split_index $SPLIT_IDX \
#   --batch_size 1 \
#   --output_csv results/gnp_tau1_split${SPLIT_IDX}_study.csv

# echo ""
# echo "================================================================="
# echo "All Evaluations Complete! Results saved in the 'results/' folder."
# echo "================================================================="



# Experience 3 changing split index



# SPLIT_IDX=4

# echo "================================================================="
# echo "Starting Evaluation 1/4: Bjork Model (High Temp) | Split: $SPLIT_IDX"
# echo "================================================================="
# python main_auto.py \
#   --model_path /home/aws_install/robust-benchmark/models/vanilla_ConvLarge_CIFAR10_1_LIP_Bjork_cifar10_tau_a250.0_T400.0_bs256_lr0.00025_eps0.01_heavy_1765292395_acc0.85.pth \
#   --model "ConvLarge_CIFAR10_1_LIP_Bjork" \
#   --dataset cifar10 \
#   --split_index $SPLIT_IDX \
#   --batch_size 1 \
#   --output_csv results/bjork_T400_split${SPLIT_IDX}_study.csv

# echo ""
# echo "================================================================="
# echo "Starting Evaluation 2/4: Bjork Model (Low Temp) | Split: $SPLIT_IDX"
# echo "================================================================="
# python main_auto.py \
#   --model_path /home/aws_install/robust-benchmark/models/vanilla_ConvLarge_CIFAR10_1_LIP_Bjork_cifar10_tau_a250.0_T1.0_bs128_lr0.001_eps0.01_heavy_1765374676_acc0.75.pth \
#   --model "ConvLarge_CIFAR10_1_LIP_Bjork" \
#   --dataset cifar10 \
#   --split_index $SPLIT_IDX \
#   --batch_size 1 \
#   --output_csv results/bjork_T1_split${SPLIT_IDX}_study.csv

# echo ""
# echo "================================================================="
# echo "Starting Evaluation 3/4: GNP Model (High Tau) | Split: $SPLIT_IDX"
# echo "================================================================="
# python main_auto.py \
#   --model_path /home/aws_install/robust-benchmark/models/vanilla_ConvLarge_CIFAR10_1_LIP_GNP_cifar10_tau_15.0_lr0.001_eps0.155_1766411827_acc0.82.pth \
#   --model "ConvLarge_CIFAR10_1_LIP_GNP" \
#   --dataset cifar10 \
#   --split_index $SPLIT_IDX \
#   --batch_size 1 \
#   --output_csv results/gnp_tau15_split${SPLIT_IDX}_study.csv

# echo ""
# echo "================================================================="
# echo "Starting Evaluation 4/4: GNP Model (Low Tau) | Split: $SPLIT_IDX"
# echo "================================================================="
# python main_auto.py \
#   --model_path /home/aws_install/robust-benchmark/models/vanilla_ConvLarge_CIFAR10_1_LIP_GNP_cifar10_tau_1.0_lr0.001_eps0.155_1766393872_acc0.75.pth \
#   --model "ConvLarge_CIFAR10_1_LIP_GNP" \
#   --dataset cifar10 \
#   --split_index $SPLIT_IDX \
#   --batch_size 1 \
#   --output_csv results/gnp_tau1_split${SPLIT_IDX}_study.csv

# echo ""
# echo "================================================================="
# echo "All Evaluations Complete! Results saved in the 'results/' folder."
# echo "================================================================="



# Experience 4 : hybrid with alpha crown



# echo "================================================================="
# echo "Starting Evaluation 1/4: Bjork Model (High Temp) | Split: $SPLIT_IDX"
# echo "================================================================="
# python main_auto.py \
#   --model_path /home/aws_install/robust-benchmark/models/vanilla_ConvLarge_CIFAR10_1_LIP_Bjork_cifar10_tau_a250.0_T400.0_bs256_lr0.00025_eps0.01_heavy_1765292395_acc0.85.pth \
#   --model "ConvLarge_CIFAR10_1_LIP_Bjork" \
#   --dataset cifar10 \
#   --split_index $SPLIT_IDX \
#   --batch_size 1 \
#   --high_tau True \
#   --sdp False \
#   --output_csv results_alpha/bjork_T400_split${SPLIT_IDX}_study_alpha.csv

# echo ""
# echo "================================================================="
# echo "Starting Evaluation 2/4: Bjork Model (Low Temp) | Split: $SPLIT_IDX"
# echo "================================================================="
# python main_auto.py \
#   --model_path /home/aws_install/robust-benchmark/models/vanilla_ConvLarge_CIFAR10_1_LIP_Bjork_cifar10_tau_a250.0_T1.0_bs128_lr0.001_eps0.01_heavy_1765374676_acc0.75.pth \
#   --model "ConvLarge_CIFAR10_1_LIP_Bjork" \
#   --dataset cifar10 \
#   --split_index $SPLIT_IDX \
#   --batch_size 1 \
#   --sdp False \
#   --output_csv results_alpha/bjork_T1_split${SPLIT_IDX}_study_alpha.csv

# echo ""
# echo "================================================================="
# echo "Starting Evaluation 3/4: GNP Model (High Tau) | Split: $SPLIT_IDX"
# echo "================================================================="
# python main_auto.py \
#   --model_path /home/aws_install/robust-benchmark/models/vanilla_ConvLarge_CIFAR10_1_LIP_GNP_cifar10_tau_15.0_lr0.001_eps0.155_1766411827_acc0.82.pth \
#   --model "ConvLarge_CIFAR10_1_LIP_GNP" \
#   --dataset cifar10 \
#   --split_index $SPLIT_IDX \
#   --batch_size 1 \
#   --high_tau True \
#   --sdp False \
#   --output_csv results_alpha/gnp_tau15_split${SPLIT_IDX}_study_alpha.csv

# echo ""
# echo "================================================================="
# echo "Starting Evaluation 4/4: GNP Model (Low Tau) | Split: $SPLIT_IDX"
# echo "================================================================="
# python main_auto.py \
#   --model_path /home/aws_install/robust-benchmark/models/vanilla_ConvLarge_CIFAR10_1_LIP_GNP_cifar10_tau_1.0_lr0.001_eps0.155_1766393872_acc0.75.pth \
#   --model "ConvLarge_CIFAR10_1_LIP_GNP" \
#   --dataset cifar10 \
#   --split_index $SPLIT_IDX \
#   --batch_size 1 \
#   --sdp False \
#   --output_csv results_alpha/gnp_tau1_split${SPLIT_IDX}_study_alpha.csv

# echo ""
# echo "================================================================="
# echo "All Evaluations Complete! Results saved in the 'results/' folder."
# echo "================================================================="

