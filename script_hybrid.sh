#!/bin/bash


# Define the split index here so it's easy to change later!
SPLIT_IDX=3


# EXPERIENCE 1 low vs high Tau

#
#echo "================================================================="
#echo "Starting Evaluation 1/2: Bjork Model (Low Temp) | Split: $SPLIT_IDX"
#echo "================================================================="
#python main_auto.py \
#  --model_path ./models/vanilla_ConvLarge_CIFAR10_1_LIP_Bjork_cifar10_tau_a250.0_T1.0_bs128_lr0.001_eps0.01_heavy_1765374676_acc0.75.pth \
#  --model "ConvLarge_CIFAR10_1_LIP_Bjork" \
#  --dataset cifar10 \
#  --split_index $SPLIT_IDX \
#  --batch_size 1 \
#  --solvers_config "{'alphacrown': False, 'heavy_certified': False, 'hybrid': True}"\
#  --output_csv results_highlowtau/bjork_T1_split${SPLIT_IDX}_study_lowtau.csv
#
#  echo "================================================================="
#echo "Starting Evaluation 2/2: Bjork Model (High Temp) | Split: $SPLIT_IDX"
#echo "================================================================="
#python main_auto.py \
#  --model_path ./models/vanilla_ConvLarge_CIFAR10_1_LIP_Bjork_cifar10_tau_a250.0_T1.0_bs128_lr0.001_eps0.01_heavy_1765374676_acc0.75.pth \
#  --model "ConvLarge_CIFAR10_1_LIP_Bjork" \
#  --dataset cifar10 \
#  --split_index $SPLIT_IDX \
#  --batch_size 1 \
#  --high_tau True \
#  --solvers_config "{'alphacrown': False, 'heavy_certified': False, 'hybrid': True}"\
#  --output_csv results_highlowtau/bjork_T1_split${SPLIT_IDX}_study_hightau.csv

# Uncomment what is placed next !!!
#
## Experience 2 conventional sdp crown
#
#
#echo "================================================================="
#echo "Starting Evaluation 1/4: Bjork Model (High Temp) | Split: $SPLIT_IDX"
#echo "================================================================="
#python main_auto.py \
#  --model_path /home/aws_install/robust-benchmark/models/vanilla_ConvLarge_CIFAR10_1_LIP_Bjork_cifar10_tau_a250.0_T400.0_bs256_lr0.00025_eps0.01_heavy_1765292395_acc0.85.pth \
#  --model "ConvLarge_CIFAR10_1_LIP_Bjork" \
#  --dataset cifar10 \
#  --split_index $SPLIT_IDX \
#  --batch_size 1 \
#  --output_csv results/bjork_T400_split${SPLIT_IDX}_study.csv
#
#echo ""
#echo "================================================================="
#echo "Starting Evaluation 2/4: Bjork Model (Low Temp) | Split: $SPLIT_IDX"
#echo "================================================================="
#python main_auto.py \
#  --model_path /home/aws_install/robust-benchmark/models/vanilla_ConvLarge_CIFAR10_1_LIP_Bjork_cifar10_tau_a250.0_T1.0_bs128_lr0.001_eps0.01_heavy_1765374676_acc0.75.pth \
#  --model "ConvLarge_CIFAR10_1_LIP_Bjork" \
#  --dataset cifar10 \
#  --split_index $SPLIT_IDX \
#  --batch_size 1 \
#  --output_csv results/bjork_T1_split${SPLIT_IDX}_study.csv
#
#echo ""
#echo "================================================================="
#echo "Starting Evaluation 3/4: GNP Model (High Tau) | Split: $SPLIT_IDX"
#echo "================================================================="
#python main_auto.py \
#  --model_path /home/aws_install/robust-benchmark/models/vanilla_ConvLarge_CIFAR10_1_LIP_GNP_cifar10_tau_15.0_lr0.001_eps0.155_1766411827_acc0.82.pth \
#  --model "ConvLarge_CIFAR10_1_LIP_GNP" \
#  --dataset cifar10 \
#  --split_index $SPLIT_IDX \
#  --batch_size 1 \
#  --output_csv results/gnp_tau15_split${SPLIT_IDX}_study.csv
#
#echo ""
#echo "================================================================="
#echo "Starting Evaluation 4/4: GNP Model (Low Tau) | Split: $SPLIT_IDX"
#echo "================================================================="
#python main_auto.py \
#  --model_path /home/aws_install/robust-benchmark/models/vanilla_ConvLarge_CIFAR10_1_LIP_GNP_cifar10_tau_1.0_lr0.001_eps0.155_1766393872_acc0.75.pth \
#  --model "ConvLarge_CIFAR10_1_LIP_GNP" \
#  --dataset cifar10 \
#  --split_index $SPLIT_IDX \
#  --batch_size 1 \
#  --output_csv results/gnp_tau1_split${SPLIT_IDX}_study.csv
#
#echo ""
#echo "================================================================="
#echo "All Evaluations Complete! Results saved in the 'results/' folder."
#echo "================================================================="



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


# Expérience 5 VGG split 13

SPLIT_IDX=13
#
#echo "================================================================="
#echo "Starting Evaluation 1/4: Bjork Model (High Temp) | Split: $SPLIT_IDX"
#echo "================================================================="
#python main_auto.py \
#  --model_path /home/aws_install/robust-benchmark/models/vanilla_VGG13_1_LIP_Bjork_CIFAR10_cifar10_tau_a250.0_T100.0_bs128_lr0.001_1771859356_acc0.86.pth \
#  --model "VGG13_1_LIP_Bjork_CIFAR10" \
#  --dataset cifar10 \
#  --split_index $SPLIT_IDX \
#  --batch_size 1 \
#  --sdp True \
#  --output_csv results/bjork_VGG_T100_split${SPLIT_IDX}_study.csv
#
#echo ""
#echo "================================================================="
#echo "Starting Evaluation 2/4: Bjork Model (Low Temp) | Split: $SPLIT_IDX"
#echo "================================================================="
#python main_auto.py \
#  --model_path /home/aws_install/robust-benchmark/models/vanilla_VGG13_1_LIP_Bjork_CIFAR10_cifar10_tau_a250.0_T5.0_bs128_lr0.001_1771861736_acc0.76.pth \
#  --model "VGG13_1_LIP_Bjork_CIFAR10" \
#  --dataset cifar10 \
#  --split_index $SPLIT_IDX \
#  --batch_size 1 \
#  --sdp True \
#  --output_csv results/bjork_VGG_T5_split${SPLIT_IDX}_study.csv
#
#echo ""
#echo "================================================================="
#echo "Starting Evaluation 3/4: GNP Model (High Tau) | Split: $SPLIT_IDX"
#echo "================================================================="
#python main_auto.py \
#  --model_path /home/aws_install/robust-benchmark/models/vanilla_VGG13_1_LIP_GNP_CIFAR10_cifar10_tau_a250.0_T15.0_bs128_lr0.001_1772009879_acc0.81.pth \
#  --model "VGG13_1_LIP_GNP_CIFAR10" \
#  --dataset cifar10 \
#  --split_index $SPLIT_IDX \
#  --batch_size 1 \
#  --sdp True \
#  --output_csv results/gnp_VGG_T15_split${SPLIT_IDX}_study.csv
#
#echo ""
#echo "================================================================="
#echo "Starting Evaluation 4/4: GNP Model (Low Tau) | Split: $SPLIT_IDX"
#echo "================================================================="
#python main_auto.py \
#  --model_path /home/aws_install/robust-benchmark/models/vanilla_VGG13_1_LIP_GNP_CIFAR10_cifar10_tau_a250.0_T1.0_bs128_lr0.001_1771927725_acc0.76.pth \
#  --model "VGG13_1_LIP_GNP_CIFAR10" \
#  --dataset cifar10 \
#  --split_index $SPLIT_IDX \
#  --batch_size 1 \
#  --sdp True \
#  --output_csv results/gnp_VGG_T1_split${SPLIT_IDX}_study.csv


# ADDING EXPERIENCE : alpha crown missing

#echo ""
#echo "================================================================="
#echo "Starting Evaluation 3/4: GNP Model (High Tau) | Split: $SPLIT_IDX"
#echo "================================================================="
#python main_auto.py \
#  --model_path ./models/vanilla_VGG13_1_LIP_GNP_CIFAR10_cifar10_tau_a250.0_T15.0_bs128_lr0.001_1772009879_acc0.81.pth \
#  --model "VGG13_1_LIP_GNP_CIFAR10" \
#  --dataset cifar10 \
#  --split_index $SPLIT_IDX \
#  --batch_size 1 \
#  --sdp False \
#  --solvers_config "{'alphacrown': False, 'heavy_certified': False, 'hybrid': True}"\
#  --output_csv results_alpha_hybrid/gnp_VGG_T15_split${SPLIT_IDX}_study.csv

echo ""
echo "================================================================="
echo "Starting Evaluation 4/4: GNP Model (Low Tau) | Split: $SPLIT_IDX"
echo "================================================================="
python main_auto.py \
  --model_path ./models/vanilla_VGG13_1_LIP_GNP_CIFAR10_cifar10_tau_a250.0_T1.0_bs128_lr0.001_1771927725_acc0.76.pth \
  --model "VGG13_1_LIP_GNP_CIFAR10" \
  --dataset cifar10 \
  --split_index $SPLIT_IDX \
  --batch_size 1 \
  --sdp False \
  --solvers_config "{'alphacrown': False, 'heavy_certified': False, 'hybrid': True}"\
  --output_csv results_alpha_hybrid/gnp_VGG_T1_split${SPLIT_IDX}_study.csv

# # Expérience 6 VGG split 15

# SPLIT_IDX=15

# echo "================================================================="
# echo "Starting Evaluation 1/4: Bjork Model (High Temp) | Split: $SPLIT_IDX"
# echo "================================================================="
# python main_auto.py \
#   --model_path /home/aws_install/robust-benchmark/models/vanilla_VGG13_1_LIP_Bjork_CIFAR10_cifar10_tau_a250.0_T100.0_bs128_lr0.001_1771859356_acc0.86.pth \
#   --model "VGG13_1_LIP_Bjork_CIFAR10" \
#   --dataset cifar10 \
#   --split_index $SPLIT_IDX \
#   --batch_size 1 \
#   --sdp True \
#   --output_csv results/bjork_VGG_T100_split${SPLIT_IDX}_study.csv

# echo ""
# echo "================================================================="
# echo "Starting Evaluation 2/4: Bjork Model (Low Temp) | Split: $SPLIT_IDX"
# echo "================================================================="
# python main_auto.py \
#   --model_path /home/aws_install/robust-benchmark/models/vanilla_VGG13_1_LIP_Bjork_CIFAR10_cifar10_tau_a250.0_T5.0_bs128_lr0.001_1771861736_acc0.76.pth \
#   --model "VGG13_1_LIP_Bjork_CIFAR10" \
#   --dataset cifar10 \
#   --split_index $SPLIT_IDX \
#   --batch_size 1 \
#   --sdp True \
#   --output_csv results/bjork_VGG_T5_split${SPLIT_IDX}_study.csv

# echo ""
# echo "================================================================="
# echo "Starting Evaluation 3/4: GNP Model (High Tau) | Split: $SPLIT_IDX"
# echo "================================================================="
# python main_auto.py \
#   --model_path /home/aws_install/robust-benchmark/models/vanilla_VGG13_1_LIP_GNP_CIFAR10_cifar10_tau_a250.0_T15.0_bs128_lr0.001_1772009879_acc0.81.pth \
#   --model "VGG13_1_LIP_GNP_CIFAR10" \
#   --dataset cifar10 \
#   --split_index $SPLIT_IDX \
#   --batch_size 1 \
#   --sdp True \
#   --output_csv results/gnp_VGG_T15_split${SPLIT_IDX}_study.csv

# echo ""
# echo "================================================================="
# echo "Starting Evaluation 4/4: GNP Model (Low Tau) | Split: $SPLIT_IDX"
# echo "================================================================="
# python main_auto.py \
#   --model_path /home/aws_install/robust-benchmark/models/vanilla_VGG13_1_LIP_GNP_CIFAR10_cifar10_tau_a250.0_T1.0_bs128_lr0.001_1771927725_acc0.76.pth \
#   --model "VGG13_1_LIP_GNP_CIFAR10" \
#   --dataset cifar10 \
#   --split_index $SPLIT_IDX \
#   --batch_size 1 \
#   --sdp True \
#   --output_csv results/gnp_VGG_T1_split${SPLIT_IDX}_study.csv

