# #!/bin/bash

# # This script runs two sets of Python commands multiple times with varying epsilon values.

# # --- Experiment 1: L-infinity Norm ---
# echo "====================================================="
# echo "Starting Experiment 1: L-infinity Norm"
# echo "====================================================="

# # --- Configuration for Experiment 1 ---
# START_EPS_1=0.00005
# END_EPS_1=0.05
# POINTS_1=40

# # --- Base Command Arguments for Experiment 1 ---
# MODEL_PATH="/home/aws_install/robustess_project/Robust_Benchmark/models/cifar10_CNNA_CIFAR10_1_LIP_temp1.0_bs256_trainacc0.0_valacc0.6.pth"
# MODEL_NAME="CNNA_CIFAR10_1_LIP"
# DATASET="cifar10"
# NORM_TYPE="inf"
# # BASE_OUTPUT_NAME_1="CNNA_LIP_Linf"

# # Generate epsilon values for Experiment 1
# EPSILON_VALUES_1=$(python3 -c "import numpy as np; print(' '.join(f'{x:.6f}' for x in np.linspace($START_EPS_1, $END_EPS_1, $POINTS_1)))")

# if [ -z "$EPSILON_VALUES_1" ]; then
#     echo "Error: Could not generate epsilon values for Experiment 1. Make sure Python and NumPy are installed."
#     exit 1
# fi

# echo "Epsilon will range from $START_EPS_1 to $END_EPS_1 in $POINTS_1 steps."

# # Loop over each epsilon value for Experiment 1
# for epsilon in $EPSILON_VALUES_1
# do
#   echo "-----------------------------------------------------"
#   echo "Executing L-inf command with epsilon = $epsilon"
#   echo "-----------------------------------------------------"

#   OUTPUT_CSV="${BASE_OUTPUT_NAME_1}_2.csv"
#   OUTPUT_PKL="${BASE_OUTPUT_NAME_1}_2.pkl"

#   python main.py \
#     --model_path "$MODEL_PATH" \
#     --model "$MODEL_NAME" \
#     --dataset "$DATASET" \
#     --epsilon "$epsilon" \
#     --output_csv "$OUTPUT_CSV" \
#     --output_pkl "$OUTPUT_PKL" \
#     --norm "$NORM_TYPE"
# done

# echo "Experiment 1 finished."


# --- Experiment 2: L2 Norm (or default) ---
# echo "====================================================="
# echo "Starting Experiment 2: L2 Norm (or default)"
# echo "====================================================="

# # --- Configuration for Experiment 2 ---
# START_EPS_2=0.4
# END_EPS_2=1
# POINTS_2=20

# # --- Base Command Arguments for Experiment 2 ---
# # Reusing model path, name, and dataset from above
# BASE_OUTPUT_NAME_2="CNNA_LIP_L2"

# # Generate epsilon values for Experiment 2
# EPSILON_VALUES_2=$(python3 -c "import numpy as np; print(' '.join(f'{x:.6f}' for x in np.linspace($START_EPS_2, $END_EPS_2, $POINTS_2)))")

# if [ -z "$EPSILON_VALUES_2" ]; then
#     echo "Error: Could not generate epsilon values for Experiment 2. Make sure Python and NumPy are installed."
#     exit 1
# fi

# echo "Epsilon will range from $START_EPS_2 to $END_EPS_2 in $POINTS_2 steps."

# # Loop over each epsilon value for Experiment 2
# for epsilon in $EPSILON_VALUES_2
# do
#   echo "-----------------------------------------------------"
#   echo "Executing L2 command with epsilon = $epsilon"
#   echo "-----------------------------------------------------"
  
#   OUTPUT_CSV="${BASE_OUTPUT_NAME_2}_2.csv"
#   OUTPUT_PKL="${BASE_OUTPUT_NAME_2}_2.pkl"

#   # Note: The --norm argument is omitted in this command
#   python main.py \
#     --model_path "$MODEL_PATH" \
#     --model "$MODEL_NAME" \
#     --dataset "$DATASET" \
#     --epsilon "$epsilon" \
#     --output_csv "$OUTPUT_CSV" \
#     --output_pkl "$OUTPUT_PKL"
# done

# echo "Experiment 2 finished."
# echo "====================================================="
# echo "Script finished all runs."
# echo "====================================================="



# # This script runs two sets of Python commands multiple times with varying epsilon values.

# # --- Experiment 3: L-infinity Norm ---
# echo "====================================================="
# echo "Starting Experiment 3: L-infinity Norm"
# echo "====================================================="

# # --- Configuration for Experiment 1 ---
# START_EPS_1=0.003
# END_EPS_1=0.013
# POINTS_1=40

# # --- Base Command Arguments for Experiment 1 ---
# MODEL_PATH="/home/aws_install/robustess_project/Robust_Benchmark/models/cifar10_CNNA_CIFAR10_1_LIP_GNP_temp0.8_bs256_trainacc0.0_valacc0.6.pth"
# MODEL_NAME="CNNA_CIFAR10_1_LIP_GNP"
# DATASET="cifar10"
# NORM_TYPE="inf"
# BASE_OUTPUT_NAME_1="CNNA_LIP_GNP_Linf"

# # Generate epsilon values for Experiment 1
# EPSILON_VALUES_1=$(python3 -c "import numpy as np; print(' '.join(f'{x:.6f}' for x in np.linspace($START_EPS_1, $END_EPS_1, $POINTS_1)))")

# if [ -z "$EPSILON_VALUES_1" ]; then
#     echo "Error: Could not generate epsilon values for Experiment 1. Make sure Python and NumPy are installed."
#     exit 1
# fi

# echo "Epsilon will range from $START_EPS_1 to $END_EPS_1 in $POINTS_1 steps."

# # Loop over each epsilon value for Experiment 1
# for epsilon in $EPSILON_VALUES_1
# do
#   echo "-----------------------------------------------------"
#   echo "Executing L-inf command with epsilon = $epsilon"
#   echo "-----------------------------------------------------"

#   OUTPUT_CSV="${BASE_OUTPUT_NAME_1}_2.csv"
#   OUTPUT_PKL="${BASE_OUTPUT_NAME_1}_2.pkl"

#   python main.py \
#     --model_path "$MODEL_PATH" \
#     --model "$MODEL_NAME" \
#     --dataset "$DATASET" \
#     --epsilon "$epsilon" \
#     --output_csv "$OUTPUT_CSV" \
#     --output_pkl "$OUTPUT_PKL" \
#     --norm "$NORM_TYPE"
# done

# echo "Experiment 3 finished."


# # --- Experiment 4: L2 Norm (or default) ---
# echo "====================================================="
# echo "Starting Experiment 4: L2 Norm (or default)"
# echo "====================================================="

# # --- Configuration for Experiment 2 ---
# START_EPS_2=0.005
# END_EPS_2=1
# POINTS_2=20

# # --- Base Command Arguments for Experiment 2 ---
# # Reusing model path, name, and dataset from above
# BASE_OUTPUT_NAME_2="CNNA_LIP_GNP_L2"

# # Generate epsilon values for Experiment 2
# EPSILON_VALUES_2=$(python3 -c "import numpy as np; print(' '.join(f'{x:.6f}' for x in np.linspace($START_EPS_2, $END_EPS_2, $POINTS_2)))")

# if [ -z "$EPSILON_VALUES_2" ]; then
#     echo "Error: Could not generate epsilon values for Experiment 2. Make sure Python and NumPy are installed."
#     exit 1
# fi

# echo "Epsilon will range from $START_EPS_2 to $END_EPS_2 in $POINTS_2 steps."

# # Loop over each epsilon value for Experiment 2
# for epsilon in $EPSILON_VALUES_2
# do
#   echo "-----------------------------------------------------"
#   echo "Executing L2 command with epsilon = $epsilon"
#   echo "-----------------------------------------------------"
  
#   OUTPUT_CSV="${BASE_OUTPUT_NAME_2}_2.csv"
#   OUTPUT_PKL="${BASE_OUTPUT_NAME_2}_2.pkl"

#   # Note: The --norm argument is omitted in this command
#   python main.py \
#     --model_path "$MODEL_PATH" \
#     --model "$MODEL_NAME" \
#     --dataset "$DATASET" \
#     --epsilon "$epsilon" \
#     --output_csv "$OUTPUT_CSV" \
#     --output_pkl "$OUTPUT_PKL"
# done

# echo "Experiment 4 finished."
# echo "====================================================="
# echo "Script finished all runs."
# echo "====================================================="

# #!/bin/bash

# # This script runs two sets of Python commands multiple times with varying epsilon values
# # for the ConvLarge_MNIST_1_LIP model.

# # --- Base Model Configuration ---
# MODEL_PATH="/home/aws_install/robustess_project/Robust_Benchmark/models/mnist_ConvLarge_MNIST_1_LIP_temp0.3_bs256_trainacc0.0_valacc0.9.pth"
# MODEL_NAME="ConvLarge_MNIST_1_LIP"
# DATASET="mnist"

# # --- Experiment 1: L-infinity Norm ---
# echo "====================================================="
# echo "Starting Experiment 1: L-infinity Norm (MNIST)"
# echo "====================================================="

# # --- Configuration for Experiment 1 ---
# START_EPS_1=0.05
# END_EPS_1=0.4
# POINTS_1=20

# # --- Base Command Arguments for Experiment 1 ---
# BASE_OUTPUT_NAME_1="ConvLarge_LIP_Mnist_Linf"
# NORM_TYPE="inf"

# # Generate epsilon values for Experiment 1
# EPSILON_VALUES_1=$(python3 -c "import numpy as np; print(' '.join(f'{x:.6f}' for x in np.linspace($START_EPS_1, $END_EPS_1, $POINTS_1)))")

# if [ -z "$EPSILON_VALUES_1" ]; then
#     echo "Error: Could not generate epsilon values for Experiment 1."
#     exit 1
# fi

# echo "Epsilon will range from $START_EPS_1 to $END_EPS_1 in $POINTS_1 steps."

# # Loop over each epsilon value for Experiment 1
# for epsilon in $EPSILON_VALUES_1
# do
#   echo "-----------------------------------------------------"
#   echo "Executing L-inf command with epsilon = $epsilon"
#   echo "-----------------------------------------------------"

#   # Generate unique output names for each epsilon
#   OUTPUT_CSV="${BASE_OUTPUT_NAME_1}.csv"
#   OUTPUT_PKL="${BASE_OUTPUT_NAME_1}_eps_${epsilon}.pkl"

#   python main.py \
#     --model_path "$MODEL_PATH" \
#     --model "$MODEL_NAME" \
#     --dataset "$DATASET" \
#     --epsilon "$epsilon" \
#     --output_csv "$OUTPUT_CSV" \
#     --output_pkl "$OUTPUT_PKL" \
#     --norm "$NORM_TYPE"
# done

# echo "Experiment 1 finished."

# # --- Experiment 2: L2 Norm (or default) ---
# echo "====================================================="
# echo "Starting Experiment 2: L2 Norm (MNIST)"
# echo "====================================================="

# # --- Configuration for Experiment 2 ---
# START_EPS_2=0.5
# END_EPS_2=3.0
# POINTS_2=20

# # --- Base Command Arguments for Experiment 2 ---
# BASE_OUTPUT_NAME_2="ConvLarge_LIP_Mnist_L2"

# # Generate epsilon values for Experiment 2
# EPSILON_VALUES_2=$(python3 -c "import numpy as np; print(' '.join(f'{x:.6f}' for x in np.linspace($START_EPS_2, $END_EPS_2, $POINTS_2)))")

# if [ -z "$EPSILON_VALUES_2" ]; then
#     echo "Error: Could not generate epsilon values for Experiment 2."
#     exit 1
# fi

# echo "Epsilon will range from $START_EPS_2 to $END_EPS_2 in $POINTS_2 steps."

# # Loop over each epsilon value for Experiment 2
# for epsilon in $EPSILON_VALUES_2
# do
#   echo "-----------------------------------------------------"
#   echo "Executing L2 command with epsilon = $epsilon"
#   echo "-----------------------------------------------------"
  
#   # Generate unique output names for each epsilon
#   OUTPUT_CSV="${BASE_OUTPUT_NAME_2}.csv"
#   OUTPUT_PKL="${BASE_OUTPUT_NAME_2}_eps_${epsilon}.pkl"

#   # Note: The --norm argument is omitted in this command
#   python main.py \
#     --model_path "$MODEL_PATH" \
#     --model "$MODEL_NAME" \
#     --dataset "$DATASET" \
#     --epsilon "$epsilon" \
#     --output_csv "$OUTPUT_CSV" \
#     --output_pkl "$OUTPUT_PKL"
# done

# echo "Experiment 2 finished."
# echo "====================================================="
# echo "Script finished all runs."
# echo "====================================================="

#!/bin/bash

# This script runs four sets of Python commands multiple times with varying epsilon values
# across two different MLP models.

# ===================================================================================
#                       START: MLP_MNIST_1_LIP_GNP Model
# ===================================================================================

# --- Base Model Configuration (GNP) ---
MODEL_PATH_GNP="/home/aws_install/robustess_project/Robust_Benchmark/models/mnist_MLP_MNIST_1_LIP_GNP_temp0.03_bs256_trainacc0.0_valacc0.8.pth"
MODEL_NAME_GNP="MLP_MNIST_1_LIP_GNP"
DATASET_GNP="mnist"

# # --- Experiment 1: L-infinity Norm (GNP) ---
# echo "====================================================="
# echo "Starting Experiment 1: L-infinity Norm (MLP GNP MNIST)"
# echo "====================================================="

# # --- Configuration for Experiment 1 ---
# # !!! Please ADJUST these epsilon values as needed for this model !!!
# START_EPS_1=0.005
# END_EPS_1=0.05
# POINTS_1=20

# # --- Base Command Arguments for Experiment 1 ---
# BASE_OUTPUT_NAME_1="MLP_LIP_GNP_Mnist_Linf"
# NORM_TYPE_1="inf"

# # Generate epsilon values for Experiment 1
# EPSILON_VALUES_1=$(python3 -c "import numpy as np; print(' '.join(f'{x:.6f}' for x in np.linspace($START_EPS_1, $END_EPS_1, $POINTS_1)))")

# if [ -z "$EPSILON_VALUES_1" ]; then
#     echo "Error: Could not generate epsilon values for Experiment 1."
#     exit 1
# fi

# echo "Epsilon will range from $START_EPS_1 to $END_EPS_1 in $POINTS_1 steps."

# # Loop over each epsilon value for Experiment 1
# for epsilon in $EPSILON_VALUES_1
# do
#   echo "-----------------------------------------------------"
#   echo "Executing L-inf command with epsilon = $epsilon"
#   echo "-----------------------------------------------------"

#   OUTPUT_CSV="${BASE_OUTPUT_NAME_1}.csv"
#   OUTPUT_PKL="${BASE_OUTPUT_NAME_1}_eps_${epsilon}.pkl"

#   python main.py \
#     --model_path "$MODEL_PATH_GNP" \
#     --model "$MODEL_NAME_GNP" \
#     --dataset "$DATASET_GNP" \
#     --epsilon "$epsilon" \
#     --output_csv "$OUTPUT_CSV" \
#     --output_pkl "$OUTPUT_PKL" \
#     --norm "$NORM_TYPE_1"
# done
# echo "Experiment 1 finished."

# ---

# --- Experiment 2: L2 Norm (GNP) ---
echo "====================================================="
echo "Starting Experiment 2: L2 Norm (MLP GNP MNIST)"
echo "====================================================="

# --- Configuration for Experiment 2 ---
# !!! Please ADJUST these epsilon values as needed for this model !!!
START_EPS_2=1.3
END_EPS_2=1.8
POINTS_2=10

# --- Base Command Arguments for Experiment 2 ---
BASE_OUTPUT_NAME_2="MLP_LIP_GNP_Mnist_L2"

# Generate epsilon values for Experiment 2
EPSILON_VALUES_2=$(python3 -c "import numpy as np; print(' '.join(f'{x:.6f}' for x in np.linspace($START_EPS_2, $END_EPS_2, $POINTS_2)))")

if [ -z "$EPSILON_VALUES_2" ]; then
    echo "Error: Could not generate epsilon values for Experiment 2."
    exit 1
fi

echo "Epsilon will range from $START_EPS_2 to $END_EPS_2 in $POINTS_2 steps."

# Loop over each epsilon value for Experiment 2
for epsilon in $EPSILON_VALUES_2
do
  echo "-----------------------------------------------------"
  echo "Executing L2 command with epsilon = $epsilon"
  echo "-----------------------------------------------------"
  
  OUTPUT_CSV="${BASE_OUTPUT_NAME_2}.csv"
  OUTPUT_PKL="${BASE_OUTPUT_NAME_2}_eps_${epsilon}.pkl"

  python main.py \
    --model_path "$MODEL_PATH_GNP" \
    --model "$MODEL_NAME_GNP" \
    --dataset "$DATASET_GNP" \
    --epsilon "$epsilon" \
    --output_csv "$OUTPUT_CSV" \
    --output_pkl "$OUTPUT_PKL"
done
echo "Experiment 2 finished."

# ===================================================================================
#                       START: MLP_MNIST_1_LIP Model (No GNP)
# ===================================================================================

# --- Base Model Configuration (No GNP) ---
MODEL_PATH_LIP="/home/aws_install/robustess_project/Robust_Benchmark/models/mnist_MLP_MNIST_1_LIP_temp0.03_bs256_trainacc0.0_valacc0.8.pth"
MODEL_NAME_LIP="MLP_MNIST_1_LIP"
DATASET_LIP="mnist" # Assuming dataset is the same

# # --- Experiment 3: L-infinity Norm (No GNP) ---
# echo "====================================================="
# echo "Starting Experiment 3: L-infinity Norm (MLP LIP MNIST)"
# echo "====================================================="

# # --- Configuration for Experiment 3 ---
# # !!! Please ADJUST these epsilon values as needed for this model !!!
# START_EPS_3=0.005
# END_EPS_3=0.05
# POINTS_3=20

# # --- Base Command Arguments for Experiment 3 ---
# BASE_OUTPUT_NAME_3="MLP_LIP_Mnist_Linf"
# NORM_TYPE_3="inf"

# # Generate epsilon values for Experiment 3
# EPSILON_VALUES_3=$(python3 -c "import numpy as np; print(' '.join(f'{x:.6f}' for x in np.linspace($START_EPS_3, $END_EPS_3, $POINTS_3)))")

# if [ -z "$EPSILON_VALUES_3" ]; then
#     echo "Error: Could not generate epsilon values for Experiment 3."
#     exit 1
# fi

# echo "Epsilon will range from $START_EPS_3 to $END_EPS_3 in $POINTS_3 steps."

# # Loop over each epsilon value for Experiment 3
# for epsilon in $EPSILON_VALUES_3
# do
#   echo "-----------------------------------------------------"
#   echo "Executing L-inf command with epsilon = $epsilon"
#   echo "-----------------------------------------------------"

#   OUTPUT_CSV="${BASE_OUTPUT_NAME_3}.csv"
#   OUTPUT_PKL="${BASE_OUTPUT_NAME_3}_eps_${epsilon}.pkl"

#   python main.py \
#     --model_path "$MODEL_PATH_LIP" \
#     --model "$MODEL_NAME_LIP" \
#     --dataset "$DATASET_LIP" \
#     --epsilon "$epsilon" \
#     --output_csv "$OUTPUT_CSV" \
#     --output_pkl "$OUTPUT_PKL" \
#     --norm "$NORM_TYPE_3"
# done
# echo "Experiment 3 finished."

# ---

# --- Experiment 4: L2 Norm (No GNP) ---
echo "====================================================="
echo "Starting Experiment 4: L2 Norm (MLP LIP MNIST)"
echo "====================================================="

# --- Configuration for Experiment 4 ---
# !!! Please ADJUST these epsilon values as needed for this model !!!
START_EPS_4=3.0
END_EPS_4=4.0
POINTS_4=10

# --- Base Command Arguments for Experiment 4 ---
BASE_OUTPUT_NAME_4="MLP_LIP_Mnist_L2"

# Generate epsilon values for Experiment 4
EPSILON_VALUES_4=$(python3 -c "import numpy as np; print(' '.join(f'{x:.6f}' for x in np.linspace($START_EPS_4, $END_EPS_4, $POINTS_4)))")

if [ -z "$EPSILON_VALUES_4" ]; then
    echo "Error: Could not generate epsilon values for Experiment 4."
    exit 1
fi

echo "Epsilon will range from $START_EPS_4 to $END_EPS_4 in $POINTS_4 steps."

# Loop over each epsilon value for Experiment 4
for epsilon in $EPSILON_VALUES_4
do
  echo "-----------------------------------------------------"
  echo "Executing L2 command with epsilon = $epsilon"
  echo "-----------------------------------------------------"
  
  OUTPUT_CSV="${BASE_OUTPUT_NAME_4}.csv"
  OUTPUT_PKL="${BASE_OUTPUT_NAME_4}_eps_${epsilon}.pkl"

  python main.py \
    --model_path "$MODEL_PATH_LIP" \
    --model "$MODEL_NAME_LIP" \
    --dataset "$DATASET_LIP" \
    --epsilon "$epsilon" \
    --output_csv "$OUTPUT_CSV" \
    --output_pkl "$OUTPUT_PKL"
done
echo "Experiment 4 finished."

# ===================================================================================
#                                 END OF SCRIPT
# ===================================================================================

echo "====================================================="
echo "Script finished all runs."
echo "====================================================="