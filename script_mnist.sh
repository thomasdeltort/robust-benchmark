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

#!/bin/bash

# This script runs two sets of Python commands multiple times with varying epsilon values
# for the ConvLarge_MNIST_1_LIP model.

# --- Base Model Configuration ---
MODEL_PATH="/home/aws_install/robustess_project/Robust_Benchmark/models/mnist_ConvLarge_MNIST_1_LIP_temp0.3_bs256_trainacc0.0_valacc0.9.pth"
MODEL_NAME="ConvLarge_MNIST_1_LIP"
DATASET="mnist"

# --- Experiment 1: L-infinity Norm ---
echo "====================================================="
echo "Starting Experiment 1: L-infinity Norm (MNIST)"
echo "====================================================="

# --- Configuration for Experiment 1 ---
START_EPS_1=0.05
END_EPS_1=0.4
POINTS_1=20

# --- Base Command Arguments for Experiment 1 ---
BASE_OUTPUT_NAME_1="ConvLarge_LIP_Mnist_Linf"
NORM_TYPE="inf"

# Generate epsilon values for Experiment 1
EPSILON_VALUES_1=$(python3 -c "import numpy as np; print(' '.join(f'{x:.6f}' for x in np.linspace($START_EPS_1, $END_EPS_1, $POINTS_1)))")

if [ -z "$EPSILON_VALUES_1" ]; then
    echo "Error: Could not generate epsilon values for Experiment 1."
    exit 1
fi

echo "Epsilon will range from $START_EPS_1 to $END_EPS_1 in $POINTS_1 steps."

# Loop over each epsilon value for Experiment 1
for epsilon in $EPSILON_VALUES_1
do
  echo "-----------------------------------------------------"
  echo "Executing L-inf command with epsilon = $epsilon"
  echo "-----------------------------------------------------"

  # Generate unique output names for each epsilon
  OUTPUT_CSV="${BASE_OUTPUT_NAME_1}.csv"
  OUTPUT_PKL="${BASE_OUTPUT_NAME_1}_eps_${epsilon}.pkl"

  python main.py \
    --model_path "$MODEL_PATH" \
    --model "$MODEL_NAME" \
    --dataset "$DATASET" \
    --epsilon "$epsilon" \
    --output_csv "$OUTPUT_CSV" \
    --output_pkl "$OUTPUT_PKL" \
    --norm "$NORM_TYPE"
done

echo "Experiment 1 finished."

# --- Experiment 2: L2 Norm (or default) ---
echo "====================================================="
echo "Starting Experiment 2: L2 Norm (MNIST)"
echo "====================================================="

# --- Configuration for Experiment 2 ---
START_EPS_2=0.5
END_EPS_2=3.0
POINTS_2=20

# --- Base Command Arguments for Experiment 2 ---
BASE_OUTPUT_NAME_2="ConvLarge_LIP_Mnist_L2"

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
  
  # Generate unique output names for each epsilon
  OUTPUT_CSV="${BASE_OUTPUT_NAME_2}.csv"
  OUTPUT_PKL="${BASE_OUTPUT_NAME_2}_eps_${epsilon}.pkl"

  # Note: The --norm argument is omitted in this command
  python main.py \
    --model_path "$MODEL_PATH" \
    --model "$MODEL_NAME" \
    --dataset "$DATASET" \
    --epsilon "$epsilon" \
    --output_csv "$OUTPUT_CSV" \
    --output_pkl "$OUTPUT_PKL"
done

echo "Experiment 2 finished."
echo "====================================================="
echo "Script finished all runs."
echo "====================================================="