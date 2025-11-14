# -*- coding: utf-8 -*-
"""
Model Robustness Evaluation Script

This script evaluates the adversarial robustness of a pre-trained PyTorch model.
It calculates the model's accuracy on clean data and its robust accuracy against
a specified epsilon-perturbation using various empirical and certified methods.

This version has been updated to also measure and record the execution time
for each evaluation method, providing valuable performance metrics.

Empirical methods include:
- Projected Gradient Descent (PGD)
- AutoAttack (AA)

Certified methods include:
- Certified Robustness Accuracy (CRA)
- LIRPA-based methods (alpha-CROWN, beta-CROWN)
- SDP-based verification (SDP-CROWN)

Results, including timings, are appended to a CSV file and optionally saved to a Pickle file.
"""
import torch
import torch.nn as nn
import numpy as np
from deel import torchlip
import argparse
import os
import os
import sys
import csv
import pickle
import time # Import the time module for timing operations

# --- Handle Local Module Imports ---
# This section adds parent directories to the Python path to locate custom modules.
# This is a common practice in research projects but can be replaced by creating a proper Python package.
# try:
    # Assumes models.py and utils.py are in the parent directory
from models import *
from project_utils import *

# --- Model Zoo ---
# A list of supported model architectures. The `--model_name` argument must match one of these.

model_zoo = {
     # --- Standard, non-Lipschitz constrained models ---
    "MNIST_MLP": MNIST_MLP,
    "MNIST_ConvSmall": MNIST_ConvSmall,
    "MNIST_ConvLarge": MNIST_ConvLarge,
    "CIFAR10_CNN_A": CIFAR10_CNN_A,
    "CIFAR10_CNN_B": CIFAR10_CNN_B,
    "CIFAR10_CNN_C": CIFAR10_CNN_C,
    "CIFAR10_ConvSmall": CIFAR10_ConvSmall,
    "CIFAR10_ConvDeep": CIFAR10_ConvDeep,
    "CIFAR10_ConvLarge": CIFAR10_ConvLarge,

    # --- 1-Lipschitz models (Spectral Normalization) ---
    "MLP_MNIST_1_LIP": MLP_MNIST_1_LIP,
    "ConvSmall_MNIST_1_LIP": ConvSmall_MNIST_1_LIP,
    "ConvLarge_MNIST_1_LIP": ConvLarge_MNIST_1_LIP,
    "ConvLarge_MNIST_1_LIP_MaxMin": ConvLarge_MNIST_1_LIP_MaxMin,
    "CNNA_CIFAR10_1_LIP": CNNA_CIFAR10_1_LIP,
    "CNNB_CIFAR10_1_LIP": CNNB_CIFAR10_1_LIP,
    "CNNC_CIFAR10_1_LIP": CNNC_CIFAR10_1_LIP,
    "ConvSmall_CIFAR10_1_LIP": ConvSmall_CIFAR10_1_LIP,
    "ConvDeep_CIFAR10_1_LIP": ConvDeep_CIFAR10_1_LIP,
    "ConvLarge_CIFAR10_1_LIP": ConvLarge_CIFAR10_1_LIP,

    # --- 1-Lipschitz models (GNP technique) ---
    "MLP_MNIST_1_LIP_GNP": MLP_MNIST_1_LIP_GNP,
    "ConvSmall_MNIST_1_LIP_GNP": ConvSmall_MNIST_1_LIP_GNP,
    "ConvLarge_MNIST_1_LIP_GNP": ConvLarge_MNIST_1_LIP_GNP,
    "CNNA_CIFAR10_1_LIP_GNP": CNNA_CIFAR10_1_LIP_GNP,
    "CNNB_CIFAR10_1_LIP_GNP": CNNB_CIFAR10_1_LIP_GNP,
    "CNNC_CIFAR10_1_LIP_GNP": CNNC_CIFAR10_1_LIP_GNP,
    "ConvSmall_CIFAR10_1_LIP_GNP": ConvSmall_CIFAR10_1_LIP_GNP,
    "ConvDeep_CIFAR10_1_LIP_GNP": ConvDeep_CIFAR10_1_LIP_GNP,
    "ConvLarge_CIFAR10_1_LIP_GNP": ConvLarge_CIFAR10_1_LIP_GNP,
}


def main():
    """
    Main function to parse arguments, load data/model, run evaluations, and save results.
    """
    # --- 1. Argument Parsing ---
    parser = argparse.ArgumentParser(
        description='Evaluate a pre-trained PyTorch model for adversarial robustness.'
    )
    parser.add_argument('--norm', default=2, choices=[2, 'inf'],
                        help="The norm to use in calculations (default: 2).")
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the saved model .pth file.')
    parser.add_argument('--model', type=str, required=True, choices=model_zoo,
                        help='Name of the model architecture to load.')
    parser.add_argument('--dataset', type=str, required=True, choices=['cifar10', 'mnist'],
                        help='Dataset to use for evaluation (cifar10 or mnist).')
    parser.add_argument('--epsilon', type=float, required=True,
                        help='Adversarial perturbation radius (L-infinity norm).')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Input batch size for evaluation. Default: 256.')
    parser.add_argument('--output_csv', type=str, 
                        help='Path to save the output CSV file. Results are appended.')
    parser.add_argument('--output_pkl', type=str,
                        help='Optional path to save the output Pickle file.')
    # SDP Crown parameters
    parser.add_argument('--start', default=0, type=int, help='start index for the dataset')

    parser.add_argument('--end', default=200, type=int, help='end index for the dataset')

    parser.add_argument('--lr_alpha', default=0.5, type=float, help='alpha learning rate')

    parser.add_argument('--lr_lambda', default=0.05, type=float, help='lambda learning rate')


    args = parser.parse_args()
    
    # For compatibility with underlying utility functions, 'epsilon' is aliased as 'radius'.
    args.radius = args.epsilon 
    norm = str(args.norm).lower()
    # --- 2. Setup Device, Data, and Model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Load the benchmark dataset and the pre-trained model
    images, targets, epsilon_rescaled, classes = load_dataset_benchmark(args)
    images = images[:2]
    targets = targets[:2]
    model = load_model(args, model_zoo, device)
    model.eval() # Set the model to evaluation mode
    # --- 3. Calculate Clean Accuracy (Baseline) ---
    print("Calculating clean accuracy on the test set...")
    with torch.no_grad():
        images = images.to(device)
        targets = targets.to(device)
        
        output = model(images)
        predictions = output.argmax(dim=1)
        
        # Calculate accuracy
        correct_predictions = torch.sum(predictions == targets).cpu().item()
        clean_accuracy = (correct_predictions / len(targets)) * 100
        
        # Get indices of correctly classified images, as robustness is only measured on these
        clean_indices = (predictions == targets).nonzero(as_tuple=True)[0]

    print(f"Clean Accuracy: {clean_accuracy:.2f}% ({correct_predictions}/{len(targets)})")
    print(f"Perturbation budget (epsilon): {args.epsilon} (rescaled to {epsilon_rescaled:.4f} for normalized data)")
    print(f"Evaluating robustness on {len(clean_indices)} correctly classified samples.")

    #REMOVE
    # clean_indices = (predictions == predictions).nonzero(as_tuple=True)[0]


    # --- 4. Run Robustness Evaluations with Timing ---
    print("\nStarting robustness evaluations...")

    L_2=1
    L = convert_lipschitz_constant(L_2, norm, input_dim=images[0].numel())

    # Note from author: It seems that the radius obtained should be rescaled to be comparable to epsilon.
    # This is a crucial step to ensure fair comparison across different data normalizations.
    _, certificates_cra, time_cra = compute_certificates_CRA(images, model, epsilon_rescaled, clean_indices, norm = norm, L = L)
    print(f"  - Certified Robustness (CRA): {certificates_cra:.2f}% | Time: {time_cra:.4f}s")
    # certificates_cra, time_cra = 0,0
    pgd_era, time_pgd = compute_pgd_era_and_time(images, targets, model, args.epsilon, clean_indices, norm = norm, dataset_name=args.dataset)
    print(f"  - Empirical Robustness (PGD): {pgd_era:.2f}% | Time: {time_pgd:.4f}s")
    # pgd_era, time_pgd = 0,0
    aa_era, time_aa = compute_autoattack_era_and_time(images, targets, model, args.epsilon, clean_indices, norm = norm, dataset_name=args.dataset)
    print(f"  - Empirical Robustness (AutoAttack): {aa_era:.2f}% | Time: {time_aa:.4f}s")
    # aa_era, time_aa = 0,0
    lirpa_alpha_vra, time_lirpa_alpha = compute_alphacrown_vra_and_time(images, targets, model, epsilon_rescaled, clean_indices, norm = args.norm)
    print(f"  - Certified Robustness (LIRPA α-CROWN): {lirpa_alpha_vra:.2f}% | Time: {time_lirpa_alpha:.4f}s")
    # lirpa_alpha_vra, time_lirpa_alpha = 0, 0
    if norm =='inf':
        lirpa_beta_vra, time_lirpa_beta = compute_alphabeta_vra_and_time(args.dataset, args.model, args.model_path, args.epsilon, "cifar_l2_norm.yaml", clean_indices)
        print(f"  - Certified Robustness (LIRPA β-CROWN): {lirpa_beta_vra:.2f}% | Time: {time_lirpa_beta:.4f}s")
        # lirpa_beta_vra, time_lirpa_beta = 0, 0
        sdp_crown_vra, time_sdp = 0, 0
    elif norm == '2':
        sdp_crown_vra, time_sdp = compute_sdp_crown_vra(images, targets, model, epsilon_rescaled, clean_indices, device, classes, args)
        print(f"  - Certified Robustness (SDP-CROWN): {sdp_crown_vra:.2f}% | Time: {time_sdp:.4f}s")
        # sdp_crown_vra, time_sdp = 0, 0
        lirpa_beta_vra, time_lirpa_beta = 0, 0
    # --- 5. Store and Save Results ---
    # The results are collected in a dictionary. The keys will become the CSV header.
    result_dict = {
        'epsilon': args.epsilon,
        'certificate': certificates_cra,
        'pgd': pgd_era,
        'aa': aa_era,
        'lirpa_alphacrown': lirpa_alpha_vra,
        'lirpa_betacrown': lirpa_beta_vra,
        'sdp': sdp_crown_vra,
        'time_cra': time_cra,
        'time_pgd': time_pgd,
        'time_aa': time_aa,
        'time_lirpa_alpha': time_lirpa_alpha,
        'time_lirpa_beta': time_lirpa_beta,
        'time_sdp': time_sdp,
    }
    # import pdb;pdb.set_trace()
    
    add_result_and_sort(result_dict, args.output_csv, norm=args.norm)
   

if __name__ == '__main__':
    main()# -*- coding: utf-8 -*-
"""
Model Robustness Evaluation Script

This script evaluates the adversarial robustness of a pre-trained PyTorch model.
It calculates the model's accuracy on clean data and its robust accuracy against
a specified epsilon-perturbation using various empirical and certified methods.

This version has been updated to also measure and record the execution time
for each evaluation method, providing valuable performance metrics.

Empirical methods include:
- Projected Gradient Descent (PGD)
- AutoAttack (AA)

Certified methods include:
- Certified Robustness Accuracy (CRA)
- LIRPA-based methods (alpha-CROWN, beta-CROWN)
- SDP-based verification (SDP-CROWN)

Results, including timings, are appended to a CSV file and optionally saved to a Pickle file.
"""
import torch
import torch.nn as nn
import numpy as np
from deel import torchlip
import argparse
import os
import os
import sys
import csv
import pickle
import time # Import the time module for timing operations

# --- Handle Local Module Imports ---
# This section adds parent directories to the Python path to locate custom modules.
# This is a common practice in research projects but can be replaced by creating a proper Python package.
# try:
    # Assumes models.py and utils.py are in the parent directory
from models import *
from project_utils import *

# --- Model Zoo ---
# A list of supported model architectures. The `--model_name` argument must match one of these.

model_zoo = {
     # --- Standard, non-Lipschitz constrained models ---
    "MNIST_MLP": MNIST_MLP,
    "MNIST_ConvSmall": MNIST_ConvSmall,
    "MNIST_ConvLarge": MNIST_ConvLarge,
    "CIFAR10_CNN_A": CIFAR10_CNN_A,
    "CIFAR10_CNN_B": CIFAR10_CNN_B,
    "CIFAR10_CNN_C": CIFAR10_CNN_C,
    "CIFAR10_ConvSmall": CIFAR10_ConvSmall,
    "CIFAR10_ConvDeep": CIFAR10_ConvDeep,
    "CIFAR10_ConvLarge": CIFAR10_ConvLarge,

    # --- 1-Lipschitz models (Spectral Normalization) ---
    "MLP_MNIST_1_LIP": MLP_MNIST_1_LIP,
    "ConvSmall_MNIST_1_LIP": ConvSmall_MNIST_1_LIP,
    "ConvLarge_MNIST_1_LIP": ConvLarge_MNIST_1_LIP,
    "CNNA_CIFAR10_1_LIP": CNNA_CIFAR10_1_LIP,
    "CNNB_CIFAR10_1_LIP": CNNB_CIFAR10_1_LIP,
    "CNNC_CIFAR10_1_LIP": CNNC_CIFAR10_1_LIP,
    "ConvSmall_CIFAR10_1_LIP": ConvSmall_CIFAR10_1_LIP,
    "ConvDeep_CIFAR10_1_LIP": ConvDeep_CIFAR10_1_LIP,
    "ConvLarge_CIFAR10_1_LIP": ConvLarge_CIFAR10_1_LIP,

    # --- 1-Lipschitz models (GNP technique) ---
    "MLP_MNIST_1_LIP_GNP": MLP_MNIST_1_LIP_GNP,
    "ConvSmall_MNIST_1_LIP_GNP": ConvSmall_MNIST_1_LIP_GNP,
    "ConvLarge_MNIST_1_LIP_GNP": ConvLarge_MNIST_1_LIP_GNP,
    "ConvLarge_MNIST_1_LIP_GNP_MaxMin": ConvLarge_MNIST_1_LIP_GNP_MaxMin,
    "CNNA_CIFAR10_1_LIP_GNP": CNNA_CIFAR10_1_LIP_GNP,
    "CNNB_CIFAR10_1_LIP_GNP": CNNB_CIFAR10_1_LIP_GNP,
    "CNNC_CIFAR10_1_LIP_GNP": CNNC_CIFAR10_1_LIP_GNP,
    "ConvSmall_CIFAR10_1_LIP_GNP": ConvSmall_CIFAR10_1_LIP_GNP,
    "ConvDeep_CIFAR10_1_LIP_GNP": ConvDeep_CIFAR10_1_LIP_GNP,
    "ConvLarge_CIFAR10_1_LIP_GNP": ConvLarge_CIFAR10_1_LIP_GNP,
}


def main():
    """
    Main function to parse arguments, load data/model, run evaluations, and save results.
    """
    # --- 1. Argument Parsing ---
    parser = argparse.ArgumentParser(
        description='Evaluate a pre-trained PyTorch model for adversarial robustness.'
    )
    parser.add_argument('--norm', default=2, choices=[2, 'inf'],
                        help="The norm to use in calculations (default: 2).")
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the saved model .pth file.')
    parser.add_argument('--model', type=str, required=True, choices=model_zoo,
                        help='Name of the model architecture to load.')
    parser.add_argument('--dataset', type=str, required=True, choices=['cifar10', 'mnist'],
                        help='Dataset to use for evaluation (cifar10 or mnist).')
    parser.add_argument('--epsilon', type=float, required=True,
                        help='Adversarial perturbation radius (L-infinity norm).')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Input batch size for evaluation. Default: 256.')
    parser.add_argument('--output_csv', type=str, 
                        help='Path to save the output CSV file. Results are appended.')
    parser.add_argument('--output_pkl', type=str,
                        help='Optional path to save the output Pickle file.')
    # SDP Crown parameters
    parser.add_argument('--start', default=0, type=int, help='start index for the dataset')

    parser.add_argument('--end', default=200, type=int, help='end index for the dataset')

    parser.add_argument('--lr_alpha', default=0.5, type=float, help='alpha learning rate')

    parser.add_argument('--lr_lambda', default=0.05, type=float, help='lambda learning rate')


    args = parser.parse_args()
    
    # For compatibility with underlying utility functions, 'epsilon' is aliased as 'radius'.
    args.radius = args.epsilon 
    norm = str(args.norm).lower()
    # --- 2. Setup Device, Data, and Model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Load the benchmark dataset and the pre-trained model
    images, targets, epsilon_rescaled, classes = load_dataset_benchmark(args)
    images = images[:2]
    targets = targets[:2]
    model = load_model(args, model_zoo, device)
    model.eval() # Set the model to evaluation mode
    # --- 3. Calculate Clean Accuracy (Baseline) ---
    print("Calculating clean accuracy on the test set...")
    with torch.no_grad():
        images = images.to(device)
        targets = targets.to(device)
        
        output = model(images)
        predictions = output.argmax(dim=1)
        
        # Calculate accuracy
        correct_predictions = torch.sum(predictions == targets).cpu().item()
        clean_accuracy = (correct_predictions / len(targets)) * 100
        
        # Get indices of correctly classified images, as robustness is only measured on these
        clean_indices = (predictions == targets).nonzero(as_tuple=True)[0]

    print(f"Clean Accuracy: {clean_accuracy:.2f}% ({correct_predictions}/{len(targets)})")
    print(f"Perturbation budget (epsilon): {args.epsilon} (rescaled to {epsilon_rescaled:.4f} for normalized data)")
    print(f"Evaluating robustness on {len(clean_indices)} correctly classified samples.")

    #REMOVE
    # clean_indices = (predictions == predictions).nonzero(as_tuple=True)[0]


    # --- 4. Run Robustness Evaluations with Timing ---
    print("\nStarting robustness evaluations...")

    L_2=1
    L = convert_lipschitz_constant(L_2, norm, input_dim=images[0].numel())

    if norm =='inf':
        lirpa_beta_vra, time_lirpa_beta = compute_alphabeta_vra_and_time(args.dataset, args.model, args.model_path, args.epsilon, "cifar_l2_norm.yaml", clean_indices)
        print(f"  - Certified Robustness (LIRPA β-CROWN): {lirpa_beta_vra:.2f}% | Time: {time_lirpa_beta:.4f}s")
   

if __name__ == '__main__':
    main()