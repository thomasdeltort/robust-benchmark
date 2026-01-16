# -*- coding: utf-8 -*-
"""
Model Robustness Evaluation Script

This script evaluates the adversarial robustness of a pre-trained PyTorch model.
It calculates the model's accuracy on clean data and its robust accuracy against
a specified epsilon-perturbation using various empirical and certified methods.

This version has been updated to also measure and record the execution time
for each evaluation method, providing valuable performance metrics.

It also now tracks exactly *which* points are robust for each method and saves
binary vectors to a pickle file.
"""
import torch
import torch.nn as nn
import numpy as np
from deel import torchlip
import argparse
import os
import sys
import csv
import pickle
import time 

# --- Handle Local Module Imports ---
from models import *
from project_utils import *
from robustness_registery import *

# --- Model Zoo ---
model_zoo = {
    # --- Standard, non-Lipschitz constrained models ---
    "MNIST_MLP": MNIST_MLP if 'MNIST_MLP' in globals() else None,
    "MNIST_ConvSmall": MNIST_ConvSmall if 'MNIST_ConvSmall' in globals() else None,
    "MNIST_ConvLarge": MNIST_ConvLarge if 'MNIST_ConvLarge' in globals() else None,
    "CIFAR10_CNN_A": CIFAR10_CNN_A if 'CIFAR10_CNN_A' in globals() else None,
    "CIFAR10_CNN_B": CIFAR10_CNN_B if 'CIFAR10_CNN_B' in globals() else None,
    "CIFAR10_CNN_C": CIFAR10_CNN_C if 'CIFAR10_CNN_C' in globals() else None,
    "CIFAR10_ConvSmall": CIFAR10_ConvSmall if 'CIFAR10_ConvSmall' in globals() else None,
    "CIFAR10_ConvDeep": CIFAR10_ConvDeep if 'CIFAR10_ConvDeep' in globals() else None,
    "CIFAR10_ConvLarge": CIFAR10_ConvLarge if 'CIFAR10_ConvLarge' in globals() else None,

    # --- 1-Lipschitz models (Spectral Normalization) ---
    "MLP_MNIST_1_LIP": MLP_MNIST_1_LIP if 'MLP_MNIST_1_LIP' in globals() else None,
    "ConvSmall_MNIST_1_LIP": ConvSmall_MNIST_1_LIP if 'ConvSmall_MNIST_1_LIP' in globals() else None,
    "ConvLarge_MNIST_1_LIP": ConvLarge_MNIST_1_LIP if 'ConvLarge_MNIST_1_LIP' in globals() else None,
    "CNNA_CIFAR10_1_LIP": CNNA_CIFAR10_1_LIP if 'CNNA_CIFAR10_1_LIP' in globals() else None,
    "CNNB_CIFAR10_1_LIP": CNNB_CIFAR10_1_LIP if 'CNNB_CIFAR10_1_LIP' in globals() else None,
    "CNNC_CIFAR10_1_LIP": CNNC_CIFAR10_1_LIP if 'CNNC_CIFAR10_1_LIP' in globals() else None,
    "ConvSmall_CIFAR10_1_LIP": ConvSmall_CIFAR10_1_LIP if 'ConvSmall_CIFAR10_1_LIP' in globals() else None,
    "ConvDeep_CIFAR10_1_LIP": ConvDeep_CIFAR10_1_LIP if 'ConvDeep_CIFAR10_1_LIP' in globals() else None,
    "ConvLarge_CIFAR10_1_LIP": ConvLarge_CIFAR10_1_LIP if 'ConvLarge_CIFAR10_1_LIP' in globals() else None,

    # --- 1-Lipschitz models (GNP technique) ---
    "MLP_MNIST_1_LIP_GNP": MLP_MNIST_1_LIP_GNP if 'MLP_MNIST_1_LIP_GNP' in globals() else None,
    "ConvSmall_MNIST_1_LIP_GNP": ConvSmall_MNIST_1_LIP_GNP if 'ConvSmall_MNIST_1_LIP_GNP' in globals() else None,
    "ConvLarge_MNIST_1_LIP_GNP": ConvLarge_MNIST_1_LIP_GNP if 'ConvLarge_MNIST_1_LIP_GNP' in globals() else None,
    "CNNA_CIFAR10_1_LIP_GNP": CNNA_CIFAR10_1_LIP_GNP if 'CNNA_CIFAR10_1_LIP_GNP' in globals() else None,
    "CNNA_CIFAR10_1_LIP_GNP_torchlip": CNNA_CIFAR10_1_LIP_GNP_torchlip if 'CNNA_CIFAR10_1_LIP_GNP_torchlip' in globals() else None,
    "CNNA_CIFAR10_1_LIP_GNP_circular": CNNA_CIFAR10_1_LIP_GNP_circular if 'CNNA_CIFAR10_1_LIP_GNP_circular' in globals() else None,
    "CNNB_CIFAR10_1_LIP_GNP": CNNB_CIFAR10_1_LIP_GNP if 'CNNB_CIFAR10_1_LIP_GNP' in globals() else None,
    "CNNC_CIFAR10_1_LIP_GNP": CNNC_CIFAR10_1_LIP_GNP if 'CNNC_CIFAR10_1_LIP_GNP' in globals() else None,
    "ConvSmall_CIFAR10_1_LIP_GNP": ConvSmall_CIFAR10_1_LIP_GNP if 'ConvSmall_CIFAR10_1_LIP_GNP' in globals() else None,
    "ConvDeep_CIFAR10_1_LIP_GNP": ConvDeep_CIFAR10_1_LIP_GNP if 'ConvDeep_CIFAR10_1_LIP_GNP' in globals() else None,
    "ConvLarge_CIFAR10_1_LIP_GNP": ConvLarge_CIFAR10_1_LIP_GNP if 'ConvLarge_CIFAR10_1_LIP_GNP' in globals() else None,
    "VGG13_1_LIP_GNP_CIFAR10" : VGG13_1_LIP_GNP_CIFAR10 if 'VGG13_1_LIP_GNP_CIFAR10' in globals() else None,
    "VGG19_1_LIP_GNP_CIFAR10" : VGG19_1_LIP_GNP_CIFAR10 if 'VGG19_1_LIP_GNP_CIFAR10' in globals() else None,

    # --- 1-Lipschitz models (Bjork technique) ---
    "MLP_MNIST_1_LIP_Bjork": MLP_MNIST_1_LIP_Bjork if 'MLP_MNIST_1_LIP_Bjork' in globals() else None,
    "ConvSmall_MNIST_1_LIP_Bjork": ConvSmall_MNIST_1_LIP_Bjork if 'ConvSmall_MNIST_1_LIP_Bjork' in globals() else None,
    "ConvLarge_MNIST_1_LIP_Bjork": ConvLarge_MNIST_1_LIP_Bjork if 'ConvLarge_MNIST_1_LIP_Bjork' in globals() else None,
    "CNNA_CIFAR10_1_LIP_Bjork": CNNA_CIFAR10_1_LIP_Bjork if 'CNNA_CIFAR10_1_LIP_Bjork' in globals() else None,
    "CNNB_CIFAR10_1_LIP_Bjork": CNNB_CIFAR10_1_LIP_Bjork if 'CNNB_CIFAR10_1_LIP_Bjork' in globals() else None,
    "CNNC_CIFAR10_1_LIP_Bjork": CNNC_CIFAR10_1_LIP_Bjork if 'CNNC_CIFAR10_1_LIP_Bjork' in globals() else None,
    "ConvSmall_CIFAR10_1_LIP_Bjork": ConvSmall_CIFAR10_1_LIP_Bjork if 'ConvSmall_CIFAR10_1_LIP_Bjork' in globals() else None,
    "ConvDeep_CIFAR10_1_LIP_Bjork": ConvDeep_CIFAR10_1_LIP_Bjork if 'ConvDeep_CIFAR10_1_LIP_Bjork' in globals() else None,
    "ConvLarge_CIFAR10_1_LIP_Bjork": ConvLarge_CIFAR10_1_LIP_Bjork if 'ConvLarge_CIFAR10_1_LIP_Bjork' in globals() else None,
    "VGG13_1_LIP_Bjork_CIFAR10": VGG13_1_LIP_Bjork_CIFAR10 if 'VGG13_1_LIP_Bjork_CIFAR10' in globals() else None,
    "VGG16_1_LIP_Bjork_CIFAR10": VGG16_1_LIP_Bjork_CIFAR10 if 'VGG16_1_LIP_Bjork_CIFAR10' in globals() else None,
    "VGG19_1_LIP_Bjork_CIFAR10": VGG19_1_LIP_Bjork_CIFAR10 if 'VGG19_1_LIP_Bjork_CIFAR10' in globals() else None,
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
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Input batch size for evaluation. Default: 32.')
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
    model = load_model(args, model_zoo, device)
    model.eval() # Set the model to evaluation mode
    
    # --- 3. Calculate Clean Accuracy (Baseline) ---
    print("Calculating clean accuracy on the test set...")
    with torch.no_grad():
        images = images.to(device)
        targets = targets.to(device)
        print(model)
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

    # --- Initialize Robustness Registry ---
    registry = RobustnessRegistry(
        model_name=args.model,
        norm=norm,
        total_dataset_size=images.shape[0],
        save_dir="./results/robust_points"
    )
    # Register "clean" points (originally correct)
    registry.register(args.epsilon, "clean", clean_indices)

    # --- 4. Run Robustness Evaluations with Timing ---
    print("\nStarting robustness evaluations...")

    # --- PGD (Currently disabled/placeholder) ---
    pgd_era, time_pgd = 0, 0

    # --- AutoAttack (MOVED TO START) ---
    # We compute empirical robustness first. If AA finds adversaries for everything (0%),
    # we can skip certified methods as they cannot be > 0%.
    aa_era, time_aa, idx_aa = compute_autoattack_era_and_time(
        images, targets, model, args.epsilon, clean_indices, 
        norm=norm, dataset_name=args.dataset, return_robust_points=True
    )
    registry.register(args.epsilon, "autoattack", idx_aa)
    print(f"  - Empirical Robustness (AutoAttack): {aa_era:.2f}% | Time: {time_aa:.4f}s")

    # --- Check if AutoAttack is 0% ---
    if aa_era > 0:
        # --- Certificate (CRA) ---
        L_2=1
        L = convert_lipschitz_constant(L_2, norm, input_dim=images[0].numel())
        
        _, certificates_cra, time_cra, idx_cert = compute_certificates_CRA(
            images, model, epsilon_rescaled, clean_indices, 
            norm=norm, L=L, return_robust_points=True
        )
        registry.register(args.epsilon, "certificate", idx_cert)
        print(f"  - Certified Robustness (CRA): {certificates_cra:.2f}% | Time: {time_cra:.4f}s")

        # --- Bounds setup for Crown methods ---
        if "mnist" in args.dataset.lower():
            # MNIST: Raw range is usually [0, 1]
            x_L = torch.zeros((1, 1, 28, 28), device=device)
            x_U = torch.ones((1, 1, 28, 28), device=device)

        elif "cifar" in args.dataset.lower():
            # CIFAR: Range [0, 1] transformed by (image - mean) / std
            MEANS = torch.tensor([0.4914, 0.4822, 0.4465], device=device).view(1, 3, 1, 1)
            STD = torch.tensor([0.225, 0.225, 0.225], device=device).view(1, 3, 1, 1)
            
            x_L = ((0.0 - MEANS) / STD).expand(1, 3, 32, 32).contiguous()
            x_U = ((1.0 - MEANS) / STD).expand(1, 3, 32, 32).contiguous()
        else:
            raise ValueError(f"Bounds not defined for dataset: {args.dataset}")

        print("Bounds shapes:", x_L.shape, x_U.shape)
        
        # --- Alpha-CROWN ---
        lirpa_alpha_vra, time_lirpa_alpha, idx_alpha = compute_alphacrown_vra_and_time(
            images, targets, model, epsilon_rescaled, clean_indices, 
            batch_size=args.batch_size, norm=args.norm, 
            return_robust_points=True, x_U=x_U, x_L=x_L
        )
        registry.register(args.epsilon, "alphacrown", idx_alpha)
        print(f"  - Certified Robustness (LIRPA α-CROWN): {lirpa_alpha_vra:.2f}% | Time: {time_lirpa_alpha:.4f}s")

        # --- Beta-CROWN / SDP-CROWN ---
        # Initialize
        lirpa_beta_vra, time_lirpa_beta = 0, 0
        sdp_crown_vra, time_sdp = 0, 0

        if norm =='inf':
            if args.dataset == 'mnist':
                config_file = 'mnist_cnn_a_adv.yaml'
            else:
                config_file = 'cifar_l2_norm.yaml'
                
            lirpa_beta_vra, time_lirpa_beta, idx_beta = compute_alphabeta_vra_and_time(
                args.dataset, args.model, args.model_path, args.epsilon, 
                config_file, clean_indices, total_samples=images.shape[0],
                norm=args.norm, return_robust_points=True
            )
            registry.register(args.epsilon, "betacrown", idx_beta)
            print(f"  - Certified Robustness (LIRPA β-CROWN): {lirpa_beta_vra:.2f}% | Time: {time_lirpa_beta:.4f}s")
            
        elif norm == '2':
            swapped_model = replace_groupsort(model, images[:1])
            # TODO Convert the model to have gs2 'sdp crown friendly'
            sdp_crown_vra, time_sdp, idx_sdp = compute_sdp_crown_vra(
                images, targets, swapped_model, epsilon_rescaled, clean_indices, 
                device, classes, args, batch_size=1, 
                return_robust_points=True, x_U=x_U, x_L=x_L
            )
            registry.register(args.epsilon, "sdp", idx_sdp)
            print(f"  - Certified Robustness (SDP-CROWN): {sdp_crown_vra:.2f}% | Time: {time_sdp:.4f}s")

    else:
        # --- Skip everything if AutoAttack is 0 ---
        print("AutoAttack ERA is 0.00%. Skipping all certified robustness calculations (setting values to 0).")
        certificates_cra, time_cra = 0, 0
        lirpa_alpha_vra, time_lirpa_alpha = 0, 0
        lirpa_beta_vra, time_lirpa_beta = 0, 0
        sdp_crown_vra, time_sdp = 0, 0
        # No robust points to register for cert methods


    # --- 5. Store and Save Results ---
    # Save the registry to the pickle file
    registry.save()

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
    
    add_result_and_sort(result_dict, args.output_csv, norm=args.norm)


if __name__ == '__main__':
    main()