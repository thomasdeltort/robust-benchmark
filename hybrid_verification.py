#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Hybrid Robustness Verification Script

This script combines a library of utility functions (provided by the user)
with a hybrid verification method.

The verification splits a 1-Lipschitz (L2) model into:
1.  A 1-Lipschitz (L2) prefix: f_1
2.  A standard suffix: f_2

It uses the 1-Lipschitz property to get a perfect L2 certificate for the
intermediate activation z_k (||delta_z||_2 <= epsilon).

The '--norm' flag controls the "hand-off" to the suffix verifier (LIRPA):

1.  --norm 2 (Lossless L2 Verification):
    -   Hand-off: L2-ball -> L2-ball (lossless).
    -   Suffix Verifier: 'alpha-CROWN' propagating L2 bounds.

2.  --norm 'inf' (Lossy L-infinity Verification):
    -   Hand-off: L2-ball -> L-infinity-box (lossy over-approximation).
    -   Suffix Verifier: 'alpha-CROWN' propagating L-inf bounds.
"""

# --- 1. USER-PROVIDED UTILITY FUNCTIONS AND IMPORTS ---
import torch
import torch.nn as nn
import numpy as np
import argparse
import os
import sys
import csv
import pickle
import time
import copy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from torch.nn.utils.parametrize import is_parametrized

# --- LIRPA Imports (from user's code) ---
# import auto_LiRPA
# from auto_LiRPA import BoundedModule, BoundedTensor
# from auto_LiRPA.perturbations import PerturbationLpNorm

# --- torchattacks Import (from user's code) ---
import torchattacks

# --- Lipschitz Layer Imports (Needed for VGG models) ---
try:
    from deel import torchlip
except ImportError:
    print("Error: Could not import 'deel.torchlip'.")
    print("Please ensure 'deel' is installed (`pip install deel-torchlip`)")
    sys.exit(1)

# --- Model Imports (from user's code) ---
try:
    from models import * # Make sure this import works from your utils.py location
except ImportError:
    print("Error: 'models.py' not found. Make sure it is in the same directory or Python path.")
try:
    from project_utils import * # Make sure this import works from your utils.py location
except ImportError:
    print("Error: 'project_utils.py' not found. Make sure it is in the same directory or Python path.")


def run_hybrid_verification(args, model_zoo):
    """
    Main function to run hybrid verification with Set Comparison.
    """
    # --- Setup Device, Data, and Model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    images, targets, epsilon_rescaled, classes = load_dataset_benchmark(args)
    
    # Slice dataset based on args
    images = images[args.start:args.end]
    targets = targets[args.start:args.end]
    
    print(f"Loading 1-Lipschitz model (pre-export): {args.model}")
    full_model = load_model(args, model_zoo, device)
    full_model.eval()

    # --- Split the Model ---
    all_layers = list(full_model.children())
    split_idx = args.split_index
    
    if split_idx <= 0 or split_idx >= len(all_layers):
        raise ValueError(f"--split_index {split_idx} is out of bounds.")

    f1_prefix = torchlip.Sequential(*all_layers[:split_idx]).to(device).eval()
    f2_suffix_lip = torchlip.Sequential(*all_layers[split_idx:]).to(device).eval()
    
    print("Converting model suffix for LIRPA using vanilla_export...")
    f2_suffix_vanilla = vanilla_export(f2_suffix_lip).to(device).eval()

    print(f"Model split complete. Prefix: {len(f1_prefix)} layers. Suffix: {len(f2_suffix_lip)} layers.")

    # --- Get Cleanly Classified Indices ---
    print("Calculating clean accuracy on the test subset...")
    with torch.no_grad():
        images = images.to(device)
        targets = targets.to(device)
        output = full_model(images)
        predictions = output.argmax(dim=1)
        # Indices relative to the current slice (0 to batch_size)
        clean_indices = (predictions == targets).nonzero(as_tuple=True)[0]
        clean_accuracy = (len(clean_indices) / len(targets)) * 100
        
    print(f"Clean Accuracy: {clean_accuracy:.2f}% ({len(clean_indices)}/{len(targets)})")
    print(f"Verifying robustness for {len(clean_indices)} correctly classified samples.")
    print(f"Input L2 Epsilon: {args.epsilon} (rescaled to {epsilon_rescaled:.4f})")
    
    # ======================================================================
    # 1. BASELINE: Global Lipschitz CRA
    # ======================================================================
    norm_str = str(args.norm).lower()
    L_2 = 1
    L = convert_lipschitz_constant(L_2, norm_str, input_dim=images[0].numel())
    
    print("\n--- Method 1: Standard CRA (Global Lipschitz) ---")
    # Note: We pass return_robust_points=True
    _, certificates_cra, time_cra, robust_idxs_cra = compute_certificates_CRA(
        images, full_model, epsilon_rescaled, clean_indices, 
        norm=norm_str, L=L, return_robust_points=True
    )
    print(f"CRA Result: {certificates_cra:.2f}% | Time: {time_cra:.4f}s")

    # ======================================================================
    # 2. HYBRID METHOD
    # ======================================================================
    print("\n--- Method 2: Hybrid Verification ---")
    
    # Step A: Run 1-Lip Prefix
    with torch.no_grad():
        z_k = f1_prefix(images.to(device))

    # Step B: Run Suffix Verification
    robust_idxs_hybrid = torch.tensor([])
    hybrid_vra = 0.0
    time_hybrid = 0.0

    if args.norm == 'inf':
        print("Mode: Hybrid L-Infinity (Alpha-CROWN)")
        hybrid_vra, time_hybrid, robust_idxs_hybrid = compute_alphacrown_vra_and_time(
            z_k, targets, f2_suffix_vanilla, epsilon_rescaled, clean_indices, 
            norm='inf', return_robust_points=True
        )
    else: # args.norm == 2
        print("Mode: Hybrid L2 (SDP-CROWN)")
        # Using the function name defined in previous step: verified_sdp_crown
        hybrid_vra, time_hybrid, robust_idxs_hybrid = verified_sdp_crown(
            z_k, targets, f2_suffix_vanilla, epsilon_rescaled, clean_indices, 
            device, classes, args, return_robust_points=True
        )
        
    print(f"Hybrid Result: {hybrid_vra:.2f}% | Time: {time_hybrid:.4f}s")

    # ======================================================================
    # 3. COMPARISON OF SETS
    # ======================================================================
    print("\n" + "="*40)
    print("ROBUSTNESS SET COMPARISON")
    print("="*40)

    # Convert tensors to Python Sets for easy comparison
    # .tolist() handles empty tensors gracefully
    set_cra = set(robust_idxs_cra.cpu().tolist())
    set_hybrid = set(robust_idxs_hybrid.cpu().tolist())

    intersection = set_cra.intersection(set_hybrid)
    only_cra = set_cra - set_hybrid
    only_hybrid = set_hybrid - set_cra
    union = set_cra.union(set_hybrid)

    total_clean = len(clean_indices)

    print(f"Total Clean Samples Evaluated: {total_clean}")
    print("-" * 30)
    print(f"Verified by BOTH methods:      {len(intersection):4d}")
    print(f"Verified by CRA ONLY:          {len(only_cra):4d}  (Hybrid failed here)")
    print(f"Verified by HYBRID ONLY:       {len(only_hybrid):4d}  (CRA failed here)")
    print("-" * 30)
    print(f"Total Unique Verified:         {len(union):4d}")
    
    if total_clean > 0:
        print(f"\nOverlap Percentage:            {(len(intersection)/total_clean)*100:.2f}%")
        print(f"Hybrid Improvement over CRA:   {((len(set_hybrid) - len(set_cra))/total_clean)*100:+.2f}% (Absolute points)")
    
    print("="*40)

    # Optional: Save indices to file for later analysis
    # torch.save({'cra': robust_idxs_cra, 'hybrid': robust_idxs_hybrid}, 'robust_indices.pth')


    
if __name__ == '__main__':
    # --- Populate the model_zoo ---
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
        "CNNA_CIFAR10_1_LIP_GNP": CNNA_CIFAR10_1_LIP_GNP,
        "CNNB_CIFAR10_1_LIP_GNP": CNNB_CIFAR10_1_LIP_GNP,
        "CNNC_CIFAR10_1_LIP_GNP": CNNC_CIFAR10_1_LIP_GNP,
        "ConvSmall_CIFAR10_1_LIP_GNP": ConvSmall_CIFAR10_1_LIP_GNP,
        "ConvDeep_CIFAR10_1_LIP_GNP": ConvDeep_CIFAR10_1_LIP_GNP,
        "ConvLarge_CIFAR10_1_LIP_GNP": ConvLarge_CIFAR10_1_LIP_GNP,

        # --- NEW VGG GNP MODELS ---
        "VGG13_1_LIP_GNP_CIFAR10": VGG13_1_LIP_GNP_CIFAR10,
        "VGG16_1_LIP_GNP_CIFAR10": VGG16_1_LIP_GNP_CIFAR10,
        "VGG19_1_LIP_GNP_CIFAR10": VGG19_1_LIP_GNP_CIFAR10,
    }
    
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description='Perform HYBRID robustness verification (1-Lip Prefix + LIRPA Suffix).'
    )
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the saved model .pth file.')
    parser.add_argument('--model', type=str, required=True, choices=model_zoo.keys(),
                        help='Name of the 1-Lipschitz model architecture.')
    parser.add_argument('--dataset', type=str, required=True, choices=['cifar10', 'mnist'],
                        help='Dataset to use for evaluation.')
    parser.add_argument('--epsilon', type=float, required=True,
                        help='Adversarial L2 perturbation radius (e.g., 0.5).')
    parser.add_argument('--split_index', type=int, required=True,
                        help='The index of the layer to split *before*.')
    
    parser.add_argument('--norm', type=str, default='2', choices=['2', 'inf'],
                        help="Propagation norm for the suffix. '2' for lossless L2, 'inf' for lossy L-inf hand-off. (Default: 2)")
    
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for verification. Default: 256.')
    parser.add_argument('--start', default=0, type=int, help='start index for the dataset')
    parser.add_argument('--end', default=200, type=int, help='end index for the dataset')

    parser.add_argument('--lr_alpha', default=0.5, type=float, help='alpha learning rate')

    parser.add_argument('--lr_lambda', default=0.05, type=float, help='lambda learning rate')

    args = parser.parse_args()
    args.radius = args.epsilon # For load_dataset_benchmark compatibility

    # Convert 'norm' string to int if needed
    if args.norm == '2':
        args.norm = 2
        
    # Run the main verification logic
    run_hybrid_verification(args, model_zoo)