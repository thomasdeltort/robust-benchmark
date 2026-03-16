#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Hybrid Robustness Verification Script
Combines 1-Lipschitz (L2) prefix verification with LIRPA/CROWN suffix verification,
and includes AutoAttack empirical baselines and CSV logging.
"""

import torch
import torch.nn as nn
import numpy as np
import argparse
import os
import sys
import csv
import time

# --- Lipschitz Layer Imports ---
try:
    from deel import torchlip
except ImportError:
    print("Warning: Could not import 'deel.torchlip'.")

# --- Model Imports (from your local files) ---
try:
    from models import *
    from project_utils import *
except ImportError:
    print("Warning: Ensure 'models.py' and 'project_utils.py' are in your path.")


def run_hybrid_verification(args, model_zoo):
    """
    Main function to run hybrid verification, empirical attacks, and save results.
    """
    # --- 1. Setup Device, Data, and Model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    images, targets, epsilon_rescaled, classes = load_dataset_benchmark(args)
    
    # Slice dataset based on args
    images = images[args.start:args.end]
    targets = targets[args.start:args.end]
    
    print(f"Loading 1-Lipschitz model: {args.model}")
    full_model = load_model(args, model_zoo, device)
    full_model.eval()

    # Print model layers to help choose split_index
    print("\n--- Model Layer Breakdown for --split_index ---")
    for i, layer in enumerate(full_model.children()):
        print(f"[{i}] {layer.__class__.__name__}: {layer}")
    print("-----------------------------------------------\n")

    # --- 2. Split the Model ---
    all_layers = list(full_model.children())
    split_idx = args.split_index
    
    if split_idx <= 0 or split_idx >= len(all_layers):
        raise ValueError(f"--split_index {split_idx} is out of bounds for this model.")

    f1_prefix = torchlip.Sequential(*all_layers[:split_idx]).to(device).eval()
    f2_suffix_lip = torchlip.Sequential(*all_layers[split_idx:]).to(device).eval()
    
    print("Converting model suffix for LIRPA using vanilla_export...")
    f2_suffix_vanilla = vanilla_export(f2_suffix_lip).to(device).eval()
    print(f"Model split complete. Prefix: {len(f1_prefix)} layers. Suffix: {len(f2_suffix_lip)} layers.")

    # ======================================================================
    # NEW: LIPSCHITZ ESTIMATION VIA POWER ITERATION
    # ======================================================================
    print("\n--- Estimating Lipschitz Constants (Power Iteration) ---")
    if "mnist" in args.dataset.lower():
        inp_shape = (1, 1, 28, 28)
    else:
        inp_shape = (1, 3, 32, 32)
        
    # 1. Estimate Full Model L (for Method 1 CRA)
    try:
        L_full_empirical = compute_model_lipschitz(full_model, input_shape=inp_shape, device=device)
        print(f"Full Model L_2 (Empirical): {L_full_empirical:.4f}")
    except Exception as e:
        print(f"Warning: PI failed for full model ({e}). Defaulting to 1.0")
        L_full_empirical = 1.0

    # 2. Estimate Prefix Model L (for Method 2 Hybrid)
    try:
        L_prefix_empirical = compute_model_lipschitz(f1_prefix, input_shape=inp_shape, device=device)
        print(f"Prefix Model L_2 (Empirical): {L_prefix_empirical:.4f}")
    except Exception as e:
        print(f"Warning: PI failed for prefix ({e}). Defaulting to 1.0")
        L_prefix_empirical = 1.0
    # ======================================================================

    # --- 3. Clean Accuracy Baseline ---
    print("\nCalculating clean accuracy on the test subset...")
    with torch.no_grad():
        images = images.to(device)
        targets = targets.to(device)
        output = full_model(images)
        predictions = output.argmax(dim=1)
        clean_indices = (predictions == targets).nonzero(as_tuple=True)[0]
        clean_accuracy = (len(clean_indices) / len(targets)) * 100
        
    print(f"Clean Accuracy: {clean_accuracy:.2f}% ({len(clean_indices)}/{len(targets)})")
    print(f"Verifying robustness for {len(clean_indices)} correctly classified samples.")
    print(f"Input Epsilon: {args.epsilon} (rescaled to {epsilon_rescaled:.4f})")
    
    # ======================================================================
    # Method 0: EMPIRICAL ROBUSTNESS (AutoAttack)
    # ======================================================================
    print("\n--- Method 0: Empirical Robustness (AutoAttack) ---")
    try:
        aa_acc, time_aa, robust_idxs_aa = compute_autoattack_era_and_time(
            images, targets, full_model, args.epsilon, clean_indices, 
            norm=str(args.norm), dataset_name=args.dataset, return_robust_points=True
        )
        print(f"AutoAttack Result: {aa_acc:.2f}% | Time: {time_aa:.4f}s")
    except NameError:
        print("Warning: compute_autoattack_era_and_time not found. Skipping AutoAttack.")
        aa_acc, time_aa, robust_idxs_aa = 0.0, 0.0, torch.tensor([])

    # ======================================================================
    # Method 1: BASELINE CERTIFIED (Global Lipschitz CRA)
    # ======================================================================
    norm_str = str(args.norm).lower()
    
    # Use the empirical L2 constant instead of a hardcoded 1
    L = convert_lipschitz_constant(L_full_empirical, norm_str, input_dim=images[0].numel())
    
    print(f"\n--- Method 1: Standard CRA (Global Lipschitz L={L:.4f}) ---")
    _, certificates_cra, time_cra, robust_idxs_cra = compute_certificates_CRA(
        images, full_model, epsilon_rescaled, clean_indices, 
        norm=norm_str, L=L, return_robust_points=True
    )
    print(f"CRA Result: {certificates_cra:.2f}% | Time: {time_cra:.4f}s")

    # ======================================================================
    # Method 2: HYBRID CERTIFIED VERIFICATION
    # ======================================================================
    print(f"\n--- Method 2: Hybrid Verification (Split at layer {args.split_index}) ---")
    
    # Scale intermediate perturbation by the Prefix Lipschitz constant
    intermediate_epsilon = float(epsilon_rescaled * L_prefix_empirical)
    print(f"Input Epsilon: {epsilon_rescaled:.4f} -> Intermediate Epsilon: {intermediate_epsilon:.4f}")

    # Step A: Run 1-Lip Prefix
    with torch.no_grad():
        z_k = f1_prefix(images)

    # Step B: Run Suffix Verification
    robust_idxs_hybrid = torch.tensor([])
    hybrid_vra = 0.0
    time_hybrid = 0.0

    if args.norm == 'inf':
        print("Mode: Hybrid L-Infinity (Alpha-CROWN)")
        hybrid_vra, time_hybrid, robust_idxs_hybrid = compute_alphacrown_vra_and_time(
            z_k, targets, f2_suffix_vanilla, intermediate_epsilon, clean_indices, args, # Scaled EPS
            batch_size=args.batch_size, norm='inf', x_U=None, x_L=None, return_robust_points=True
        )
    else:
    #     print("Mode: Hybrid L2")
    #     # groupsort = "GNP" in args.model or "Bjork" in args.model
    #     # print("SDP")
    #     # hybrid_vra, time_hybrid, robust_idxs_hybrid = compute_sdp_crown_vra(
    #     #     z_k, targets, f2_suffix_vanilla, float(intermediate_epsilon), clean_indices, 
    #     #     device, classes, args, batch_size=1, return_robust_points=True, x_U=None, x_L=None, groupsort=groupsort
    #     # )
    #     print("ALPHA")
    #     hybrid_vra, time_hybrid, robust_idxs_hybrid = compute_alphacrown_vra_and_time(
    #         z_k, targets, f2_suffix_vanilla, intermediate_epsilon, clean_indices, args, # Scaled EPS
    #         batch_size=args.batch_size, norm=2, x_U=None, x_L=None, return_robust_points=True
    #     )
    # print(f"Hybrid Result: {hybrid_vra:.2f}% | Time: {time_hybrid:.4f}s")
        print(f"\n[Hybrid L2 Challenge] Running Alpha-CROWN vs SDP-CROWN")
        groupsort = "GNP" in args.model or "Bjork" in args.model

        # 1. Run SDP-CROWN
        print(" -> Executing SDP-CROWN...")
        vra_sdp, t_sdp, idx_sdp = compute_sdp_crown_vra(
            z_k, targets, f2_suffix_vanilla, float(intermediate_epsilon), clean_indices, 
            device, classes, args, batch_size=1, return_robust_points=True, x_U=None, x_L=None, groupsort=groupsort
        )

        # 2. Run Alpha-CROWN
        print(" -> Executing Alpha-CROWN...")
        vra_alpha, t_alpha, idx_alpha = compute_alphacrown_vra_and_time(
            z_k, targets, f2_suffix_vanilla, intermediate_epsilon, clean_indices, args,
            batch_size=args.batch_size, norm=2, x_U=None, x_L=None, return_robust_points=True
        )

        # 3. SET COMPARISON LOGIC
        set_sdp = set(idx_sdp.cpu().tolist())
        set_alpha = set(idx_alpha.cpu().tolist())

        only_sdp = set_sdp - set_alpha
        only_alpha = set_alpha - set_sdp
        both = set_sdp.intersection(set_alpha)

        # 4. CLEAR REPORTING
        print("\n" + "="*50)
        print("VERIFIER PERFORMANCE SUMMARY")
        print("="*50)
        print(f"Alpha-CROWN Robust Acc : {vra_alpha:.2f}% (Time: {t_alpha:.4f}s)")
        print(f"SDP-CROWN   Robust Acc : {vra_sdp:.2f}% (Time: {t_sdp:.4f}s)")
        print("-" * 50)
        print(f"Verified by BOTH      : {len(both)}")
        print(f"Verified ONLY by SDP  : {len(only_sdp)}  <-- The 'SDP Gain'")
        print(f"Verified ONLY by Alpha: {len(only_alpha)}")
        
        if vra_sdp > vra_alpha:
            print(f"\nWINNER: SDP-CROWN (+{vra_sdp - vra_alpha:.2f}%)")
        elif vra_alpha > vra_sdp:
            print(f"\nWINNER: Alpha-CROWN (+{vra_alpha - vra_sdp:.2f}%)")
        else:
            print("\nRESULT: TIE (Results are mathematically identical)")
        print("="*50 + "\n")

        # Assign the best result to the main variable for logging
        hybrid_vra, time_hybrid, robust_idxs_hybrid = (vra_sdp, t_sdp, idx_sdp) if vra_sdp >= vra_alpha else (vra_alpha, t_alpha, idx_alpha)

    # ======================================================================
    # COMPARISON OF SETS
    # ======================================================================
    print("\n" + "="*40)
    print("ROBUSTNESS SET COMPARISON")
    print("="*40)

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
    print(f"Total Unique Verified (Union): {len(union):4d}")
    
    verified_union_acc = (len(union) / total_clean) * 100 if total_clean > 0 else 0.0

    if total_clean > 0:
        print(f"\nOverlap Percentage (of clean): {(len(intersection)/total_clean)*100:.2f}%")
        print(f"Hybrid Improvement over CRA:   {((len(set_hybrid) - len(set_cra))/total_clean)*100:+.2f}% (Absolute points)")
    
    print("="*40)

    # ======================================================================
    # CSV LOGGING
    # ======================================================================
    results_dict = {
        'model': args.model,
        'dataset': args.dataset,
        'norm': args.norm,
        'epsilon': args.epsilon,
        'split_index': args.split_index,
        'L_full': L_full_empirical,     # Added empirically computed L
        'L_prefix': L_prefix_empirical, # Added empirically computed prefix L
        'total_samples': len(targets),
        'clean_samples': total_clean,
        'clean_acc': clean_accuracy,
        'aa_acc': aa_acc,
        'time_aa': time_aa,
        'cra_acc': certificates_cra,
        'time_cra': time_cra,
        'hybrid_acc': hybrid_vra,
        'time_hybrid': time_hybrid,
        'verified_union_acc': verified_union_acc
    }

    # Ensure directory exists
    os.makedirs(os.path.dirname(args.output_csv) or '.', exist_ok=True)
    
    file_exists = os.path.isfile(args.output_csv)
    with open(args.output_csv, mode='a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=results_dict.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(results_dict)
        
    print(f"-> Successfully appended results to: {args.output_csv}")


if __name__ == '__main__':
    # --- Model Zoo ---
    model_zoo = {
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
    
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description='Perform HYBRID robustness verification (1-Lip Prefix + LIRPA Suffix).'
    )
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved model .pth file.')
    parser.add_argument('--model', type=str, required=True, choices=model_zoo.keys(), help='Name of the 1-Lipschitz model architecture.')
    parser.add_argument('--dataset', type=str, required=True, choices=['cifar10', 'mnist'], help='Dataset to use for evaluation.')
    parser.add_argument('--epsilon', type=float, required=True, help='Adversarial L2 perturbation radius (e.g., 0.5).')
    parser.add_argument('--split_index', type=int, required=True, help='The index of the layer to split *before*.')
    
    parser.add_argument('--norm', type=str, default='2', choices=['2', 'inf'],
                        help="Propagation norm for the suffix. '2' for lossless L2, 'inf' for lossy L-inf hand-off. (Default: 2)")
    
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for verification. Default: 256.')
    parser.add_argument('--start', default=0, type=int, help='start index for the dataset')
    parser.add_argument('--end', default=200, type=int, help='end index for the dataset')

    parser.add_argument('--lr_alpha', default=0.5, type=float, help='alpha learning rate')
    parser.add_argument('--lr_lambda', default=0.05, type=float, help='lambda learning rate')
    parser.add_argument('--high_tau', default=False, type=bool, help='Training temperature high/low')
    
    # --- CSV ARGUMENT ---
    parser.add_argument('--output_csv', type=str, default='results/hybrid_run_results.csv', 
                        help='Path to the output CSV file for saving results.')

    args = parser.parse_args()
    args.radius = args.epsilon # For load_dataset_benchmark compatibility

    # Convert 'norm' string to int if needed
    if args.norm == '2':
        args.norm = 2
        
    # Run the main verification logic
    run_hybrid_verification(args, model_zoo)

    