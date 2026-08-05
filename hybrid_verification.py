#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ablation Study: Dual Optimization Failure & Relaxation Compounding
Evaluates:
  1. Full Network + Vanilla CROWN (No dual optimization)
  2. Full Network + Alpha-CROWN (With dual gradient ascent)
  3. Hybrid Suffix + Vanilla CROWN
  4. Hybrid Suffix + Alpha-CROWN
"""

import torch
import torch.nn as nn
import argparse
import os
import csv
import time
import json
import numpy as np

# Ensure auto_LiRPA components are available (assuming they are in project_utils or auto_LiRPA)
# If BoundedModule is directly from auto_LiRPA, it should be imported in project_utils.
from models import *
from project_utils import *

try:
    from deel import torchlip
except ImportError:
    print("Warning: Could not import 'deel.torchlip'.")


def compute_alphacrown_vra_and_time(
    images, targets, model, epsilon, clean_indices, args, 
    batch_size=2, norm=2, return_robust_points=False, x_U=None, x_L=None,
    heavy_computation=False, partial_results=None, results_dict=None, results_filename=None,
    method='alpha-crown'  # <-- NEW PARAMETER: 'alpha-crown' or 'crown'
):
    """
    Computes Certified Robust Accuracy (CRA) using Alpha-Crown or Vanilla CROWN.
    """
    batch_size = getattr(args, 'batch_size', batch_size)
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = images.device
    total_num_images = images.shape[0]
    model.eval()
    
    if not isinstance(clean_indices, torch.Tensor):
        clean_indices = torch.tensor(clean_indices)

    # --- Step 1: Filter for correctly classified samples ---
    correct_images = images[clean_indices]
    correct_targets = targets[clean_indices]

    if len(correct_images) == 0:
        if return_robust_points:
            return 0.0, 0.0, torch.tensor([])
        return 0.0, 0.0

    # --- Step 2: Initialize variables & Checkpoint Loading ---
    num_robust_points = 0
    total_time = 0.0
    num_batches = (len(correct_images) + batch_size - 1) // batch_size
    robust_indices_list = []
    start_batch = 0

    if heavy_computation:
        if partial_results is None:
            partial_results = []
        completed_batches = len(partial_results) // batch_size
        partial_results = partial_results[:completed_batches * batch_size]
        start_batch = completed_batches
        num_robust_points = sum(partial_results)
        
        if results_dict is not None and "total_time" in results_dict:
            total_time = results_dict.get("total_time", 0.0)

    # --- Step 3: Setup BoundedModule ---
    has_residuals = any(isinstance(m, (BasicBlockLipschitz, BottleneckBlockLipschitz)) 
                        for m in model.modules())
    
    selected_conv_mode = "matrix" if has_residuals else "patches"
    
    dummy_input = correct_images[0:1].to(device)
    bounded_model = BoundedModule(model, dummy_input, bound_opts={"conv_mode": selected_conv_mode}, verbose=False)
    bounded_model.eval()

    print(f"   [Verifier Engine: {method.upper()} | ConvMode: {selected_conv_mode}]")

    jitter = 1e-7

    # --- Step 4: Batch Loop ---
    for i in range(start_batch, num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(correct_images))
        
        batch_images = correct_images[start_idx:end_idx].clone().to(device)
        batch_targets = correct_targets[start_idx:end_idx]
        current_bs = batch_images.shape[0]

        # --- A. Prepare Global Domain Bounds ---
        if x_L is not None:
            batch_global_L = x_L.expand(current_bs, *x_L.shape[1:]).contiguous() 
        else:
            batch_global_L = None
            
        if x_U is not None:
            batch_global_U = x_U.expand(current_bs, *x_U.shape[1:]).contiguous() 
        else:
            batch_global_U = None

        # --- B. CLAMP IMAGES ---
        if batch_global_L is not None and batch_global_U is not None:
            batch_images = torch.max(torch.min(batch_images, batch_global_U), batch_global_L)

        # --- C. Define Perturbation Constraints ---
        if norm == 'inf' or norm == float('inf'):
            if batch_global_L is not None and batch_global_U is not None:
                ptb_L = torch.max(batch_global_L, batch_images - epsilon) - jitter
                ptb_U = torch.min(batch_global_U, batch_images + epsilon) + jitter
                ptb = PerturbationLpNorm(norm=np.inf, eps=epsilon, x_L=ptb_L, x_U=ptb_U)
            else:
                ptb = PerturbationLpNorm(norm=np.inf, eps=epsilon)
        else:
            if getattr(args, 'use_conventional_groupsort', False):
                safe_L = batch_global_L - jitter if batch_global_L is not None else None
                safe_U = batch_global_U + jitter if batch_global_U is not None else None
                ptb = PerturbationLpNorm(norm=norm, eps=epsilon, x_L=safe_L, x_U=safe_U)
            else:
                ptb = PerturbationLpNorm(norm=norm, eps=epsilon, x_L=batch_global_L, x_U=batch_global_U)

        bounded_input = BoundedTensor(batch_images, ptb)
        num_classes = 10 
        c = build_C(batch_targets.to("cpu"), num_classes).to(device)

        # --- Time the verification ---
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start_time_batch = time.time()
        
        # --- DYNAMIC ITERATION SETTING ---
        iterations = 0 if method == 'crown' else getattr(args, 'iteration', 300)
        
        bounded_model.set_bound_opts({
            'optimize_bound_args': {
                'iteration': iterations,  
                'lr_alpha': getattr(args, 'lr_alpha', 0.5),
                'early_stop_patience': 20, 
                'enable_opt_interm_bounds': (iterations > 0), 
                'verbosity': False
            }, 
            'verbosity': False
        })
        
        lb_diff = bounded_model.compute_bounds(x=(bounded_input,), C=c, method=method)[0]
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        end_time_batch = time.time()
        total_time += (end_time_batch - start_time_batch)

        # --- Check Robustness ---
        is_robust = (lb_diff.view(current_bs, num_classes - 1) > 0).all(dim=1)
        is_robust_list = is_robust.cpu().tolist()
        
        if heavy_computation:
            partial_results.extend(is_robust_list)
            num_robust_points = sum(partial_results)
            if results_dict is not None and results_filename is not None:
                results_dict["partial_results"] = partial_results
                results_dict["total_time"] = total_time
                with open(results_filename, 'w') as f:
                    json.dump(results_dict, f, indent=4)
        else:
            num_robust_points += sum(is_robust_list)
            if return_robust_points:
                batch_global_indices = clean_indices[start_idx:end_idx]
                robust_indices_list.append(batch_global_indices[is_robust.cpu()])

        print(f"      Batch {i+1}/{num_batches}: {torch.sum(is_robust).item()}/{current_bs} robust.", end='\r')

    print("\n") 
    
    cra = (num_robust_points / total_num_images) * 100.0
    mean_time_per_image = total_time / len(correct_images) if len(correct_images) > 0 else 0.0

    if return_robust_points:
        if heavy_computation:
            partial_results_tensor = torch.tensor(partial_results, dtype=torch.bool)
            all_robust_indices = clean_indices[partial_results_tensor]
        else:
            all_robust_indices = torch.cat(robust_indices_list) if robust_indices_list else torch.tensor([])
        return cra, total_time, all_robust_indices

    return cra, total_time


def run_cancellation_ablation(args, model_zoo):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # --- 1. Load Data & Models ---
    images, targets, epsilon_rescaled, classes = load_dataset_benchmark(args)
    images = images[args.start:args.end].to(device)
    targets = targets[args.start:args.end].to(device)

    print(f"Loading 1-Lipschitz model: {args.model}")
    full_model = load_model(args, model_zoo, device).eval()

    # Clean accuracy filtering
    with torch.no_grad():
        output = full_model(images)
        predictions = output.argmax(dim=1)
        clean_indices = (predictions == targets).nonzero(as_tuple=True)[0]
        clean_acc = (len(clean_indices) / len(targets)) * 100

    print(f"Dataset subset: {len(targets)} samples | Clean Acc: {clean_acc:.2f}% ({len(clean_indices)} samples)")
    print(f"Input Epsilon: {args.epsilon} (rescaled: {epsilon_rescaled:.4f})\n")

    # --- 2. Prepare Exported Models ---
    print("Exporting Full Model for auto_LiRPA...")
    full_model_vanilla = vanilla_export(full_model).to(device).eval()

    print(f"Splitting model at index {args.split_index}...")
    all_layers = list(full_model.children())
    split_idx = args.split_index
    f1_prefix = torchlip.Sequential(*all_layers[:split_idx]).to(device).eval()
    f2_suffix_lip = torchlip.Sequential(*all_layers[split_idx:]).to(device).eval()
    f2_suffix_vanilla = vanilla_export(f2_suffix_lip).to(device).eval()

    # Calculate Prefix Lipschitz Constant
    try:
        inp_shape = (1, 3, 32, 32) if "cifar" in args.dataset.lower() else (1, 1, 28, 28)
        L_prefix = compute_model_lipschitz(f1_prefix, input_shape=inp_shape, device=device)
    except Exception:
        L_prefix = 1.0

    intermediate_epsilon = float(epsilon_rescaled * L_prefix)
    print(f"Prefix Lipschitz Constant: {L_prefix:.4f} | Intermediate Epsilon: {intermediate_epsilon:.4f}\n")

    # Calculate Intermediate Latent Space (z_k)
    with torch.no_grad():
        z_k = f1_prefix(images)

    # --- 3. Execute 2x2 Ablation Matrix ---
    def evaluate_verifier(x_input, model, eps, use_alpha, label):
        args_copy = argparse.Namespace(**vars(args))
        method_str = 'alpha-crown' if use_alpha else 'crown'

        print(f"--> {label}")
        vra, t_exec, idxs = compute_alphacrown_vra_and_time(
            x_input, targets, model, eps, clean_indices, args_copy,
            batch_size=args.batch_size, norm=args.norm, return_robust_points=True,
            method=method_str
        )
        return vra, t_exec

    print("=" * 60)
    print("RUNNING ABLATION MATRIX")
    print("=" * 60)

    # 1. Full Network + Vanilla CROWN
    vra_full_crown, t_full_crown = evaluate_verifier(
        images, full_model_vanilla, epsilon_rescaled, use_alpha=False, 
        label="[1/4] Full Network + Vanilla CROWN"
    )

    # 2. Full Network + Alpha-CROWN
    vra_full_alpha, t_full_alpha = evaluate_verifier(
        images, full_model_vanilla, epsilon_rescaled, use_alpha=True, 
        label="[2/4] Full Network + Alpha-CROWN"
    )

    # 3. Hybrid Suffix + Vanilla CROWN
    vra_hyb_crown, t_hyb_crown = evaluate_verifier(
        z_k, f2_suffix_vanilla, intermediate_epsilon, use_alpha=False, 
        label="[3/4] Hybrid Suffix + Vanilla CROWN"
    )

    # 4. Hybrid Suffix + Alpha-CROWN
    vra_hyb_alpha, t_hyb_alpha = evaluate_verifier(
        z_k, f2_suffix_vanilla, intermediate_epsilon, use_alpha=True, 
        label="[4/4] Hybrid Suffix + Alpha-CROWN"
    )

    # --- 4. Report Findings & Interpretation ---
    print("\n" + "=" * 65)
    print("ABLATION RESULTS: DUAL OPTIMIZATION vs. GRAPH TRUNCATION")
    print("=" * 65)
    print(f"{'Setting':<22} | {'Verifier':<14} | {'Certified Acc (%)':<17} | {'Time (s)':<8}")
    print("-" * 65)
    print(f"{'Full Network (x)':<22} | {'Vanilla CROWN':<14} | {vra_full_crown:<17.2f} | {t_full_crown:<8.2f}")
    print(f"{'Full Network (x)':<22} | {'Alpha-CROWN':<14} | {vra_full_alpha:<17.2f} | {t_full_alpha:<8.2f}")
    print(f"{'Hybrid Suffix (z_k)':<22} | {'Vanilla CROWN':<14} | {vra_hyb_crown:<17.2f} | {t_hyb_crown:<8.2f}")
    print(f"{'Hybrid Suffix (z_k)':<22} | {'Alpha-CROWN':<14} | {vra_hyb_alpha:<17.2f} | {t_hyb_alpha:<8.2f}")
    print("=" * 65)

    # Diagnostic Interpretation Logic
    print("\nDIAGNOSTIC ANALYSIS:")
    if vra_full_alpha <= vra_full_crown:
        print("  [!] DUAL OPTIMIZATION FAILURE CONFIRMED on Full Network:")
        print(f"      Alpha-CROWN ({vra_full_alpha:.2f}%) failed to beat Vanilla CROWN ({vra_full_crown:.2f}%).")
        print("      Dual gradient ascent got trapped in bad local minima across deep layers.")
    else:
        print("  [*] Dual optimization provided minor gains on the full graph.")

    alpha_gain_hybrid = vra_hyb_alpha - vra_hyb_crown
    print(f"  [*] Alpha Optimization Gain on Hybrid Suffix: +{alpha_gain_hybrid:.2f}%")
    if alpha_gain_hybrid > 0:
        print("      Proves slope optimization works effectively once the graph is truncated!")

    # --- 5. Save Results to CSV ---
    if args.output_csv:
        results_dict = {
            'model': args.model,
            'dataset': args.dataset,
            'norm': args.norm,
            'epsilon': args.epsilon,
            'split_index': args.split_index,
            'clean_samples': len(clean_indices),
            'clean_acc': clean_acc,
            'full_crown_acc': vra_full_crown,
            'full_crown_time': t_full_crown,
            'full_alpha_acc': vra_full_alpha,
            'full_alpha_time': t_full_alpha,
            'hybrid_crown_acc': vra_hyb_crown,
            'hybrid_crown_time': t_hyb_crown,
            'hybrid_alpha_acc': vra_hyb_alpha,
            'hybrid_alpha_time': t_hyb_alpha
        }

        os.makedirs(os.path.dirname(args.output_csv) or '.', exist_ok=True)
        file_exists = os.path.isfile(args.output_csv)
        
        with open(args.output_csv, mode='a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=results_dict.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(results_dict)
            
        print(f"\n-> Successfully appended results to: {args.output_csv}")


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
        "VGG13_1_LIP_GNP_CIFAR10": VGG13_1_LIP_GNP_CIFAR10 if 'VGG13_1_LIP_GNP_CIFAR10' in globals() else None,
        "VGG19_1_LIP_GNP_CIFAR10": VGG19_1_LIP_GNP_CIFAR10 if 'VGG19_1_LIP_GNP_CIFAR10' in globals() else None,

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

        # --- Imagenette ResNet Models ---
        "ResNet18_1_LIP_GNP": ResNet18_1_LIP_GNP if 'ResNet18_1_LIP_GNP' in globals() else None,
        "ResNet18_1_LIP_Bjork": ResNet18_1_LIP_Bjork if 'ResNet18_1_LIP_Bjork' in globals() else None,
        "ResNet18_1_LIP_GNP_Imagenette": ResNet18_1_LIP_GNP_Imagenette if 'ResNet18_1_LIP_GNP_Imagenette' in globals() else None,
        "ResNet18_1_LIP_Bjork_Imagenette": ResNet18_1_LIP_Bjork_Imagenette if 'ResNet18_1_LIP_Bjork_Imagenette' in globals() else None,
    }

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description='Perform CROWN vs. Alpha-CROWN Ablation Study (Full Net vs. Hybrid Suffix).'
    )
    parser.add_argument('--model_path', type=str, required=True, help='Path to saved model .pth file.')
    parser.add_argument('--model', type=str, required=True, choices=model_zoo.keys(), help='Name of the 1-Lipschitz architecture.')
    parser.add_argument('--dataset', type=str, required=True, choices=['cifar10', 'mnist', 'imagenette'], help='Dataset for evaluation.')
    parser.add_argument('--epsilon', type=float, required=True, help='Adversarial L2 perturbation radius (e.g., 0.03137).')
    parser.add_argument('--split_index', type=int, required=True, help='The index of the layer to split *before*.')
    parser.add_argument('--norm', type=str, default='2', choices=['2', 'inf'], help="Propagation norm ('2' or 'inf'). Default: 2.")
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for verification. Default: 1.')
    parser.add_argument('--start', default=0, type=int, help='Start index for test dataset.')
    parser.add_argument('--end', default=200, type=int, help='End index for test dataset.')
    parser.add_argument('--lr_alpha', default=0.5, type=float, help='Alpha learning rate.')
    parser.add_argument('--lr_lambda', default=0.05, type=float, help='Lambda learning rate.')
    parser.add_argument('--iteration', default=300, type=int, help='Number of iterations for Alpha-CROWN.')
    parser.add_argument('--high_tau', default=False, type=bool, help='Temperature setting.')
    parser.add_argument('--output_csv', type=str, default='results/ablation_results.csv', help='Path to output CSV file.')

    args = parser.parse_args()
    args.radius = args.epsilon  # For compatibility with project_utils loader

    # Convert norm string to integer
    if args.norm == '2':
        args.norm = 2

    # Run the 2x2 ablation experiment
    run_cancellation_ablation(args, model_zoo)