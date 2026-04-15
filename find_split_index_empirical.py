#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ground-Truth Hybrid VRA Profiler
Calculates and plots the actual Verified Robust Accuracy (VRA) for 
Alpha-CROWN, SDP-CROWN, and pure 1-Lipschitz CRA across all candidate split layers.
Candidate layers are restricted to those immediately preceding an activation.
Includes strict OOM protection.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import argparse
import os

# --- Import your local project utilities ---
try:
    from deel import torchlip
except ImportError:
    print("Warning: Could not import 'deel.torchlip'.")

from models import *
from project_utils import *

def profile_true_vras(args, model_zoo):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ======================================================================
    # 1. LOAD DATA AND MODEL
    # ======================================================================
    print("Loading dataset and model...")
    images, targets, epsilon_rescaled, classes = load_dataset_benchmark(args)
    
    # Slice dataset to save time during profiling
    images = images[args.start:args.end].to(device)
    targets = targets[args.start:args.end].to(device)
    
    full_model = load_model(args, model_zoo, device)
    full_model.eval()
    
    # Input shape for Lipschitz estimation
    if "mnist" in args.dataset.lower():
        inp_shape = (1, 1, 28, 28)
    elif "imagenette" in args.dataset.lower():
        inp_shape = (1, 3, 224, 224)
    else:
        inp_shape = (1, 3, 32, 32)

    # ======================================================================
    # 2. GET CLEAN ACCURACY & CLEAN INDICES
    # ======================================================================
    print("\nCalculating clean accuracy baseline...")
    with torch.no_grad():
        imgs_dev = images.to(device)
        targs_dev = targets.to(device)
        output = full_model(imgs_dev)
        predictions = output.argmax(dim=1)
        clean_indices = (predictions == targs_dev).nonzero(as_tuple=True)[0]
        
    print(f"Clean samples to verify: {len(clean_indices)} / {len(targets)}")
    if len(clean_indices) == 0:
        print("No clean samples to verify. Exiting.")
        return

    # ======================================================================
    # 3. COMPUTE GLOBAL 1-LIPSCHITZ CRA (BASELINE)
    # ======================================================================
    print("\n--- Computing Global 1-Lipschitz CRA Baseline ---")
    
    # Removed generic try/except. Let it crash if compute_model_lipschitz is broken!
    L_full = compute_model_lipschitz(full_model, input_shape=inp_shape, device=device)
        
    norm_str = str(args.norm).lower()
    L_converted = convert_lipschitz_constant(L_full, norm_str, input_dim=images[0].numel())
    
    clean_indices = clean_indices.to(device)

    _, cra_acc, _, _ = compute_certificates_CRA(
        images, full_model, epsilon_rescaled, clean_indices.to(device), 
        norm=norm_str, L=L_converted, return_robust_points=True
    )
    print(f"Global CRA Baseline: {cra_acc:.2f}%")

    # ======================================================================
    # 4. FILTER FOR CANDIDATE SPLIT INDICES
    # ======================================================================
    print("\n" + "="*80)
    print(" FILTERING CANDIDATE SPLIT INDICES")
    print("="*80)
    
    all_layers = list(full_model.children())
    candidate_indices = [0]
    
    for k in range(1, len(all_layers)):
        # We split at 'k'. Prefix ends at k-1, Suffix starts at k.
        layer_before_split = all_layers[k-1]
        layer_after_split = all_layers[k]
        
        # A valid candidate is splitting AFTER a Conv/Linear and BEFORE an Activation
        if isinstance(layer_before_split, (nn.Conv2d, nn.Linear)):
            # If the next layer is NOT a Conv, Linear, or Flatten, we assume it is an activation/pool
            if not isinstance(layer_after_split, (nn.Conv2d, nn.Linear, nn.Flatten)):
                candidate_indices.append(k)
                print(f"[*] Layer {k:2d} added to candidates: {layer_before_split.__class__.__name__} -> {layer_after_split.__class__.__name__}")
    if len(all_layers) not in candidate_indices:
        candidate_indices.append(len(all_layers))
        
    print(f"\nReduced search space from {len(all_layers)-1} layers down to {len(candidate_indices)} candidate splits.")

#    # ======================================================================
#    # GLOBAL BOUNDS SETUP FOR CROWN VERIFICATION
#    # ======================================================================
#    x_L = torch.zeros(inp_shape, device=device)
#    x_U = torch.ones(inp_shape, device=device)
#    
#    if "cifar" in args.dataset.lower():
#        MEANS = torch.tensor([0.4914, 0.4822, 0.4465], device=device).view(1, 3, 1, 1)
#        STD = torch.tensor([0.225, 0.225, 0.225], device=device).view(1, 3, 1, 1)
#        x_L = ((0.0 - MEANS) / STD).expand(*inp_shape).contiguous()
#        x_U = ((1.0 - MEANS) / STD).expand(*inp_shape).contiguous()
#    elif "imagenette" in args.dataset.lower():
#        MEANS = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
#        STD = torch.tensor([0.225, 0.225, 0.225], device=device).view(1, 3, 1, 1)
#        x_L = ((0.0 - MEANS) / STD).expand(*inp_shape).contiguous()
#        x_U = ((1.0 - MEANS) / STD).expand(*inp_shape).contiguous()

    # ======================================================================
    # 5. ITERATE THROUGH CANDIDATE SPLITS AND COMPUTE VRA
    # ======================================================================
    print("\n" + "="*80)
    print(" TRUE VRA PROFILING: ALPHA vs SDP ON CANDIDATES")
    print("="*80)
    
    alpha_vras = []
    sdp_vras = []
    evaluated_indices = [] # Keep track of strictly successful layers

    print(sorted(candidate_indices, reverse=True))
    for k in sorted(candidate_indices, reverse=True):
        if k == len(all_layers):
            print(f"\n--- Evaluating Candidate Split at Layer {k} (FULL 1-LIP PREFIX) ---")
            print(f"      No suffix to evaluate! Using Global CRA Baseline.")
            print(f"      Alpha VRA: {cra_acc:.2f}%")
            
            alpha_vras.append(cra_acc)
            if getattr(args, 'sdp', False):
                sdp_vras.append(cra_acc)
            evaluated_indices.append(k)
            continue # Skip the Alpha-CROWN stuff and go to the next index
            
        layer_name = all_layers[k-1].__class__.__name__
        print(f"\n--- Evaluating Candidate Split at Layer {k} ({layer_name}) ---")
        
        # Split the model
        f1_prefix = torchlip.Sequential(*all_layers[:k]).to(device).eval()
        f2_suffix_lip = torchlip.Sequential(*all_layers[k:]).to(device).eval()
        f2_suffix_vanilla = vanilla_export(f2_suffix_lip).to(device).eval()
        
        # Estimate prefix Lipschitz constant
        L_prefix = compute_model_lipschitz(f1_prefix, input_shape=inp_shape, device=device)
            
        intermediate_epsilon = float(epsilon_rescaled * L_prefix)
        print(f"   Prefix L2: {L_prefix:.4f} | Intermediate Eps: {intermediate_epsilon:.4f}")

        # Propagate clean inputs through the prefix
        with torch.no_grad():
            z_k = f1_prefix(imgs_dev)

        # --- ALPHA-CROWN ---
        print("   -> Executing Alpha-CROWN...")
        try:
            vra_alpha, _, _ = compute_alphacrown_vra_and_time(
                z_k, targets, f2_suffix_vanilla, intermediate_epsilon, clean_indices, args,
                batch_size=args.batch_size, norm=2, x_U=None, x_L=None, return_robust_points=True
            )
        except torch.cuda.OutOfMemoryError:
            print(f"\n[!] OOM ERROR: Alpha-CROWN ran out of memory at Layer {k}.")
            print("--> Stopping evaluation early and proceeding to plot available data.")
            torch.cuda.empty_cache()
            break
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\n[!] OOM ERROR: Alpha-CROWN ran out of memory at Layer {k}.")
                print("--> Stopping evaluation early and proceeding to plot available data.")
                torch.cuda.empty_cache()
                break
            else:
                raise e # Real bug! Crash!
                
        print(f"      Alpha VRA: {vra_alpha:.2f}%")

        if getattr(args, 'sdp', False):
            # --- SDP-CROWN ---
            groupsort = True if ("GNP" in args.model or "Bjork" in args.model) else False

            # Dynamically inject Identity layer for SDP-CROWN if needed
            if not starts_with_affine(f2_suffix_vanilla):
                f2_suffix_sdp = wrap_with_identity(f2_suffix_vanilla, z_k)
            else:
                f2_suffix_sdp = f2_suffix_vanilla

            print("   -> Executing SDP-CROWN...")
            try:
                vra_sdp, _, _ = compute_sdp_crown_vra(
                    z_k, targets, f2_suffix_sdp, float(intermediate_epsilon), clean_indices, 
                    device, classes, args, batch_size=1, return_robust_points=True, x_U=None, x_L=None, groupsort=groupsort
                )
            except torch.cuda.OutOfMemoryError:
                print(f"\n[!] OOM ERROR: SDP-CROWN ran out of memory at Layer {k}.")
                print("--> Stopping evaluation early and proceeding to plot available data.")
                torch.cuda.empty_cache()
                break
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"\n[!] OOM ERROR: SDP-CROWN ran out of memory at Layer {k}.")
                    print("--> Stopping evaluation early and proceeding to plot available data.")
                    torch.cuda.empty_cache()
                    break
                else:
                    raise e # Real bug! Crash!
                    
            print(f"      SDP VRA:   {vra_sdp:.2f}%")
        
        # If BOTH Alpha and SDP succeed without OOM, we append to our final lists
        alpha_vras.append(vra_alpha)
        if getattr(args, 'sdp', False):
            sdp_vras.append(vra_sdp)
        evaluated_indices.append(k)

    # ======================================================================
    # 6. GENERATE AND SAVE THE PLOT
    # ======================================================================
    print("\nGenerating final VRA plot...")
    plt.figure(figsize=(10, 6))
    
    # Plot the curves using evaluated_indices on the X-axis
    plt.plot(evaluated_indices, alpha_vras, marker='s', linestyle='--', color='orange', linewidth=2, label='Alpha-CROWN Hybrid VRA')
    if getattr(args, 'sdp', False):
        plt.plot(evaluated_indices, sdp_vras, marker='^', linestyle='-.', color='green', linewidth=2, label='SDP-CROWN Hybrid VRA')
    
    # Plot the global CRA baseline as a horizontal line
    plt.axhline(y=cra_acc, color='blue', linestyle='-', linewidth=2, label=f'Global 1-Lip CRA ({cra_acc:.2f}%)')
    
    # Make sure X-axis ticks only show our evaluated candidate layers
    plt.xticks(evaluated_indices)
    
    plt.title(f"Hybrid Verification VRA vs. Candidate Split Index\n(Model: {args.model}, $\epsilon$: {args.epsilon})")
    plt.xlabel("Candidate Split Index (End of Prefix)")
    plt.ylabel("Verified Robust Accuracy (VRA %)")
    plt.ylim(0, 105) # Cap at slightly above 100% for readability
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    
    save_path = f"true_vra_curves_candidates_{args.model}_tau_{args.high_tau}.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Plot successfully saved to: {save_path}")
    print("="*80 + "\n")

    # ======================================================================
    # 5. SAVE DATA AS A DICTIONARY (JSON)
    # ======================================================================
    import json

    results_dict = {
        "model": args.model,
        "epsilon": args.epsilon,
        "high_tau": args.high_tau,
        "evaluated_indices": evaluated_indices,
        "alpha_vras": alpha_vras,
    }
    
    if getattr(args, 'sdp', False):
        results_dict["sdp_vras"] = sdp_vras

    data_save_path = f"true_vra_curves_candidates_{args.model}_tau_{args.high_tau}.json"
    with open(data_save_path, 'w') as f:
        json.dump(results_dict, f, indent=4)
        
    print(f"Data dictionary saved to: {data_save_path}")

if __name__ == '__main__':
    # Add your models to the zoo mapping here just like in your main script
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
        "ConvLarge_Bottleneck_1_LIP_GNP" : ConvLarge_Bottleneck_1_LIP_GNP if 'ConvLarge_Bottleneck_1_LIP_GNP' in globals() else None,

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
        
        # --- NEW: Imagenette ResNet Models ---
        "ResNet18_1_LIP_GNP": ResNet18_1_LIP_GNP if 'ResNet18_1_LIP_GNP' in globals() else None,
        "ResNet18_1_LIP_Bjork": ResNet18_1_LIP_Bjork if 'ResNet18_1_LIP_Bjork' in globals() else None,
        "ResNet18_1_LIP_GNP_Imagenette": ResNet18_1_LIP_GNP_Imagenette if 'ResNet18_1_LIP_GNP_Imagenette' in globals() else None,
        "ResNet18_1_LIP_Bjork_Imagenette": ResNet18_1_LIP_Bjork_Imagenette if 'ResNet18_1_LIP_Bjork_Imagenette' in globals() else None,
    }
    
    parser = argparse.ArgumentParser(description='Profile true VRA curves for hybrid verification on candidate splits.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved model .pth file.')
    parser.add_argument('--model', type=str, required=True, help='Name of the 1-Lipschitz model architecture.')
    
    # --- UPDATED: Added imagenette to choices ---
    parser.add_argument('--dataset', type=str, required=True, choices=['cifar10', 'mnist', 'imagenette'])
    
    parser.add_argument('--epsilon', type=float, required=True)
    parser.add_argument('--norm', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=1)
    
    # Highly recommended to keep the subset small (e.g., --end 10) for the initial test run!
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=200, type=int) 
    
    # Project-specific parameters
    parser.add_argument('--radius', type=float, default=0.0) 
    parser.add_argument('--lr_alpha', default=0.5, type=float)
    parser.add_argument('--lr_lambda', default=0.05, type=float)
    parser.add_argument('--high_tau', action='store_true', help='Enable high tau')
    
    # --- RESTORED: SDP Argument (was missing from pasted snippet) ---
    parser.add_argument('--sdp', action='store_true', help='Enable sdp')

    args = parser.parse_args()
    
    # Sync radius with epsilon if not provided separately
    if args.radius == 0.0:
        args.radius = args.epsilon

    profile_true_vras(args, model_zoo)