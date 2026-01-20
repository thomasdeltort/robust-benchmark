import torch
import numpy as np
import argparse
import os
import time
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
def find_max_epsilon_binary_CRA(images, model, clean_indices, args, L, tol=0.005):
    """
    Finds the largest epsilon where Certified Robust Accuracy (CRA) is still > 0.
    Automatically expands the search range if the initial max is insufficient.
    """
    print(f"\n[Phase 1] Binary Search for CERTIFIED Boundary (Norm: {args.norm})")
    
    # 1. Setup Scaling
    is_cifar = "cifar" in args.dataset.lower()
    scale_factor = 0.225 if is_cifar else 1.0
    
    low = 0.00001
    high = 4  # Start with the arg default (e.g., 4.0)

    # 2. Rescale for internal checking
    low_rescaled = low / scale_factor
    high_rescaled = high / scale_factor
    
    # 3. Check Upper Bound & Expand if Necessary
    # We loop until we find an epsilon 'high' where the model is fully broken (CRA == 0)
    while True:
        _, cra, _ = compute_certificates_CRA(
            images, model, high_rescaled, clean_indices, 
            norm=str(args.norm), L=L
        )
        
        if cra <= 0:
            print(f"  Boundary found within [0, {high:.2f}] (Internal: {high_rescaled:.2f})")
            break
        else:
            print(f"  Warning: Model still robust (CRA={cra:.2f}%) at {high:.2f}. Doubling search range...")
            low = high
            low_rescaled = high_rescaled
            
            high *= 2
            high_rescaled *= 2
            
            # Safety break to prevent infinite loops on degenerate models
            if high > 1000: 
                print("  ! Safety limit reached. Stopping expansion.")
                break

    # 4. Perform Binary Search
    best_eps = low_rescaled
    
    # Standard binary search between the now-confirmed [low, high]
    while (high_rescaled - low_rescaled) > (tol / scale_factor):
        mid = (low_rescaled + high_rescaled) / 2
        
        _, cra, _ = compute_certificates_CRA(
            images, model, mid, clean_indices, 
            norm=str(args.norm), L=L
        )
        # print(f"  Testing Epsilon: {mid * scale_factor:.4f} | CRA: {cra:.2f}%")
        
        if cra > 0:
            best_eps = mid
            low_rescaled = mid
        else:
            high_rescaled = mid
            
    final_eps = best_eps * scale_factor
    print(f"Found Certified Boundary at Epsilon: {final_eps:.4f}")
    
    return final_eps

# def main():
#     parser = argparse.ArgumentParser(description='Intelligent Robustness Evaluation')
#     parser.add_argument('--norm', default=2, choices=[2, 'inf'])
#     parser.add_argument('--model_path', type=str, required=True)
#     parser.add_argument('--model', type=str, required=True)
#     parser.add_argument('--dataset', type=str, required=True, choices=['cifar10', 'mnist'])
#     parser.add_argument('--batch_size', type=int, default=2)
#     parser.add_argument('--output_csv', type=str, default='results/study.csv')
#     parser.add_argument('--num_points', type=int, default=110, help='Points to sample in linear paving')

#      # SDP Crown parameters
#     parser.add_argument('--start', default=0, type=int, help='start index for the dataset')
#     parser.add_argument('--end', default=200, type=int, help='end index for the dataset')
#     parser.add_argument('--lr_alpha', default=0.5, type=float, help='alpha learning rate')
#     parser.add_argument('--lr_lambda', default=0.05, type=float, help='lambda learning rate')
#     args = parser.parse_args()

#     # Setup
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     images, targets, classes = load_dataset_benchmark_auto(args)
#     model = load_model(args, model_zoo, device)
#     model.eval()

#     # Clean Accuracy Baseline
#     with torch.no_grad():
#         images, targets = images.to(device), targets.to(device)
#         output = model(images)
#         predictions = output.argmax(dim=1)
#         clean_indices = (predictions == targets).nonzero(as_tuple=True)[0]
#         clean_acc = (len(clean_indices) / len(targets)) * 100
#         print(f"Clean Accuracy: {clean_acc:.2f}% ({len(clean_indices)} samples)")

# # --- Initialize Robustness Registry ---
#     registry = RobustnessRegistry(
#         model_name=args.model,
#         norm=str(args.norm),
#         total_dataset_size=images.shape[0],
#         save_dir="./results2/robust_points"
#     )
#     registry.register(0.0, "clean", clean_indices)

#     # --- PRE-CALCULATE LIPSCHITZ CONSTANT ---
#     L_2 = 1 
#     L = convert_lipschitz_constant(L_2, str(args.norm), images[0].numel())
#     print(f"Lipschitz Constant (Norm {args.norm}): {L:.4f}")

#     # Phase 1: Find Baseline Boundary using CRA
#     # We use this to set the resolution of our linspace. 
#     cra_boundary = find_max_epsilon_binary_CRA(images, model, clean_indices, args, L)
    
#     # STRATEGY: Define paving max slightly higher than the CRA boundary.
#     # This gives "Heavy Certified" methods a chance to prove robustness 
#     # beyond the theoretical certificate if possible.
#     if args.norm == 'inf':
#         paving_max = cra_boundary * 2
#     else:
#         paving_max = cra_boundary * 1.1
    
#     # Phase 2: Systematic Paving (0 -> Max)
#     # Note: We start from 0.0 (where acc is 100%)
#     epsilon_range = np.linspace(0.0001, paving_max, args.num_points)
    
#     solvers = {
#         "aa": True, 
#         "cra": True, 
#         "alphacrown": True, 
#         "heavy_certified": True 
#     }

#     print(f"\n[Phase 2] Paving Range [0, {paving_max:.4f}]")
#     print("         Stopping when Best Certified (Union) reaches 0%.")

#     # CHANGE 1: Forward Loop (Low Eps -> High Eps)
#     for eps in epsilon_range:
#         print(f"\n--- EVALUATING EPSILON: {eps:.5f} ---")
#         eps_rescaled = eps / 0.225 if "cifar" in args.dataset.lower() else eps
#         result_dict = {'epsilon': eps}
        
#         certified_indices_union = set()

#         # 1. EMPIRICAL FILTER (AutoAttack)
#         if solvers["aa"]:
#             # Note: Use the updated RELATIVE accuracy function here
#             era, t_aa, idx_aa = compute_autoattack_era_and_time(
#                 images, targets, model, eps, clean_indices, 
#                 norm=str(args.norm), dataset_name=args.dataset, return_robust_points=True
#             )
#             result_dict['aa'], result_dict['time_aa'] = era, t_aa
#             registry.register(eps, "autoattack", idx_aa)
            
#             # Optimization: If AA breaks completely, we usually stop AA, 
#             # but we continue others until they also break.
#             if era <= 0: solvers["aa"] = False
#         else:
#             result_dict['aa'], result_dict['time_aa'] = 0.0, 0.0

#         # 2. CERTIFIED METHODS
#         # We perform certification as long as AutoAttack hasn't fully broken the model
#         # OR if we want to see the certifiers drop to 0 independently.
#         # Since we want to stop only when Best Certified is 0, we run them if enabled.
        
#         # --- Lipschitz CRA ---
#         if solvers["cra"]:
#             # Note: Use the updated RELATIVE accuracy function here
#             _, cra_val, t_cra, idx_cra = compute_certificates_CRA(
#                 images, model, eps_rescaled, clean_indices, norm=str(args.norm), L=L, return_robust_points=True
#             )
#             result_dict['certificate'], result_dict['time_cra'] = cra_val, t_cra
#             registry.register(eps, "certificate", idx_cra)
#             certified_indices_union.update(idx_cra.cpu().tolist())

#             if cra_val <= 0: solvers["cra"] = False
#         else:
#             result_dict['certificate'] = 0.0

#         # --- Setup for Crown ---
#         x_L = torch.zeros((1, 1, 28, 28), device=device)
#         x_U = torch.ones((1, 1, 28, 28), device=device)
#         if "cifar" in args.dataset.lower():
#             MEANS = torch.tensor([0.4914, 0.4822, 0.4465], device=device).view(1, 3, 1, 1)
#             STD = torch.tensor([0.225, 0.225, 0.225], device=device).view(1, 3, 1, 1)
#             x_L = ((0.0 - MEANS) / STD).expand(1, 3, 32, 32).contiguous()
#             x_U = ((1.0 - MEANS) / STD).expand(1, 3, 32, 32).contiguous()

#         # --- Alpha-CROWN ---
#         if solvers["alphacrown"]:
#             vra, t_v, idx_alpha = compute_alphacrown_vra_and_time(
#                 images, targets, model, eps_rescaled, clean_indices, 
#                 batch_size=args.batch_size, norm=args.norm, x_U=x_U, x_L=x_L, return_robust_points=True
#             )
#             result_dict['lirpa_alphacrown'], result_dict['time_lirpa_alpha'] = vra, t_v
#             registry.register(eps, "alphacrown", idx_alpha)
#             certified_indices_union.update(idx_alpha.cpu().tolist())
            
#             print(f"    [Alpha-CROWN] Acc: {vra:.2f}%")

#             if vra <= 0: solvers["alphacrown"] = False
#         else:
#             result_dict['lirpa_alphacrown'] = 0.0

#         # --- Beta/SDP CROWN (Heavy) ---
#         if solvers["heavy_certified"]:
#             if str(args.norm) == 'inf':
#                 v_acc, t_v, idx_beta = compute_alphabeta_vra_and_time(
#                     args.dataset, args.model, args.model_path, eps, 
#                     'cifar_l2_norm.yaml', clean_indices, total_samples=images.shape[0], return_robust_points=True
#                 )
#                 result_dict['lirpa_betacrown'], result_dict['time_lirpa_beta'] = v_acc, t_v
#                 registry.register(eps, "betacrown", idx_beta)
#                 certified_indices_union.update(idx_beta.cpu().tolist())
#                 print(f"    [Beta-CROWN]  Acc: {v_acc:.2f}%")
#             else:
#                 swapped_model = replace_groupsort(model, images[:1]).to(device)
#                 v_acc, t_v, idx_sdp = compute_sdp_crown_vra(
#                     images, targets, swapped_model, float(eps_rescaled), clean_indices, 
#                     device, classes, args, batch_size=1, return_robust_points=True, x_U=x_U, x_L=x_L
#                 )
#                 result_dict['sdp'], result_dict['time_sdp'] = v_acc, t_v
#                 registry.register(eps, "sdp", idx_sdp)
#                 certified_indices_union.update(idx_sdp.cpu().tolist())
#                 print(f"    [SDP-CROWN]   Acc: {v_acc:.2f}%")
            
#             # Optimization: Only turn off if THIS solver hits 0
#             if v_acc <= 0: solvers["heavy_certified"] = False
#         else:
#             result_dict['lirpa_betacrown'] = 0.0
#             result_dict['sdp'] = 0.0

#         # --- 3. CALCULATE BEST CERTIFIED (UNION) ---
#         total_samples = images.shape[0]
#         best_certified_acc = (len(certified_indices_union) / total_samples) * 100.0
        
#         result_dict['best_certified'] = best_certified_acc
        
#         if len(certified_indices_union) > 0:
#             union_tensor = torch.tensor(sorted(list(certified_indices_union)), dtype=torch.long)
#         else:
#             union_tensor = torch.tensor([], dtype=torch.long)
            
#         registry.register(eps, "best_certified", union_tensor)
#         print(f"  > Best Certified (Union): {best_certified_acc:.2f}%")

#         registry.save()
#         add_result_and_sort(result_dict, args.output_csv, norm=args.norm)

#         # CHANGE 2: Dynamic Stop Condition
#         # If the Union of all certificates is 0.0%, there is no point continuing
#         # to higher epsilons.
#         if best_certified_acc <= 0.0:
#             print(f"  > Stopping Early: All certified methods reached 0% at epsilon {eps:.5f}.")
#             break

#     print("\nStudy Complete. Results saved to:", args.output_csv)
#     create_robustness_plot_v3(f"{os.path.splitext(args.output_csv)[0]}_norm_{args.norm}.csv", f"{os.path.splitext(args.output_csv)[0]}_norm_{args.norm}.png")
def main():
    parser = argparse.ArgumentParser(description='Intelligent Robustness Evaluation')
    parser.add_argument('--norm', default=2, choices=[2, 'inf'])
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True, choices=['cifar10', 'mnist'])
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--output_csv', type=str, default='results/study.csv')
    parser.add_argument('--num_points', type=int, default=50, help='Points to sample in linear paving')
    parser.add_argument('--start', default=0, type=int, help='start index for the dataset')
    parser.add_argument('--end', default=200, type=int, help='end index for the dataset')
    parser.add_argument('--lr_alpha', default=0.5, type=float, help='alpha learning rate')
    parser.add_argument('--lr_lambda', default=0.05, type=float, help='lambda learning rate')
    args = parser.parse_args()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images, targets, classes = load_dataset_benchmark_auto(args)
    model = load_model(args, model_zoo, device)
    model.eval()

    # Clean Accuracy Baseline
    with torch.no_grad():
        images, targets = images.to(device), targets.to(device)
        output = model(images)
        predictions = output.argmax(dim=1)
        clean_indices = (predictions == targets).nonzero(as_tuple=True)[0]
        clean_acc = (len(clean_indices) / len(targets)) * 100
        print(f"Clean Accuracy: {clean_acc:.2f}% ({len(clean_indices)} samples)")

    # --- Initialize Robustness Registry ---
    registry = RobustnessRegistry(
        model_name=args.model,
        norm=str(args.norm),
        total_dataset_size=images.shape[0],
        save_dir="./results2/robust_points"
    )
    registry.register(0.0, "clean", clean_indices)

    # --- 1. ORIGINAL LIPSCHITZ CONSTANT (Theoretical) ---
    L_2 = 1 
    L = convert_lipschitz_constant(L_2, str(args.norm), images[0].numel())
    print(f"Theoretical Lipschitz Constant (L): {L:.4f}")

    # --- 2. NEW LIPSCHITZ CONSTANT (Power Iteration) ---
    print("\n[Setup] Computing Lipschitz Constant via Power Iteration...")
    if "mnist" in args.dataset.lower():
        inp_shape = (1, 1, 28, 28)
    else:
        inp_shape = (1, 3, 32, 32)

    L_2_empirical = compute_model_lipschitz(model, input_shape=inp_shape, device=device)
    L_PI = convert_lipschitz_constant(L_2_empirical, str(args.norm), images[0].numel())
    print(f"Power Iteration Lipschitz Constant (L_PI): {L_PI:.4f}")

    # Phase 1: Find Baseline Boundary using CRA (Using original L to set range)
    cra_boundary = find_max_epsilon_binary_CRA(images, model, clean_indices, args, L)
    
    # STRATEGY: Define paving max slightly higher than the CRA boundary.
    if args.norm == 'inf':
        paving_max = cra_boundary * 2
    else:
        paving_max = cra_boundary * 1.1
    
    # Phase 2: Systematic Paving (0 -> Max)
    epsilon_range = np.linspace(0.0001, paving_max, args.num_points)
    
    solvers = {
        "aa": True, 
        "cra": True, 
        "cra_pi": True,  # Enable new solver
        "alphacrown": True, 
        "heavy_certified": True 
    }

    print(f"\n[Phase 2] Paving Range [0, {paving_max:.4f}]")
    print("         Stopping when Best Certified (Union) reaches 0%.")

    for eps in epsilon_range:
        print(f"\n--- EVALUATING EPSILON: {eps:.5f} ---")
        eps_rescaled = eps / 0.225 if "cifar" in args.dataset.lower() else eps
        result_dict = {'epsilon': eps}
        
        certified_indices_union = set()

        # 1. EMPIRICAL FILTER (AutoAttack)
        if solvers["aa"]:
            era, t_aa, idx_aa = compute_autoattack_era_and_time(
                images, targets, model, eps, clean_indices, 
                norm=str(args.norm), dataset_name=args.dataset, return_robust_points=True
            )
            result_dict['aa'], result_dict['time_aa'] = era, t_aa
            registry.register(eps, "autoattack", idx_aa)
            
            if era <= 0: solvers["aa"] = False
        else:
            result_dict['aa'], result_dict['time_aa'] = 0.0, 0.0

        # 2. CERTIFIED METHODS
        # --- A. Original Lipschitz CRA (Theoretical L) ---
        if solvers["cra"]:
            _, cra_val, t_cra, idx_cra = compute_certificates_CRA(
                images, model, eps_rescaled, clean_indices, norm=str(args.norm), L=L, return_robust_points=True
            )
            result_dict['certificate'], result_dict['time_cra'] = cra_val, t_cra
            registry.register(eps, "certificate", idx_cra)
            certified_indices_union.update(idx_cra.cpu().tolist())

            if cra_val <= 0: solvers["cra"] = False
        else:
            result_dict['certificate'] = 0.0
            result_dict['time_cra'] = 0.0

        # --- B. New Power Iteration CRA (L_PI) ---
        if solvers["cra_pi"]:
            _, cra_pi_val, t_cra_pi, idx_cra_pi = compute_certificates_CRA(
                images, model, eps_rescaled, clean_indices, norm=str(args.norm), L=L_PI, return_robust_points=True
            )
            result_dict['certificate_pi'], result_dict['time_cra_pi'] = cra_pi_val, t_cra_pi
            registry.register(eps, "certificate_pi", idx_cra_pi)
            
            # Note: We usually DON'T add L_PI certificates to the UNION if they are not theoretically guaranteed
            # (since L_PI is an estimate). But if you trust it, uncomment the line below:
            # certified_indices_union.update(idx_cra_pi.cpu().tolist())

            if cra_pi_val <= 0: solvers["cra_pi"] = False
        else:
            result_dict['certificate_pi'] = 0.0
            result_dict['time_cra_pi'] = 0.0

        # --- Setup for Crown ---
        x_L = torch.zeros((1, 1, 28, 28), device=device)
        x_U = torch.ones((1, 1, 28, 28), device=device)
        if "cifar" in args.dataset.lower():
            MEANS = torch.tensor([0.4914, 0.4822, 0.4465], device=device).view(1, 3, 1, 1)
            STD = torch.tensor([0.225, 0.225, 0.225], device=device).view(1, 3, 1, 1)
            x_L = ((0.0 - MEANS) / STD).expand(1, 3, 32, 32).contiguous()
            x_U = ((1.0 - MEANS) / STD).expand(1, 3, 32, 32).contiguous()

        # --- Alpha-CROWN ---
        if solvers["alphacrown"]:
            vra, t_v, idx_alpha = compute_alphacrown_vra_and_time(
                images, targets, model, eps_rescaled, clean_indices, 
                batch_size=args.batch_size, norm=args.norm, x_U=x_U, x_L=x_L, return_robust_points=True
            )
            result_dict['lirpa_alphacrown'], result_dict['time_lirpa_alpha'] = vra, t_v
            registry.register(eps, "alphacrown", idx_alpha)
            certified_indices_union.update(idx_alpha.cpu().tolist())
            print(f"    [Alpha-CROWN] Acc: {vra:.2f}%")

            if vra <= 0: solvers["alphacrown"] = False
        else:
            result_dict['lirpa_alphacrown'] = 0.0
            result_dict['time_lirpa_alpha'] = 0.0

        # --- Beta/SDP CROWN (Heavy) ---
        if solvers["heavy_certified"]:
            if str(args.norm) == 'inf':
                v_acc, t_v, idx_beta = compute_alphabeta_vra_and_time(
                    args.dataset, args.model, args.model_path, eps, 
                    'cifar_l2_norm.yaml', clean_indices, total_samples=images.shape[0], return_robust_points=True
                )
                result_dict['lirpa_betacrown'], result_dict['time_lirpa_beta'] = v_acc, t_v
                registry.register(eps, "betacrown", idx_beta)
                certified_indices_union.update(idx_beta.cpu().tolist())
                print(f"    [Beta-CROWN]  Acc: {v_acc:.2f}%")
            else:
                swapped_model = replace_groupsort(model, images[:1]).to(device)
                v_acc, t_v, idx_sdp = compute_sdp_crown_vra(
                    images, targets, swapped_model, float(eps_rescaled), clean_indices, 
                    device, classes, args, batch_size=args.batch_size, return_robust_points=True, x_U=x_U, x_L=x_L
                )
                result_dict['sdp'], result_dict['time_sdp'] = v_acc, t_v
                registry.register(eps, "sdp", idx_sdp)
                certified_indices_union.update(idx_sdp.cpu().tolist())
                print(f"    [SDP-CROWN]   Acc: {v_acc:.2f}%")
            
            if v_acc <= 0: solvers["heavy_certified"] = False
        else:
            result_dict['lirpa_betacrown'] = 0.0
            result_dict['time_lirpa_beta'] = 0.0
            result_dict['sdp'] = 0.0
            result_dict['time_sdp'] = 0.0

        # --- 3. CALCULATE BEST CERTIFIED (UNION) ---
        total_samples = images.shape[0]
        best_certified_acc = (len(certified_indices_union) / total_samples) * 100.0
        
        result_dict['best_certified'] = best_certified_acc
        
        if len(certified_indices_union) > 0:
            union_tensor = torch.tensor(sorted(list(certified_indices_union)), dtype=torch.long)
        else:
            union_tensor = torch.tensor([], dtype=torch.long)
            
        registry.register(eps, "best_certified", union_tensor)
        print(f"  > Best Certified (Union): {best_certified_acc:.2f}% (PI-Cert: {cra_pi_val:.2f}%)")

        registry.save()
        add_result_and_sort(result_dict, args.output_csv, norm=args.norm)

        # Dynamic Stop Condition
        if best_certified_acc <= 0.0:
            print(f"  > Stopping Early: All certified methods reached 0% at epsilon {eps:.5f}.")
            break

    print("\nStudy Complete. Results saved to:", args.output_csv)
    create_robustness_plot_v3(f"{os.path.splitext(args.output_csv)[0]}_norm_{args.norm}.csv", f"{os.path.splitext(args.output_csv)[0]}_norm_{args.norm}.png")

if __name__ == '__main__':
    main()