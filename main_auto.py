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
# --- 1. Binary Search Strategy ---
def find_max_epsilon_binary(images, targets, model, clean_indices, args, tol=0.005):
    """
    Finds the largest epsilon where AutoAttack ERA is still > 0.
    This defines the upper bound for our systematic paving.
    """
    print(f"\n[Phase 1] Binary Search for Empirical Boundary (Norm: {args.norm})")
    low = 0.00001
    low_rescaled = low / 0.225 if "cifar" in args.dataset.lower() else low
    high_rescaled = args.max_search_epsilon / 0.225 if "cifar" in args.dataset.lower() else args.max_search_epsilon
    best_eps = low_rescaled
    
    # Check if high is already broken
    era, _ = compute_autoattack_era_and_time(
        images, targets, model, high_rescaled, clean_indices, 
        norm=str(args.norm), dataset_name=args.dataset
    )
    if era > 0:
        print(f"  Warning: Model still robust at hard cap ({args.max_search_epsilon}). Using hard cap as max.")
        return args.max_search_epsilon

    while (high_rescaled - low_rescaled) > tol:
        mid = (low_rescaled + high_rescaled) / 2
        era, _ = compute_autoattack_era_and_time(
            images, targets, model, mid, clean_indices, 
            norm=str(args.norm), dataset_name=args.dataset
        )
        print(f"  Testing Epsilon: {mid:.4f} | AA-ERA: {era:.2f}%")
        
        if era > 0:
            best_eps = mid
            low_rescaled = mid  # Model survives, try larger epsilon
        else:
            high_rescaled = mid # Model is fully broken, go smaller
            
    print(f"Found Empirical Boundary at Epsilon: {best_eps:.4f}")
    best_eps = best_eps * 0.225 if "cifar" in args.dataset.lower() else best_eps
    return best_eps

# --- 2. Main Logic ---
def main():
    parser = argparse.ArgumentParser(description='Intelligent Robustness Evaluation')
    parser.add_argument('--norm', default=2, choices=[2, 'inf'])
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True, choices=['cifar10', 'mnist'])
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--output_csv', type=str, default='results/study.csv')
    parser.add_argument('--num_points', type=int, default=10, help='Points to sample in linear paving')
    parser.add_argument('--max_search_epsilon', type=float, default=3.0, help='Max limit for binary search')

     # SDP Crown parameters
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
    # This will track exactly which points are robust across all epsilon steps
    registry = RobustnessRegistry(
        model_name=args.model,
        norm=str(args.norm),
        total_dataset_size=images.shape[0],
        save_dir="./results2/robust_points"
    )
    # Register "clean" points as a baseline (epsilon 0 equivalent)
    registry.register(0.0, "clean", clean_indices)

    # Phase 1: Binary Search for Boundary
    max_eps = find_max_epsilon_binary(images, targets, model, clean_indices, args)

    # Phase 2: Systematic Paving
    # Here we are in the space of rescaled epsilon (already /0.255 if cifar, nothing if mnist)
    epsilon_range = np.linspace(0.00001, max_eps, args.num_points)
    
    solvers = {
        "aa": True, 
        "cra": True, 
        "alphacrown": False, 
        "heavy_certified": True 
    }

    print(f"\n[Phase 2] Paving Range [0, {max_eps:.4f}] with {args.num_points} steps.")

    for eps in epsilon_range[::-1]:
        print(f"\n--- EVALUATING EPSILON: {eps:.5f} ---")
        eps_rescaled = eps / 0.225 if "cifar" in args.dataset.lower() else eps
        result_dict = {'epsilon': eps}

        # 1. EMPIRICAL FILTER (AutoAttack)
        if solvers["aa"]:
            era, t_aa, idx_aa = compute_autoattack_era_and_time(
                images, targets, model, eps, clean_indices, 
                norm=str(args.norm), dataset_name=args.dataset, return_robust_points=True
            )
            result_dict['aa'], result_dict['time_aa'] = era, t_aa
            registry.register(eps, "autoattack", idx_aa) # Register indices
            
            if era <= 0: solvers["aa"] = False
        else:
            result_dict['aa'], result_dict['time_aa'] = 0.0, 0.0

        # 2. CERTIFIED METHODS (Only run if Empirical > 0)
        if result_dict['aa'] > 0:
            # --- Lipschitz CRA ---
            if solvers["cra"]:
                L_2 = 1
                L = convert_lipschitz_constant(L_2, str(args.norm), images[0].numel())
                _, cra_val, t_cra, idx_cra = compute_certificates_CRA(
                    images, model, eps_rescaled, clean_indices, norm=str(args.norm), L=L, return_robust_points=True
                )
                result_dict['certificate'], result_dict['time_cra'] = cra_val, t_cra
                registry.register(eps, "certificate", idx_cra) # Register indices
                
                if cra_val <= 0: solvers["cra"] = False
            else:
                result_dict['certificate'] = 0.0

            # --- Bounds setup for Crown methods ---
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
                registry.register(eps, "alphacrown", idx_alpha) # Register indices
                
                if vra <= 0: solvers["alphacrown"] = False
            else:
                result_dict['lirpa_alphacrown'] = 0.0

            # --- Beta/SDP CROWN ---
            if solvers["heavy_certified"]:
                if str(args.norm) == 'inf':
                    v_acc, t_v, idx_beta = compute_alphabeta_vra_and_time(
                        args.dataset, args.model, args.model_path, eps, 
                        'cifar_l2_norm.yaml', clean_indices, total_samples=images.shape[0], return_robust_points=True
                    )
                    result_dict['lirpa_betacrown'], result_dict['time_lirpa_beta'] = v_acc, t_v
                    registry.register(eps, "betacrown", idx_beta)
                else:
                    swapped_model = replace_groupsort(model, images[:1]).to(device)
                    # print("check : ", model(images[:1]), swapped_model(images[:1]))
                    v_acc, t_v, idx_sdp = compute_sdp_crown_vra(
                        images, targets, swapped_model, float(eps_rescaled), clean_indices, 
                        device, classes, args, batch_size=1, return_robust_points=True, x_U=x_U, x_L=x_L
                    )
                    result_dict['sdp'], result_dict['time_sdp'] = v_acc, t_v
                    registry.register(eps, "sdp", idx_sdp)
                
                if v_acc <= 0: solvers["heavy_certified"] = False
            else:
                result_dict['lirpa_betacrown'] = 0.0
                result_dict['sdp'] = 0.0

        else:
            print("  Skipping certifications (Model fully broken by AA)")
            result_dict.update({
                'certificate': 0.0, 'lirpa_alphacrown': 0.0, 
                'lirpa_betacrown': 0.0, 'sdp': 0.0
            })

        # Save registry state to disk after every epsilon step (prevents data loss)
        registry.save()
        
        # Save current point to CSV
        add_result_and_sort(result_dict, args.output_csv, norm=args.norm)

    print("\nStudy Complete. Results saved to:", args.output_csv)
    create_robustness_plot_v3(f"{os.path.splitext(args.output_csv)[0]}_norm_{args.norm}.csv", f"{os.path.splitext(args.output_csv)[0]}_norm_{args.norm}.png")

if __name__ == '__main__':
    main()