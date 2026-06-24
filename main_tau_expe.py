import torch
import numpy as np
import argparse
import os
import gc
import re
import glob
import pandas as pd
from collections import defaultdict
from models import *
from project_utils import *
from robustness_registery import *

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
    "VGG13_1_LIP_CIFAR10" : VGG13_1_LIP_CIFAR10 if 'VGG13_1_LIP_CIFAR10' in globals() else None,

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
    
    # --- Imagenette ResNet Models ---
    "ResNet18_1_LIP_GNP": ResNet18_1_LIP_GNP if 'ResNet18_1_LIP_GNP' in globals() else None,
    "ResNet18_1_LIP_Bjork": ResNet18_1_LIP_Bjork if 'ResNet18_1_LIP_Bjork' in globals() else None,
    "ResNet18_1_LIP_GNP_Imagenette": ResNet18_1_LIP_GNP_Imagenette if 'ResNet18_1_LIP_GNP_Imagenette' in globals() else None,
    "ResNet18_1_LIP_Bjork_Imagenette": ResNet18_1_LIP_Bjork_Imagenette if 'ResNet18_1_LIP_Bjork_Imagenette' in globals() else None,
}

def main():
    parser = argparse.ArgumentParser(description='Evaluate CRA and ERA across varying temperatures (tau)')
    parser.add_argument('--norm', default=2, choices=[2, 'inf'], help='Norm for attacks/certificates')
    parser.add_argument('--models_dir', type=str, required=True, help='Directory containing the trained models')
    parser.add_argument('--model', type=str, required=True, help='Model architecture from model_zoo')
    parser.add_argument('--dataset', type=str, required=True, choices=['cifar10', 'mnist', 'imagenette'])
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--eps', type=float, default=8/255, help='Fixed epsilon for evaluation')
    parser.add_argument('--output_csv', type=str, default='results/tau_study.csv')
    
    args = parser.parse_args()

    # --- 1. SET ENVIRONMENT ---
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    # --- 2. LOAD DATASET ---
    images, targets, classes = load_dataset_benchmark_auto(args)
    print(f"Dataset loaded: {len(targets)} samples.")

    # --- 3. PARSE & GROUP MODELS BY TEMPERATURE ---
    search_pattern = os.path.join(args.models_dir, "*.pth")
    all_files = glob.glob(search_pattern)
    
    tau_groups = defaultdict(list)
    target_prefix = f"vanilla_{args.model}"
    
    for mf in all_files:
        filename = os.path.basename(mf)
        if not filename.startswith(target_prefix):
            continue
            
        match = re.search(r'_T(\d+\.?\d*)_', filename)
        if match:
            tau = float(match.group(1))
            tau_groups[tau].append(mf)
            
    if not tau_groups:
        print(f"No models found starting with '{target_prefix}' that have a valid '_T<val>_' tag. Exiting.")
        return

    # Sort the unique temperatures
    unique_taus = sorted(list(tau_groups.keys()))
    print(f"Found {len(unique_taus)} unique temperatures to evaluate.")

    results = []

    # --- 4. EVALUATION LOOP ---
    for tau in unique_taus:
        models_for_tau = tau_groups[tau]
        print(f"\n=== Evaluating Temperature T={tau} ({len(models_for_tau)} models found) ===")
        
        # Sort files to ensure runs are processed in a consistent order
        models_for_tau.sort()
        
        for run_idx, model_path in enumerate(models_for_tau, start=1):
            filename = os.path.basename(model_path)
            print(f"\n  [T={tau} | Run {run_idx}/{len(models_for_tau)}] File: {filename}")
            
            torch.cuda.empty_cache()
            gc.collect()

            args.model_path = model_path
            model = load_model(args, model_zoo, device)
            model.eval()

            # Clean Accuracy
            with torch.no_grad():
                imgs_dev, tgts_dev = images.to(device), targets.to(device)
                output = model(imgs_dev)
                predictions = output.argmax(dim=1)
                clean_indices = (predictions == tgts_dev).nonzero(as_tuple=True)[0].cpu()
                clean_acc = (len(clean_indices) / len(targets)) * 100
                print(f"    > Clean Accuracy: {clean_acc:.2f}%")

            # Lipschitz Constant setup
            L_2 = 1 
            L_theory = convert_lipschitz_constant(L_2, str(args.norm), images[0].numel())
            
            inp_shape = (1, 3, 224, 224) if "imagenette" in args.dataset.lower() else ((1, 1, 28, 28) if "mnist" in args.dataset.lower() else (1, 3, 32, 32))
            try:
                L_2_empirical = compute_model_lipschitz(model, input_shape=inp_shape, device=device)
                L = convert_lipschitz_constant(L_2_empirical, str(args.norm), images[0].numel())
            except Exception:
                L = L_theory

            is_cifar_or_imagenet = "cifar" in args.dataset.lower() or "imagenette" in args.dataset.lower()
            eps_rescaled = args.eps / 0.225 if is_cifar_or_imagenet else args.eps

            # Empirical Robust Accuracy (ERA)
            era_val, _, _ = compute_autoattack_era_and_time(
                images, targets, model, args.eps, clean_indices, 
                norm=str(args.norm), dataset_name=args.dataset, return_robust_points=True
            )
            print(f"    > ERA: {era_val:.2f}%")

            # Certified Robust Accuracy (CRA)
            _, cra_val, _, _ = compute_certificates_CRA(
                images, model, eps_rescaled, clean_indices, norm=str(args.norm), L=L, return_robust_points=True
            )
            print(f"    > CRA: {cra_val:.2f}%")

            # Save the results
            results.append({
                'tau': tau,
                'run_idx': run_idx,
                'filename': filename,
                'clean_acc': clean_acc,
                'era': era_val,
                'cra': cra_val
            })

    # --- 5. SAVE TO CSV ---
    df = pd.DataFrame(results)
    df.to_csv(args.output_csv, index=False)
    print(f"\nEvaluation complete. Data saved to {args.output_csv}")

if __name__ == '__main__':
    main()