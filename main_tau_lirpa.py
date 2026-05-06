import torch
import numpy as np
import argparse
import os
import gc
import re
import glob
from models import *
from project_utils import *
import ast
from robustness_registery import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

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
        
        # --- Imagenette ResNet Models ---
        "ResNet18_1_LIP_GNP": ResNet18_1_LIP_GNP if 'ResNet18_1_LIP_GNP' in globals() else None,
        "ResNet18_1_LIP_Bjork": ResNet18_1_LIP_Bjork if 'ResNet18_1_LIP_Bjork' in globals() else None,
        "ResNet18_1_LIP_GNP_Imagenette": ResNet18_1_LIP_GNP_Imagenette if 'ResNet18_1_LIP_GNP_Imagenette' in globals() else None,
        "ResNet18_1_LIP_Bjork_Imagenette": ResNet18_1_LIP_Bjork_Imagenette if 'ResNet18_1_LIP_Bjork_Imagenette' in globals() else None,
    }

def main():
    parser = argparse.ArgumentParser(description='Evaluate ERA, CRA, Alpha-CROWN, and SDP-CROWN across varying temperatures')
    parser.add_argument('--norm', default=2, choices=[2, 'inf'], help='Norm for attacks/certificates')
    parser.add_argument('--models_dir', type=str, required=True, help='Directory containing the trained models')
    parser.add_argument('--model', type=str, required=True, help='Model architecture from model_zoo')
    parser.add_argument('--dataset', type=str, required=True, choices=['cifar10', 'mnist', 'imagenette'])
    
    # Removed the --taus argument to evaluate all points automatically
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--eps', type=float, default=8/255, help='Fixed epsilon for evaluation')
    parser.add_argument('--output_csv', type=str, default='results/tau_study_advanced.csv')
    parser.add_argument('--output_plot_temp', type=str, default='results/metrics_vs_tau.png')
    parser.add_argument('--lr_alpha', default=0.5, type=float)
    parser.add_argument('--lr_lambda', default=0.05, type=float)
    
    args = parser.parse_args()

    # --- 1. SET ENVIRONMENT ---
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_plot_temp), exist_ok=True)

    # --- 2. LOAD DATASET ---
    images, targets, classes = load_dataset_benchmark_auto(args)
    images, targets = images.to(device), targets.to(device)
    
    print(f"Dataset loaded: {len(targets)} samples.")

    # --- 3. PARSE ALL MODELS IN DIRECTORY ---
    search_pattern = os.path.join(args.models_dir, "*.pth")
    all_files = glob.glob(search_pattern)
    
    tau_models = []
    target_prefix = f"vanilla_{args.model}"
    
    for mf in all_files:
        filename = os.path.basename(mf)
        if not filename.startswith(target_prefix):
            continue
            
        match = re.search(r'_T(\d+\.?\d*)_', filename)
        if match:
            tau = float(match.group(1))
            tau_models.append((tau, mf))
                
    # Sort models by temperature (tau) in ascending order
    tau_models.sort(key=lambda x: x[0])
    
    if not tau_models:
        print(f"No models found starting with '{target_prefix}' that have a valid '_T<val>_' tag. Exiting.")
        return

    print(f"Found {len(tau_models)} total models. Evaluating at fixed epsilon: {args.eps:.5f}")
    
    # --- NEW: RESUME FROM CHECKPOINT LOGIC ---
    results = []
    evaluated_taus = set()

    if os.path.exists(args.output_csv):
        print(f"--> Found existing results file at {args.output_csv}. Loading...")
        try:
            existing_df = pd.read_csv(args.output_csv)
            if 'tau' in existing_df.columns:
                # Get the taus we've already finished (rounding slightly to avoid float precision mismatch)
                evaluated_taus = set(existing_df['tau'].round(5).values)
                # Seed our results list with the existing data
                results = existing_df.to_dict('records')
                print(f"--> Successfully loaded {len(evaluated_taus)} previously evaluated temperatures.")
        except Exception as e:
            print(f"--> Warning: Could not read existing CSV ({e}). Starting fresh.")

    # Filter out models that have already been evaluated
    tau_models_to_run = []
    for tau, path in tau_models:
        if round(tau, 5) not in evaluated_taus:
            tau_models_to_run.append((tau, path))
            
    print(f"--> Remaining models to evaluate in this run: {len(tau_models_to_run)}")

    # --- 4. EVALUATION LOOP ---
    for i, (tau, model_path) in enumerate(tau_models_to_run):
        print(f"\n[{i+1}/{len(tau_models_to_run)}] Evaluating model T={tau}: {os.path.basename(model_path)}")
        
        torch.cuda.empty_cache()
        gc.collect()

        args.model_path = model_path
        model = load_model(args, model_zoo, device)
        model.eval()

        # Clean Accuracy
        with torch.no_grad():
            output = model(images)
            predictions = output.argmax(dim=1)
            clean_indices = (predictions == targets).nonzero(as_tuple=True)[0]
            clean_acc = (len(clean_indices) / len(targets)) * 100
            print(f"  > Clean Accuracy: {clean_acc:.2f}%")

        # Lipschitz Constants
        L_theory = convert_lipschitz_constant(1, str(args.norm), images[0].numel())
        inp_shape = (1, 3, 224, 224) if "imagenette" in args.dataset.lower() else ((1, 1, 28, 28) if "mnist" in args.dataset.lower() else (1, 3, 32, 32))
        try:
            L_2_empirical = compute_model_lipschitz(model, input_shape=inp_shape, device=device)
            L = convert_lipschitz_constant(L_2_empirical, str(args.norm), images[0].numel())
        except Exception:
            L = L_theory

        eps_rescaled = args.eps / 0.225 if ("cifar" in args.dataset.lower() or "imagenette" in args.dataset.lower()) else args.eps

        # A. Empirical Robust Accuracy (ERA)
        era_val, _, _ = compute_autoattack_era_and_time(
            images, targets, model, args.eps, clean_indices, 
            norm=str(args.norm), dataset_name=args.dataset, return_robust_points=True
        )
        print(f"  > ERA (AutoAttack): {era_val:.2f}%")

        # B. Certified Robust Accuracy (CRA)
        _, cra_val, _, _ = compute_certificates_CRA(
            images, model, eps_rescaled, clean_indices, norm=str(args.norm), L=L, return_robust_points=True
        )
        print(f"  > CRA (Lipschitz):  {cra_val:.2f}%")

        # --- SETUP BOUNDS FOR CROWN ---
        x_L = torch.zeros((1, 1, 28, 28), device=device)
        x_U = torch.ones((1, 1, 28, 28), device=device)
        if "cifar" in args.dataset.lower():
            MEANS = torch.tensor([0.4914, 0.4822, 0.4465], device=device).view(1, 3, 1, 1)
            STD = torch.tensor([0.225, 0.225, 0.225], device=device).view(1, 3, 1, 1)
            x_L = ((0.0 - MEANS) / STD).expand(1, 3, 32, 32).contiguous()
            x_U = ((1.0 - MEANS) / STD).expand(1, 3, 32, 32).contiguous()
        elif "imagenette" in args.dataset.lower():
            MEANS = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
            STD = torch.tensor([0.225, 0.225, 0.225], device=device).view(1, 3, 1, 1)
            x_L = ((0.0 - MEANS) / STD).expand(1, 3, 224, 224).contiguous()
            x_U = ((1.0 - MEANS) / STD).expand(1, 3, 224, 224).contiguous()

        # C. Alpha-CROWN
        alpha_val = 0.0
        try:
            alpha_val, _, _ = compute_alphacrown_vra_and_time(
                images, targets, model, eps_rescaled, clean_indices, args, 
                batch_size=args.batch_size, norm=args.norm, x_U=x_U, x_L=x_L, return_robust_points=True
            )
            print(f"  > Alpha-CROWN VRA:  {alpha_val:.2f}%")
        except Exception as e:
            print(f"  > ? [Alpha-CROWN] Failed or OOM: {e}")

        # D. SDP-CROWN (Best of high_tau=False and high_tau=True)
        sdp_val = 0.0
        groupsort = True if ("GNP" in args.model or "Bjork" in args.model) else False
        
        # 1. Evaluate with high_tau = False
        args.high_tau = False
        sdp_val_low = 0.0
        try:
            sdp_val_low, _, _ = compute_sdp_crown_vra(
                images, targets, model, float(eps_rescaled), clean_indices, 
                device, classes, args, batch_size=1, return_robust_points=True, x_U=x_U, x_L=x_L, groupsort=groupsort
            )
            print(f"  > SDP-CROWN VRA (high_tau=False): {sdp_val_low:.2f}%")
        except Exception as e:
            print(f"  > ? [SDP-CROWN high_tau=False] Failed or OOM: {e}")

        # 2. Evaluate with high_tau = True
        args.high_tau = True
        sdp_val_high = 0.0
        try:
            sdp_val_high, _, _ = compute_sdp_crown_vra(
                images, targets, model, float(eps_rescaled), clean_indices, 
                device, classes, args, batch_size=1, return_robust_points=True, x_U=x_U, x_L=x_L, groupsort=groupsort
            )
            print(f"  > SDP-CROWN VRA (high_tau=True):  {sdp_val_high:.2f}%")
        except Exception as e:
            print(f"  > ? [SDP-CROWN high_tau=True] Failed or OOM: {e}")
        print('CHECKPOINT : ', sdp_val_low, ' low temp vs ', sdp_val_high, 'high temp')
        # 3. Take the maximum (best) metric
        sdp_val = max(sdp_val_low, sdp_val_high)
        print(f"  > SDP-CROWN VRA (Best):           {sdp_val:.2f}%")

        # Store iteration results
        results.append({
            'tau': tau,
            'clean_acc': clean_acc,
            'era': era_val,
            'cra': cra_val,
            'alpha_crown': alpha_val,
            'sdp_crown': sdp_val
        })

        df_temp = pd.DataFrame(results)
        df_temp.to_csv(args.output_csv, index=False)
        print(f"  > Checkpoint: Results saved to {args.output_csv}")

    # --- 5. DATA AGGREGATION & INITIALIZATION ---
    if len(results) < 2:
        print("Need at least 2 models to plot robust accuracy metrics.")
        return

    df = pd.DataFrame(results)
    df.to_csv(args.output_csv, index=False)
    print(f"\nFinal raw data saved to {args.output_csv}")
    
    # --- 6. IDENTIFY MODELS WITH POTENTIAL & PLOTTING ---
    print(f"\n" + "="*50)
    print(f"--- POTENTIAL ANALYSIS FOR {args.model} ---")
    
    # Find the maximum CRA across all temperatures
    max_cra = df['cra'].max()
    print(f"Maximum Certified Robust Accuracy (CRA) observed: {max_cra:.2f}%")
    
    sorted_df = df.sort_values(by='tau')
    taus_sorted = sorted_df['tau'].values
    eras_sorted = sorted_df['era'].values
    cras_sorted = sorted_df['cra'].values
    alphas_sorted = sorted_df['alpha_crown'].values
    sdps_sorted = sorted_df['sdp_crown'].values

    # Find the potential region (where ERA > Max CRA)
    potential_df = sorted_df[sorted_df['era'] > max_cra].copy()
    has_potential = not potential_df.empty

    # Dynamically determine the bounds of the potential region
    if has_potential:
        min_tau_pot = potential_df['tau'].min()
        max_tau_pot = potential_df['tau'].max()
    else:
        min_tau_pot, max_tau_pot = 0, 0

    base_dir = os.path.dirname(args.output_plot_temp)
    base_name, ext = os.path.splitext(os.path.basename(args.output_plot_temp))
    if not ext:
        ext = '.png'

    # ==========================================
    # PLOT 1: All Metrics vs ALL Temperatures
    # ==========================================
    plt.figure(figsize=(11, 7))

    plt.plot(taus_sorted, eras_sorted, marker='o', markersize=8, linestyle='-', color='crimson', linewidth=2, label='Empirical (ERA)')
    plt.plot(taus_sorted, alphas_sorted, marker='^', markersize=8, linestyle='-', color='darkorange', linewidth=2, label='Alpha-CROWN')
    plt.plot(taus_sorted, sdps_sorted, marker='d', markersize=8, linestyle='-', color='mediumpurple', linewidth=2, label='SDP-CROWN')
    plt.plot(taus_sorted, cras_sorted, marker='s', markersize=8, linestyle='-', color='teal', linewidth=2, label='Lipschitz CRA')
    
    plt.axhline(y=max_cra, color='dimgrey', linestyle='--', linewidth=2, label=f'Max CRA ({max_cra:.2f}%)')

    if has_potential:
        plt.axvspan(min_tau_pot * 0.9, max_tau_pot * 1.1, color='gold', alpha=0.15, label='Potential Region')

    plt.xlabel('Temperature (Tau)')
    plt.ylabel('Accuracy (%)')
    plt.title(f'Robust Accuracy Metrics vs Temperature\nModel: {args.model} | Dataset: {args.dataset} | Epsilon: {args.eps:.4f}')
    plt.legend(loc='best')
    plt.grid(True, linestyle=':', alpha=0.7)

    output_plot_main = os.path.join(base_dir, f"{base_name}_all_metrics{ext}")
    plt.savefig(output_plot_main, dpi=300, bbox_inches='tight')
    print(f"Plot 1 (All Temperatures) saved successfully to: {output_plot_main}")
    plt.close()

    # ==========================================
    # PLOT 2: Auto-Zoomed Potential Region
    # ==========================================
    if has_potential:
        plt.figure(figsize=(9, 6))
        
        tau_span = max_tau_pot - min_tau_pot
        margin = tau_span * 0.5 if tau_span > 0 else (taus_sorted[-1] - taus_sorted[0]) * 0.05
        
        zoom_min = max(taus_sorted[0], min_tau_pot - margin)
        zoom_max = min(taus_sorted[-1], max_tau_pot + margin)
        
        zoom_df = sorted_df[(sorted_df['tau'] >= zoom_min) & (sorted_df['tau'] <= zoom_max)]
        
        plt.plot(zoom_df['tau'], zoom_df['era'], marker='o', markersize=9, linestyle='-', color='crimson', linewidth=2, label='ERA')
        plt.plot(zoom_df['tau'], zoom_df['alpha_crown'], marker='^', markersize=9, linestyle='-', color='darkorange', linewidth=2, label='Alpha-CROWN')
        plt.plot(zoom_df['tau'], zoom_df['sdp_crown'], marker='d', markersize=9, linestyle='-', color='mediumpurple', linewidth=2, label='SDP-CROWN')
        plt.plot(zoom_df['tau'], zoom_df['cra'], marker='s', markersize=9, linestyle='-', color='teal', linewidth=2, label='CRA')
        
        plt.axhline(y=max_cra, color='dimgrey', linestyle='--', linewidth=2, label=f'Max CRA ({max_cra:.2f}%)')
        
        plt.xlabel('Temperature (Tau)')
        plt.ylabel('Accuracy (%)')
        plt.title('Zoomed-In: LiRPA & Lipschitz Metrics in Potential Region')
        plt.legend(loc='best')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        output_plot_zoomed = os.path.join(base_dir, f"{base_name}_zoomed{ext}")
        plt.savefig(output_plot_zoomed, dpi=300, bbox_inches='tight')
        print(f"Plot 2 (Zoomed View) saved successfully to: {output_plot_zoomed}")
        plt.close()
    else:
        print("Skipping Plot 2 (Zoomed View): No potential region found to zoom in on.")

    # ==========================================
    # PLOT 3: Delta Plot (ERA - Max CRA)
    # ==========================================
    plt.figure(figsize=(10, 5))
    deltas = eras_sorted - max_cra
    
    markerline, stemlines, baseline = plt.stem(taus_sorted, deltas, basefmt="k-")
    plt.setp(stemlines, 'color', 'grey', 'linestyle', ':', 'alpha', 0.7, 'linewidth', 2)
    plt.setp(baseline, 'linewidth', 2)
    
    for i, d in enumerate(deltas):
        m_color = 'mediumseagreen' if d > 0 else 'lightcoral'
        plt.plot(taus_sorted[i], d, marker='o', color=m_color, markersize=8)

    plt.xlabel('Temperature (Tau)')
    plt.ylabel('Delta: ERA - Max CRA (%)')
    plt.title('Performance Delta: Does ERA beat the best overall Lipschitz CRA?')
    plt.grid(True, linestyle=':', alpha=0.5)

    output_plot_delta = os.path.join(base_dir, f"{base_name}_delta{ext}")
    plt.savefig(output_plot_delta, dpi=300, bbox_inches='tight')
    print(f"Plot 3 (Performance Delta) saved successfully to: {output_plot_delta}")
    plt.close()

    # --- Print Terminal Results & Save CSV ---
    if has_potential:
        print(f"\nFound {len(potential_df)} interesting temperature(s) where ERA > Max CRA:")
        for index, row in potential_df.iterrows():
            print(f"  > Tau: {row['tau']} | ERA: {row['era']:.2f}% | Alpha: {row['alpha_crown']:.2f}% | SDP: {row['sdp_crown']:.2f}% | CRA: {row['cra']:.2f}% | Delta (ERA): +{(row['era'] - max_cra):.2f}%")
            
        potential_csv_path = args.output_csv.replace('.csv', '_high_potential.csv')
        potential_df.to_csv(potential_csv_path, index=False)
        print(f"\nList of interesting models saved to: {potential_csv_path}")
    else:
        print("\nNo temperatures found where ERA > Max CRA.")
    print("="*50 + "\n")

if __name__ == '__main__':
    main()