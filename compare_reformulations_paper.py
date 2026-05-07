import torch
import numpy as np
import argparse
import gc
import time
import csv
import os
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from models import * 
from project_utils import *
from main_auto import *

def replace_groupsort_conventional(model):
    """ 
    Recursively replaces GroupSort_General with the Conventional variant.
    """
    for name, module in model.named_children():
        if isinstance(module, GroupSort_General):
            setattr(model, name, GroupSort2Conventional())
        else:
            replace_groupsort_conventional(module)

def compute_vra_and_time(images, targets, model, epsilon, clean_indices, args, batch_size=2, norm=2, x_U=None, x_L=None, bound_method='alpha-crown'):
    """ 
    Computes VRA using the specified bound_method.
    Includes VRAM clearing and numerical jitter to dodge auto_LiRPA bugs.
    """
    device = next(model.parameters()).device
    total_num_images = images.shape[0]
    model.eval()
    
    if not isinstance(clean_indices, torch.Tensor):
        clean_indices = torch.tensor(clean_indices)

    correct_images = images[clean_indices]
    correct_targets = targets[clean_indices]
    if len(correct_images) == 0: return 0.0, 0.0

    num_robust_points = 0
    total_time = 0.0
    num_batches = (len(correct_images) + batch_size - 1) // batch_size

    # Initialize auto_LiRPA BoundedModule
    dummy_input = correct_images[0:1].to(device)
    bounded_model = BoundedModule(model, dummy_input, bound_opts={"conv_mode": "patches"}, verbose=False)
    bounded_model.eval()

    for i in range(num_batches):
        start_idx, end_idx = i * batch_size, min((i + 1) * batch_size, len(correct_images))
        batch_images = correct_images[start_idx:end_idx].clone().to(device)
        batch_targets = correct_targets[start_idx:end_idx]
        current_bs = batch_images.shape[0]

        # Setup Global Bounds
        batch_global_L = x_L.expand(current_bs, *x_L.shape[1:]).contiguous() if x_L is not None else None
        batch_global_U = x_U.expand(current_bs, *x_U.shape[1:]).contiguous() if x_U is not None else None

        # Clamp center point to global bounds
        if batch_global_L is not None and batch_global_U is not None:
            batch_images = torch.max(torch.min(batch_images, batch_global_U), batch_global_L)

        # --- BUG FIX 1: Add Microscopic Noise to dodge alpha-CROWN MinMax AssertionError ---
        jitter = 1e-7
        
        if str(norm) == 'inf':
            ptb_L = torch.max(batch_global_L, batch_images - epsilon) - jitter
            ptb_U = torch.min(batch_global_U, batch_images + epsilon) + jitter
            ptb = PerturbationLpNorm(norm=np.inf, eps=epsilon, x_L=ptb_L, x_U=ptb_U)
        else:
            # For L2, we add jitter to the global domain bounds
            safe_L = batch_global_L - jitter if batch_global_L is not None else None
            safe_U = batch_global_U + jitter if batch_global_U is not None else None
            ptb = PerturbationLpNorm(norm=norm, eps=epsilon, x_L=safe_L, x_U=safe_U)

        bounded_input = BoundedTensor(batch_images, ptb)
        num_classes = model[-1].out_features 
        c = build_C(batch_targets.to("cpu"), num_classes).to(device)

        if device.type == 'cuda': torch.cuda.synchronize()
        start_t = time.time()
        
        try:
            lb_diff = bounded_model.compute_bounds(x=(bounded_input,), C=c, method=bound_method)[0]
            
            if device.type == 'cuda': torch.cuda.synchronize()
            total_time += (time.time() - start_t)

            is_robust = (lb_diff.view(current_bs, num_classes - 1) > 0).all(dim=1)
            num_robust_points += torch.sum(is_robust).item()
            
        except Exception as e:
            error_msg = str(e).split('\n')[0]
            print(f"      [!] auto_LiRPA solver failed on batch {i}. Error: {error_msg}")
            if device.type == 'cuda': torch.cuda.synchronize()
            total_time += (time.time() - start_t)
            
        # --- BUG FIX 2: Aggressive VRAM Clearing to dodge CUDA OOM ---
        del batch_images, bounded_input, ptb, c
        if 'lb_diff' in locals(): del lb_diff
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    cra = (num_robust_points / total_num_images) * 100.0
    return cra, (total_time / len(correct_images))


def main():
    parser = argparse.ArgumentParser(description="Compute single-point alpha-CROWN VRA for Conventional GroupSort")
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--norm', default=2, help="Norm to use (e.g., 2 or inf)")
    parser.add_argument('--batch_size', type=int, default=1) 
    # Added argument to allow custom CSV naming
    parser.add_argument('--csv_name', type=str, default='alpha_crown_benchmark.csv', help="Name of the output CSV file")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Data
    images, targets, _ = load_dataset_benchmark_auto(args)
    images = images.to(device)
    targets = targets.to(device)
    
    # Setup Global Domain Bounds
    x_L = torch.zeros_like(images[0:1]).to(device)
    x_U = torch.ones_like(images[0:1]).to(device)
    if "cifar" in args.dataset.lower():
        m, s = [0.4914, 0.4822, 0.4465], [0.225, 0.225, 0.225]
        x_L = (torch.tensor(0.0) - torch.tensor(m).view(1,3,1,1)) / torch.tensor(s).view(1,3,1,1)
        x_U = (torch.tensor(1.0) - torch.tensor(m).view(1,3,1,1)) / torch.tensor(s).view(1,3,1,1)
    x_L, x_U = x_L.to(device), x_U.to(device)

    # Load Base Model & Determine Clean Indices
    model = load_model(args, model_zoo, device).eval()
    clean_idx = (model(images).argmax(1) == targets).nonzero(as_tuple=True)[0]

    # Convert to Conventional GroupSort
    print(f"\n--- PREPARING MODEL ---")
    print("Replacing GroupSort with GroupSort2Conventional...")
    replace_groupsort_conventional(model)
    model.to(device).eval()

    # Calculate Epsilon
    target_eps = 8.0 / 255.0
    eps_rescaled = target_eps / 0.225 if "cifar" in args.dataset.lower() else target_eps

    # Run Verification
    print(f"\n--- RUNNING ALPHA-CROWN ---")
    print(f"Target Epsilon: 8/255 (~{target_eps:.4f})")
    print(f"Rescaled Epsilon: {eps_rescaled:.4f} (Used for solver)")
    
    vra, t_v = compute_vra_and_time(
        images, targets, model, eps_rescaled, clean_idx, args, 
        batch_size=args.batch_size, norm=args.norm, x_L=x_L, x_U=x_U, bound_method='alpha-crown'
    )
    
    print(f"\n✅ RESULTS:")
    print(f"Verified Robust Accuracy: {vra:.2f}%")
    print(f"Average Time per Image:   {t_v:.4f}s")

    # --- SAVE TO CSV LOGIC ---
    file_exists = os.path.isfile(args.csv_name)
    
    with open(args.csv_name, mode='a', newline='') as f:
        writer = csv.writer(f)
        # Write the header if the file is new
        if not file_exists:
            writer.writerow(['Model', 'Dataset', 'Norm', 'Epsilon', 'VRA (%)', 'Time per Image (s)'])
        
        # Write the data row
        writer.writerow([args.model, args.dataset, args.norm, '8/255', round(vra, 2), round(t_v, 4)])
        
    print(f"💾 Appended results to {args.csv_name}")

if __name__ == "__main__":
    main()