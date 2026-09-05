import os
import sys
import time
import json
import csv
import argparse
import re
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader, Subset

# --- External Libraries ---
try:
    from deel import torchlip
except ImportError:
    import torch.nn as torchlip # fallback
    
from orthogonium.layers.conv.AOC import AdaptiveOrthoConv2d
from orthogonium.reparametrizers import DEFAULT_ORTHO_PARAMS

from models import *
from project_utils import *

sys.path.insert(0, "/lustre/fswork/projects/rech/syo/utf64nw/robust-benchmark/SDP-CROWN/")
import auto_LiRPA
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from sdp_crown import verified_sdp_crown

# ==========================================
# 1. Automatic Configuration Inference
# ==========================================
def auto_detect_model_config(model_path, checkpoint):
    """
    Infers hyper-parameters automatically from checkpoint metadata or filename patterns.
    """
    config = {
        'arch': 'VGG13',
        'dataset': 'cifar10',
        'split_idx': 4,
        'lip_constant': 1.0,
        'head_act': 'ReLU'
    }

    if isinstance(checkpoint, dict) and 'config' in checkpoint:
        print("[?] Automatically loaded hyperparameters from checkpoint metadata.")
        return checkpoint['config']

    filename = os.path.basename(model_path)
    
    for a in ['ConvLarge', 'VGG16', 'VGG13']:
        if a in filename:
            config['arch'] = a
            break
            
    for d in ['imagenette', 'cifar10', 'mnist']:
        if d in filename:
            config['dataset'] = d
            break

    split_match = re.search(r'split(\d+)', filename)
    if split_match:
        config['split_idx'] = int(split_match.group(1))

    lip_match = re.search(r'_L([\d\.]+)', filename)
    if lip_match:
        config['lip_constant'] = float(lip_match.group(1))

    if 'GroupSort' in filename or 'ActGroupSort' in filename:
        config['head_act'] = 'GroupSort'
    else:
        act_match = re.search(r'Act([A-Za-z0-9]+)', filename)
        if act_match:
            config['head_act'] = act_match.group(1)

    print(f"[?] Auto-detected settings from filename:")
    print(f"    Arch: {config['arch']} | Dataset: {config['dataset']} | Split: {config['split_idx']} | Lip: {config['lip_constant']} | Head Act: {config['head_act']}\n")
    
    return config

# ==========================================
# 2. Architecture Definitions
# ==========================================
class LayerScale(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = scale
    def forward(self, x):
        return x * self.scale

ARCH_CONFIGS = {
    'VGG13': [
        ('conv', 64, 1, 3, 1), ('conv', 64, 2, 3, 1),
        ('conv', 128, 1, 3, 1), ('conv', 128, 2, 3, 1),
        ('conv', 256, 1, 3, 1), ('conv', 256, 2, 3, 1),
        ('conv', 512, 1, 3, 1), ('conv', 512, 2, 3, 1),
        ('conv', 512, 1, 3, 1), ('conv', 512, 2, 3, 1),
        ('linear', 512), ('linear', 512), ('linear', 10)
    ],
    'VGG16': [
        ('conv', 64, 1, 3, 1), ('conv', 64, 2, 3, 1),
        ('conv', 128, 1, 3, 1), ('conv', 128, 2, 3, 1),
        ('conv', 256, 1, 3, 1), ('conv', 256, 1, 3, 1), ('conv', 256, 2, 3, 1),
        ('conv', 512, 1, 3, 1), ('conv', 512, 1, 3, 1), ('conv', 512, 2, 3, 1),
        ('conv', 512, 1, 3, 1), ('conv', 512, 1, 3, 1), ('conv', 512, 2, 3, 1),
        ('linear', 512), ('linear', 512), ('linear', 10)
    ],
    'ConvLarge': [
        ('conv', 32, 1, 3, 1), ('conv', 32, 2, 4, 1),
        ('conv', 64, 1, 3, 1), ('conv', 64, 2, 4, 1),
        ('linear', 512), ('linear', 512), ('linear', 10)
    ]
}

class GroupSort_General(nn.Module):
    """
    Applies GroupSort specifically on the channel dimension.
    """
    def __init__(self, axis=1):
        super(GroupSort_General, self).__init__()
        self.axis = axis
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dims = list(range(x.dim()))
        channel_dim = dims.pop(self.axis) 
        dims.append(channel_dim)
        
        x_permuted = x.permute(dims).contiguous()
        permuted_shape = x_permuted.shape
        batch_size = permuted_shape[0]
        
        x_flat = x_permuted.reshape(batch_size, -1)
        reshaped_x = x_flat.reshape(batch_size, -1, 2)
        
        x1s = reshaped_x[..., 0]
        x2s = reshaped_x[..., 1]
        
        diff = x2s + (-1*x1s)
        relu_diff = self.relu(diff)
        
        y1 = x2s + (-1*relu_diff)
        y2 = x1s + relu_diff 
        
        sorted_pairs = torch.stack((y1, y2), dim=2)
        sorted_flat = sorted_pairs.reshape(batch_size, -1)
        
        output_permuted = sorted_flat.reshape(permuted_shape)
        
        inv_dims = list(range(x.dim()))
        last_dim = inv_dims.pop(-1)
        inv_dims.insert(self.axis, last_dim)
        
        output = output_permuted.permute(inv_dims)
        return output

class FlexibleHybridModel(nn.Module):
    def __init__(self, arch='VGG13', dataset='cifar10', split_idx=4, head_act='ReLU', lip_constant=1.0, num_classes=10):
        super().__init__()
        
        config = ARCH_CONFIGS[arch]
        in_channels = 3
        input_size = 224 if dataset.lower() in ['imagenette', 'imagenet'] else 32

        # Calculate number of constrained layers based on the split index
        N = min(split_idx, len(config))
        scale_layers = N if N < len(config) else N - 1
        per_layer_lip = (lip_constant ** (1.0 / scale_layers)) if scale_layers > 0 else 1.0

        prefix_layers = []
        suffix_layers = []
        
        current_in_conv = in_channels
        curr_spatial = input_size
        flattened = False
        current_in_lin = 0

        for idx, layer_cfg in enumerate(config):
            is_constrained = idx < split_idx
            layer_type = layer_cfg[0]
            is_last_layer = idx == (len(config) - 1)
            
            step_modules = []

            if layer_type == 'conv':
                _, out_c, stride, kernel, pad = layer_cfg
                
                if is_constrained:
                    step_modules.append(AdaptiveOrthoConv2d(
                        current_in_conv, out_c, kernel_size=kernel, stride=stride, 
                        padding=pad, padding_mode='zeros', ortho_params=DEFAULT_ORTHO_PARAMS
                    ))
                    step_modules.append(GroupSort_General())
                    step_modules.append(LayerScale(per_layer_lip))
                else:
                    step_modules.append(nn.Conv2d(
                        current_in_conv, out_c, kernel_size=kernel, stride=stride, padding=pad
                    ))
                    step_modules.append(nn.ReLU() if head_act.lower() == 'relu' else GroupSort_General())

                current_in_conv = out_c
                curr_spatial = (curr_spatial + 2 * pad - kernel) // stride + 1

            elif layer_type == 'linear':
                if not flattened:
                    step_modules.append(nn.Flatten())
                    current_in_lin = current_in_conv * curr_spatial * curr_spatial
                    flattened = True
                
                out_features = layer_cfg[1]
                if is_last_layer:
                    out_features = num_classes
                
                if is_constrained:
                    step_modules.append(torchlip.SpectralLinear(current_in_lin, out_features))
                    if not is_last_layer:
                        step_modules.append(GroupSort_General())
                        step_modules.append(LayerScale(per_layer_lip))
                else:
                    step_modules.append(nn.Linear(current_in_lin, out_features))
                    if not is_last_layer:
                        step_modules.append(nn.ReLU() if head_act.lower() == 'relu' else GroupSort_General())
                
                current_in_lin = out_features

            if is_constrained:
                prefix_layers.extend(step_modules)
            else:
                suffix_layers.extend(step_modules)

        self.prefix = nn.Sequential(*prefix_layers)
        self.suffix = nn.Sequential(*suffix_layers)

    def forward(self, x):
        features = self.prefix(x)
        logits = self.suffix(features)
        return logits

# ==========================================
# 3. Hybrid Verifier
# ==========================================
def compute_hybrid_vra_flexible(images, targets, model, eps_rescaled, clean_indices, device, classes, args):
    """
    Directly verifies FlexibleHybridModel architectures.
    Propagates exactly through the prefix, and runs LiRPA/SDP-CROWN on the suffix.
    """
    start_time = time.time()
    hybrid_backend = args.hybrid_backend.lower()

    # Directly access the pre-split components
    f1_prefix = model.prefix.eval()
    f2_suffix = model.suffix.eval()

    # Compute Intermediate Epsilon
    L_total = args.lip_constant 
    
    if str(args.norm) == 'inf':
        input_dim = images[0].numel()
        eps_backbone = eps_rescaled * np.sqrt(input_dim) 
    else:
        eps_backbone = eps_rescaled

    intermediate_epsilon = float(eps_backbone * L_total)

    # Forward prefix
    with torch.no_grad():
        z_k = f1_prefix(images.to(device))

    # Check Affine Wrapper Requirement
    if not starts_with_affine(f2_suffix):
        f2_suffix_ready = wrap_with_identity(f2_suffix, z_k)
        print("Wrapped unconstrained head with Identity Affine layer.")
    else:
        f2_suffix_ready = f2_suffix

    has_groupsort = any(isinstance(m, GroupSort_General) for m in f2_suffix_ready.modules())

    print(f"Running {hybrid_backend.upper()} on intermediate features (Shape: {list(z_k.shape)}, Radius: {intermediate_epsilon:.5f})")
    
    if hybrid_backend == 'sdp':
        vra, t_v, idx_robust = compute_sdp_crown_vra(
            z_k, targets, f2_suffix_ready, intermediate_epsilon, clean_indices, 
            device, classes, args, batch_size=1, return_robust_points=True, 
            x_U=None, x_L=None, groupsort=has_groupsort
        )
    elif hybrid_backend == 'alphacrown':
        vra, t_v, idx_robust = compute_alphacrown_vra_and_time(
            z_k, targets, f2_suffix_ready, intermediate_epsilon, clean_indices, args, 
            batch_size=args.batch_size, norm=2, x_U=None, x_L=None, return_robust_points=True
        )
    elif hybrid_backend == 'ibpcrown':
        vra, t_v, idx_robust = compute_ibpcrown_vra_and_time(
            z_k, targets, f2_suffix_ready, intermediate_epsilon, clean_indices, args, 
            batch_size=args.batch_size, norm=2, x_U=None, x_L=None, return_robust_points=True
        )
    else:
        raise ValueError(f"Unknown hybrid backend selected: {hybrid_backend}")

    total_time = time.time() - start_time
    return vra, total_time, idx_robust

# ==========================================
# 4. Main Execution Block
# ==========================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Automated Hybrid Verification & Attack Script")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the saved .pth file")
    parser.add_argument('--eps', type=float, default=0.141, help="Epsilon radius")
    parser.add_argument('--norm', type=str, default='2', choices=['2', 'inf'])
    parser.add_argument('--hybrid_backend', type=str, default='sdp', choices=['sdp', 'alphacrown', 'ibpcrown'])
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--high_tau', action='store_true', help="Use high tau setting for SDP-CROWN")
    parser.add_argument('--no_aa', action='store_true', help="Disable pre-verification AutoAttack")
    
    # NEW FLAG: Run full verification on ConvLarge
    parser.add_argument('--run_full_sdp', action='store_true', help="Also run SDP-CROWN on the FULL model (ConvLarge only)")
    
    # Optional verifier flags
    parser.add_argument('--lr_alpha', type=float, default=0.1)
    parser.add_argument('--lr_lambda', type=float, default=0.1)
    parser.add_argument('--lr_beta', type=float, default=0.05)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    raw_checkpoint = torch.load(args.model_path, map_location=device)
    config = auto_detect_model_config(args.model_path, raw_checkpoint)

    # Populate missing verification arguments dynamically
    default_verifier_args = {
        'model': config['arch'],
        'arch': config['arch'],
        'dataset': config['dataset'],
        'split_idx': config['split_idx'],
        'lip_constant': config['lip_constant'],
        'head_act': config['head_act'],
        'high_tau': getattr(args, 'high_tau', False),
        'lr_alpha': 0.1,
        'lr_lambda': 0.1,
        'lr_beta': 0.05,
        'use_conventional_groupsort': False,
        'remove_sparse_ibp': False,
        'otherpoints': False,
        'start': 0,
        'end': None
    }

    for key, val in default_verifier_args.items():
        if not hasattr(args, key) or getattr(args, key) is None:
            setattr(args, key, val)

    print(f"Loading {args.dataset}...")
    dataset, labels, classes = load_dataset_benchmark_auto(args)
    dataset = dataset.to(device)
    labels = labels.to(device)

    model = vanilla_export(FlexibleHybridModel(
        arch=args.arch,
        dataset=args.dataset,
        split_idx=args.split_idx,
        head_act=args.head_act,
        lip_constant=args.lip_constant,
        num_classes=classes
    ))

    print(f"Loading weights from {args.model_path}...")
    state_dict = raw_checkpoint['state_dict'] if isinstance(raw_checkpoint, dict) and 'state_dict' in raw_checkpoint else raw_checkpoint

    # Convert model structure if checkpoint contains unparametrized weights
    if any("prefix" in k and ".weight" in k and "parametrizations" not in k for k in state_dict.keys()):
        model = vanilla_export(model)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Find Cleanly Classified Indices
    with torch.no_grad():
        outputs = model(dataset)
        preds = outputs.argmax(dim=1)
        clean_mask = (preds == labels)
        clean_indices = clean_mask.nonzero(as_tuple=False).squeeze()
        clean_acc = (len(clean_indices) / len(dataset)) * 100.0
        
    print(f"Clean Accuracy: {clean_acc:.2f}% ({len(clean_indices)}/{len(dataset)})")

    # Rescale epsilon for normalization
    eps_rescaled = args.eps / 0.225 if ("cifar" in args.dataset.lower() or "imagenette" in args.dataset.lower()) else args.eps

    verification_candidates = clean_indices
    era = None

    # Option 1: Run Empirical AutoAttack from project_utils first
    if not args.no_aa:
        print(f"Running AutoAttack (eps={args.eps}, norm=L{args.norm})...")
        era, aa_time, aa_robust_indices = compute_autoattack_era_and_time(
            images=dataset,
            targets=labels,
            model=model,
            epsilon=args.eps,
            clean_indices=clean_indices,
            norm=args.norm,
            dataset_name=args.dataset,
            return_robust_points=True
        )
        print(f"AutoAttack Finished | ERA: {era:.2f}% | Avg Time/Image: {aa_time:.4f}s")

        # EARLY STOPPING: Skip formal verification if AutoAttack broke all clean samples
        if len(aa_robust_indices) == 0 or era == 0.0:
            print("\n[!] AutoAttack reduced empirical accuracy to 0.00%.")
            print("    Skipping formal verification (VRA is guaranteed to be 0.00%).")
            print(f"\n? Final Results:")
            print(f"  • ERA (AutoAttack) : 0.00%")
            print(f"  • VRA (Hybrid)     : 0.00%")
            sys.exit(0)
            
        verification_candidates = aa_robust_indices

    # Option 2: Run Formal Verification on Surviving Candidates
    vra, exec_time, robust_idx = compute_hybrid_vra_flexible(
        images=dataset, 
        targets=labels, 
        model=model, 
        eps_rescaled=eps_rescaled, 
        clean_indices=verification_candidates, 
        device=device, 
        classes=classes, 
        args=args
    )

    # Option 3: Full Network Verification (Optional)
    full_vra = None
    full_exec_time = None
    if getattr(args, 'run_full_sdp', False) and config['arch'] == 'ConvLarge':
        print(f"\n--- Running FULL Network SDP-CROWN for {config['arch']} ---")
        
        # In case the model isn't recognized as starting with an affine layer by auto_LiRPA
        try:
            full_model_ready = wrap_with_identity(model, dataset[0:1].to(device))
            print("Wrapped full model with Identity Affine layer.")
        except Exception:
            full_model_ready = model
            print("Using model as-is for full verification.")

        has_groupsort_full = any(isinstance(m, GroupSort_General) for m in full_model_ready.modules())
        
        # For full verification, the epsilon bound applies directly to the image input
        if str(args.norm) == 'inf':
            input_dim = dataset[0].numel()
            full_eps = eps_rescaled * np.sqrt(input_dim) 
        else:
            full_eps = eps_rescaled

        start_time_full = time.time()
        
        # ?? INDEPENDENT STUDY: Pass clean_indices directly, bypassing verification_candidates (AA results)
        full_vra, _, _ = compute_sdp_crown_vra(
            dataset, labels=labels, model=full_model_ready, radius=full_eps, 
            clean_output=clean_indices, device=device, classes=classes, 
            args=args, batch_size=args.batch_size, return_robust_points=True, 
            x_U=None, x_L=None, groupsort=has_groupsort_full
        )
        full_exec_time = time.time() - start_time_full

    print(f"\n? Final Results:")
    print(f"  • Backend          : {args.hybrid_backend.upper()}")
    print(f"  • Epsilon          : {args.eps} (Scaled to {eps_rescaled:.5f} in normalized space)")
    print(f"  • Norm             : L{args.norm}")
    if era is not None:
        print(f"  • ERA (AutoAttack) : {era:.2f}%")
    print(f"  • VRA (Hybrid)     : {vra:.2f}%")
    print(f"  • Time (Hybrid)    : {exec_time:.2f}s")
    
    if full_vra is not None:
        print(f"  • VRA (Full SDP)   : {full_vra:.2f}%")
        print(f"  • Time (Full SDP)  : {full_exec_time:.2f}s")

   # --- Save Results to CSV ---
    results_file = "verification_results.csv"
    
    # Extract the actual model name from the file path
    model_name = os.path.basename(args.model_path)
    
    # Structure the data with full model characterization
    results_data = {
        "Model": model_name, 
        "Dataset": config['dataset'],
        "Split_Idx": config['split_idx'],          # <--- Added Split
        "Lip_Constant": config['lip_constant'],    # <--- Added Lipschitz Constant
        "Head_Act": config['head_act'],            # <--- Added Head Activation
        "Backend": args.hybrid_backend.upper(),
        "Epsilon": args.eps,
        "Norm": f"L{args.norm}",
        "Clean_Acc": clean_acc,
        "ERA": era if era is not None else "",
        "VRA": vra,
        "Time_s": exec_time,
        "Full_VRA": full_vra if full_vra is not None else "",
        "Full_Time_s": full_exec_time if full_exec_time is not None else ""
    }
    
    # Convert to DataFrame and append to CSV
    df = pd.DataFrame([results_data])
    file_exists = os.path.isfile(results_file)
    
    df.to_csv(results_file, mode='a', header=not file_exists, index=False)
    print(f" ?? Results appended to {results_file}")