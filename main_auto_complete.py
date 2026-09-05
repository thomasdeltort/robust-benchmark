import os
import sys
import time
import re
import copy
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrize import is_parametrized

# --- External Libraries ---
try:
    from deel import torchlip
except ImportError:
    import torch.nn as torchlip # fallback
    
from orthogonium.layers.conv.AOC import AdaptiveOrthoConv2d
from orthogonium.reparametrizers import DEFAULT_ORTHO_PARAMS

# --- Handle local module imports ---
sys.path.append('./..')
try:
    from models import *
    from project_utils import *
    sys.path.insert(0, "/lustre/fswork/projects/rech/syo/utf64nw/robust-benchmark/SDP-CROWN/")
    import auto_LiRPA
    from auto_LiRPA import BoundedModule, BoundedTensor
    from sdp_crown import verified_sdp_crown
except ImportError:
    print("Warning: Local modules or SDP-CROWN path not found.")

# ==========================================
# 1. Automatic Configuration Inference
# ==========================================
def auto_detect_model_config(model_path, checkpoint, args):
    """Infers hyper-parameters from checkpoint metadata or filename."""
    config = {
        'arch': getattr(args, 'arch', 'VGG13'),
        'dataset': getattr(args, 'dataset', 'cifar10'),
        'split_idx': getattr(args, 'split_idx', 4),
        'b1_type': getattr(args, 'b1_type', 'aoc'),
        'b2_type': getattr(args, 'b2_type', 'unconstrained'),
        'b1_lip': getattr(args, 'b1_lip', 1.0),
        'b2_lip': getattr(args, 'b2_lip', 1.0),
        'b1_act': getattr(args, 'b1_act', 'GroupSort'),
        'b2_act': getattr(args, 'b2_act', 'ReLU')
    }

    if isinstance(checkpoint, dict) and 'config' in checkpoint:
        print("[?] Automatically loaded hyperparameters from checkpoint metadata.")
        saved_config = checkpoint['config']
        for k, v in saved_config.items():
            if k in config: config[k] = v
            # Backward compatibility
            elif k == 'lip_constant': config['b1_lip'] = v; config['b2_lip'] = v
            elif k == 'head_act': config['b2_act'] = v
    else:
        print("[!] No internal metadata found. Falling back to filename regex parsing...")
        filename = os.path.basename(model_path).lower()
        
        for a in ['convlarge', 'vgg16', 'vgg13']:
            if a in filename: config['arch'] = ('ConvLarge' if a == 'convlarge' else a.upper())
        for d in ['imagenette', 'cifar10', 'mnist']:
            if d in filename: config['dataset'] = d
        split_match = re.search(r'split(\d+)', filename)
        if split_match: config['split_idx'] = int(split_match.group(1))
        lip_match = re.search(r'_l([\d\.]+)', filename)
        if lip_match:
            config['b1_lip'] = config['b2_lip'] = float(lip_match.group(1))
        act_match = re.search(r'act([a-z0-9]+)', filename)
        if act_match:
            config['b2_act'] = 'GroupSort' if 'groupsort' in act_match.group(1) else 'ReLU'

    print(f"\n--- Active Configuration ---")
    print(f"  Arch: {config['arch']} | Dataset: {config['dataset']} | Split: {config['split_idx']}")
    print(f"  B1: {config['b1_type'].upper()} | Lip: {config['b1_lip']} | Act: {config['b1_act']}")
    print(f"  B2: {config['b2_type'].upper()} | Lip: {config['b2_lip']} | Act: {config['b2_act']}")
    print("----------------------------\n")
    return config

# ==========================================
# 2. Architecture Definitions
# ==========================================
class LayerScale(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = scale
    def forward(self, x): return x * self.scale

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
    def __init__(self, axis=1):
        super().__init__()
        self.axis = axis

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dims = list(range(x.dim()))
        channel_dim = dims.pop(self.axis) 
        dims.append(channel_dim)
        
        x_perm = x.permute(dims).contiguous()
        b = x_perm.shape[0]
        reshaped = x_perm.reshape(b, -1, 2)
        x1, x2 = reshaped[..., 0], reshaped[..., 1]
        
        relu_diff = F.relu(x2 - x1)
        y1, y2 = x2 - relu_diff, x1 + relu_diff
        
        sorted_flat = torch.stack((y1, y2), dim=2).reshape(b, -1)
        output_perm = sorted_flat.reshape(x_perm.shape)
        
        inv_dims = list(range(x.dim()))
        inv_dims.insert(self.axis, inv_dims.pop(-1))
        return output_perm.permute(inv_dims)

class FlexibleHybridModel(nn.Module):
    def __init__(self, arch, dataset, split_idx, b1_type, b2_type, b1_lip, b2_lip, b1_act, b2_act, num_classes=10):
        super().__init__()
        config = ARCH_CONFIGS[arch]
        in_c, spatial = 3, (224 if dataset.lower() in ['imagenette', 'imagenet'] else 32)
        
        b1_count = min(split_idx, len(config))
        b2_count = max(0, len(config) - b1_count)
        b1_scale = (b1_lip ** (1.0 / b1_count)) if b1_count > 0 else 1.0
        b2_scale = (b2_lip ** (1.0 / b2_count)) if b2_count > 0 else 1.0

        p_layers, s_layers = [], []
        flattened = False

        for idx, layer_cfg in enumerate(config):
            is_b1 = idx < split_idx
            is_last = idx == (len(config) - 1)
            b_type = b1_type.lower() if is_b1 else b2_type.lower()
            scale = b1_scale if is_b1 else b2_scale
            act = b1_act if is_b1 else b2_act
            
            l_type = layer_cfg[0]
            out_f = layer_cfg[1]
            
            steps = []
            if l_type == 'conv':
                stride, kernel, pad = layer_cfg[2], layer_cfg[3], layer_cfg[4]
                
                if b_type == 'aoc':
                    steps.append(AdaptiveOrthoConv2d(in_c, out_f, kernel, stride, pad, padding_mode='zeros', ortho_params=DEFAULT_ORTHO_PARAMS))
                elif b_type == 'torchlip':
                    steps.append(torchlip.SpectralConv2d(in_c, out_f, kernel, stride, pad))
                else:
                    steps.append(nn.Conv2d(in_c, out_f, kernel, stride, pad))
                
                steps.append(nn.ReLU() if act == 'ReLU' else GroupSort_General())
                if b_type in ['aoc', 'torchlip'] and scale != 1.0: steps.append(LayerScale(scale))
                in_c, spatial = out_f, (spatial + 2 * pad - kernel) // stride + 1

            elif l_type == 'linear':
                if not flattened:
                    steps.append(nn.Flatten())
                    in_c = in_c * spatial * spatial
                    flattened = True
                
                out_dim = num_classes if is_last else out_f
                if b_type in ['aoc', 'torchlip']:
                    steps.append(torchlip.SpectralLinear(in_c, out_dim))
                else:
                    steps.append(nn.Linear(in_c, out_dim))
                    
                if not is_last: steps.append(nn.ReLU() if act == 'ReLU' else GroupSort_General())
                if b_type in ['aoc', 'torchlip'] and scale != 1.0: steps.append(LayerScale(scale))
                in_c = out_dim

            (p_layers if is_b1 else s_layers).extend(steps)

        self.prefix, self.suffix = nn.Sequential(*p_layers), nn.Sequential(*s_layers)

    def forward(self, x): return self.suffix(self.prefix(x))

def vanilla_export(model1):
    model1.eval()
    model2 = copy.deepcopy(model1).eval()
    dict_modified_layers = {}
    for (n1, p1), (n2, p2) in zip(model1.named_modules(), model2.named_modules()):
        if isinstance(p1, nn.Conv2d) and is_parametrized(p1):
            new_layer = nn.Conv2d(p1.in_channels, p1.out_channels, p1.kernel_size, p1.stride, p1.padding, bias=(p1.bias is not None))
            new_layer.weight.data = p1.weight.data.clone()
            if p1.bias is not None: new_layer.bias.data = p1.bias.data.clone()
            dict_modified_layers[n2] = new_layer
        elif isinstance(p1, nn.Linear) and is_parametrized(p1):
            new_layer = nn.Linear(p1.in_features, p1.out_features, bias=(p1.bias is not None))
            new_layer.weight.data = p1.weight.data.clone()
            if p1.bias is not None: new_layer.bias.data = p1.bias.data.clone()
            dict_modified_layers[n2] = new_layer
            
    for n2, new_layer in dict_modified_layers.items():
        attrs = n2.split('.')
        setattr(getattr(model2, '.'.join(attrs[:-1])) if len(attrs) > 1 else model2, attrs[-1], new_layer)
    return model2

# ==========================================
# 3. Verifier & Analysis Tools
# ==========================================
def compute_hybrid_vra_flexible(images, targets, model, eps_rescaled, clean_indices, device, classes, args, config):
    start_time = time.time()
    f1_prefix, f2_suffix = model.prefix.eval(), model.suffix.eval()

    intermediate_epsilon = float(eps_rescaled * config['b1_lip'])
    if str(args.norm) == 'inf':
        intermediate_epsilon *= np.sqrt(images[0].numel()) 

    with torch.no_grad():
        z_k = f1_prefix(images.to(device))

    f2_suffix_ready = wrap_with_identity(f2_suffix, z_k) if not starts_with_affine(f2_suffix) else f2_suffix
    has_groupsort = any(isinstance(m, GroupSort_General) for m in f2_suffix_ready.modules())
    
    print(f"Running {args.hybrid_backend.upper()} on intermediate features (Radius: {intermediate_epsilon:.5f})")
    
    if args.hybrid_backend == 'sdp':
        vra, t_v, idx_robust = compute_sdp_crown_vra(
            z_k, targets, f2_suffix_ready, intermediate_epsilon, clean_indices, 
            device, classes, args, batch_size=1, return_robust_points=True, 
            x_U=None, x_L=None, groupsort=has_groupsort
        )
    elif args.hybrid_backend == 'alphacrown':
        vra, t_v, idx_robust = compute_alphacrown_vra_and_time(
            z_k, targets, f2_suffix_ready, intermediate_epsilon, clean_indices, args, 
            batch_size=args.batch_size, norm=2, return_robust_points=True
        )
    return vra, time.time() - start_time, idx_robust

def compute_analytical_cras(model, images, labels, eps_req, config, device):
    """Computes Global CRA and Last-Layer Normalized (LLN) CRA dynamically."""
    model.eval()
    
    b1_lip = config['b1_lip']
    b2_lip = config['b2_lip']
    is_fully_constrained = (config['b1_type'] != 'unconstrained') and (config['b2_type'] != 'unconstrained')
    
    if is_fully_constrained:
        print("\n[i] Fully constrained network detected. Computing strict Global CRA.")
        L_global = b1_lip * b2_lip
        L_h = None # LLN doesn't apply cleanly in the same way without SN tracking
    else:
        # Head is unconstrained
        if any(isinstance(m, nn.Conv2d) for m in model.suffix.modules()):
            print("\n[!] Convolutional layer detected in unconstrained head. Doing nothing (Skipping SN & CRA).")
            return None, None
            
        suffix_linears = [m for m in model.suffix.modules() if isinstance(m, nn.Linear)]
        if not suffix_linears:
            return None, None
            
        print("\n--- Unconstrained Layers Spectral Norms ---")
        sn_list = [torch.linalg.matrix_norm(m.weight, 2).item() for m in suffix_linears]
        for idx, sn in enumerate(sn_list):
            print(f"  • Suffix Linear {idx+1} SN : {sn:.4f}")
            
        L_h = b1_lip * np.prod(sn_list[:-1]) if len(sn_list) > 1 else b1_lip
        L_global = L_h * sn_list[-1]
        
        print(f"  • Constrained Backbone Lip : {b1_lip:.4f}")
        print(f"  • Cumulative L_h (Penult)  : {L_h:.4f}")
        print(f"  • Cumulative L_global      : {L_global:.4f}")
        print("-------------------------------------------\n")
    
    batch_size = 128
    global_robust_correct, lln_robust_correct = 0, 0
    total_samples = len(images)
    
    if not is_fully_constrained:
        last_weight = suffix_linears[-1].weight.detach()

    with torch.no_grad():
        for i in range(0, total_samples, batch_size):
            batch_img, batch_lab = images[i:i+batch_size].to(device), labels[i:i+batch_size].to(device)
            logits = model(batch_img)
            preds = logits.argmax(dim=1)
            correct_mask = (preds == batch_lab)
            N_b = logits.shape[0]
            
            # --- Global CRA ---
            logits_clone = logits.clone()
            logits_clone[torch.arange(N_b), batch_lab] = -float('inf')
            margins = logits[torch.arange(N_b), batch_lab] - logits_clone.max(dim=1)[0]
            
            global_radii = margins / (np.sqrt(2) * L_global)
            global_robust_correct += ((global_radii > eps_req) & correct_mask).sum().item()
            
            # --- LLN CRA (Only if Unconstrained Linears exist) ---
            if not is_fully_constrained:
                W_y = last_weight[batch_lab]
                diffs = W_y.unsqueeze(1) - last_weight.unsqueeze(0)
                norms = torch.norm(diffs, p=2, dim=2)
                norms[torch.arange(N_b), batch_lab] = float('inf') 
                norms[norms == 0] = float('inf') 
                
                lln_radii_all = (logits[torch.arange(N_b), batch_lab].unsqueeze(1) - logits) / (L_h * norms)
                lln_radii_all[torch.arange(N_b), batch_lab] = float('inf')
                lln_radii = lln_radii_all.min(dim=1)[0]
                lln_robust_correct += ((lln_radii > eps_req) & correct_mask).sum().item()
            
    global_cra = (global_robust_correct / total_samples) * 100.0
    lln_cra = (lln_robust_correct / total_samples) * 100.0 if not is_fully_constrained else None
    
    return global_cra, lln_cra

# ==========================================
# 4. Main Execution Block
# ==========================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Automated Hybrid Verification Script")
    parser.add_argument('--model_path', type=str, required=True)
    
    # Optional overrides
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--arch', type=str)
    parser.add_argument('--split_idx', type=int)
    parser.add_argument('--b1_type', type=str)
    parser.add_argument('--b2_type', type=str)
    parser.add_argument('--b1_lip', type=float)
    parser.add_argument('--b2_lip', type=float)
    parser.add_argument('--b1_act', type=str)
    parser.add_argument('--b2_act', type=str)
    
    # Attack & Verification
    parser.add_argument('--eps', type=float, default=0.141)
    parser.add_argument('--norm', type=str, default='2', choices=['2', 'inf'])
    parser.add_argument('--hybrid_backend', type=str, default='sdp', choices=['sdp', 'alphacrown', 'ibpcrown'])
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--no_aa', action='store_true')
    parser.add_argument('--run_full_sdp', action='store_true')

    parser.add_argument('--lr_alpha', type=float, default=0.1)
    parser.add_argument('--lr_lambda', type=float, default=0.1)
    
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    raw_checkpoint = torch.load(args.model_path, map_location=device)
    config = auto_detect_model_config(args.model_path, raw_checkpoint, args)

    # Updates the 'args' object with the auto-detected metadata
    for k, v in config.items():
        setattr(args, k, v)
    # --------------------------

    print(f"Loading {config['dataset']}...")
    dataset, labels, classes = load_dataset_benchmark_auto(args)
    dataset, labels = dataset.to(device), labels.to(device)

    model = FlexibleHybridModel(
        arch=config['arch'], dataset=config['dataset'], split_idx=config['split_idx'],
        b1_type=config['b1_type'], b2_type=config['b2_type'], 
        b1_lip=config['b1_lip'], b2_lip=config['b2_lip'], 
        b1_act=config['b1_act'], b2_act=config['b2_act'], num_classes=classes
    )
    model = vanilla_export(model)

    state_dict = raw_checkpoint['state_dict'] if isinstance(raw_checkpoint, dict) and 'state_dict' in raw_checkpoint else raw_checkpoint
    if any("prefix" in k and ".weight" in k and "parametrizations" not in k for k in state_dict.keys()):
        model = vanilla_export(model)

    model.load_state_dict(state_dict)
    model.to(device).eval()

    with torch.no_grad():
        preds = model(dataset).argmax(dim=1)
        clean_indices = (preds == labels).nonzero(as_tuple=False).squeeze()
        clean_acc = (len(clean_indices) / len(dataset)) * 100.0
    print(f"Clean Accuracy: {clean_acc:.2f}%")

    eps_rescaled = args.eps / 0.225 if ("cifar" in config['dataset'].lower() or "imagenette" in config['dataset'].lower()) else args.eps
    eps_backbone = eps_rescaled * np.sqrt(dataset[0].numel()) if str(args.norm) == 'inf' else eps_rescaled

    print(f"Computing Analytical Certificates...")
    global_cra, lln_cra = compute_analytical_cras(model, dataset, labels, eps_backbone, config, device)

    era, verification_candidates = None, clean_indices

    if not args.no_aa:
        print(f"Running AutoAttack (eps={args.eps}, L{args.norm})...")
        era, aa_time, aa_robust_indices = compute_autoattack_era_and_time(
            images=dataset, targets=labels, model=model, epsilon=args.eps,
            clean_indices=clean_indices, norm=args.norm, dataset_name=config['dataset'],
            return_robust_points=True
        )
        print(f"AutoAttack Finished | ERA: {era:.2f}%")
        if len(aa_robust_indices) == 0:
            print("[!] ERA is 0. Skipping formal verification.")
            sys.exit(0)
        verification_candidates = aa_robust_indices

    vra, exec_time, _ = compute_hybrid_vra_flexible(
        dataset, labels, model, eps_rescaled, verification_candidates, 
        device, classes, args, config
    )

    print(f"\n? Final Results:")
    print(f"  • VRA (Hybrid)     : {vra:.2f}%")
    
    # Save Results
    results_file = "verification_results_updated.csv"
    results_data = {
        "Model": os.path.basename(args.model_path), 
        "Dataset": config['dataset'],
        "Split_Idx": config['split_idx'],          
        "B1_Type": config['b1_type'], "B2_Type": config['b2_type'],
        "B1_Lip": config['b1_lip'], "B2_Lip": config['b2_lip'],    
        "Backend": args.hybrid_backend.upper(),
        "Epsilon": args.eps,
        "Clean_Acc": clean_acc,
        "ERA": era if era is not None else "",
        "Global_CRA": global_cra if global_cra is not None else "N/A",
        "LLN_CRA": lln_cra if lln_cra is not None else "N/A",
        "VRA": vra,
        "Time_s": exec_time
    }
    
    df = pd.DataFrame([results_data])
    df.to_csv(results_file, mode='a', header=not os.path.isfile(results_file), index=False)
    print(f" ?? Results appended to {results_file}")