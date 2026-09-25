import os
import sys
import time
import json
import csv
import copy
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrize import is_parametrized
from torchinfo import summary
from torch.optim.lr_scheduler import MultiStepLR
import itertools
from contextlib import redirect_stdout

# --- External Libraries ---
import schedulefree
from deel import torchlip
from deel.torchlip import TauCrossEntropyLoss

# --- Handle local module imports ---
sys.path.append('./..')
try:
    from project_utils import load_dataset, load_dataset_benchmark_auto, wrap_with_identity, starts_with_affine, compute_sdp_crown_vra
    from orthogonium.layers.conv.AOC import AdaptiveOrthoConv2d
    from orthogonium.reparametrizers import DEFAULT_ORTHO_PARAMS
except ImportError:
    print("Warning: Local modules not found. Ensure 'project_utils' and 'orthogonium' are in the parent directory.")

# sys.path.insert(0, "/lustre/fswork/projects/rech/syo/utf64nw/robust-benchmark/SDP-CROWN/")
# sys.path.append("/home/thomas.deltort/SDP-CROWN")
# # import auto_LiRPA
# # from auto_LiRPA import BoundedModule, BoundedTensor
# from sdp_crown import compute_sdp_crown_vra
# from project_utils import load_dataset_benchmark_auto, wrap_with_identity, starts_with_affine
VRA_AVAILABLE = True


# ==========================================
# 1. Custom Layers & Losses
# ==========================================
class GroupSort_General(nn.Module):
    """Applies GroupSort specifically on the channel dimension."""
    def __init__(self, axis=1):
        super(GroupSort_General, self).__init__()
        self.axis = axis

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dims = list(range(x.dim()))
        channel_dim = dims.pop(self.axis) 
        dims.append(channel_dim)
        
        x_permuted = x.permute(dims).contiguous()
        batch_size = x_permuted.shape[0]
        
        reshaped_x = x_permuted.reshape(batch_size, -1, 2)
        x1s, x2s = reshaped_x[..., 0], reshaped_x[..., 1]
        
        relu_diff = F.relu(x2s - x1s)
        y1 = x2s - relu_diff
        y2 = x1s + relu_diff 
        
        sorted_flat = torch.stack((y1, y2), dim=2).reshape(batch_size, -1)
        output_permuted = sorted_flat.reshape(x_permuted.shape)
        
        inv_dims = list(range(x.dim()))
        inv_dims.insert(self.axis, inv_dims.pop(-1))
        return output_permuted.permute(inv_dims)

class LayerScale(nn.Module):
    """Scales output of a single block by c = L^(1/N) to distribute the Lipschitz constant."""
    def __init__(self, scale: float):
        super().__init__()
        self.scale = scale
        
    def forward(self, x):
        return x * self.scale

class HKRMultiLossLSE(nn.Module):
    """Hinge-Kantorovich-Rubinstein loss for robust Lipschitz training."""
    def __init__(self, alpha: float = 1., temperature: float = 1., penalty: float = 1., margin: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.penalty = penalty
        self.margin = margin

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_pred = y_pred * self.temperature
        pos = y_pred[y_true == 1]
        
        neg = torch.where(y_true == 1, -float('inf'), y_pred)
        t = torch.log(y_pred.new_tensor(y_pred.size(1) - 1)) / (self.margin * self.penalty)
        neg_soft = 1/t * torch.logsumexp(t * neg, dim=1)
        
        hinge_loss = torch.mean(F.relu(self.margin - pos)) + torch.mean(F.relu(self.margin + neg_soft))
        kr = torch.mean(pos) - torch.mean(neg_soft)
        
        return (1 - 1./self.alpha) * hinge_loss - (1./self.alpha) * kr

# ==========================================
# 2. Modular Hybrid Architecture Builder
# ==========================================
ARCH_CONFIGS = {
#    'VGG13': [
#        ('conv', 64, 1, 3, 1), ('conv', 64, 2, 3, 1),
#        ('conv', 128, 1, 3, 1), ('conv', 128, 2, 3, 1),
#        ('conv', 256, 1, 3, 1), ('conv', 256, 2, 3, 1),
#        ('conv', 512, 1, 3, 1), ('conv', 512, 2, 3, 1),
#        ('conv', 512, 1, 3, 1), ('conv', 512, 2, 3, 1),
#        ('linear', 512), ('linear', 512), ('linear', 10)
#    ],
#    'VGG13': [
#        ('conv', 64, 1, 3, 1), ('conv', 64, 1, 3, 1), ('conv', 64, 2, 3, 1),
#        ('conv', 128, 1, 3, 1), ('conv', 128, 1, 3, 1), ('conv', 128, 2, 3, 1),
#        ('conv', 256, 1, 3, 1), ('conv', 256, 1, 3, 1), ('conv', 256, 2, 3, 1),
#        ('linear', 512), ('linear', 512), ('linear', 10)
#    ],
##    'VGG16': [
#        ('conv', 64, 1, 3, 1), ('conv', 64, 2, 3, 1),
#        ('conv', 128, 1, 3, 1), ('conv', 128, 2, 3, 1),
#        ('conv', 256, 1, 3, 1), ('conv', 256, 1, 3, 1), ('conv', 256, 2, 3, 1),
#        ('conv', 512, 1, 3, 1), ('conv', 512, 1, 3, 1), ('conv', 512, 2, 3, 1),
#        ('conv', 512, 1, 3, 1), ('conv', 512, 1, 3, 1), ('conv', 512, 2, 3, 1),
#        ('linear', 512), ('linear', 512), ('linear', 10)
#    ],
    'VGG13_CIFAR': [
        # 3 Blocks (9 Convolutions) for 32x32 images
        ('conv', 64, 1, 3, 1), ('conv', 64, 1, 3, 1), ('conv', 64, 2, 3, 1),
        ('conv', 128, 1, 3, 1), ('conv', 128, 1, 3, 1), ('conv', 128, 2, 3, 1),
        ('conv', 256, 1, 3, 1), ('conv', 256, 1, 3, 1), ('conv', 256, 2, 3, 1),
        ('linear', 512), ('linear', 512), ('linear', 10)
    ],
    'VGG13_IMAGENETTE': [
        # 5 Blocks (10 Convolutions) matching VGG13_1_LIP_GNP_Imagenette
        ('conv', 64, 1, 3, 1), ('conv', 64, 2, 3, 1),
        ('conv', 128, 1, 3, 1), ('conv', 128, 2, 3, 1),
        ('conv', 256, 1, 3, 1), ('conv', 256, 2, 3, 1),
        ('conv', 512, 1, 3, 1), ('conv', 512, 2, 3, 1),
        ('conv', 512, 1, 3, 1), ('conv', 512, 2, 3, 1),
        ('linear', 512), ('linear', 512), ('linear', 10)
    ],
    'VGG16': [
        # Block 1 (64) - 3 Convs
        ('conv', 64, 1, 3, 1), ('conv', 64, 1, 3, 1), ('conv', 64, 2, 3, 1),
        # Block 2 (128) - 3 Convs
        ('conv', 128, 1, 3, 1), ('conv', 128, 1, 3, 1), ('conv', 128, 2, 3, 1),
        # Block 3 (256) - 4 Convs
        ('conv', 256, 1, 3, 1), ('conv', 256, 1, 3, 1), ('conv', 256, 1, 3, 1), ('conv', 256, 2, 3, 1),
        # Block 4 (512) - 4 Convs
        ('conv', 512, 1, 3, 1), ('conv', 512, 1, 3, 1), ('conv', 512, 1, 3, 1), ('conv', 512, 2, 3, 1),
        # Block 5 (512) - 4 Convs
        ('conv', 512, 1, 3, 1), ('conv', 512, 1, 3, 1), ('conv', 512, 1, 3, 1), ('conv', 512, 2, 3, 1),
        # Classifier - 3 Linears
        ('linear', 512), ('linear', 512), ('linear', 10)
    ],
    'ConvLarge': [
        ('conv', 32, 1, 3, 1), ('conv', 32, 2, 4, 1),
        ('conv', 64, 1, 3, 1), ('conv', 64, 2, 4, 1),
        ('linear', 512), ('linear', 512), ('linear', 10)
    ]
}

class CleanFlexibleHybridModel(nn.Module):
    def __init__(self, arch, dataset, split_idx, b1_type, b2_type, b1_lip, b2_lip, b1_act, b2_act, num_classes=10, use_bn=False):
        super().__init__()
        
        if arch == 'VGG13':
            if dataset.lower() in ['imagenette', 'imagenet']:
                config = ARCH_CONFIGS['VGG13_IMAGENETTE']
            else:
                config = ARCH_CONFIGS['VGG13_CIFAR']
        else:
            config = ARCH_CONFIGS[arch]
            
        in_channels = 3
        input_size = 224 if dataset.lower() in ['imagenette', 'imagenet'] else 32

        b1_count = min(split_idx, len(config))
        b2_count = max(0, len(config) - b1_count)
        
        b1_scale = (b1_lip ** (1.0 / b1_count)) if b1_count > 0 else 1.0
        b2_scale = (b2_lip ** (1.0 / b2_count)) if b2_count > 0 else 1.0

        prefix_layers = []
        suffix_layers = []
        
        current_in_conv = in_channels
        curr_spatial = input_size
        flattened = False
        current_in_lin = 0

        def get_act(act_str):
            return nn.ReLU() if act_str.lower() == 'relu' else GroupSort_General()

        for idx, layer_cfg in enumerate(config):
            is_b1 = idx < split_idx
            is_last = idx == (len(config) - 1)
            
            b_type = b1_type.lower() if is_b1 else b2_type.lower()
            b_act_str = b1_act if is_b1 else b2_act
            b_scale = b1_scale if is_b1 else b2_scale
            
            layer_type = layer_cfg[0]
            step_modules = []

            if layer_type == 'conv':
                _, out_c, stride, kernel, pad = layer_cfg
                
                if b_type == 'aoc':
                    step_modules.append(AdaptiveOrthoConv2d(
                        current_in_conv, out_c, kernel_size=kernel, stride=stride, 
                        padding=pad, padding_mode='zeros', ortho_params=DEFAULT_ORTHO_PARAMS
                    ))
                elif b_type == 'torchlip':
                    step_modules.append(torchlip.SpectralConv2d(
                        current_in_conv, out_c, kernel_size=kernel, stride=stride, padding=pad
                    ))
                elif b_type == 'spectral':
                    step_modules.append(torchlip.SpectralConv2d(
                        current_in_conv, out_c, kernel_size=kernel, stride=stride, padding=pad, eps_bjorck=None
                    ))
                else:
                    step_modules.append(nn.Conv2d(
                        current_in_conv, out_c, kernel_size=kernel, stride=stride, padding=pad
                    ))
                    
                step_modules.append(get_act(b_act_str))
                
                if b_type in ['aoc', 'torchlip', 'spectral'] and b_scale != 1.0:
                    step_modules.append(LayerScale(b_scale))

                current_in_conv = out_c
                curr_spatial = (curr_spatial + 2 * pad - kernel) // stride + 1

            elif layer_type == 'linear':
                if not flattened:
                    step_modules.append(nn.Flatten())
                    current_in_lin = current_in_conv * curr_spatial * curr_spatial
                    flattened = True
                
                out_features = num_classes if is_last else layer_cfg[1]
                
                if b_type in ['aoc', 'torchlip']:
                    step_modules.append(torchlip.SpectralLinear(current_in_lin, out_features))
                elif b_type == 'spectral':
                    step_modules.append(torchlip.SpectralLinear(current_in_lin, out_features, eps_bjorck=None))
                else: 
                    # 1. Determine if we need a bias
                    layer_bias = not (use_bn and not is_last)
                    
                    # 2. Create the linear layer
                    lin_layer = nn.Linear(current_in_lin, out_features, bias=layer_bias)
                    
                    # 3. Initialize weights to exactly 1-Lipschitz (Orthogonal)
                    torch.nn.init.orthogonal_(lin_layer.weight)
                    
                    # If the layer has a bias, initialize it to zeros (standard practice)
                    if lin_layer.bias is not None:
                        torch.nn.init.zeros_(lin_layer.bias)
                        
                    # 4. Append the layer
                    step_modules.append(lin_layer)
                    
                    # 5. Append BatchNorm with affine=False
                    if not is_last and use_bn:
                        step_modules.append(torch.nn.BatchNorm1d(out_features, affine=True))


                    
                if not is_last:
                    step_modules.append(get_act(b_act_str))
                
                if b_type in ['aoc', 'torchlip', 'spectral'] and b_scale != 1.0:
                    step_modules.append(LayerScale(b_scale))
                    
                current_in_lin = out_features

            if is_b1:
                prefix_layers.extend(step_modules)
            else:
                suffix_layers.extend(step_modules)

        self.prefix = nn.Sequential(*prefix_layers)
        self.suffix = nn.Sequential(*suffix_layers)

        print(f"\n{'='*70}\n??  MODEL SPLIT FLAG: Cut made at layer index {split_idx}")
        print(f"  [PREFIX] Layers 0 to {max(0, b1_count-1)} | Type: {b1_type.upper()} | Lip: {b1_lip} | Act: {b1_act}")
        print(f"  [SUFFIX] Layers {split_idx} to {len(config)-1} | Type: {b2_type.upper()} | Lip: {b2_lip} | Act: {b2_act}")
        print(f"{'='*70}\n")

    def forward(self, x):
        return self.suffix(self.prefix(x))

def vanilla_export(model1):
    """Exports a parametrized model to a standard PyTorch model for fast inference."""
    model1.eval()
    model2 = copy.deepcopy(model1)
    model2.eval()
    
    dict_modified_layers = {}
    for (n1, p1), (n2, p2) in zip(model1.named_modules(), model2.named_modules()):
        assert n1 == n2
        if isinstance(p1, nn.Conv2d) and is_parametrized(p1):
            new_conv = nn.Conv2d(
                p1.in_channels, p1.out_channels, 
                kernel_size=p1.kernel_size, stride=p1.stride, 
                padding=p1.padding, padding_mode=p1.padding_mode, bias=(p1.bias is not None)
            )
            new_conv.weight.data = p1.weight.data.clone()
            if p1.bias is not None: new_conv.bias.data = p1.bias.data.clone()
            dict_modified_layers[n2] = new_conv
            
        elif isinstance(p1, nn.Linear) and is_parametrized(p1):
            new_lin = nn.Linear(
                p1.in_features, p1.out_features, bias=(p1.bias is not None)
            )
            new_lin.weight.data = p1.weight.data.clone()
            if p1.bias is not None: new_lin.bias.data = p1.bias.data.clone()
            dict_modified_layers[n2] = new_lin
            
    for n2, new_layer in dict_modified_layers.items():
        split_hierarchy = n2.split('.')
        lay = model2
        for h in split_hierarchy[:-1]:
            lay = getattr(lay, h)
        setattr(lay, split_hierarchy[-1], new_layer)
        
    return model2

# ==========================================
# 3. Main Training & Bi-phase Logic
# ==========================================
def get_criterion(name, args, is_b2=False):
    """Helper to initialize the requested loss function."""
    temp = args.b2_temperature if is_b2 else args.temperature
    if name == 'tau': return TauCrossEntropyLoss(temp)
    if name == 'HKR': return HKRMultiLossLSE(args.alpha, temp)
    return nn.CrossEntropyLoss()

def load_cross_constraint_weights(model, vanilla_state_dict):
    """Maps unconstrained weights into the latent parameters based on layer order."""
    model_dict = model.state_dict()
    mapped_count = 0
    
    # Extract all weight keys from the checkpoint in sequential order
    vanilla_weights = [k for k in vanilla_state_dict.keys() if k.endswith('.weight')]
    
    # Get all weight-bearing modules in the new model in sequential order
    target_modules = [(name, m) for name, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    
    for i, (name, module) in enumerate(target_modules):
        if i >= len(vanilla_weights):
            break
            
        v_w_key = vanilla_weights[i]
        v_b_key = v_w_key.replace('.weight', '.bias')
        vanilla_w = vanilla_state_dict[v_w_key]
        
        # 1. Map Weights
        if hasattr(module, 'parametrizations') and 'weight' in module.parametrizations:
            target_key = f"{name}.parametrizations.weight.original"
        elif hasattr(module, 'weight_orig'):
            target_key = f"{name}.weight_orig"
        else:
            target_key = f"{name}.weight"
            
        if target_key in model_dict:
            # Check shape safety
            if model_dict[target_key].shape == vanilla_w.shape:
                model_dict[target_key].copy_(vanilla_w)
                mapped_count += 1
            else:
                print(f"[!] Shape mismatch at {name}: Expected {model_dict[target_key].shape}, got {vanilla_w.shape}")

        # 2. Map Biases (if they exist)
        target_b_key = f"{name}.bias"
        if v_b_key in vanilla_state_dict and target_b_key in model_dict:
            model_dict[target_b_key].copy_(vanilla_state_dict[v_b_key])
            
    model.load_state_dict(model_dict, strict=False)
    print(f"[?] Successfully cross-mapped {mapped_count} weight tensors!")

def replace_bn_with_linear(module):
    """Fuses BatchNorm1d directly into the preceding nn.Linear layer for safe verification."""
    if isinstance(module, nn.Sequential):
        layers = list(module.children())
        new_layers = []
        
        i = 0
        while i < len(layers):
            layer = layers[i]
            
            # If we see a Linear followed immediately by a BatchNorm1d
            if isinstance(layer, nn.Linear) and (i + 1 < len(layers)) and isinstance(layers[i+1], nn.BatchNorm1d):
                bn = layers[i+1]
                
                # 1. Extract BN parameters
                gamma = bn.weight if bn.weight is not None else torch.ones_like(bn.running_var)
                beta = bn.bias if bn.bias is not None else torch.zeros_like(bn.running_mean)
                w = gamma / torch.sqrt(bn.running_var + bn.eps)
                b = beta - bn.running_mean * w
                
                # 2. Create the fused Linear layer on the same device
                fused_linear = nn.Linear(layer.in_features, layer.out_features, bias=True).to(layer.weight.device)
                
                # Multiply the weights (Scale each row by w)
                fused_linear.weight.data = layer.weight.data * w.unsqueeze(1)
                
                # Multiply the bias and add the BN shift
                if layer.bias is not None:
                    fused_linear.bias.data = layer.bias.data * w + b
                else:
                    fused_linear.bias.data = b
                    
                new_layers.append(fused_linear)
                i += 2  # Skip the BN layer
            else:
                new_layers.append(layer)
                i += 1
                
        return nn.Sequential(*new_layers)
    return module

def run_benchmark_eval(model, images, labels, eps, b1_lip, args, device):
    # 1. Export to Vanilla (Required for auto_LiRPA and accurate SN)
    v_model = vanilla_export(model).to(device)
    v_model.eval()
    
    N = images.shape[0]
    
    # 2. Get Clean Indices on Benchmark
    with torch.no_grad():
        preds = v_model(images).argmax(dim=1)
        clean_indices = (preds == labels).nonzero(as_tuple=False).squeeze()
        bench_clean_acc = (len(clean_indices) / N) * 100.0
        
    if len(clean_indices.shape) == 0: clean_indices = clean_indices.unsqueeze(0)

    # 3. Compute Global CRA
    l_head = 1.0
    for m in v_model.suffix.modules():
        if isinstance(m, nn.Linear):
            l_head *= torch.linalg.matrix_norm(m.weight, 2).item()
        elif isinstance(m, nn.BatchNorm1d):
            # Max(|gamma| / sqrt(running_var + eps))
            bn_lip = torch.max(torch.abs(m.weight) / torch.sqrt(m.running_var + m.eps)).item()
            l_head *= bn_lip
            
    l_global = b1_lip * l_head
    print(l_global)
    cert_threshold = (2 ** 0.5) * eps * l_global

    with torch.no_grad():
        logits = v_model(images)
        logits_true = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
        logits_clone = logits.clone()
        logits_clone.scatter_(1, labels.unsqueeze(1), -float('inf'))
        logits_next = logits_clone.max(dim=1)[0]
        margins = logits_true - logits_next
        
        is_robust = (preds == labels) & (margins > cert_threshold)
        bench_cra = (is_robust.sum().item() / N) * 100.0

    # 4. Compute Hybrid VRA (SDP-CROWN)
    bench_vra = 0.0
    if VRA_AVAILABLE and len(clean_indices) > 0:
        with torch.no_grad():
            z_k = v_model.prefix(images)
            
        f2_suffix_ready = wrap_with_identity(v_model.suffix, z_k) if not starts_with_affine(v_model.suffix) else v_model.suffix
        has_groupsort = any(isinstance(m, GroupSort_General) for m in f2_suffix_ready.modules())
        intermediate_eps = float(eps * b1_lip)
        
        # bench_vra, _ = compute_sdp_crown_vra(
        #         z_k, labels, f2_suffix_ready, intermediate_eps, clean_indices, 
        #         device, 10, args, batch_size=args.batch_size, 
        #         return_robust_points=False, x_U=None, x_L=None, groupsort=has_groupsort)

        # Mute all print statements coming from the SDP-CROWN verifier
        if args.use_bn:
          f2_suffix_ready = replace_bn_with_linear(f2_suffix_ready)

        with open(os.devnull, 'w') as fnull, redirect_stdout(fnull):
            bench_vra, _ = compute_sdp_crown_vra(
                z_k, labels, f2_suffix_ready, intermediate_eps, clean_indices, 
                device, 10, args, batch_size=args.batch_size, 
                return_robust_points=False, x_U=None, x_L=None, groupsort=has_groupsort
            )
    return bench_clean_acc, bench_cra, bench_vra

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not hasattr(args, 'model'):
        args.model = f"{args.dataset}_{args.arch}"
    if not hasattr(args, 'norm'):
        args.norm = 2  # Default to L2 norm for your benchmark
    if not hasattr(args, 'high_tau'):
        args.high_tau = False


    train_loader, val_loader, test_loader = load_dataset(args.dataset, args.batch_size, aug_level=args.aug_level)

    if args.eval_benchmark:
        print("[*] Loading 200-point verification benchmark...")
        bench_dataset, bench_labels, bench_classes = load_dataset_benchmark_auto(args)
        bench_dataset, bench_labels = bench_dataset.to(device), bench_labels.to(device)

    # 1. Choose Model Architecture
    if args.model_class:
        try:
            ModelClass = globals()[args.model_class]
            model = ModelClass().to(device)
            print(f"\n[*] Using custom architecture from models.py: {args.model_class}")
        except KeyError:
            raise ValueError(f"Model class '{args.model_class}' not found in models.py!")
    else:
        model = CleanFlexibleHybridModel(
            args.arch, args.dataset, args.split_idx,
            args.b1_type, args.b2_type, args.b1_lip, args.b2_lip, args.b1_act, args.b2_act,
        use_bn=args.use_bn).to(device)

    model.eval()
#    summary(model, input_size=(1, 3, 32, 32))


    # 2. Load Pretrained Weights Dynamically
    if args.pretrained_weights:
        # --- NEW: Strict ReLU Pretraining Ban ---
        if args.b1_act.lower() == 'relu' or args.b2_act.lower() == 'relu':
            print(f"\n[!] SAFETY TRIGGER: ReLU detected in architecture (B1: {args.b1_act}, B2: {args.b2_act}).")
            print("    -> IGNORING pretrained weights and initializing from scratch!")
            args.pretrained_weights = None
        # ----------------------------------------
        else:
            print(f"\n[*] Loading pretrained weights from: {args.pretrained_weights}")
            checkpoint = torch.load(args.pretrained_weights, map_location=device)
            # state_dict = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
            # --- ROBUST WEIGHT EXTRACTION ---
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint: state_dict = checkpoint['state_dict']
                elif 'net' in checkpoint: state_dict = checkpoint['net']
                elif 'model' in checkpoint: state_dict = checkpoint['model']
                elif 'model_state_dict' in checkpoint: state_dict = checkpoint['model_state_dict']
                else: state_dict = checkpoint
            else:
                state_dict = checkpoint
            # --------------------------------

            is_vanilla = any(k.endswith('.weight') for k in state_dict.keys()) and not any('parametrizations' in k for k in state_dict.keys())
            if is_vanilla:
                print("[i] Vanilla checkpoint detected. Adapting weights to target constraints...")
                load_cross_constraint_weights(model, state_dict)
            else:
                print("[i] Constrained checkpoint detected. Remapping keys to hybrid structure...")
                new_state_dict = {}
                model_keys = list(model.state_dict().keys())
                ckpt_keys = list(state_dict.keys())
                
                # If the old checkpoint doesn't use prefix/suffix, map them sequentially by index
                if not any("prefix" in k for k in ckpt_keys):
                    if len(model_keys) == len(ckpt_keys):
                        for mk, ck in zip(model_keys, ckpt_keys):
                            new_state_dict[mk] = state_dict[ck]
                    else:
                        print(f"[!] Warning: Key count mismatch (Model: {len(model_keys)}, Ckpt: {len(ckpt_keys)}).")
                        new_state_dict = state_dict # Fallback
                else:
                    new_state_dict = state_dict
                    
                missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
                print(f"[?] Loaded constrained weights. Missing keys: {len(missing)}")

    # 3. Setup Optimizers based on training mode
    if args.train_mode == 'biphase':
        # opt_prefix = schedulefree.AdamWScheduleFree(
        #     model.prefix.parameters(), 
        #     lr=args.lr, 
        #     weight_decay=0.0,
        #     betas=(0.0001, 0.999) # <-- 0.0 disables momentum (first moment)
        # )
        opt_prefix = schedulefree.AdamWScheduleFree(model.prefix.parameters(), lr=args.lr, weight_decay=0.0)
        b2_wd = args.wd if args.b2_type == 'unconstrained' else 0.0
        opt_suffix = schedulefree.AdamWScheduleFree(model.suffix.parameters(), lr=args.lr, weight_decay=b2_wd)
        # opt_suffix = schedulefree.AdamWScheduleFree(
        #     model.suffix.parameters(), 
        #     lr=args.lr, 
        #     weight_decay=0.0,
        #     betas=(0.0001, 0.999) # <-- 0.0 disables momentum (first moment)
        # )
    elif args.train_mode == 'head_only':
        print("[i] HEAD ONLY mode: Freezing prefix permanently.")
        for p in model.prefix.parameters(): p.requires_grad = False
        opt = schedulefree.AdamWScheduleFree(model.suffix.parameters(), lr=args.lr, weight_decay=args.wd)
        # opt = schedulefree.AdamWScheduleFree(
        #     model.suffix.parameters(), 
        #     lr=args.lr, 
        #     weight_decay=0.0,
        #     betas=(0.0001, 0.999) # <-- 0.0 disables momentum (first moment)
        # )
    else:
        if args.b2_type == 'unconstrained':
            print("[i] Unconstrained Head detected: Applying weight decay ONLY to the suffix.")
            param_groups = [
                {'params': model.prefix.parameters(), 'weight_decay': 0.0, 'lr':args.lr},
                {'params': model.suffix.parameters(), 'weight_decay': args.wd, 'lr':args.lr}
            ]
        else:
            print("[i] Fully constrained model: Disabling weight decay everywhere.")
            param_groups = [
                {'params': model.parameters(), 'weight_decay': 0.0, 'lr':args.lr}
            ]
        # opt = torch.optim.AdamW(param_groups)
        # scheduler = MultiStepLR(opt, milestones=[30, 45], gamma=0.1)
        opt = schedulefree.AdamWScheduleFree(param_groups)

    crit_b1 = get_criterion(args.criterion, args, is_b2=False)
    crit_b2 = get_criterion(args.b2_criterion, args, is_b2=True)

    print(f"--- Starting {args.train_mode.upper()} Training on {device} ---")

  # --- 1. SETUP RUN DIRECTORY ---
    if not args.no_save:
        # Combine dataset and architecture so they don't overwrite/mix!
        model_name = args.model_class if args.model_class else f"{args.dataset}_{args.arch}"
        
        # Create the base save_dir (defaults to './saved_models')
        os.makedirs(args.save_dir, exist_ok=True)
        
        # Create a detailed name for this specific run
        run_name = f"Hyb_{args.train_mode}_{args.b1_type}L{args.b1_lip}_{args.b2_type}L{args.b2_lip}_split{args.split_idx}_{int(time.time())}"
        
        # Nest it: saved_models / VGG13 / run_name /
        run_dir = os.path.join(args.save_dir, model_name, run_name)
        os.makedirs(run_dir, exist_ok=True)
        
        # Save Hyperparameters to JSON
        with open(os.path.join(run_dir, 'config.json'), 'w') as f:
            json.dump(vars(args), f, indent=4)
            
        # Initialize Metrics CSV
        metrics_file = os.path.join(run_dir, 'metrics.csv')
        with open(metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Train_Acc', 'Val_Clean', 'Val_CRA', 'Bench_Clean', 'Bench_CRA', 'Bench_VRA'])
            
        print(f"[*] Logging run data to: {run_dir}")
    # ------------------------------

    if "cifar" in args.dataset.lower() or "imagenette" in args.dataset.lower():
        eps_rescaled = args.epsilon / 0.225
    else:
        eps_rescaled = args.epsilon

    for epoch in range(args.epochs):
        # print("Starting training")
        model.train()
        if args.train_mode == 'biphase':
            opt_prefix.train()
            opt_suffix.train()
        else:
            opt.train()
        m_acc = 0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            target_oh = F.one_hot(target, 10).float()
            
            if args.train_mode == 'biphase':
                # --- PHASE 1: Update HEAD (Accuracy) ---
                for p in model.prefix.parameters(): p.requires_grad = False
                for p in model.suffix.parameters(): p.requires_grad = True
                
                with torch.no_grad():
                    features = model.prefix(data)
                
                out_head = model.suffix(features)
                loss_acc = crit_b2(out_head, target if args.b2_criterion == 'CE' else target_oh)
                
                opt_suffix.zero_grad() # <-- Suffix Opt
                loss_acc.backward()
                opt_suffix.step()      # <-- Suffix Opt

                # --- PHASE 2: Update BACKBONE (Robustness) ---
                for p in model.prefix.parameters(): p.requires_grad = True
                for p in model.suffix.parameters(): p.requires_grad = False
                
                out_backbone = model(data)
                loss_rob = crit_b1(out_backbone, target if args.criterion == 'CE' else target_oh)
                
                opt_prefix.zero_grad() # <-- Prefix Opt
                loss_rob.backward()
                opt_prefix.step()

                m_acc += (out_head.argmax(1) == target).float().mean().item()

            elif args.train_mode == 'head_only':
                # --- TRANSFER LEARNING (Train Suffix Only) ---
                opt.zero_grad()
                
                # Pass through frozen prefix without tracking gradients (saves VRAM & time)
                with torch.no_grad():
                    features = model.prefix(data)
                
                # Train only the suffix
                out = model.suffix(features)
                loss = crit_b2(out, target if args.b2_criterion == 'CE' else target_oh)
                
                loss.backward()
                opt.step()
                
                m_acc += (out.argmax(1) == target).float().mean().item()

            else:
                # --- CONVENTIONAL TRAINING (Joint Update) ---
                for p in model.parameters(): p.requires_grad = True
                
                opt.zero_grad()
                out = model(data)
                loss = crit_b1(out, target if args.criterion == 'CE' else target_oh)
                loss.backward()
                opt.step()
                
                m_acc += (out.argmax(1) == target).float().mean().item()

        # ==========================================
        # Validation & Logging Step
        # ==========================================
        val_acc_str = "---"
        val_cra_str = "---"
        val_acc_str = "---"
        val_cra_str = "---"
        bench_clean_str = "---"
        bench_cra_str = "---"
        bench_vra_str = "---"
        
        # Determine if the entire network is Lipschitz bounded
        is_fully_constrained = (args.b1_type != 'unconstrained') and (args.b2_type != 'unconstrained')
        
        # Run validation only every 10 epochs or on the last epoch
        if (epoch + 1) % 15 == 1 or (epoch + 1) == args.epochs:
            # print("Starting Validation :")
            model.eval()
            if args.train_mode == 'biphase':
                opt_prefix.eval()
                opt_suffix.eval()
            else:
                opt.eval()

            # 2. ScheduleFree BatchNorm Sync (ONLY if BN is used)
            if getattr(args, 'use_bn', False):
                model.train()  # Model must be in train mode to update BN running stats
                with torch.no_grad():
                    # Run 50 batches from the training loader
                    for d, _ in itertools.islice(train_loader, 50):
                        model(d.to(device))

            # 3. Switch model to eval mode for the actual validation metrics
            model.eval()

            correct_clean = 0
            correct_robust = 0
            total = 0
            
            if is_fully_constrained:
                total_lip = args.b1_lip * args.b2_lip

                cert_threshold = (2 ** 0.5) * eps_rescaled * total_lip

            with torch.no_grad():
                for d, t in val_loader:
                    d, t = d.to(device), t.to(device)
                    logits = model(d)
                    
                    # 1. Clean Accuracy (Always Computed)
                    preds = logits.argmax(dim=1)
                    is_correct = (preds == t)
                    correct_clean += is_correct.sum().item()
                    
                    # 2. Certified Robust Accuracy (Computed ONLY if valid)
                    if is_fully_constrained:
                        logits_true = logits.gather(1, t.unsqueeze(1)).squeeze(1)
                        logits_clone = logits.clone()
                        logits_clone.scatter_(1, t.unsqueeze(1), -float('inf'))
                        logits_next = logits_clone.max(dim=1)[0]
                        margin = logits_true - logits_next
                        
                        is_robust = is_correct & (margin > cert_threshold)
                        correct_robust += is_robust.sum().item()
                        
                    total += t.size(0)
                    
            val_acc_str = f"{correct_clean / total:.3f}"
            val_cra_str = f"{correct_robust / total:.3f}" if is_fully_constrained else "N/A"

            # bench_clean_str = "---"
            # bench_cra_str = "---"
            # bench_vra_str = "---"
            
            # Run benchmark evaluation if flag is passed
            if getattr(args, 'eval_benchmark', False):
                print(f"\n[!] Running Formal Verification on 200 Benchmark Points...")
                b_clean, b_cra, b_vra = run_benchmark_eval(
                    model, bench_dataset, bench_labels, eps_rescaled, args.b1_lip, args, device
                )
                bench_clean_str = f"{b_clean:.2f}%"
                bench_cra_str = f"{b_cra:.2f}%"
                bench_vra_str = f"{b_vra:.2f}%"
            
            # ==========================================
            # Save Checkpoint with Epoch Stamp
            # ==========================================
            if not args.no_save:
                # Use run_dir instead of args.save_dir, and keep filenames clean!
                lip_path = os.path.join(run_dir, f"model_ep{epoch+1:03d}.pth")
                vanilla_path = os.path.join(run_dir, f"vanilla_ep{epoch+1:03d}.pth")
                
                # 1. Save standard checkpoint
                checkpoint_data = {'state_dict': model.state_dict(), 'config': vars(args)}
                torch.save(checkpoint_data, lip_path)
                
                # 2. Save Vanilla export (Safely deepcopied to CPU!)
                model_copy = copy.deepcopy(model).cpu()
                vanilla_data = {'state_dict': vanilla_export(model_copy).state_dict(), 'config': vars(args)}
                torch.save(vanilla_data, vanilla_path)
                
                print(f"[i] Saved Checkpoints for Epoch {epoch+1:03d}")
                
        print(f"Epoch {epoch+1:03d}/{args.epochs} | Train Acc: {m_acc/len(train_loader):.3f} | Val Clean Acc: {val_acc_str}")
        if getattr(args, 'eval_benchmark', False):
            print(f" ? Benchmark (200 pts) | Clean: {bench_clean_str} | CRA: {bench_cra_str} | VRA: {bench_vra_str}")
        # --- 2. APPEND METRICS TO CSV ---
        if not args.no_save:
            with open(metrics_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                        epoch + 1, 
                        f"{m_acc/len(train_loader):.3f}", 
                        val_acc_str, 
                        val_cra_str, 
                        bench_clean_str if 'bench_clean_str' in locals() else "N/A", 
                        bench_cra_str if 'bench_cra_str' in locals() else "N/A", 
                        bench_vra_str if 'bench_vra_str' in locals() else "N/A"
                ])
           
    # Final Test
    model.eval()
    if args.train_mode == 'biphase':
        opt_prefix.eval()
        opt_suffix.eval()
    else:
        opt.eval()
    with torch.no_grad():
        test_acc = (torch.cat([model(d.to(device)).argmax(1).cpu() for d, _ in test_loader]) == torch.cat([t for _, t in test_loader])).float().mean().item()
    
    print(f"\nFINAL TEST ACCURACY: {test_acc:.3f}")
    
    # ==========================================
    # Final Model & Metadata Saving
    # ==========================================
    if not args.no_save:
        # 1. Compute the Final Unconstrained Head & Global Lipschitz constants
        l_head = 1.0
        for m in model.suffix.modules():
            if isinstance(m, nn.Linear):
                # We can compute this directly on the un-exported weight
                weight = m.weight_orig if hasattr(m, 'weight_orig') else m.weight
                l_head *= torch.linalg.matrix_norm(weight, 2).item()
            elif isinstance(m, nn.BatchNorm1d):
                bn_lip = torch.max(torch.abs(m.weight) / torch.sqrt(m.running_var + m.eps)).item()
                l_head *= bn_lip
                
        final_l_global = args.b1_lip * l_head
        
        # 2. Attach these computed metrics to the config
        args.final_l_head = l_head
        args.final_l_global = final_l_global
        
        print(f"\n[i] Final Head Lipschitz: {l_head:.4f}")
        print(f"[i] Final Global Network Lipschitz: {final_l_global:.4f}")

        # 3. Overwrite the config.json to include these new values
        with open(os.path.join(run_dir, 'config.json'), 'w') as f:
            json.dump(vars(args), f, indent=4)

        # 4. Save the PyTorch Checkpoints
        lip_path = os.path.join(run_dir, "model_final.pth")
        checkpoint_data = {
            'state_dict': model.state_dict(),
            'config': vars(args)  # <--- Now includes the final Lipschitz values!
        }
        torch.save(checkpoint_data, lip_path)
        print(f"Saved Parametrized Model: {lip_path}")
        
        # Save vanilla export similarly
        vanilla_data = {
            'state_dict': vanilla_export(model.cpu()).state_dict(),
            'config': vars(args)
        }
        vanilla_path = os.path.join(run_dir, "vanilla_final.pth")
        torch.save(vanilla_data, vanilla_path)
        print(f"Saved Vanilla Export: {vanilla_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # --- Base Architecture ---
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'imagenette'])
    parser.add_argument('--arch', type=str, default='VGG13', choices=['VGG13', 'VGG16', 'ConvLarge'])
    parser.add_argument('--split_idx', type=int, default=4, help='Index to split Block 1 (Prefix) and Block 2 (Suffix)')
    
    # --- Block 1 (Prefix) ---
    parser.add_argument('--b1_type', type=str, default='aoc', choices=['aoc', 'torchlip', 'spectral', 'unconstrained'])
    parser.add_argument('--b1_lip', type=float, default=1.0)
    parser.add_argument('--b1_act', type=str, default='GroupSort', choices=['GroupSort', 'ReLU'])

    # --- Block 2 (Suffix) ---
    parser.add_argument('--b2_type', type=str, default='unconstrained', choices=['aoc', 'torchlip', 'spectral', 'unconstrained'])
    parser.add_argument('--b2_lip', type=float, default=1.0)
    parser.add_argument('--b2_act', type=str, default='ReLU', choices=['GroupSort', 'ReLU'])
    parser.add_argument('--use_bn', action='store_true', help='Enable BatchNorm1d in the unconstrained linear layers.')
    
    # --- Training Modes & Hyperparameters ---
    parser.add_argument('--train_mode', type=str, default='conventional', choices=['conventional', 'biphase', 'head_only'])
    parser.add_argument('--criterion', type=str, default='tau', choices=['tau', 'HKR', 'CE'], help="Main loss (B1 Robustness)")
    parser.add_argument('--b2_criterion', type=str, default='CE', choices=['tau', 'HKR', 'CE'], help="Head loss (B2 Accuracy, biphase only)")
    parser.add_argument('--temperature', type=float, default=1.0, help="Temperature for B1")
    parser.add_argument('--b2_temperature', type=float, default=10.0, help="Temperature for B2")
    parser.add_argument('--alpha', type=float, default=250.0)
    
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--wd', type=float, default=0.0)
    parser.add_argument('--aug_level', type=str, default='heavy')

    parser.add_argument('--epsilon', type=float, default=0.03137)
    
    # --- I/O ---
    parser.add_argument('--no_save', action='store_true', help='Disable model saving.')
    parser.add_argument('--save_dir', type=str, default='./saved_models')

    parser.add_argument('--lr_alpha', type=float, default=0.5)
    parser.add_argument('--lr_lambda', type=float, default=0.05)
    parser.add_argument('--high_tau', action='store_true', help='Use high tau for SDP-CROWN')

    # --- Pretrained & Custom Models ---
    parser.add_argument('--model_class', type=str, default=None)
    parser.add_argument('--pretrained_weights', type=str, default=None)
    parser.add_argument('--eval_benchmark', action='store_true', help='Run CRA and VRA on the 200-point benchmark during validation')

    main(parser.parse_args())