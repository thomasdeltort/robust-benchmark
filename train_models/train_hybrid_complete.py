import os
import sys
import time
import copy
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrize import is_parametrized

# --- External Libraries ---
import schedulefree
from deel import torchlip
from deel.torchlip import TauCrossEntropyLoss

# --- Handle local module imports ---
sys.path.append('./..')
try:
    from project_utils import load_dataset
    from orthogonium.layers.conv.AOC import AdaptiveOrthoConv2d
    from orthogonium.reparametrizers import DEFAULT_ORTHO_PARAMS
except ImportError:
    print("Warning: Local modules not found. Ensure 'project_utils' and 'orthogonium' are in the parent directory.")

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

class CleanFlexibleHybridModel(nn.Module):
    def __init__(self, arch, dataset, split_idx, b1_type, b2_type, b1_lip, b2_lip, b1_act, b2_act, num_classes=10):
        super().__init__()
        
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
                else:
                    step_modules.append(nn.Conv2d(
                        current_in_conv, out_c, kernel_size=kernel, stride=stride, padding=pad
                    ))
                    
                step_modules.append(get_act(b_act_str))
                if b_type in ['aoc', 'torchlip'] and b_scale != 1.0:
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
                else: 
                    step_modules.append(nn.Linear(current_in_lin, out_features))
                    
                if not is_last:
                    step_modules.append(get_act(b_act_str))
                
                if b_type in ['aoc', 'torchlip'] and b_scale != 1.0:
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
    """Maps unconstrained weights into the latent parameters of constrained layers."""
    model_dict = model.state_dict()
    mapped_count = 0
    
    for name, module in model.named_modules():
        v_weight_key = f"{name}.weight"
        v_bias_key = f"{name}.bias"
        
        if v_weight_key in vanilla_state_dict:
            vanilla_w = vanilla_state_dict[v_weight_key]
            if hasattr(module, 'parametrizations') and 'weight' in module.parametrizations:
                target_key = f"{name}.parametrizations.weight.original"
                if target_key in model_dict:
                    model_dict[target_key].copy_(vanilla_w)
                    mapped_count += 1
            elif hasattr(module, 'weight_orig'):
                target_key = f"{name}.weight_orig"
                if target_key in model_dict:
                    model_dict[target_key].copy_(vanilla_w)
                    mapped_count += 1
            else:
                if v_weight_key in model_dict:
                    model_dict[v_weight_key].copy_(vanilla_w)
                    mapped_count += 1
                    
        if v_bias_key in vanilla_state_dict and v_bias_key in model_dict:
            model_dict[v_bias_key].copy_(vanilla_state_dict[v_bias_key])
                
    model.load_state_dict(model_dict, strict=False)
    print(f"[?] Successfully cross-mapped {mapped_count} weight tensors!")

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = load_dataset(args.dataset, args.batch_size, aug_level=args.aug_level)

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
            args.b1_type, args.b2_type, args.b1_lip, args.b2_lip, args.b1_act, args.b2_act
        ).to(device)

    # 2. Load Pretrained Weights Dynamically
    if args.pretrained_weights:
        print(f"\n[*] Loading pretrained weights from: {args.pretrained_weights}")
        checkpoint = torch.load(args.pretrained_weights, map_location=device)
        state_dict = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
        
        is_vanilla = any(k.endswith('.weight') for k in state_dict.keys()) and not any('parametrizations' in k for k in state_dict.keys())
        if is_vanilla:
            print("[i] Vanilla checkpoint detected. Adapting weights to target constraints...")
            load_cross_constraint_weights(model, state_dict)
        else:
            print("[i] Constrained checkpoint detected. Loading directly...")
            model.load_state_dict(state_dict, strict=False)

    # 3. Setup Separate Optimizers (DO NOT add a global 'opt')
    opt_prefix = schedulefree.AdamWScheduleFree(model.prefix.parameters(), lr=args.lr, weight_decay=0.0)
    opt_suffix = schedulefree.AdamWScheduleFree(model.suffix.parameters(), lr=args.lr, weight_decay=args.wd)

    crit_b1 = get_criterion(args.criterion, args, is_b2=False)
    crit_b2 = get_criterion(args.b2_criterion, args, is_b2=True)

    print(f"--- Starting {args.train_mode.upper()} Training on {device} ---")

    for epoch in range(args.epochs):
        model.train()
        opt.train()
        opt_prefix.train()
        opt_suffix.train()
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
                
                opt.zero_grad() # <-- Suffix Opt
                loss_acc.backward()
                opt.step()      # <-- Suffix Opt

                # --- PHASE 2: Update BACKBONE (Robustness) ---
                for p in model.prefix.parameters(): p.requires_grad = True
                for p in model.suffix.parameters(): p.requires_grad = False
                
                out_backbone = model(data)
                loss_rob = crit_b1(out_backbone, target if args.criterion == 'CE' else target_oh)
                
                opt.zero_grad() # <-- Prefix Opt
                loss_rob.backward()
                opt.step()

                m_acc += (out_backbone.argmax(1) == target).float().mean().item()

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
        
        # Determine if the entire network is Lipschitz bounded
        is_fully_constrained = (args.b1_type != 'unconstrained') and (args.b2_type != 'unconstrained')
        
        # Run validation only every 10 epochs or on the last epoch
        if (epoch + 1) % 10 == 0 or (epoch + 1) == args.epochs:
            model.eval()
            opt_prefix.eval()
            opt_suffix.eval()
            opt.eval()
            
            correct_clean = 0
            correct_robust = 0
            total = 0
            
            if is_fully_constrained:
                total_lip = args.b1_lip * args.b2_lip
                cert_threshold = (2 ** 0.5) * args.epsilon * total_lip

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
            
        print(f"Epoch {epoch+1:03d}/{args.epochs} | Train Acc: {m_acc/len(train_loader):.3f} | Val Clean Acc: {val_acc_str} | Val CRA (eps={args.epsilon}): {val_cra_str}")
    # Final Test
    model.eval()
    opt.eval()
    opt_prefix.eval()
    opt_suffix.eval()
    with torch.no_grad():
        test_acc = (torch.cat([model(d.to(device)).argmax(1).cpu() for d, _ in test_loader]) == torch.cat([t for _, t in test_loader])).float().mean().item()
    
    print(f"\nFINAL TEST ACCURACY: {test_acc:.3f}")
    
    if not args.no_save:
        os.makedirs(args.save_dir, exist_ok=True)
        
        # 1. Format the loss string depending on the training mode
        if args.train_mode == 'biphase':
            loss_str = f"{args.criterion}T{args.temperature}_{args.b2_criterion}T{args.b2_temperature}"
        else:
            loss_str = f"{args.criterion}T{args.temperature}"
            
        # 2. Create the highly detailed base filename
        base = f"Hyb_{args.train_mode}_{loss_str}_{args.b1_type}L{args.b1_lip}_{args.b2_type}L{args.b2_lip}_split{args.split_idx}_{int(time.time())}"
        
        lip_path = os.path.join(args.save_dir, f"{base}.pth")
        
        # Save a rich dictionary with both weights and hyperparameters
        checkpoint_data = {
            'state_dict': model.state_dict(),
            'config': vars(args)  # <--- This guarantees the eval script knows everything
        }
        
        torch.save(checkpoint_data, lip_path)
        print(f"Saved Parametrized Model: {lip_path}")
        
        # Save vanilla export similarly
        vanilla_data = {
            'state_dict': vanilla_export(model.cpu()).state_dict(),
            'config': vars(args)
        }
        vanilla_path = os.path.join(args.save_dir, f"vanilla_{base}.pth")
        torch.save(vanilla_data, vanilla_path)
        print(f"Saved Vanilla Export: {vanilla_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # --- Base Architecture ---
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'imagenette'])
    parser.add_argument('--arch', type=str, default='VGG13', choices=['VGG13', 'VGG16', 'ConvLarge'])
    parser.add_argument('--split_idx', type=int, default=4, help='Index to split Block 1 (Prefix) and Block 2 (Suffix)')
    
    # --- Block 1 (Prefix) ---
    parser.add_argument('--b1_type', type=str, default='aoc', choices=['aoc', 'torchlip', 'unconstrained'])
    parser.add_argument('--b1_lip', type=float, default=1.0)
    parser.add_argument('--b1_act', type=str, default='GroupSort', choices=['GroupSort', 'ReLU'])

    # --- Block 2 (Suffix) ---
    parser.add_argument('--b2_type', type=str, default='unconstrained', choices=['aoc', 'torchlip', 'unconstrained'])
    parser.add_argument('--b2_lip', type=float, default=1.0)
    parser.add_argument('--b2_act', type=str, default='ReLU', choices=['GroupSort', 'ReLU'])
    
    # --- Training Modes & Hyperparameters ---
    parser.add_argument('--train_mode', type=str, default='conventional', choices=['conventional', 'biphase'])
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

    # --- Pretrained & Custom Models ---
    parser.add_argument('--model_class', type=str, default=None)
    parser.add_argument('--pretrained_weights', type=str, default=None)

    main(parser.parse_args())