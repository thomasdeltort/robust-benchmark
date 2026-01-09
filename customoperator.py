import torch
import torch.nn as nn
import sys
import os

# =============================================================================
# SETUP PATHS
# =============================================================================
SDP_CROWN_PATH = '/home/aws_install/robustess_project/SDP-CROWN'
sys.path.insert(0, SDP_CROWN_PATH)

from auto_LiRPA import BoundedModule, BoundedTensor, register_custom_op
from auto_LiRPA.operators import BoundTwoPieceLinear
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.patches import Patches

# =============================================================================
# 1. MODELS DEFINITION
# =============================================================================

# class GroupSortOp(torch.autograd.Function):
#     """ Custom Autograd Function for the Custom Operator Version """
#     @staticmethod
#     def symbolic(g, x, axis):
#         return g.op('onnx::GroupSortGeneral', x, axis_i=axis).setType(x.type())

#     @staticmethod
#     def forward(ctx, x, axis):
#         ctx.axis = axis
#         with torch.no_grad():
#             dims = list(range(x.dim()))
#             c_dim = dims.pop(axis)
#             dims.append(c_dim)
#             x_p = x.permute(dims).contiguous()
#             s = x_p.shape
#             x_flat = x_p.view(s[0], -1, 2)
#             x1, x2 = x_flat[..., 0], x_flat[..., 1]
#             mask = (x1 <= x2)
#             y1, y2 = torch.where(mask, x1, x2), torch.where(mask, x2, x1)
#             out_flat = torch.stack((y1, y2), dim=-1).view(s)
#             inv_dims = list(range(x.dim()))
#             inv_dims.insert(axis, inv_dims.pop(-1))
#             out = out_flat.permute(inv_dims).contiguous()
#         ctx.save_for_backward(mask)
#         return out

class GroupSortOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, axis):
        dims = list(range(x.dim()))
        channel_dim = dims.pop(axis)
        dims.append(channel_dim)
        x_p = x.permute(dims).contiguous()
        s = x_p.shape
        x_flat = x_p.view(s[0], -1, 2)
        x1, x2 = x_flat[..., 0], x_flat[..., 1]
        diff = torch.relu(x2 - x1)
        y1, y2 = x2 - diff, x1 + diff
        out = torch.stack((y1, y2), dim=-1).view(s)
        inv_dims = list(range(x.dim()))
        inv_dims.insert(axis, inv_dims.pop(-1))
        return out.permute(inv_dims)

class GroupSort_Custom(nn.Module):
    def __init__(self, axis=1):
        super().__init__()
        self.axis = axis
    def forward(self, x):
        return GroupSortOp.apply(x, self.axis)

class GroupSort_Decomposed(nn.Module):
    """ Standard PyTorch implementation that auto_LiRPA will trace node-by-node """
    def __init__(self, axis=1):
        super().__init__()
        self.axis = axis
        self.relu = nn.ReLU()
    def forward(self, x):
        dims = list(range(x.dim()))
        channel_dim = dims.pop(self.axis) 
        dims.append(channel_dim)
        x_p = x.permute(dims).contiguous()
        s = x_p.shape
        x_flat = x_p.view(s[0], -1, 2)
        x1, x2 = x_flat[..., 0], x_flat[..., 1]
        diff = x2 + (-1*x1)
        relu_diff = self.relu(diff)
        y1, y2 = x2 + (-1*relu_diff), x1 + relu_diff
        out_flat = torch.stack((y1, y2), dim=-1).view(s)
        inv_dims = list(range(x.dim()))
        inv_dims.insert(self.axis, inv_dims.pop(-1))
        return out_flat.permute(inv_dims)

# =============================================================================
# 2. CUSTOM BOUND OPERATOR (For the Custom Model)
# =============================================================================

class BoundGroupSort_General(BoundTwoPieceLinear):
    def __init__(self, attr, inputs, output_index, options):
        super().__init__(attr, inputs, output_index, options)
        self.axis = attr.get('axis', 1)
        self.alpha_size = 2 
        self.init_d = None 

    def forward(self, x):
        return GroupSortOp.forward(None, x, self.axis)

    def interval_propagate(self, *v):
        h_L, h_U = v[0]
        l, u = GroupSortOp.forward(None, h_L, self.axis), GroupSortOp.forward(None, h_U, self.axis)
        if hasattr(self.inputs[0], 'output_rho'):
            self.output_rho = self.inputs[0].output_rho
        return l, u

    def bound_backward(self, last_lA, last_uA, x=None, start_node=None, unstable_idx=None, **kwargs):
        A_obj = last_lA if last_lA is not None else last_uA
        if A_obj is None: return None, 0, 0

        if self.opt_stage in ['opt', 'reuse']:
            selected_alpha, _ = self.select_alpha_by_idx(last_lA, last_uA, unstable_idx, start_node)
            d = selected_alpha[0]
        else:
            num_pairs = self.inputs[0].lower.shape[self.axis] // 2
            d = torch.full((num_pairs,), 0.5, device=A_obj.patches.device if isinstance(A_obj, Patches) else A_obj.device)
            if self.init_d is None:
                h, w = self.inputs[0].lower.shape[-2:]
                self.init_d = d.view(1, 1, num_pairs, 1, 1).expand(1, 1, num_pairs, h, w).detach()

        A_y1, A_y2 = self._split_A(A_obj)
        A_diff = A_y2 - A_y1
        d_broadcast = self._align_slope(d, A_obj)
        new_A_x1, new_A_x2 = A_y2 - A_diff * d_broadcast, A_y1 + A_diff * d_broadcast
        final_A = self._merge_A(new_A_x1, new_A_x2, A_obj)

        total_bias = 0
        if hasattr(self, 'input_rho') and self.input_rho is not None and self.opt_stage == 'opt':
            selected_lam = self.select_lam_by_idx(last_lA, last_uA, unstable_idx, start_node)
            total_bias = self.sdp_crown_bias(A_diff, A_diff * d_broadcast, selected_lam[0], start_node, A_diff.shape)
        return [(final_A, final_A)], total_bias, total_bias

    def _split_A(self, A):
        if isinstance(A, Patches):
            s = A.patches.shape
            p_reshaped = A.patches.view(s[0], s[1], s[2], s[3], s[4] // 2, 2, s[5], s[6])
            return p_reshaped[..., 0, :, :], p_reshaped[..., 1, :, :]
        dims = list(range(A.dim()))
        dims.append(dims.pop(2))
        A_r = A.permute(dims).reshape(*A.shape[:2], -1, 2)
        return A_r[..., 0], A_r[..., 1]

    def _align_slope(self, d, A_obj):
        if isinstance(A_obj, Patches):
            if d.dim() > 1: d = d.mean(dim=(-1, -2))
            return d.view(1, 1, 1, 1, -1, 1, 1)
        return d.view(1, 1, -1, 1, 1) if d.dim() == 1 else d

    def _merge_A(self, A1, A2, orig):
        if isinstance(orig, Patches):
            new_p = torch.stack([A1, A2], dim=-3).view(orig.patches.shape)
            return Patches(new_p, orig.stride, orig.padding, orig.shape, orig.identity, orig.unstable_idx, orig.output_shape)
        merged = torch.stack([A1, A2], dim=-1).flatten(start_dim=-2)
        inv_dims = list(range(merged.dim()))
        inv_dims.insert(2, inv_dims.pop(-1))
        return merged.permute(inv_dims)

register_custom_op("onnx::GroupSortGeneral", BoundGroupSort_General)

# =============================================================================
# 3. COMPARISON SUITE
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("COMPARISON: DECOMPOSED VS CUSTOM OPERATOR")
    print("="*60)

    # Inputs: 4 channels (2 pairs)
    dummy_input = torch.randn(1, 4, 8, 8)
    eps = 0.1
    ptb = PerturbationLpNorm(norm=2, eps=eps)
    x = BoundedTensor(dummy_input, ptb)

    # Models: Sequential with Conv -> GroupSort
    conv = nn.Conv2d(4, 4, 3, padding=1)
    model_dec = nn.Sequential(conv, GroupSort_Decomposed(axis=1))
    model_cus = nn.Sequential(conv, GroupSort_Custom(axis=1))

    lirpa_dec = BoundedModule(model_dec, dummy_input)
    lirpa_cus = BoundedModule(model_cus, dummy_input)

    # --- STEP 1: CROWN ---
    print("\n--- STEP 1: CROWN (Static) ---")
    lb_dec, _ = lirpa_dec.compute_bounds(x=(x,), method='CROWN')
    lb_cus, _ = lirpa_cus.compute_bounds(x=(x,), method='CROWN')
    
    print(f"Decomposed LB: {lb_dec.flatten()[0]:.6f}")
    print(f"Custom Op  LB: {lb_cus.flatten()[0]:.6f}")
    print(f"Difference:    {torch.abs(lb_dec - lb_cus).max().item():.2e}")

    # --- STEP 2: ALPHA-CROWN ---
    print("\n--- STEP 2: ALPHA-CROWN (Optimized) ---")
    opt_opts = {'optimize_bound_args': {'iteration': 20, 'lr_alpha': 0.1, 'sparse_intermediate_bounds': False}}
    lirpa_dec.set_bound_opts(opt_opts)
    lirpa_cus.set_bound_opts(opt_opts)

    lb_dec_opt, _ = lirpa_dec.compute_bounds(x=(x,), method='alpha-CROWN')
    lb_cus_opt, _ = lirpa_cus.compute_bounds(x=(x,), method='alpha-CROWN')

    print(f"Decomposed Optimized LB: {lb_dec_opt.flatten()[0]:.6f}")
    print(f"Custom Op  Optimized LB: {lb_cus_opt.flatten()[0]:.6f}")
    print(f"Difference:              {torch.abs(lb_dec_opt - lb_cus_opt).max().item():.2e}")


    # --- STEP 3: SDP-CROWN ---
    print("\n--- STEP 2: SDP-CROWN (Optimized) ---")
    opt_opts = {'optimize_bound_args': {
        'iteration': 5,
        'enable_SDP_crown': True,
        'sparse_intermediate_bounds': False,
    }}
    lirpa_dec.set_bound_opts(opt_opts)
    lirpa_cus.set_bound_opts(opt_opts)

    lb_dec_opt, _ = lirpa_dec.compute_bounds(x=(x,), method='CROWN-Optimized')
    lb_cus_opt, _ = lirpa_cus.compute_bounds(x=(x,), method='CROWN-Optimized')

    print(f"Decomposed Optimized LB: {lb_dec_opt.flatten()[0]:.6f}")
    print(f"Custom Op  Optimized LB: {lb_cus_opt.flatten()[0]:.6f}")
    print(f"Difference:              {torch.abs(lb_dec_opt - lb_cus_opt).max().item():.2e}")

    # --- FINAL CHECK ---
    max_diff = torch.abs(lb_dec_opt - lb_cus_opt).max().item()
    if max_diff < 1e-5:
        print("\n[SUCCESS] Custom Operator and Decomposed logic are mathematically equivalent.")
    else:
        print("\n[WARNING] Significant difference detected. Verify broadcasting logic.")