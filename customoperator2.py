import torch
import torch.nn as nn
import sys

# Update path to your local SDP-CROWN repository
SDP_CROWN_PATH = '/home/aws_install/robustess_project/SDP-CROWN'
sys.path.insert(0, SDP_CROWN_PATH)

from auto_LiRPA import BoundedModule, BoundedTensor, register_custom_op
from auto_LiRPA.operators import BoundTwoPieceLinear
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.patches import Patches

# =============================================================================
# STEP 1: Custom GroupSort Logic
# =============================================================================
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

class GroupSort_General(nn.Module):
    def __init__(self, axis=1):
        super().__init__()
        self.axis = axis
    def forward(self, x):
        return GroupSortOp.apply(x, self.axis)

# =============================================================================
# STEP 2: Custom Bound Class (Unified Sandwich Logic)
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
        l = GroupSortOp.forward(None, h_L, self.axis)
        u = GroupSortOp.forward(None, h_U, self.axis)
        if hasattr(self.inputs[0], 'output_rho'):
            self.output_rho = self.inputs[0].output_rho
        return l, u

    def bound_backward(self, last_lA, last_uA, x=None, start_node=None, unstable_idx=None, **kwargs):
        A_obj = last_lA if last_lA is not None else last_uA
        if A_obj is None: return None, 0, 0

        # 1. Determine Slope d
        if self.opt_stage in ['opt', 'reuse']:
            selected_alpha, _ = self.select_alpha_by_idx(last_lA, last_uA, unstable_idx, start_node)
            d = selected_alpha[0]
        else:
            # For C/2 pairs, d should be per-channel
            num_pairs = self.inputs[0].lower.shape[self.axis] // 2
            d = torch.full((num_pairs,), 0.5, device=A_obj.patches.device if isinstance(A_obj, Patches) else A_obj.device)
            if self.init_d is None:
                # Shape requirement for Alpha-CROWN init: [1, 1, C/2, H, W]
                h, w = self.inputs[0].lower.shape[-2:]
                self.init_d = d.view(1, 1, num_pairs, 1, 1).expand(1, 1, num_pairs, h, w).detach()

        # 2. Split Sensitivities
        A_y1, A_y2 = self._split_A(A_obj)
        A_diff = A_y2 - A_y1
        
        # 3. Apply Sandwich Relaxation
        # Crucial broadcasting fix:
        d_broadcast = self._align_slope(d, A_obj)
        new_A_x1 = A_y2 - A_diff * d_broadcast
        new_A_x2 = A_y1 + A_diff * d_broadcast

        final_A = self._merge_A(new_A_x1, new_A_x2, A_obj)

        # 4. SDP-CROWN Bias
        total_bias = 0
        if hasattr(self, 'input_rho') and self.input_rho is not None and self.opt_stage == 'opt':
            selected_lam = self.select_lam_by_idx(last_lA, last_uA, unstable_idx, start_node)
            total_bias = self.sdp_crown_bias(A_diff, A_diff * d_broadcast, selected_lam[0], start_node, A_diff.shape)

        return [(final_A, final_A)], total_bias, total_bias

    def _split_A(self, A):
        if isinstance(A, Patches):
            p = A.patches # [out_c, batch, out_h, out_w, in_c, k_h, k_w]
            s = p.shape
            p_reshaped = p.view(s[0], s[1], s[2], s[3], s[4] // 2, 2, s[5], s[6])
            return p_reshaped[..., 0, :, :], p_reshaped[..., 1, :, :]
        else:
            # [spec, batch, C, H, W] or [spec, batch, C]
            dims = list(range(A.dim()))
            dims.append(dims.pop(2)) # Move C to end
            A_p = A.permute(dims)
            A_r = A_p.reshape(*A_p.shape[:-1], -1, 2)
            return A_r[..., 0], A_r[..., 1]

    def _align_slope(self, d, A_obj):
        if isinstance(A_obj, Patches):
            # Patches shape is 7D. Slope must align with in_c (index 4)
            # d is [num_pairs] or [spec, batch, num_pairs, H, W]
            if d.dim() > 1:
                # If d is optimized, take mean over spatial to allow broadcasting in patches
                d = d.mean(dim=(-1, -2))
            return d.view(1, 1, 1, 1, -1, 1, 1)
        else:
            # Tensor case
            if d.dim() == 1:
                return d.view(1, 1, -1, 1, 1)
            return d

    def _merge_A(self, A1, A2, orig):
        if isinstance(orig, Patches):
            # Stack along the new 'pair' dimension then flatten back to in_c
            new_p = torch.stack([A1, A2], dim=-3).view(orig.patches.shape)
            return Patches(new_p, orig.stride, orig.padding, orig.shape, orig.identity, orig.unstable_idx, orig.output_shape)
        else:
            merged = torch.stack([A1, A2], dim=-1).flatten(start_dim=-2)
            inv_dims = list(range(merged.dim()))
            inv_dims.insert(2, inv_dims.pop(-1))
            return merged.permute(inv_dims)

register_custom_op("onnx::GroupSortGeneral", BoundGroupSort_General)

# =============================================================================
# MAIN TESTING SECTION
# =============================================================================
if __name__ == "__main__":
    print("\n--- TEST 1: ARCHITECTURE ---")
    model = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),
        GroupSort_General(axis=1),
        nn.Flatten(),
        nn.Linear(8*32*32, 2)
    )
    dummy_input = torch.randn(1, 3, 32, 32)
    lirpa_model = BoundedModule(model, dummy_input)
    print("[✓] Model registered.")

    print("\n--- TEST 2: CROWN PASS ---")
    x = BoundedTensor(dummy_input, PerturbationLpNorm(norm=2, eps=0.1))
    lb, ub = lirpa_model.compute_bounds(x=(x,), method='alpha-CROWN')
    print(f"[✓] CROWN pass successful. LB: {lb[0][0].item():.4f}")

    print("\n--- TEST 3: SDP-CROWN PASS ---")
    lirpa_model.set_bound_opts({'optimize_bound_args': {
        'iteration': 5,
        'enable_SDP_crown': True,
        'sparse_intermediate_bounds': False,
    }})
    lb_opt, _ = lirpa_model.compute_bounds(x=(x,), method='CROWN-Optimized')
    print(f"[✓] Optimized pass successful. LB: {lb_opt[0][0].item():.4f}")
