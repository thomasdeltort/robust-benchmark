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
# 1. Standard PyTorch Operation (Autograd)
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

    @staticmethod
    def symbolic(g, x, axis):
        # Crucial for auto_LiRPA to map this to our custom class
        return g.op("GroupSortGeneral", x, axis_i=axis)

class GroupSort_General(nn.Module):
    def __init__(self, axis=1):
        super().__init__()
        self.axis = axis
    def forward(self, x):
        return GroupSortOp.apply(x, self.axis)


class BoundGroupSort_General(BoundTwoPieceLinear):
    def __init__(self, attr, inputs, output_index, options):
        super().__init__(attr, inputs, output_index, options)
        self.axis = attr.get('axis', 1)
        self.alpha_size = 2 
        self.init_d = None 

    def forward(self, x):
        self.shape = x.shape[1:] 
        return GroupSortOp.forward(None, x, self.axis)

    def interval_propagate(self, *v):
        h_L, h_U = v[0]
        l = GroupSortOp.forward(None, h_L, self.axis)
        u = GroupSortOp.forward(None, h_U, self.axis)
        if hasattr(self.inputs[0], 'output_rho'):
            self.output_rho = self.inputs[0].output_rho
        return l, u

    def get_unstable_idx(self):
        x_l, x_u = self.inputs[0].lower, self.inputs[0].upper
        l1, l2 = self._split_A(x_l)
        u1, u2 = self._split_A(x_u)
        v_l, v_u = l2 - u1, u2 - l1
        self.alpha_indices = torch.logical_and(v_l < 0, v_u > 0).any(dim=0).nonzero(as_tuple=True)

    def _relu_upper_bound(self, lb, ub):
        """ Robust CROWN upper bound for internal ReLU (x2 - x1). """
        # Numerical Stability: ensure ub > lb
        lb_r = lb.clamp(max=0)
        ub_r = ub.clamp(min=0)
        diff = (ub_r - lb_r).clamp(min=1e-8) 
        
        d_up = ub_r / diff
        b_up = -lb_r * d_up
        return d_up, b_up

    def bound_backward(self, last_lA, last_uA, x=None, start_node=None, unstable_idx=None, **kwargs):
        A_obj = last_lA if last_lA is not None else last_uA
        if A_obj is None: return None, 0, 0
        
        # --- 1. Pre-activation Bounds for inner ReLU ---
        x_L, x_U = self.inputs[0].lower, self.inputs[0].upper
        l1, l2 = self._split_A(x_L)
        u1, u2 = self._split_A(x_U)
        
        # Worst-case interval for v = x2 - x1
        v_L = l2 - u1
        v_U = u2 - l1
        
        # Calculate CROWN Upper Bound Line (Triangle Relaxation)
        d_up, b_up = self._relu_upper_bound(v_L, v_U)

        # --- 2. Slope Selection & Stability Enforcement ---
        # A. Identify Stable Neurons
        is_stable_pos = v_L >= 0 # ReLU is identity
        is_stable_neg = v_U <= 0 # ReLU is zero
        
        # B. Retrieve/Initialize Optimized Lower Slopes (Alpha)
        if self.opt_stage in ['opt', 'reuse']:
            selected_alpha, _ = self.select_alpha_by_idx(last_lA, last_uA, unstable_idx, start_node)
            d_low_l, d_low_u = selected_alpha[0], selected_alpha[1]
            
            if self.alpha_indices is not None:
                full_shape = [d_low_l.shape[0], d_low_l.shape[1]] + list(self.shape)
                full_shape[2] = full_shape[2] // 2 
                d_low_l = self.reconstruct_full_alpha(d_low_l, full_shape, self.alpha_indices)
                d_low_u = self.reconstruct_full_alpha(d_low_u, full_shape, self.alpha_indices)
        else:
            if self.init_d is None:
                s = list(x_L.shape)
                d_shape = [1, s[0], s[self.axis] // 2] + s[2:]
                self.init_d = torch.full(d_shape, 0.5, device=x_L.device)
            d_low_l = d_low_u = self.init_d[0]

        # C. OVERRIDE Slopes for Stable Neurons
        # Stable Positive -> Slope 1
        d_low_l = torch.where(is_stable_pos, torch.ones_like(d_low_l), d_low_l)
        d_low_u = torch.where(is_stable_pos, torch.ones_like(d_low_u), d_low_u)
        
        # Stable Negative -> Slope 0
        d_low_l = torch.where(is_stable_neg, torch.zeros_like(d_low_l), d_low_l)
        d_low_u = torch.where(is_stable_neg, torch.zeros_like(d_low_u), d_low_u)

        # --- 3. Center Transformation ---
        original_xc = self.xc
        x_center = (self.inputs[0].lower + self.inputs[0].upper) / 2.0
        xc1, xc2 = self._split_A(x_center)
        self.xc = xc2 - xc1 

        lbias = ubias = 0
        final_lA = final_uA = None
        opt_args = self.options.get('optimize_bound_args', {})
        enable_sdp = opt_args.get('enable_SDP_crown', False)

        # --- 4. Backward Propagation with Bias Correction ---
        if last_lA is not None:
            lA_y1, lA_y2 = self._split_A(last_lA)
            lA_diff = lA_y2 - lA_y1
            
            # LB Logic: 
            # If sensitivity > 0, we need ReLU Lower Bound -> d_low_l
            # If sensitivity < 0, we need ReLU Upper Bound -> d_up
            l_slope = torch.where(lA_diff >= 0, d_low_l, d_up)
            l_bias_map = torch.where(lA_diff < 0, lA_diff * b_up, torch.zeros_like(b_up))
            
            lbias = l_bias_map.reshape(lA_diff.shape[0], lA_diff.shape[1], -1).sum(dim=-1)

            if enable_sdp and self.opt_stage == 'opt' and hasattr(self, 'input_rho'):
                selected_lam = self.select_lam_by_idx(last_lA, last_uA, unstable_idx, start_node)
                lbias += self.sdp_crown_bias(lA_diff, lA_diff * l_slope, selected_lam[0], 
                                             start_node, [lA_diff.shape[0], lA_diff.shape[1]], sign=-1)
            
            final_lA = self._merge_A(lA_y2 - lA_diff * l_slope, lA_y1 + lA_diff * l_slope, last_lA)

        if last_uA is not None:
            uA_y1, uA_y2 = self._split_A(last_uA)
            uA_diff = uA_y2 - uA_y1
            
            # UB Logic (Signs flip):
            # If sensitivity > 0, we need ReLU Upper Bound -> d_up
            # If sensitivity < 0, we need ReLU Lower Bound -> d_low_u
            u_slope = torch.where(uA_diff >= 0, d_up, d_low_u)
            u_bias_map = torch.where(uA_diff >= 0, uA_diff * b_up, torch.zeros_like(b_up))
            
            ubias = u_bias_map.reshape(uA_diff.shape[0], uA_diff.shape[1], -1).sum(dim=-1)

            if enable_sdp and self.opt_stage == 'opt' and hasattr(self, 'input_rho'):
                selected_lam = self.select_lam_by_idx(last_lA, last_uA, unstable_idx, start_node)
                ubias += self.sdp_crown_bias(uA_diff, uA_diff * u_slope, selected_lam[1], 
                                             start_node, [uA_diff.shape[0], uA_diff.shape[1]], sign=+1)
            
            final_uA = self._merge_A(uA_y2 - uA_diff * u_slope, uA_y1 + uA_diff * u_slope, last_uA)

        self.xc = original_xc
        return [(final_lA, final_uA)], lbias, ubias

    # --- Shape Helpers ---
    def _split_A(self, A):
        if isinstance(A, Patches):
            p = A.patches 
            s = list(p.shape)
            p_r = p.view(s[0], s[1], s[2], s[3], s[4] // 2, 2, s[5], s[6])
            return p_r[..., 0, :, :], p_r[..., 1, :, :]
        s = list(A.shape)
        if len(s) == 5: 
            A_r = A.view(s[0], s[1], s[2] // 2, 2, s[3], s[4])
            return A_r[:, :, :, 0], A_r[:, :, :, 1]
        if len(s) == 4: 
            A_r = A.view(s[0], s[1] // 2, 2, s[2], s[3])
            return A_r[:, :, 0], A_r[:, :, 1]
        return A

    def _align_slope(self, d, A_obj):
        if isinstance(A_obj, Patches):
            if d.dim() == 5: d = d.mean(dim=(-1, -2))
            return d.view(d.size(0), d.size(1), 1, 1, -1, 1, 1)
        return d 

    def _merge_A(self, A1, A2, orig):
        if isinstance(orig, Patches):
            new_p = torch.stack([A1, A2], dim=5).view(orig.patches.shape)
            return Patches(new_p, orig.stride, orig.padding, orig.shape, 
                           orig.identity, orig.unstable_idx, orig.output_shape)
        res = torch.stack([A1, A2], dim=3 if orig.dim()==5 else 2)
        return res.view(orig.shape)
    def clip_alpha(self):
        for v in self.alpha.values():
            v.data = torch.clamp(v.data, 0., 1.)
    
    def clip_lam(self):
        with torch.no_grad():
            for v in self.lam.values():
                v.clamp_(min=1e-6)
# Register the logic
register_custom_op("onnx::GroupSortGeneral", BoundGroupSort_General)

# =============================================================================
# MAIN TESTING SECTION
# =============================================================================
if __name__ == "__main__":
    print("\n" + "="*50)
    print("TEST 1: ARCHITECTURE INITIALIZATION")
    print("="*50)
    model = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),
        GroupSort_General(axis=1),
        nn.Flatten(),
        nn.Linear(8*32*32, 2)
    )
    dummy_input = torch.randn(1, 3, 32, 32)
    lirpa_model = BoundedModule(model, dummy_input)
    
    # Check if registered
    found = any(isinstance(n, BoundGroupSort_General) for n in lirpa_model.nodes())
    print(f"[✓] Custom node found: {found}")

    eps = 0.1
    ptb = PerturbationLpNorm(norm=2, eps=eps)
    x = BoundedTensor(dummy_input, ptb)
    print(model(dummy_input))
    print("\n" + "="*50)
    print("TEST 2: ALPHA-CROWN PASS")
    print("="*50)
    lirpa_model.set_bound_opts({
            'optimize_bound_args': {
                'iteration': 200, 
                'early_stop_patience': 30, 
                'enable_opt_interm_bounds': True, 
                'verbosity': False
            }, 
            'verbosity': False
        })
    lb, ub = lirpa_model.compute_bounds(x=(x,), method='alpha-CROWN')
    print(f"Neuron 0: LB = {lb[0][0].item():.4f}")
    print(f"Neuron 0: UB = {ub[0][0].item():.4f}")
    print(f"Neuron 1: LB = {lb[0][1].item():.4f}")
    print(f"Neuron 1: UB = {ub[0][1].item():.4f}")

    print("\n" + "="*50)
    print("TEST 3: SDP-CROWN PASS")
    print("="*50)
    # lirpa_model.set_bound_opts({'optimize_bound_args': {'iteration': 20, 'enable_SDP_crown': True}})
    # Recommended settings for a tight final benchmark
    lirpa_model.set_bound_opts({'optimize_bound_args': {
            'iteration': 300, 
            'lr_alpha': 0.5, 
            'early_stop_patience': 20, 
            'fix_interm_bounds': False, 
            'enable_opt_interm_bounds': True, 
            'enable_SDP_crown': True, 
            'lr_lambda': 0.05
        }})
    lb_opt, ub_opt = lirpa_model.compute_bounds(x=(x,), method='CROWN-Optimized')
    print(f"Neuron 0: LB = {lb_opt[0][0].item():.4f}")
    print(f"Neuron 0: UB = {ub_opt[0][0].item():.4f}")
    print(f"Neuron 1: LB = {lb_opt[0][1].item():.4f}")
    print(f"Neuron 1: UB = {ub_opt[0][1].item():.4f}")