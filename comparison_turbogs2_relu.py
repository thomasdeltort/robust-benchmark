import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from auto_LiRPA import BoundedModule, BoundedTensor, register_custom_op
from auto_LiRPA.bound_ops import Bound
from auto_LiRPA.perturbations import PerturbationLpNorm

# --- Step 1: Define a `torch.autograd.Function` for the custom operator ---
class GroupSortOp(torch.autograd.Function):
    @staticmethod
    def symbolic(g, x):
        output = g.op('custom::GroupSort', x)
        output.setType(x.type())
        return output

    @staticmethod
    def forward(ctx, x):
        original_shape = x.shape
        x_reshaped = x.contiguous().view(-1, 2)
        sorted_x, _ = torch.sort(x_reshaped, dim=-1)
        return sorted_x.view(original_shape)

# --- Step 2: Define an `nn.Module` that uses the custom operator ---
class GroupSort(nn.Module):
    def forward(self, x):
        return GroupSortOp.apply(x)

# --- Step 3: Implement the complete Bound class for the custom operator ---
class BoundGroupSort(Bound):
    def __init__(self, attr, inputs, output_index, options):
        super().__init__(attr, inputs, output_index, options)

    def forward(self, x):
        return GroupSortOp.forward(None, x)

    def interval_propagate(self, *v):
        lower, upper = v[0]
        # --- ROBUSTNESS FIX 1 ---
        # If the input bounds to this function have not been computed, we cannot proceed.
        if lower is None or upper is None:
            return None, None
        
        lower_reshaped = lower.contiguous().view(-1, 2)
        upper_reshaped = upper.contiguous().view(-1, 2)
        final_lower = torch.sort(lower_reshaped, dim=-1)[0].view(lower.shape)
        final_upper = torch.sort(upper_reshaped, dim=-1)[0].view(upper.shape)
        return final_lower, final_upper

    def bound_backward(self, last_lA, last_uA, x, *args, **kwargs):
        def _bound_oneside(last_A):
            if last_A is None: return None, 0
            l, u = x.lower, x.upper
            
            A_reshaped = last_A.view(last_A.shape[0], last_A.shape[1], -1, 2)
            A_new = torch.zeros_like(A_reshaped)
            A_y1, A_y2 = A_reshaped[..., 0], A_reshaped[..., 1]

            # --- ROBUSTNESS FIX 2 ---
            # If concrete bounds for the input x are missing, we MUST default
            # to the most general (unstable) relaxation to avoid a crash.
            if l is None or u is None:
                A_new[..., 0] = torch.min(A_y1, A_y2)
                A_new[..., 1] = torch.max(A_y1, A_y2)
            else:
                l_reshaped = l.contiguous().view(l.shape[0], -1, 2)
                u_reshaped = u.contiguous().view(u.shape[0], -1, 2)

                case1 = (u_reshaped[..., 0] <= l_reshaped[..., 1]).unsqueeze(0).unsqueeze(0)
                case2 = (u_reshaped[..., 1] <= l_reshaped[..., 0]).unsqueeze(0).unsqueeze(0)
                case3 = ~case1 & ~case2
                
                A_new[..., 0] = torch.where(case1, A_y1, A_new[..., 0])
                A_new[..., 1] = torch.where(case1, A_y2, A_new[..., 1])
                A_new[..., 0] = torch.where(case2, A_y2, A_new[..., 0])
                A_new[..., 1] = torch.where(case2, A_y1, A_new[..., 1])
                A_new[..., 0] = torch.where(case3, torch.min(A_y1, A_y2), A_new[..., 0])
                A_new[..., 1] = torch.where(case3, torch.max(A_y1, A_y2), A_new[..., 1])

            return A_new.view_as(last_A), 0
        
        lA, lbias = _bound_oneside(last_lA)
        uA, ubias = _bound_oneside(last_uA)
        return [(lA, uA)], lbias, ubias

# --- Step 4: Register the custom operator name with its Bound class ---
register_custom_op("custom::GroupSort", BoundGroupSort)

# --- Experiment Function ---
def run_final_comparison(in_features=10, out_features=10, epsilon=0.1, num_trials=500):
    relu_widths, groupsort_widths = [], []
    for i in range(num_trials):
        linear_layer = nn.Linear(in_features, out_features)
        model_relu = nn.Sequential(linear_layer, nn.ReLU())
        model_groupsort = nn.Sequential(linear_layer, GroupSort())
        
        x0 = torch.randn(1, in_features)
        ptb = PerturbationLpNorm(norm=torch.inf, eps=epsilon)
        bounded_input = BoundedTensor(x0, ptb)
        
        lirpa_model_relu = BoundedModule(model_relu, x0)
        lb_relu, ub_relu = lirpa_model_relu.compute_bounds(x=(bounded_input,), method='CROWN')
        relu_widths.append(torch.mean(ub_relu - lb_relu).item())
        
        lirpa_model_gs = BoundedModule(model_groupsort, x0)
        lb_gs, ub_gs = lirpa_model_gs.compute_bounds(x=(bounded_input,), method='CROWN')
        groupsort_widths.append(torch.mean(ub_gs - lb_gs).item())

    return relu_widths, groupsort_widths

# --- Run and Visualize ---
if __name__ == '__main__':
    INPUT_DIM, OUTPUT_DIM, EPSILON, NUM_TRIALS = 20, 20, 0.05, 1000
    print("Running comparison with DEFINITIVE Custom GroupSort handler...")
    relu_results, groupsort_results = run_final_comparison(
        in_features=INPUT_DIM, out_features=OUTPUT_DIM, epsilon=EPSILON, num_trials=NUM_TRIALS
    )
    print("Done.\n")
    
    mean_relu, mean_gs = np.mean(relu_results), np.mean(groupsort_results)
    print("--- Average Bound Width Results (lower is better) ---")
    print(f"ReLU                 : {mean_relu:.6f}")
    print(f"Custom GroupSort     : {mean_gs:.6f}")
    
    if mean_gs < mean_relu:
        improvement = (1 - mean_gs / mean_relu) * 100
        print(f"\n✅ Custom GroupSort produced bounds that were on average {improvement:.2f}% tighter than ReLU.")
    else:
        improvement = (1 - mean_relu / mean_gs) * 100
        print(f"\n❌ ReLU bounds were unexpectedly tighter by {improvement:.2f}%.")

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 6))
    sns.histplot(relu_results, color="skyblue", kde=True, label=f'ReLU (Mean: {mean_relu:.4f})', bins=50)
    sns.histplot(groupsort_results, color="violet", kde=True, label=f'Custom GroupSort (Mean: {mean_gs:.4f})', bins=50)
    plt.title('Distribution of CROWN Bound Widths', fontsize=16)
    plt.xlabel('Average Bound Width')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

