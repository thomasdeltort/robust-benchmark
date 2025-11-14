import torch
import torch.nn as nn
import numpy as np
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
import warnings

# Suppress auto_LiRPA warnings for cleaner output
warnings.filterwarnings("ignore")

# --- Model Definitions (from your documentation) ---

class GroupSort(nn.Module):
    """
    Computes min/max using ONE SHARED ReLU node.
    This is expected to be the TIGHTEST formulation
    for bounding pairs under alpha-CROWN.
    """
    def __init__(self):
        super(GroupSort, self).__init__()
        self.relu = nn.ReLU() # ONE shared ReLU
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        batch_size = original_shape[0]
        num_features = np.prod(original_shape[1:])
        if num_features % 2 != 0: raise ValueError("Total features must be even.")
        x_flat = x.reshape(batch_size, -1)
        reshaped_x = x_flat.reshape(batch_size, -1, 2)
        x1s = reshaped_x[..., 0]
        x2s = reshaped_x[..., 1]
        
        diff = x1s - x2s
        relu_diff = self.relu(diff) # Computed ONCE
        
        y1_min = x1s - relu_diff # min = a - ReLU(a-b)
        y2_max = x2s + relu_diff # max = b + ReLU(a-b)
        
        sorted_pairs = torch.stack((y1_min, y2_max), dim=2)
        sorted_flat = sorted_pairs.reshape(batch_size, -1)
        output = sorted_flat.reshape(original_shape)
        return output
    
# class MaxMin(nn.Module):
#     """
#     Computes min/max using torch.min and torch.max.
#     These are bounded INDEPENDENTLY, losing the
#     sum(min, max) = a + b correlation.
#     """
#     def forward(self, x):
#         original_shape = x.shape
#         batch_size = original_shape[0]
#         num_features = np.prod(original_shape[1:])
#         if num_features % 2 != 0: raise ValueError("Total features must be even.")
#         x_flat = x.reshape(batch_size, -1)
#         x_pairs = x_flat.reshape(batch_size, -1, 2)
#         a = x_pairs[..., 0]
#         b = x_pairs[..., 1]
        
#         min_vals = torch.min(a, b)
#         max_vals = torch.max(a, b)
        
#         sorted_pairs = torch.stack((min_vals, max_vals), dim=-1)
#         sorted_flat = sorted_pairs.reshape(batch_size, -1)
#         return sorted_flat.reshape(original_shape)

class MaxMin(nn.Module):
    """
    Computes min/max using torch.split.
    
    WARNING: This will still cause the broadcasting RuntimeError
    in auto_LiRPA when batch size > 1.
    """
    def forward(self, x):
        original_shape = x.shape
        batch_size = original_shape[0]
        num_features = np.prod(original_shape[1:])
        
        if num_features % 2 != 0: 
            raise ValueError("Total features must be even.")
            
        x_flat = x.reshape(batch_size, -1)
        x_pairs = x_flat.reshape(batch_size, -1, 2)
        
        # --- Using torch.split ---
        # x_pairs has shape [batch_size, num_pairs, 2]
        # This splits along dim=-1 into two tensors of size 1
        a_tensor, b_tensor = torch.split(x_pairs, 1, dim=-1)
        
        # Squeeze the last dim to get shape [batch_size, num_pairs]
        a = a_tensor.squeeze(-1)
        b = b_tensor.squeeze(-1)
        
        # --- This is the part that causes the error ---
        # These operations trigger the buggy handler in auto_LiRPA
        min_vals = -torch.max(-a, -b)
        max_vals = torch.max(a, b)
        
        # --- End of problematic part ---
        
        sorted_pairs = torch.stack((min_vals, max_vals), dim=-1)
        sorted_flat = sorted_pairs.reshape(batch_size, -1)
        return sorted_flat.reshape(original_shape)

class Model_JustMax(nn.Module):
    """
    Computes *only* the max of pairs using torch.max.
    Input: (batch, features)
    Output: (batch, features / 2)
    """
    def forward(self, x):
        batch_size = x.shape[0]
        num_features = np.prod(x.shape[1:])
        if num_features % 2 != 0: raise ValueError("Total features must be even.")
        
        x_flat = x.reshape(batch_size, -1)
        x_pairs = x_flat.reshape(batch_size, -1, 2)
        
        a = x_pairs[..., 0]
        b = x_pairs[..., 1]
        
        max_vals = torch.max(a, b)
        return max_vals # Output shape is (batch, features / 2)

class Model_Max_via_ReLU(nn.Module):
    """
    Computes *only* the max of pairs, using the ReLU reformulation.
    Input: (batch, features)
    Output: (batch, features / 2)
    """
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        
    def forward(self, x):
        batch_size = x.shape[0]
        num_features = np.prod(x.shape[1:])
        if num_features % 2 != 0: raise ValueError("Total features must be even.")
        
        x_flat = x.reshape(batch_size, -1)
        x_pairs = x_flat.reshape(batch_size, -1, 2)
        
        a = x_pairs[..., 0]
        b = x_pairs[..., 1]
        
        # The exact b + ReLU(a - b) formulation
        # Note: this is max(a,b) = b + ReLU(a-b)
        # The other formulation is max(a,b) = a + ReLU(b-a)
        max_vals = b + self.relu(a - b)
        return max_vals # Output shape is (batch, features / 2)

class Model_Max_via_ReLU_Symmetric(nn.Module):
    """
    Computes *only* the max of pairs, using the symmetric ReLU reformulation.
    Input: (batch, features)
    Output: (batch, features / 2)
    """
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        
    def forward(self, x):
        batch_size = x.shape[0]
        num_features = np.prod(x.shape[1:])
        if num_features % 2 != 0: raise ValueError("Total features must be even.")
        
        x_flat = x.reshape(batch_size, -1)
        x_pairs = x_flat.reshape(batch_size, -1, 2)
        
        a = x_pairs[..., 0]
        b = x_pairs[..., 1]
        
        # The symmetric a + ReLU(b - a) formulation
        max_vals = a + self.relu(b - a)
        return max_vals # Output shape is (batch, features / 2)

def print_comparison_table(title, model_names_map, results_dict, methods):
    """Prints a clean, formatted table of the results."""
    print("\n" + "="*80)
    print(f"--- {title} ---")
    print("="*80 + "\n")
    
    # Header
    print(f"{'Model':<48} {'Method':<12} {'Lower Bound':<40} {'Upper Bound':<40}")
    print("-"*140)
    
    # Data
    for model_key, model_name in model_names_map.items():
        for method in methods:
            if model_key not in results_dict[method]:
                # Handle cases where a model failed to run (e.g., if we skipped it)
                lb_str = "N/A (CRASHED)"
                ub_str = "N/A (CRASHED)"
            else:
                lb, ub = results_dict[method][model_key]
                # Format tensors for clean printing
                lb_str = " ".join(f"{x:7.4f}" for x in lb.flatten())
                ub_str = " ".join(f"{x:7.4f}" for x in ub.flatten())
                
                # Pad lb/ub strings to align columns
                lb_str = f"[{lb_str.strip()}]"
                ub_str = f"[{ub_str.strip()}]"
            
            print(f"{model_name:<48} {method:<12} {lb_str:<40} {ub_str:<40}")
        print("") # Add a newline for readability between models


# --- Main execution to compare bounds for ONE point ---
if __name__ == '__main__':
    
    # --- 1. Setup ---
    IN_FEATURES = 8  # (Must be even)
    EPSILON = 0.1    # L-infinity perturbation radius
    METHODS_TO_TEST = ['alpha-crown']
    # 'ibp', 'crown', 

    # Use a fixed seed for reproducible results
    torch.manual_seed(123)
    
    # --- Define all models ---
    
    # Comparison 1: "Max Only" models
    models_comp1 = {
        'torch_max': nn.Sequential(Model_JustMax()),
        'relu_max': nn.Sequential(Model_Max_via_ReLU()),
        'relu_max_sym': nn.Sequential(Model_Max_via_ReLU_Symmetric())
    }
    model_names_comp1 = {
        'torch_max': 'Model_JustMax (torch.max)',
        'relu_max': 'Model_Max_via_ReLU (b + ReLU(a-b))',
        'relu_max_sym': 'Model_Max_via_ReLU_Symmetric (a + ReLU(b-a))'
    }
    
    # Comparison 2: "Min/Max Pair" models
    models_comp2 = {
        'torch_mm': nn.Sequential(MaxMin()),
        'gs_shared': nn.Sequential(GroupSort()),
    }
    model_names_comp2 = {
        'torch_mm': 'MaxMin (torch.min/max)',
        'gs_shared': 'GroupSort (shared ReLU)',
    }
    
    # --- Define Test Cases (different input tensors) ---
    
    # This tensor tests all 3 ReLU states:
    # Pair 1: [0.9, 1.1] vs [-1.3, -1.1] -> d in [2.0, 2.4] (Stable Active)
    # Pair 2: [-0.05, 0.15] vs [0.0, 0.2] -> d in [-0.25, 0.15] (Unstable)
    # Pair 3: [0.7, 0.9] vs [-0.4, -0.2] -> d in [0.9, 1.3] (Stable Active)
    # Pair 4: [-0.1, 0.1] vs [1.4, 1.6] -> d in [-1.7, -1.3] (Stable Inactive)
    x0_mixed = torch.tensor([[1.0, -1.2, 0.05, 0.1, 0.8, -0.3, 0.0, 1.5]]) 
    
    # This tensor tests 4 different "Unstable" ReLU states
    # Pair 1: [-0.1, 0.1] vs [0.0, 0.2] -> d in [-0.3, 0.1] (Unstable)
    # Pair 2: [-0.1, 0.1] vs [-0.1, 0.1] -> d in [-0.2, 0.2] (Unstable)
    # Pair 3: [0.0, 0.2] vs [-0.1, 0.1] -> d in [-0.1, 0.3] (Unstable)
    # Pair 4: [0.4, 0.6] vs [0.4, 0.6] -> d in [-0.2, 0.2] (Unstable)
    x0_unstable = torch.tensor([[0.0, 0.1, 0.0, 0.0, 0.1, 0.0, 0.5, 0.5]])
    
    test_cases = {
        "TEST CASE 1: Mixed ReLU States (Active, Inactive, Unstable)": x0_mixed,
        "TEST CASE 2: All Unstable ReLU States": x0_unstable
    }

    ptb = PerturbationLpNorm(norm=torch.inf, eps=EPSILON)

    # --- 2. Run All Computations for All Test Cases ---
    
    for case_name, x0 in test_cases.items():
        
        print("\n" + "#"*80)
        print(f"### {case_name}")
        print(f"### Input point (x0): {x0}")
        print("#"*80)

        bounded_input = BoundedTensor(x0, ptb)
        
        # --- Check Forward Pass (Sanity Check) ---
        print("\n--- Forward Pass Check ---")
        with torch.no_grad():
            out_torch_max_only = models_comp1['torch_max'](x0)
            out_relu_max_only = models_comp1['relu_max'](x0)
            out_relu_max_only_sym = models_comp1['relu_max_sym'](x0)
            print(f"  Max-only outputs are identical: {torch.allclose(out_torch_max_only, out_relu_max_only) and torch.allclose(out_torch_max_only, out_relu_max_only_sym)}")
            
            out_torch_maxmin = models_comp2['torch_mm'](x0)
            out_relu_groupsort = models_comp2['gs_shared'](x0)
            print(f"  Min/Max Pair outputs are identical: {torch.allclose(out_torch_maxmin, out_relu_groupsort)}\n")

        
        # --- Compute All Bounds ---
        results_comp1 = {method: {} for method in METHODS_TO_TEST}
        results_comp2 = {method: {} for method in METHODS_TO_TEST}

        for method in METHODS_TO_TEST:
            print(f"--- Running bound computation for method: '{method}' ---")
            
            # # Run Comparison 1
            # for key, model in models_comp1.items():
            #     print(f"  Computing bounds for {model_names_comp1[key]}...")
            #     lirpa_model = BoundedModule(model, x0)
            #     lb, ub = lirpa_model.compute_bounds(x=(bounded_input,), method=method)
            #     results_comp1[method][key] = (lb.detach(), ub.detach())

            # Run Comparison 2
            for key, model in models_comp2.items():
                print(f"  Computing bounds for {model_names_comp2[key]}...")
                
                bound_opts = {}
                # --- WORKAROUND FOR auto_LiRPA BUG ---
                # Apply a workaround for a known alpha-CROWN bug in the min/max operator
                if key == 'torch_mm' and method == 'alpha-crown':
                    print("    -> Applying 'no_alpha_layers' workaround for MaxMin.")
                    # The model is nn.Sequential(MaxMin()), so its module name is '0'
                    bound_opts = {
                        'optimize_bound_args': {
                             'no_alpha_layers': ['0'] 
                        }
                    }
                
                lirpa_model = BoundedModule(model, x0, bound_opts=bound_opts)
                # --- END OF WORKAROUND ---

                lb, ub = lirpa_model.compute_bounds(x=(bounded_input,), method=method)
                results_comp2[method][key] = (lb.detach(), ub.detach())

        # --- Print All Results ---
        
        print_comparison_table(
            title=f"COMPARISON 1: 'MAX ONLY' MODELS ({case_name})",
            model_names_map=model_names_comp1,
            results_dict=results_comp1,
            methods=METHODS_TO_TEST
        )
        
        print_comparison_table(
            title=f"COMPARISON 2: 'MIN/MAX PAIR' MODELS ({case_name})",
            model_names_map=model_names_comp2,
            results_dict=results_comp2,
            methods=METHODS_TO_TEST
        )

    print("\n--- End of Analysis ---")

