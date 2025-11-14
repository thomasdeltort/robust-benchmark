# import torch
# import torch.nn as nn
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from auto_LiRPA import BoundedModule, BoundedTensor
# from auto_LiRPA.perturbations import PerturbationLpNorm
# import warnings

# # Suppress warnings for cleaner output
# warnings.filterwarnings("ignore")


# # --- Definition for Model 1: The "Loose" Independent ReLUs Model ---
# class GroupSortIndependentReLUs(nn.Module):
#     """
#     A GroupSort implementation that computes min and max independently,
#     by reformulating each with its *own* separate nn.ReLU module.
    
#     HYPOTHESIS: This will be "loose" and equivalent to MaxMin.
#     """
#     def __init__(self):
#         super().__init__()
#         # We define two different nn.ReLU instances.
#         self.relu_for_min = nn.ReLU()
#         self.relu_for_max = nn.ReLU()

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         original_shape = x.shape
#         batch_size = original_shape[0]
        
#         num_features = np.prod(original_shape[1:])
#         if num_features % 2 != 0:
#             raise ValueError(
#                 f"The total number of features must be even, but got {num_features} "
#                 f"for shape {original_shape}."
#             )

#         x_flat = x.reshape(batch_size, -1)
#         reshaped_x = x_flat.reshape(batch_size, -1, 2)
        
#         a = reshaped_x[..., 0]
#         b = reshaped_x[..., 1]

#         # --- Independent Min/Max-via-ReLU Logic ---
        
#         # 1. Compute min(a, b) using the identity: a - ReLU(a - b)
#         diff_for_min = a - b
#         relu_diff_min = self.relu_for_min(diff_for_min)
#         min_vals = a - relu_diff_min

#         # 2. Compute max(a, b) using the identity: b + ReLU(a - b)
#         diff_for_max = a - b
#         relu_diff_max = self.relu_for_max(diff_for_max)
#         max_vals = b + relu_diff_max
        
#         # Stack them back in the same (min, max) order
#         sorted_pairs = torch.stack((min_vals, max_vals), dim=2)
        
#         sorted_flat = sorted_pairs.reshape(batch_size, -1)
#         output = sorted_flat.reshape(original_shape)
        
#         return output


# # --- Definition for Model 2: The "Tight" Shared ReLU Model ---
# class GroupSort(nn.Module):
#     """
#     The standard, verifier-friendly GroupSort that *shares* one ReLU.
    
#     HYPOTHESIS: This will be the "tightest" model.
#     """
#     def __init__(self):
#         super(GroupSort, self).__init__()
#         # ONE shared ReLU module
#         self.relu = nn.ReLU()

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         original_shape = x.shape
#         batch_size = original_shape[0]
        
#         num_features = np.prod(original_shape[1:])
        
#         if num_features % 2 != 0:
#             raise ValueError(
#                 f"The total number of features must be even, but got {num_features} "
#                 f"for shape {original_shape}."
#             )

#         x_flat = x.reshape(batch_size, -1)
#         reshaped_x = x_flat.reshape(batch_size, -1, 2)
        
#         x1s = reshaped_x[..., 0]
#         x2s = reshaped_x[..., 1]
        
#         # --- Shared ReLU Logic ---
#         diff = x1s - x2s
#         relu_diff = self.relu(diff) # <-- The ONE ReLU node
        
#         # Both outputs are derived from the *same* relu_diff
#         y1 = x1s - relu_diff  # min(x1s, x2s)
#         y2 = x2s + relu_diff  # max(x1s, x2s)
        
#         sorted_pairs = torch.stack((y1, y2), dim=2)
#         sorted_flat = sorted_pairs.reshape(batch_size, -1)
#         output = sorted_flat.reshape(original_shape)
        
#         return output


# # --- Definition for Model 3: The "Loose" torch.min/max Model ---
# class MaxMin(nn.Module):
#     """
#     Custom activation layer that uses torch.min and torch.max.
    
#     HYPOTHESIS: This will be "loose" because auto_LiRPA will create
#     two independent relaxation nodes (BoundMin, BoundMax).
#     """
#     def forward(self, x):
#         original_shape = x.shape
#         batch_size = original_shape[0]
        
#         num_features = np.prod(original_shape[1:])
#         if num_features % 2 != 0:
#             raise ValueError(
#                 f"The total number of features must be even, but got {num_features} "
#                 f"for shape {original_shape}."
#             )
            
#         x_flat = x.reshape(batch_size, -1)
#         x_pairs = x_flat.reshape(batch_size, -1, 2)

#         a = x_pairs[..., 0]
#         b = x_pairs[..., 1]
        
#         # --- Independent Min/Max Logic ---
#         min_vals = torch.min(a, b) # <-- Node 1
#         max_vals = torch.max(a, b) # <-- Node 2

#         sorted_pairs = torch.stack((min_vals, max_vals), dim=-1)
        
#         sorted_flat = sorted_pairs.reshape(batch_size, -1)
#         return sorted_flat.reshape(original_shape)


# # --- Main Experiment (MODIFIED) ---
# def run_crown_comparison(in_features=10, out_features=10, epsilon=0.1, num_trials=500):
#     """
#     Compares the tightness of CROWN bounds for all three GroupSort models.
#     """
#     # We will store results in three separate lists
#     gs_shared_widths = []
#     maxmin_widths = []
#     gs_independent_widths = []
    
#     if out_features % 2 != 0:
#         raise ValueError("out_features must be an even number for GroupSort(2).")

#     for i in range(num_trials):
#         # 1. Create a single linear layer to be shared for fairness
#         linear_layer = nn.Linear(in_features, out_features)
        
#         # 2. Define the three models, all sharing the same linear layer
#         model_gs_shared = nn.Sequential(linear_layer, GroupSort())
#         model_maxmin = nn.Sequential(linear_layer, MaxMin())
#         model_gs_independent = nn.Sequential(linear_layer, GroupSortIndependentReLUs())
        
#         # 3. Define random input and its perturbation
#         x0 = torch.randn(1, in_features)
#         ptb = PerturbationLpNorm(norm=torch.inf, eps=epsilon)
#         bounded_input = BoundedTensor(x0, ptb)

#         # --- [NEW] Check for forward pass equivalence ---
#         with torch.no_grad(): # Disable gradient tracking for this check
#             output_shared = model_gs_shared(x0)
#             output_maxmin = model_maxmin(x0)
#             output_independent = model_gs_independent(x0)
            
#         # Check that all models produce the same output
#         check1 = torch.allclose(output_shared, output_maxmin, atol=1e-6)
#         check2 = torch.allclose(output_shared, output_independent, atol=1e-6)
        
#         if not (check1 and check2):
#             print(f"--- FAILED equivalence check at trial {i+1} ---")
#             print(f"Max diff (Shared vs MaxMin): {torch.max(torch.abs(output_shared - output_maxmin))}")
#             print(f"Max diff (Shared vs Independent): {torch.max(torch.abs(output_shared - output_independent))}")
#             raise AssertionError("Model outputs do not match! Implementations are not equivalent.")
        
#         # Print confirmation on the first trial only
#         if i == 0:
#             print("✅ Forward pass equivalence check passed for all models.")
#         # --- [END NEW] ---
        
#         # --- Analyze Model 1: GroupSort (Shared ReLU) ---
#         lirpa_model_shared = BoundedModule(model_gs_shared, x0)
#         lb_shared, ub_shared = lirpa_model_shared.compute_bounds(x=(bounded_input,), method='crown')
#         avg_width_shared = torch.mean(ub_shared - lb_shared).item()
#         gs_shared_widths.append(avg_width_shared)
        
#         # --- Analyze Model 2: MaxMin (torch.min/max) ---
#         lirpa_model_maxmin = BoundedModule(model_maxmin, x0)
#         lb_mm, ub_mm = lirpa_model_maxmin.compute_bounds(x=(bounded_input,), method='crown')
#         avg_width_mm = torch.mean(ub_mm - lb_mm).item()
#         maxmin_widths.append(avg_width_mm)
        
#         # --- Analyze Model 3: GroupSort (Independent ReLUs) ---
#         lirpa_model_independent = BoundedModule(model_gs_independent, x0)
#         lb_indep, ub_indep = lirpa_model_independent.compute_bounds(x=(bounded_input,), method='crown')
#         avg_width_indep = torch.mean(ub_indep - lb_indep).item()
#         gs_independent_widths.append(avg_width_indep)
        
#         if (i + 1) % 100 == 0:
#             print(f"Completed trial {i+1}/{num_trials}")
            
#     return gs_shared_widths, maxmin_widths, gs_independent_widths


# # --- Run and Visualize (MODIFIED) ---
# if __name__ == '__main__':
#     # Experiment parameters
#     INPUT_DIM = 20
#     OUTPUT_DIM = 20  # Must be even
#     EPSILON = 0.05   # Input perturbation radius
#     NUM_TRIALS = 1000

#     print("Running CROWN comparison for 3 models using auto_LiRPA...")
#     print("Adding sanity check for forward pass equivalence...")
    
#     results_shared, results_maxmin, results_independent = run_crown_comparison(
#         in_features=INPUT_DIM,
#         out_features=OUTPUT_DIM,
#         epsilon=EPSILON,
#         num_trials=NUM_TRIALS
#     )
#     print("Done.\n")

#     # --- Analysis ---
    
#     # Filter out top 1% outliers for cleaner stats and plots
#     q = 0.99
#     clean_shared = [r for r in results_shared if r < np.quantile(results_shared, q)]
#     clean_maxmin = [g for g in results_maxmin if g < np.quantile(results_maxmin, q)]
#     clean_independent = [i for i in results_independent if i < np.quantile(results_independent, q)]
    
#     mean_shared = np.mean(clean_shared)
#     mean_maxmin = np.mean(clean_maxmin)
#     mean_independent = np.mean(clean_independent)

#     print("--- Average Bound Width Results (lower is better) ---")
#     print(f"GroupSort (1 Shared ReLU)   : {mean_shared:.6f}  <-- TIGHTEST")
#     print(f"GroupSort (2 Indep. ReLUs): {mean_independent:.6f}  <-- LOOSE")
#     print(f"MaxMin (torch.min/max)      : {mean_maxmin:.6f}  <-- LOOSE")
    
#     # Check if the two "loose" models are indeed similar
#     diff_percent = (abs(mean_maxmin - mean_independent) / mean_maxmin) * 100
#     print(f"\nDifference between the two 'loose' models: {diff_percent:.2f}% (expecting < 1-2%)")
    
#     improvement = (1 - mean_shared / mean_maxmin) * 100
#     print(f"📈 The Shared ReLU model is {improvement:.2f}% tighter than the MaxMin model.")


#     # --- Plotting ---
#     plt.style.use('seaborn-v0_8-whitegrid')
#     fig, ax = plt.subplots(figsize=(12, 7))

#     sns.histplot(clean_shared, color="blue", kde=True, 
#                  label=f'GroupSort (Shared ReLU) (Mean: {mean_shared:.4f})', 
#                  ax=ax, bins=50, alpha=0.6, line_kws={'lw': 2.5})
    
#     sns.histplot(clean_maxmin, color="red", kde=True, 
#                  label=f'MaxMin (torch.min/max) (Mean: {mean_maxmin:.4f})', 
#                  ax=ax, bins=50, alpha=0.6, line_kws={'lw': 2.5})
    
#     sns.histplot(clean_independent, color="green", kde=True, 
#                  label=f'GroupSort (Indep. ReLUs) (Mean: {mean_independent:.4f})', 
#                  ax=ax, bins=50, alpha=0.6, line_kws={'lw': 2.5}, linestyle='--')

#     ax.set_title(f'Distribution of CROWN Output Bound Widths ({NUM_TRIALS} trials)', fontsize=16)
#     ax.set_xlabel('Average Bound Width', fontsize=12)
#     ax.set_ylabel('Frequency', fontsize=12)
#     ax.legend()
#     plt.tight_layout()
#     plt.show()

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


# --- Original Models (for reference) ---

class GroupSortIndependentReLUs(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu_for_min = nn.ReLU()
        self.relu_for_max = nn.ReLU()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        batch_size = original_shape[0]
        num_features = np.prod(original_shape[1:])
        if num_features % 2 != 0: raise ValueError("Total features must be even.")
        x_flat = x.reshape(batch_size, -1)
        reshaped_x = x_flat.reshape(batch_size, -1, 2)
        a = reshaped_x[..., 0]
        b = reshaped_x[..., 1]
        diff_for_min = a - b
        relu_diff_min = self.relu_for_min(diff_for_min)
        min_vals = a - relu_diff_min
        diff_for_max = a - b
        relu_diff_max = self.relu_for_max(diff_for_max)
        max_vals = b + relu_diff_max
        sorted_pairs = torch.stack((min_vals, max_vals), dim=2)
        sorted_flat = sorted_pairs.reshape(batch_size, -1)
        output = sorted_flat.reshape(original_shape)
        return output

class GroupSort(nn.Module):
    def __init__(self):
        super(GroupSort, self).__init__()
        self.relu = nn.ReLU()
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
        relu_diff = self.relu(diff)
        y1 = x1s - relu_diff
        y2 = x2s + relu_diff
        sorted_pairs = torch.stack((y1, y2), dim=2)
        sorted_flat = sorted_pairs.reshape(batch_size, -1)
        output = sorted_flat.reshape(original_shape)
        return output

class MaxMin(nn.Module):
    def forward(self, x):
        original_shape = x.shape
        batch_size = original_shape[0]
        num_features = np.prod(original_shape[1:])
        if num_features % 2 != 0: raise ValueError("Total features must be even.")
        x_flat = x.reshape(batch_size, -1)
        x_pairs = x_flat.reshape(batch_size, -1, 2)
        a = x_pairs[..., 0]
        b = x_pairs[..., 1]
        min_vals = torch.min(a, b)
        max_vals = torch.max(a, b)
        sorted_pairs = torch.stack((min_vals, max_vals), dim=-1)
        sorted_flat = sorted_pairs.reshape(batch_size, -1)
        return sorted_flat.reshape(original_shape)


# --- [NEW] Models for the torch.max vs ReLU-formulation test ---

class Model_JustMax(nn.Module):
    """
    Computes *only* the max of pairs.
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
        max_vals = b + self.relu(a - b)
        return max_vals # Output shape is (batch, features / 2)


# --- [NEW] Experiment 2: Compare max(a,b) vs b + ReLU(a-b) ---

def run_max_vs_relu_comparison(in_features=10, out_features=10, epsilon=0.1, num_trials=500):
    """
    Compares the CROWN bounds for torch.max vs its ReLU reformulation.
    """
    max_widths = []
    max_relu_widths = []
    
    if out_features % 2 != 0:
        raise ValueError("out_features must be an even number.")

    for i in range(num_trials):
        # 1. Create a shared linear layer
        linear_layer = nn.Linear(in_features, out_features)
        
        # 2. Define the two models sharing the layer
        model_max = nn.Sequential(linear_layer, GroupSortIndependentReLUs())
        model_max_relu = nn.Sequential(linear_layer, GroupSort())
        
        # 3. Define random input and perturbation
        x0 = torch.randn(1, in_features)
        ptb = PerturbationLpNorm(norm=torch.inf, eps=epsilon)
        bounded_input = BoundedTensor(x0, ptb)

        # --- [NEW] Check for forward pass equivalence ---
        with torch.no_grad():
            output_max = model_max(x0)
            output_max_relu = model_max_relu(x0)
            
        check = torch.allclose(output_max, output_max_relu, atol=1e-6)
        
        if not check:
            print(f"--- FAILED equivalence check at trial {i+1} ---")
            print(f"Max diff: {torch.max(torch.abs(output_max - output_max_relu))}")
            raise AssertionError("Model outputs do not match!")
        
        if i == 0:
            print("✅ Forward pass equivalence check passed for max vs. max-via-ReLU.")
        # --- [END NEW] ---
        
        # --- Analyze Model 1: torch.max ---
        lirpa_model_max = BoundedModule(model_max, x0)
        lb_max, ub_max = lirpa_model_max.compute_bounds(x=(bounded_input,), method='alpha-crown')
        avg_width_max = torch.mean(ub_max - lb_max).item()
        max_widths.append(avg_width_max)
        
        # --- Analyze Model 2: b + ReLU(a-b) ---
        lirpa_model_max_relu = BoundedModule(model_max_relu, x0)
        lb_relu, ub_relu = lirpa_model_max_relu.compute_bounds(x=(bounded_input,), method='alpha-crown')
        avg_width_relu = torch.mean(ub_relu - lb_relu).item()
        max_relu_widths.append(avg_width_relu)
        
        if (i + 1) % 100 == 0:
            print(f"Completed trial {i+1}/{num_trials}")
            
    return max_widths, max_relu_widths


# --- Run and Visualize (MODIFIED to run the new experiment) ---
if __name__ == '__main__':
    # Experiment parameters
    INPUT_DIM = 20
    OUTPUT_DIM = 20  # Must be even
    EPSILON = 0.05   # Input perturbation radius
    NUM_TRIALS = 1000

    print("Running CROWN comparison for torch.max vs. b + ReLU(a-b)...")
    
    results_max, results_max_relu = run_max_vs_relu_comparison(
        in_features=INPUT_DIM,
        out_features=OUTPUT_DIM,
        epsilon=EPSILON,
        num_trials=NUM_TRIALS
    )
    print("Done.\n")

    # --- Analysis ---
    q = 0.99
    clean_max = [r for r in results_max if r < np.quantile(results_max, q)]
    clean_max_relu = [g for g in results_max_relu if g < np.quantile(results_max_relu, q)]
    
    mean_max = np.mean(clean_max)
    mean_max_relu = np.mean(clean_max_relu)

    print("--- Average Bound Width Results (lower is better) ---")
    print(f"Model_JustMax (torch.max)  : {mean_max:.6f}")
    print(f"Model_Max_via_ReLU         : {mean_max_relu:.6f}")
    
    diff_percent = (abs(mean_max - mean_max_relu) / (mean_max + 1e-9)) * 100
    print(f"\nDifference between models: {diff_percent:.4f}%")
    
    if diff_percent < 0.01:
        print("✅ SUCCESS: Bounds are identical, as hypothesized.")
    else:
        print("❌ FAILED: Bounds are different.")


    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    sns.histplot(clean_max, color="red", kde=True, 
                 label=f'torch.max (Mean: {mean_max:.4f})', 
                 ax=ax, bins=50, alpha=0.6, line_kws={'lw': 2.5})
    
    sns.histplot(clean_max_relu, color="purple", kde=True, 
                 label=f'b + ReLU(a-b) (Mean: {mean_max_relu:.4f})', 
                 ax=ax, bins=50, alpha=0.6, line_kws={'lw': 2.5}, linestyle='--')

    ax.set_title(f'Distribution of CROWN Output Bound Widths ({NUM_TRIALS} trials)', fontsize=16)
    ax.set_xlabel('Average Bound Width', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.legend()
    plt.tight_layout()
    plt.show()