import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm

import seaborn as sns
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm

# Define the GroupSort(2) activation as a custom PyTorch module.
# This is necessary for auto_LiRPA to trace and bound it.
class GroupSort(nn.Module):
    """
    A universal, auto_lirpa-compatible PyTorch module that sorts pairs of features.

    This module can handle inputs of any shape (e.g., 2D, 4D). It works by 
    temporarily flattening the feature dimensions, applying the sort logic in a 
    verifier-friendly way (with ReLU on a 2D tensor), and then reshaping the 
    output back to the original input shape.

    The total number of features (product of dimensions after the batch dim) must be even.
    """
    def __init__(self):
        super(GroupSort, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        batch_size = original_shape[0]
        
        num_features = np.prod(original_shape[1:])
        
        if num_features % 2 != 0:
            raise ValueError(
                f"The total number of features must be even, but got {num_features} "
                f"for shape {original_shape}."
            )

        # Utiliser .reshape() pour gérer les tenseurs non-contigus
        x_flat = x.reshape(batch_size, -1)

        # --- Logique de tri ---
        
        # .reshape() est aussi plus sûr ici, par précaution
        reshaped_x = x_flat.reshape(batch_size, -1, 2)
        
        x1s = reshaped_x[..., 0]
        x2s = reshaped_x[..., 1]
        
        diff = x2s - x1s
        relu_diff = self.relu(diff)
        
        y1 = x2s - relu_diff
        y2 = x1s + relu_diff
        
        sorted_pairs = torch.stack((y1, y2), dim=2)
        
        # .reshape() est aussi plus sûr ici
        sorted_flat = sorted_pairs.reshape(batch_size, -1)

        # --- Fin de la logique ---

        # Restaurer la forme originale en utilisant .reshape()
        output = sorted_flat.reshape(original_shape)
        
        return output

class MaxMin(nn.Module):
    """
    Custom activation layer that sorts features in pairs.
    Equivalent to torchlip.GroupSort2() but compatible with auto_LiRPA.
    """
    def forward(self, x):
        # Ensure the last dimension has an even number of features
        # if x.shape[-1] % 2 != 0:
        #     raise ValueError("The last dimension must be even for MaxMin activation.")

        # Reshape the tensor to group the last dimension into pairs
        # (batch_size, ..., num_features) -> (batch_size, ..., num_features/2, 2)
        x_pairs = x.view(*x.shape[:-1], -1, 2)

        # Separate the pairs and compute min and max
        a = x_pairs[..., 0]
        b = x_pairs[..., 1]
        min_vals = torch.min(a, b)
        max_vals = torch.max(a, b)

        # Stack them back together to form sorted pairs
        # The result has min followed by max for each pair
        sorted_pairs = torch.stack((min_vals, max_vals), dim=-1)

        # Reshape back to the original input shape
        return sorted_pairs.view(x.shape)

# --- Main Experiment ---
def run_crown_comparison(in_features=10, out_features=10, epsilon=0.1, num_trials=500):
    """
    Compares the tightness of CROWN bounds for ReLU and GroupSort activations.
    """
    relu_widths = []
    groupsort_widths = []
    
    if out_features % 2 != 0:
        raise ValueError("out_features must be an even number for GroupSort(2).")

    for i in range(num_trials):
        # 1. Create a single linear layer to be shared for fairness
        linear_layer = nn.Linear(in_features, out_features)
        
        # 2. Define the two models
        model_relu = nn.Sequential(linear_layer, MaxMin())
        # nn.ReLU()
        model_groupsort = nn.Sequential(linear_layer, GroupSort())
        
        # 3. Define random input and its perturbation
        x0 = torch.randn(1, in_features)
        
        # Define the L_infinity perturbation
        ptb = PerturbationLpNorm(norm=torch.inf, eps=epsilon)
        bounded_input = BoundedTensor(x0, ptb)
        
        # --- Analyze ReLU Network ---
        lirpa_model_relu = BoundedModule(model_relu, x0)
        # We specify method='CROWN' to avoid using IBP-CROWN
        lb_relu, ub_relu = lirpa_model_relu.compute_bounds(x=(bounded_input,), method='crown')
        avg_width_relu = torch.mean(ub_relu - lb_relu).item()
        relu_widths.append(avg_width_relu)
        
        # --- Analyze GroupSort Network ---
        lirpa_model_gs = BoundedModule(model_groupsort, x0)
        lb_gs, ub_gs = lirpa_model_gs.compute_bounds(x=(bounded_input,), method='crown')
        avg_width_gs = torch.mean(ub_gs - lb_gs).item()
        groupsort_widths.append(avg_width_gs)
        
        if (i + 1) % 100 == 0:
            print(f"Completed trial {i+1}/{num_trials}")
            
    return relu_widths, groupsort_widths

# --- Run and Visualize ---
if __name__ == '__main__':
    # Experiment parameters
    INPUT_DIM = 20
    OUTPUT_DIM = 20  # Must be even for GroupSort(2)
    EPSILON = 0.05   # Input perturbation radius
    NUM_TRIALS = 1000

    print("Running CROWN comparison using auto_LiRPA...")
    relu_results, groupsort_results = run_crown_comparison(
        in_features=INPUT_DIM,
        out_features=OUTPUT_DIM,
        epsilon=EPSILON,
        num_trials=NUM_TRIALS
    )
    print("Done.\n")

    # --- Analysis ---
    mean_relu = np.mean(relu_results)
    mean_gs = np.mean(groupsort_results)
    
    # Filter out potential outliers for cleaner stats if necessary
    # (e.g., cases where bounds might be exceptionally large)
    clean_relu = [r for r in relu_results if r < np.quantile(relu_results, 0.99)]
    clean_gs = [g for g in groupsort_results if g < np.quantile(groupsort_results, 0.99)]
    clean_mean_relu = np.mean(clean_relu)
    clean_mean_gs = np.mean(clean_gs)


    print("--- Average Bound Width Results (lower is better) ---")
    print(f"ReLU      : {clean_mean_relu:.6f}")
    print(f"GroupSort : {clean_mean_gs:.6f}")
    
    if clean_mean_gs < clean_mean_relu:
        improvement = (1 - clean_mean_gs / clean_mean_relu) * 100
        print(f"\n📈 GroupSort produced CROWN bounds that were on average {improvement:.2f}% tighter than ReLU.")
    else:
        improvement = (1 - clean_mean_relu / clean_mean_gs) * 100
        print(f"\n📉 ReLU produced CROWN bounds that were on average {improvement:.2f}% tighter than GroupSort.")


    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.histplot(clean_relu, color="skyblue", kde=True, label=f'ReLU (Mean: {clean_mean_relu:.4f})', ax=ax, bins=50)
    sns.histplot(clean_gs, color="red", kde=True, label=f'GroupSort(2) (Mean: {clean_mean_gs:.4f})', ax=ax, bins=50)

    ax.set_title(f'Distribution of CROWN Output Bound Widths ({NUM_TRIALS} trials)', fontsize=16)
    ax.set_xlabel('Average Bound Width', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.legend()
    plt.tight_layout()
    plt.show()
    