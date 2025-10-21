import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm



import torch
import torch.nn as nn
from torch.autograd import Function

# Step 1: Define the custom autograd function.
# This makes 'NativeGroupSort' a primitive operation in the computation graph.
class NativeGroupSortFunction(Function):
    @staticmethod
    def forward(ctx, x):
        # The forward pass is simple: just sort pairs of features.
        original_shape = x.shape
        x_reshaped = x.view(-1, 2)
        sorted_x, _ = torch.sort(x_reshaped, dim=-1)
        return sorted_x.view(original_shape)

    @staticmethod
    def backward(ctx, grad_output):
        # The backward pass is not needed for auto_LiRPA's CROWN,
        # but it's good practice to have for standard training.
        # This can be left as None if not training through this model.
        return None

# Step 2: Wrap the function in an nn.Module for easy use in models.
class NativeGroupSort(nn.Module):
    def forward(self, x):
        return NativeGroupSortFunction.apply(x)
    

from auto_LiRPA.bound_ops import Bound

# This is the handler function that computes the bounds for our custom op.
def bound_native_groupsort(self, A=None, C=None, IBP=False, forward=False):
    if IBP or forward:
        # For IBP, we can simply sort the lower and upper bounds.
        # This is a basic but valid bounding method.
        lower, upper = self.inputs[0].lower, self.inputs[0].upper
        lower_reshaped = lower.view(-1, 2)
        upper_reshaped = upper.view(-1, 2)
        # Sorting the bounds gives valid new bounds for the sorted outputs.
        # Note: This step still has the dependency issue we discussed, but
        # it's the CROWN part below that provides the real advantage.
        self.lower = torch.sort(lower_reshaped, dim=-1)[0].view(lower.shape)
        self.upper = torch.sort(upper_reshaped, dim=-1)[0].view(upper.shape)

    # --- CROWN Backpropagation Logic ---
    if A is not None:
        # A is the matrix of coefficients from the next layer that we are backpropagating.
        # Its shape is (batch, num_outputs, num_neurons_in_this_layer).
        
        # Get the concrete bounds of the input to this layer.
        l, u = self.inputs[0].lower, self.inputs[0].upper
        l_reshaped = l.view(l.shape[0], -1, 2)
        u_reshaped = u.view(u.shape[0], -1, 2)
        
        # The backpropagated coefficients A also need to be paired up.
        # A has shape (spec, out_dim, num_features). We reshape num_features.
        A_reshaped = A.view(A.shape[0], A.shape[1], -1, 2)
        
        # We need to compute the new A matrix for the previous layer.
        A_new = torch.zeros_like(A_reshaped)

        # Coefficients for the sorted outputs (y1, y2)
        A_y1 = A_reshaped[..., 0]
        A_y2 = A_reshaped[..., 1]

        # Case 1: No overlap, x1 is always smaller than x2 (u1 <= l2)
        # The sort is an identity operation.
        case1 = u_reshaped[..., 0] <= l_reshaped[..., 1]
        A_new[..., 0][case1] = A_y1[case1]
        A_new[..., 1][case1] = A_y2[case1]

        # Case 2: No overlap, x2 is always smaller than x1 (u2 <= l1)
        # The sort is a permutation.
        case2 = u_reshaped[..., 1] <= l_reshaped[..., 0]
        A_new[..., 0][case2] = A_y2[case2]
        A_new[..., 1][case2] = A_y1[case2]

        # Case 3: The intervals overlap. This is the tricky part.
        # We must take a convex relaxation.
        # The key property is that the sum is preserved: y1+y2 = x1+x2
        # This implies that the sum of coefficients is also preserved.
        case3 = ~case1 & ~case2
        
        # For the overlapping case, the tightest relaxation is:
        # The new coefficient for x1 is min(A_y1, A_y2)
        # The new coefficient for x2 is max(A_y1, A_y2)
        # This is for the upper bound. For the lower bound, it's reversed.
        # To handle both, we check the sign of A. We can simplify by noting
        # the relationship is a permutation that minimizes the dot product.
        # The new coefficients are simply the sorted old ones.
        A_new[..., 0][case3] = torch.min(A_y1, A_y2)[case3]
        A_new[..., 1][case3] = torch.max(A_y1, A_y2)[case3]
        
        # Reshape A_new back to its original shape and return
        return A_new.view_as(A), None
import seaborn as sns
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm

# Register our custom op handler in auto_LiRPA's dictionary.
# This is the magic that links the function to the bounding logic.
BoundedModule.bound_op_map[NativeGroupSortFunction] = bound_native_groupsort

def run_final_comparison(in_features=10, out_features=10, epsilon=0.1, num_trials=500):
    relu_widths = []
    groupsort_widths = []

    for i in range(num_trials):
        linear_layer = nn.Linear(in_features, out_features)
        
        # Model 1: Standard ReLU
        model_relu = nn.Sequential(linear_layer, nn.ReLU())
        
        # Model 2: Our Native GroupSort!
        model_groupsort = nn.Sequential(linear_layer, NativeGroupSort())
        
        x0 = torch.randn(1, in_features)
        ptb = PerturbationLpNorm(norm=torch.inf, eps=epsilon)
        bounded_input = BoundedTensor(x0, ptb)
        
        # --- Analyze ReLU Network ---
        lirpa_model_relu = BoundedModule(model_relu, x0)
        lb_relu, ub_relu = lirpa_model_relu.compute_bounds(x=(bounded_input,), method='CROWN')
        relu_widths.append(torch.mean(ub_relu - lb_relu).item())
        
        # --- Analyze Native GroupSort Network ---
        lirpa_model_gs = BoundedModule(model_groupsort, x0)
        lb_gs, ub_gs = lirpa_model_gs.compute_bounds(x=(bounded_input,), method='CROWN')
        groupsort_widths.append(torch.mean(ub_gs - lb_gs).item())

    return relu_widths, groupsort_widths    

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
        model_relu = nn.Sequential(linear_layer, nn.ReLU())
        model_groupsort = nn.Sequential(linear_layer, GroupSort())
        
        # 3. Define random input and its perturbation
        x0 = torch.randn(1, in_features)
        
        # Define the L_infinity perturbation
        ptb = PerturbationLpNorm(norm=torch.inf, eps=epsilon)
        bounded_input = BoundedTensor(x0, ptb)
        
        # --- Analyze ReLU Network ---
        lirpa_model_relu = BoundedModule(model_relu, x0)
        # We specify method='CROWN' to avoid using IBP-CROWN
        lb_relu, ub_relu = lirpa_model_relu.compute_bounds(x=(bounded_input,), method='CROWN')
        avg_width_relu = torch.mean(ub_relu - lb_relu).item()
        relu_widths.append(avg_width_relu)
        
        # --- Analyze GroupSort Network ---
        lirpa_model_gs = BoundedModule(model_groupsort, x0)
        lb_gs, ub_gs = lirpa_model_gs.compute_bounds(x=(bounded_input,), method='CROWN')
        avg_width_gs = torch.mean(ub_gs - lb_gs).item()
        groupsort_widths.append(avg_width_gs)
        
        if (i + 1) % 100 == 0:
            print(f"Completed trial {i+1}/{num_trials}")
            
    return relu_widths, groupsort_widths

# --- Run and Visualize ---
if __name__ == '__main__':
    # # Experiment parameters
    # INPUT_DIM = 20
    # OUTPUT_DIM = 20  # Must be even for GroupSort(2)
    # EPSILON = 0.05   # Input perturbation radius
    # NUM_TRIALS = 1000

    # print("Running CROWN comparison using auto_LiRPA...")
    # relu_results, groupsort_results = run_crown_comparison(
    #     in_features=INPUT_DIM,
    #     out_features=OUTPUT_DIM,
    #     epsilon=EPSILON,
    #     num_trials=NUM_TRIALS
    # )
    # print("Done.\n")

    # # --- Analysis ---
    # mean_relu = np.mean(relu_results)
    # mean_gs = np.mean(groupsort_results)
    
    # # Filter out potential outliers for cleaner stats if necessary
    # # (e.g., cases where bounds might be exceptionally large)
    # clean_relu = [r for r in relu_results if r < np.quantile(relu_results, 0.99)]
    # clean_gs = [g for g in groupsort_results if g < np.quantile(groupsort_results, 0.99)]
    # clean_mean_relu = np.mean(clean_relu)
    # clean_mean_gs = np.mean(clean_gs)


    # print("--- Average Bound Width Results (lower is better) ---")
    # print(f"ReLU      : {clean_mean_relu:.6f}")
    # print(f"GroupSort : {clean_mean_gs:.6f}")
    
    # if clean_mean_gs < clean_mean_relu:
    #     improvement = (1 - clean_mean_gs / clean_mean_relu) * 100
    #     print(f"\n📈 GroupSort produced CROWN bounds that were on average {improvement:.2f}% tighter than ReLU.")
    # else:
    #     improvement = (1 - clean_mean_relu / clean_mean_gs) * 100
    #     print(f"\n📉 ReLU produced CROWN bounds that were on average {improvement:.2f}% tighter than GroupSort.")


    # # --- Plotting ---
    # plt.style.use('seaborn-v0_8-whitegrid')
    # fig, ax = plt.subplots(figsize=(10, 6))

    # sns.histplot(clean_relu, color="skyblue", kde=True, label=f'ReLU (Mean: {clean_mean_relu:.4f})', ax=ax, bins=50)
    # sns.histplot(clean_gs, color="red", kde=True, label=f'GroupSort(2) (Mean: {clean_mean_gs:.4f})', ax=ax, bins=50)

    # ax.set_title(f'Distribution of CROWN Output Bound Widths ({NUM_TRIALS} trials)', fontsize=16)
    # ax.set_xlabel('Average Bound Width', fontsize=12)
    # ax.set_ylabel('Frequency', fontsize=12)
    # ax.legend()
    # plt.tight_layout()
    # plt.show()
    INPUT_DIM = 20
    OUTPUT_DIM = 20
    EPSILON = 0.05
    NUM_TRIALS = 1000

    print("Running final comparison with NATIVE GroupSort handler...")
    relu_results, groupsort_results = run_final_comparison(
        in_features=INPUT_DIM, out_features=OUTPUT_DIM, epsilon=EPSILON, num_trials=NUM_TRIALS
    )
    print("Done.\n")
    
    mean_relu = np.mean(relu_results)
    mean_gs = np.mean(groupsort_results)

    print("--- Average Bound Width Results (lower is better) ---")
    print(f"ReLU             : {mean_relu:.6f}")
    print(f"Native GroupSort : {mean_gs:.6f}")
    
    if mean_gs < mean_relu:
        improvement = (1 - mean_gs / mean_relu) * 100
        print(f"\n✅ Native GroupSort produced CROWN bounds that were on average {improvement:.2f}% tighter than ReLU.")
    else:
        print("\n❌ Something is unexpected. ReLU bounds were tighter.")

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.histplot(relu_results, color="skyblue", kde=True, label=f'ReLU (Mean: {mean_relu:.4f})', bins=50)
    sns.histplot(groupsort_results, color="violet", kde=True, label=f'Native GroupSort (Mean: {mean_gs:.4f})', bins=50)
    plt.title('Distribution of CROWN Bound Widths: Native Handler', fontsize=16)
    plt.xlabel('Average Bound Width')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()