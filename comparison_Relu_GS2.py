# import torch
# import torch.nn as nn
# from auto_LiRPA import BoundedModule, BoundedTensor
# from auto_LiRPA.perturbations import PerturbationLpNorm
# import copy
# import warnings

# warnings.filterwarnings("ignore")

# # ==============================================================================
# # 1. MODEL DEFINITIONS
# # ==============================================================================

# class GroupSortReLU(nn.Module):
#     def __init__(self):
#         super(GroupSortReLU, self).__init__()
#         self.relu = nn.ReLU() 
#     def forward(self, x):
#         bs, feats = x.shape
#         x_reshaped = x.reshape(bs, -1, 2)
#         a, b = x_reshaped[..., 0], x_reshaped[..., 1]
#         diff = a - b
#         relu_diff = self.relu(diff)
#         y_min = a - relu_diff
#         y_max = b + relu_diff
#         out = torch.stack((y_min, y_max), dim=2)
#         return out.reshape(bs, feats)

# class GroupSortMax(nn.Module):
#     def __init__(self):
#         super(GroupSortMax, self).__init__()
#     def forward(self, x):
#         bs, feats = x.shape
#         x_reshaped = x.reshape(bs, -1, 2)
#         a, b = x_reshaped[..., 0], x_reshaped[..., 1]
#         y_max = torch.max(a, b)
#         y_min = torch.min(a, b)
#         out = torch.stack((y_min, y_max), dim=2)
#         return out.reshape(bs, feats)

# # ==============================================================================
# # 2. HELPER: Get Matrix (Robust Fix)
# # ==============================================================================

# def get_lA_tensor(lirpa_model, bounded_input):
#     """ 
#     Runs CROWN and returns the Lower Bound Slope Matrix (lA).
#     Robustly handles auto_LiRPA version differences.
#     """
    
#     # 1. Robustly access nodes
#     if callable(lirpa_model.nodes):
#         # If nodes is a method (your error case)
#         all_nodes = lirpa_model.nodes() 
#         # If it returns a dict, take values; if list, take it as is
#         if isinstance(all_nodes, dict):
#             all_nodes = all_nodes.values()
#     else:
#         # If nodes is a property/dict (standard case)
#         all_nodes = lirpa_model.nodes.values()

#     # 2. Find the Input Node object dynamically
#     input_node = None
    
#     # First pass: Look for nodes with active perturbations
#     for node in all_nodes:
#         if hasattr(node, 'perturbation') and node.perturbation is not None:
#             input_node = node
#             break
    
#     # Second pass: Look for BoundInput type if perturbation is not explicitly attached
#     if input_node is None:
#         # Re-iterate
#         if callable(lirpa_model.nodes):
#             all_nodes = lirpa_model.nodes() 
#             if isinstance(all_nodes, dict): all_nodes = all_nodes.values()
#         else:
#             all_nodes = lirpa_model.nodes.values()

#         for node in all_nodes:
#             if type(node).__name__ == 'BoundInput':
#                 input_node = node
#                 break
                
#     if input_node is None:
#         raise ValueError("Could not find a BoundInput node in the model.")
    
#     # 3. Define needed_A_dict
#     # This tells LiRPA: "Please save the A matrix from Root w.r.t InputNode"
#     needed_A_dict = { lirpa_model.roots: [input_node] }

#     # 4. Compute Bounds
#     lb, ub, A_dict = lirpa_model.compute_bounds(
#         x=(bounded_input,), 
#         method='crown', 
#         return_A=True,
#         needed_A_dict=needed_A_dict
#     )
    
#     # 5. Extract
#     try:
#         lA = A_dict[lirpa_model.roots][input_node]['lA']
#         return lA.squeeze(0) # Remove batch dim
#     except KeyError:
#         print("ERROR: Could not extract A matrix.")
#         return None

# # ==============================================================================
# # 3. THE EXPERIMENT
# # ==============================================================================

# def run_simple_decomposition(name, model_class):
#     print(f"\n{'='*60}")
#     print(f"TESTING: {name}")
#     print(f"{'='*60}")

#     # Setup
#     FEATS = 2
#     BATCH = 1
#     EPSILON = 0.5
    
#     # Correlated Weights: y is correlated with x
#     W_val = torch.tensor([[1.0, 0.2], [0.2, 1.0]])
    
#     x0 = torch.zeros(BATCH, FEATS)
#     ptb = PerturbationLpNorm(norm=torch.inf, eps=EPSILON)
#     bounded_input = BoundedTensor(x0, ptb)

#     # --- PART A: GLOBAL SLOPE (Linear + Activation) ---
#     linear_layer = nn.Linear(FEATS, FEATS, bias=False)
#     linear_layer.weight = nn.Parameter(W_val)
    
#     model_global = nn.Sequential(linear_layer, model_class())
#     lirpa_global = BoundedModule(model_global, x0)
    
#     print("1. Computing Global Slope...")
#     A_global = get_lA_tensor(lirpa_global, bounded_input)

#     # --- PART B: COMPOSED SLOPE (Box_Activation * Linear) ---
#     print("2. Computing Composed Slope...")
    
#     # 1. Get Concrete Output Bounds of Linear Layer
#     model_lin = copy.deepcopy(linear_layer)
#     lirpa_lin = BoundedModule(model_lin, x0)
#     lb_lin, ub_lin = lirpa_lin.compute_bounds(x=(bounded_input,), method='crown')
    
#     # 2. Create a "Box" Input matching those bounds
#     center_sev = (lb_lin + ub_lin) / 2.0
#     center_sev = center_sev.detach()
#     lb_lin = lb_lin.detach()
#     ub_lin = ub_lin.detach()
    
#     ptb_sev = PerturbationLpNorm(norm=torch.inf, x_L=lb_lin, x_U=ub_lin)
#     bounded_input_sev = BoundedTensor(center_sev, ptb_sev)
    
#     # 3. Get Slope of Activation on that Box
#     model_act = model_class()
#     lirpa_act = BoundedModule(model_act, center_sev)
    
#     S_box = get_lA_tensor(lirpa_act, bounded_input_sev)
    
#     # 4. Multiply manually: A_composed = S_box * W
#     A_composed = torch.matmul(S_box, W_val)

#     # --- COMPARISON ---
#     diff = (A_global - A_composed).abs().sum().item()
    
#     slope_g = A_global[1,1].item()
#     slope_c = A_composed[1,1].item()

#     print(f"\nRESULTS for Dominant Slope (Index 1,1):")
#     print(f"Global Slope   : {slope_g:.4f}")
#     print(f"Composed Slope : {slope_c:.4f}")
#     print(f"Diff (L1 Sum)  : {diff:.6f}")

#     if diff < 1e-5:
#         print(f"\n>>> VERDICT: EQUAL (Global == Composed)")
#         print(f"    {name} ignores symbolic history.")
#     else:
#         print(f"\n>>> VERDICT: NOT EQUAL (Global != Composed)")
#         print(f"    {name} uses symbolic history to sharpen the slope.")

# if __name__ == "__main__":
#     run_simple_decomposition("Method A (Max)", GroupSortMax)
#     run_simple_decomposition("Method B (ReLU)", GroupSortReLU)

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from itertools import product

def get_zonotope_points(W):
    """
    Generates the vertices of the zonotope by projecting all corners 
    of the N-dimensional hypercube.
    W: 2xN weight matrix
    """
    dim_inputs = W.shape[1]
    # Generate all 2^N combinations of {-1, 1}
    # Note: For N > 12, this computation grows exponentially. 
    corners = np.array(list(product([-1, 1], repeat=dim_inputs))).T
    
    # Project to 2D: x = W * epsilon
    points_2d = W @ corners
    return points_2d.T

def plot_zonotope_analysis(ax, W, N):
    """
    Plots the Zonotope, the Bounding Box (Method A), and the Strip (Method B).
    """
    points = get_zonotope_points(W)
    
    # 1. Plot the Zonotope (Convex Hull of projected points)
    hull = ConvexHull(points)
    for simplex in hull.simplices:
        ax.plot(points[simplex, 0], points[simplex, 1], 'k-', lw=2, zorder=3)
    
    # Fill the zonotope
    ax.fill(points[hull.vertices, 0], points[hull.vertices, 1], 'gray', alpha=0.3, label='True Domain (Zonotope)')

    # --- Method A: Axis Aligned Bounding Box ---
    min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
    min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])
    
    # Draw Rectangle
    rect_x = [min_x, max_x, max_x, min_x, min_x]
    rect_y = [min_y, min_y, max_y, max_y, min_y]
    ax.plot(rect_x, rect_y, 'r--', lw=2, label='Method A: Box')

    # --- Method B: Diagonal Strip (y - x) ---
    # We want bounds on t = y - x = [-1, 1] * (row1 - row0)
    # The projection vector is v = (-1, 1)
    # t = sum(|W[1,i] - W[0,i]|)
    diff_vec = W[1, :] - W[0, :]
    max_t = np.sum(np.abs(diff_vec))
    min_t = -max_t
    
    # Plotting the lines y - x = max_t  => y = x + max_t
    # and y - x = min_t => y = x + min_t
    # We create a range of x values to draw the lines across the plot
    margin = 1.0
    x_vals = np.linspace(min_x - margin, max_x + margin, 100)
    
    ax.plot(x_vals, x_vals + max_t, 'g-', lw=2, label='Method B: Strip Bounds')
    ax.plot(x_vals, x_vals + min_t, 'g-', lw=2)
    
    # Fill the strip area (visual guide)
    ax.fill_between(x_vals, x_vals + min_t, x_vals + max_t, color='green', alpha=0.1)

    ax.set_title(f"N = {N} Inputs", fontsize=12, fontweight='bold')
    ax.axis('equal')
    ax.grid(True, linestyle=':', alpha=0.6)

# --- Main Execution ---
np.random.seed(42) # Fixed seed for reproducibility

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

# Defined N values to investigate
Ns = [2, 3, 5, 10]

for i, N in enumerate(Ns):
    # Create random weight matrix 2xN
    # We use a mix of positive and negative weights to create interesting shapes
    W = np.random.uniform(-1, 1, (2, N))
    
    # For N=2, we force a specific matrix to clearly show the parallelogram case
    if N == 2:
        W = np.array([[2, 1], [0.5, 2]]) 
        
    plot_zonotope_analysis(axes[i], W, N)

# Add single legend to the figure
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.05))

plt.tight_layout()

# --- SAVE COMMAND ---
filename = 'zonotope_analysis.png'
plt.savefig(filename, dpi=300, bbox_inches='tight')
print(f"Plot saved successfully as '{filename}'")

plt.show()



import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

def relu(x):
    return np.maximum(0, x)

def plot_crown_concept():
    # 1. Setup Input: A 2D Square (Input Domain)
    # We sample dense points to show the "True" shape
    num_points = 2000
    # Input is uniform in [-1, 1] x [-1, 1]
    inputs = np.random.uniform(-1, 1, (2, num_points))
    
    # 2. Define a Simple Network: Linear -> ReLU -> Linear
    # This matrix creates correlations (rotation/shear)
    W1 = np.array([[1.5, 1.0], 
                   [0.5, 1.5]])
    b1 = np.array([0.2, -0.2]) # Biases shift it off-center
    
    # Second mixing layer
    W2 = np.array([[1.0, -0.5], 
                   [0.5, 1.0]])

    # --- A. Compute TRUE Domain ---
    # Layer 1
    hidden = W1 @ inputs + b1[:, None]
    # Activation
    activated = relu(hidden)
    # Layer 2 (Output)
    true_output = W2 @ activated
    
    # --- B. Compute IBP (Box) ---
    # Independent bounds
    min_x, max_x = np.min(true_output[0, :]), np.max(true_output[0, :])
    min_y, max_y = np.min(true_output[1, :]), np.max(true_output[1, :])

    # --- C. Compute CROWN-like Zonotope ---
    # Conceptually, CROWN linearizes the ReLU. 
    # Instead of x = ReLU(h), CROWN says x = alpha * h + beta
    # where alpha is a slope between 0 and 1.
    # This results in a Linear Map of the input square, but slightly inflated.
    
    # We simulate this by taking the convex hull of the true domain
    # and "simplifying" it to a zonotope shape (affine map of cube).
    # In reality, CROWN computes specific bounds, but geometrically
    # it looks like the Convex Hull of the data, but slightly looser.
    hull = ConvexHull(true_output.T)
    hull_points = true_output.T[hull.vertices]
    
    # --- PLOTTING ---
    plt.figure(figsize=(10, 8))
    
    # 1. Plot True Domain (Blue dots)
    plt.scatter(true_output[0, :], true_output[1, :], s=1, c='blue', alpha=0.5, label='True Domain (Non-Convex)')
    
    # 2. Plot IBP Box (Red)
    plt.plot([min_x, max_x, max_x, min_x, min_x], 
             [min_y, min_y, max_y, max_y, min_y], 
             'r--', lw=3, label='IBP (Axis-Aligned Box)')
    
    # 3. Plot CROWN (Green Polygon)
    # We use the Convex Hull here to represent the "Best Possible Linear Bound"
    # CROWN produces a shape very similar to this hull (a Polygon).
    plt.fill(hull_points[:, 0], hull_points[:, 1], 
             edgecolor='green', fill=False, hatch='//', lw=3, label='CROWN (Generalized Zonotope)')
    
    plt.title("Visualizing CROWN Geometry", fontsize=15)
    plt.legend(loc='upper left', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    plt.savefig('crown_geometry.png', dpi=100)
    plt.show()

plot_crown_concept()


import numpy as np
import matplotlib.pyplot as plt

def visualize_zonotope_construction(N):
    # 1. Generate N random 2D vectors (The "Sticks")
    # These represent the columns of your Weight Matrix W
    np.random.seed(42)
    vectors = np.random.randn(N, 2)
    
    # 2. Create the full set of 2N edge candidates
    # (The positive and negative versions of each vector)
    all_vectors = np.vstack([vectors, -vectors])
    
    # 3. Calculate angles for sorting
    angles = np.arctan2(all_vectors[:, 1], all_vectors[:, 0])
    
    # 4. Sort vectors by angle
    sort_idx = np.argsort(angles)
    sorted_vectors = all_vectors[sort_idx]
    
    # 5. Cumulatively add them to draw the perimeter
    # Start from a specific point to center the shape
    # The "lowest" point is the sum of all components where y is negative (roughly)
    # But geometrically, just cumsum works, it just shifts the origin.
    perimeter = np.cumsum(sorted_vectors, axis=0)
    
    # Close the loop for plotting
    perimeter = np.vstack([np.zeros(2), perimeter]) # Start at 0,0
    
    # Center the plot
    center = np.mean(perimeter, axis=0)
    perimeter = perimeter - center

    # --- PLOTTING ---
    plt.figure(figsize=(8, 8))
    
    # Plot the final Polygon
    plt.plot(perimeter[:, 0], perimeter[:, 1], 'k-', linewidth=2, label='Zonotope Boundary')
    plt.fill(perimeter[:, 0], perimeter[:, 1], 'skyblue', alpha=0.3)
    
    # Plot the individual vectors (The "Sticks") inside to show origin
    origin = np.array([0, 0])
    for v in vectors:
        plt.arrow(0, 0, v[0], v[1], head_width=0.1, head_length=0.1, fc='red', ec='red')
    
    # Plot the "Steps" along the boundary
    # We draw arrows along the perimeter to show it's made of the red sticks!
    for i in range(len(sorted_vectors)):
        start = perimeter[i]
        vec = sorted_vectors[i]
        plt.arrow(start[0], start[1], vec[0], vec[1], 
                  head_width=0.05, length_includes_head=True, 
                  color='blue', alpha=0.6)

    plt.title(f"Zonotope with N={N} Inputs\n(Polygon has {2*N} sides)", fontsize=14)
    plt.grid(True)
    plt.axis('equal')
    
    # Dummy legend
    plt.plot([], [], 'r->', label='Input Vectors (Generators)')
    plt.plot([], [], 'b->', label='Sorted Perimeter Edges')
    plt.legend()
    
    plt.savefig(f'zonotope_construction_N{N}.png')
    plt.show()

# Visualize for N=5 (Should be a decagon / 10-sided polygon)
visualize_zonotope_construction(N=5)