import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import Polygon, Rectangle
from collections import OrderedDict
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from scipy.spatial import ConvexHull

# --- Custom Activation Function ---
class GroupSort2(nn.Module):
    """
    A custom activation function that sorts a 2-element vector.
    For an input (x1, x2), the output is (min(x1, x2), max(x1, x2)).
    The input tensor must have its last dimension of size 2.
    This version is reformulated using ReLU operations to be compatible
    with auto_LiRPA.
    """
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        # Reformulate sorting using ReLU identities to be traceable
        # y_min = x1 - ReLU(x1 - x2)
        # y_max = x2 + ReLU(x1 - x2)
        x1 = x[..., 0:1]
        x2 = x[..., 1:2]
        
        relu_diff = self.relu(x1 - x2)
        
        y_min = x1 - relu_diff
        y_max = x2 + relu_diff
        
        return torch.cat([y_min, y_max], dim=-1)

# --- 1. Define the Neural Network Creators ---
def create_relu_network():
    """Creates the PyTorch model with orthogonal layers and ReLU activations."""
    theta = np.pi / 4  # 45-degree rotation
    W1 = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    b1 = np.array([0.0, 0.0])
    theta = np.pi / 3
    W2 = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    b2 = np.array([-0.5, 0.2])

    model = nn.Sequential(OrderedDict([
        ('layer1_linear', nn.Linear(2, 2)),
        ('layer1_relu',   nn.ReLU()),
        ('layer2_linear', nn.Linear(2, 2)),
        ('layer2_relu',   nn.ReLU())
    ]))

    # Load the pre-defined weights
    model.layer1_linear.weight.data = torch.from_numpy(W1).float()
    model.layer1_linear.bias.data = torch.from_numpy(b1).float()
    model.layer2_linear.weight.data = torch.from_numpy(W2).float()
    model.layer2_linear.bias.data = torch.from_numpy(b2).float()
    
    model.eval()
    return model

def create_gs_network():
    """Creates the PyTorch model with orthogonal layers and GroupSort2 activations."""
    theta = np.pi / 4  # 45-degree rotation
    W1 = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    b1 = np.array([0.0, 0.0])
    theta = np.pi / 3
    W2 = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    b2 = np.array([-0.5, 0.2])

    model = nn.Sequential(OrderedDict([
        ('layer1_linear', nn.Linear(2, 2)),
        ('layer1_gs',     GroupSort2()),
        ('layer2_linear', nn.Linear(2, 2)),
        ('layer2_gs',     GroupSort2())
    ]))

    # Load the pre-defined weights
    model.layer1_linear.weight.data = torch.from_numpy(W1).float()
    model.layer1_linear.bias.data = torch.from_numpy(b1).float()
    model.layer2_linear.weight.data = torch.from_numpy(W2).float()
    model.layer2_linear.bias.data = torch.from_numpy(b2).float()
    
    model.eval()
    return model


# --- Geometric Polytope Transformation Functions ---
def apply_relu_to_polytope(vertices):
    """Computes the convex hull of a polytope's vertices after applying ReLU."""
    relu_vertices = np.maximum(0, vertices)
    if len(np.unique(relu_vertices, axis=0)) >= 3:
        hull = ConvexHull(relu_vertices)
        return relu_vertices[hull.vertices]
    else:
        return np.unique(relu_vertices, axis=0)

def apply_gs2_to_polytope(vertices):
    """
    Computes an approximation of the output polytope after applying GroupSort2.
    This is done by sampling a dense cloud of points within the input polytope,
    transforming them, and finding their convex hull.
    """
    if vertices.shape[0] < 3:
        # Not enough points to form a 2D shape, just sort the existing points
        return np.sort(vertices, axis=1)
        
    # 1. Create a grid of sample points within the bounding box of the polytope
    min_coords = vertices.min(axis=0)
    max_coords = vertices.max(axis=0)
    x_samples = np.linspace(min_coords[0], max_coords[0], 100)
    y_samples = np.linspace(min_coords[1], max_coords[1], 100)
    grid_x, grid_y = np.meshgrid(x_samples, y_samples)
    sample_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T

    # 2. Filter the points to keep only those inside the actual polytope
    path = Path(vertices)
    inside_mask = path.contains_points(sample_points)
    points_inside_polytope = sample_points[inside_mask]

    if points_inside_polytope.shape[0] < 3:
        # If very few points are inside, fall back to sorting vertices
        return np.sort(vertices, axis=1)

    # 3. Apply the GroupSort2 transformation to all interior points
    sorted_points = np.sort(points_inside_polytope, axis=1)

    # 4. The new polytope is the convex hull of the transformed points
    hull = ConvexHull(sorted_points)
    return sorted_points[hull.vertices]


def visualize_propagation(model, activation_name, activation_applier, output_filename):
    """
    Main function to run visualization of both true polytopes and ideal bounds.
    """
    epsilon = 1.0
    initial_polytope = np.array([
        [-epsilon, -epsilon], [epsilon, -epsilon],
        [epsilon, epsilon], [-epsilon, epsilon]
    ])
    
    fig, axes = plt.subplots(1, len(model) + 1, figsize=(22, 5))
    fig.suptitle(f"Propagation of Polytopes and Ideal Bounds (Activation: {activation_name})", fontsize=16)

    polytopes = {'current': initial_polytope}

    # Plot Initial State
    ax = axes[0]
    ax.set_title("Input L-inf Ball")
    ax.add_patch(Polygon(polytopes['current'], closed=True, facecolor='lightblue', edgecolor='blue', label='True Polytope'))
    manual_lb = polytopes['current'].min(axis=0)
    manual_ub = polytopes['current'].max(axis=0)
    ax.add_patch(Rectangle(manual_lb, manual_ub[0] - manual_lb[0], manual_ub[1] - manual_lb[1], 
                           facecolor='none', edgecolor='red', linestyle='--', label='Axis-Aligned Bound'))

    # Iterate through layers and plot each step
    for i, (name, layer) in enumerate(model.named_children()):
        ax = axes[i + 1]
        
        if isinstance(layer, nn.Linear):
            W = layer.weight.data.numpy()
            b = layer.bias.data.numpy()
            polytopes['current'] = polytopes['current'] @ W.T + b
            ax.set_title(f"After Linear: {name}")
        else: # Handle activations
            polytopes['current'] = activation_applier(polytopes['current'])
            ax.set_title(f"After {activation_name}: {name}")

        ax.add_patch(Polygon(polytopes['current'], closed=True, facecolor='lightblue', edgecolor='blue', label='True Polytope'))
        
        if polytopes['current'].shape[0] > 0:
            manual_lb = polytopes['current'].min(axis=0)
            manual_ub = polytopes['current'].max(axis=0)
            ax.add_patch(Rectangle(manual_lb, manual_ub[0] - manual_lb[0], manual_ub[1] - manual_lb[1], 
                                   facecolor='none', edgecolor='red', linestyle='--', label='Axis-Aligned Bound'))

    for ax in axes:
        ax.grid(True)
        ax.set_aspect('equal', adjustable='box')
        ax.autoscale_view()
        ax.legend()
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    plt.savefig(output_filename, dpi=300)
    print(f"Geometric visualization saved to {output_filename}")


def run_autolirpa_analysis(model, model_name):
    """
    Runs the numerical analysis using auto_LiRPA and prints the results.
    """
    print("\n" + "="*50)
    print(f"Running auto_LiRPA analysis for {model_name} Network...")
    
    dummy_input = torch.zeros(1, 2)
    bounded_model = BoundedModule(model, global_input=dummy_input)
    epsilon = 1.0
    pt = PerturbationLpNorm(norm=np.inf, eps=epsilon)
    bounded_input = BoundedTensor(dummy_input, pt)

    for method in ["IBP", "CROWN"]:
        print(f"\n--- Computing all bounds with {method} ---")
        try:
            bounded_model.compute_bounds(x=(bounded_input,), method=method)
            for i, (name, layer) in enumerate(model.named_children()):
                node_name = f'/{i}'
                node = bounded_model[node_name]
                lb = node.lower.detach().numpy().flatten()
                ub = node.upper.detach().numpy().flatten()
                print(f"Bounds for node '{node_name}' (Layer: {name}):")
                print(f"  Lower: [{lb[0]:.3f}, {lb[1]:.3f}] | Upper: [{ub[0]:.3f}, {ub[1]:.3f}]")
        except Exception as e:
            print(f"Could not compute intermediate bounds for {method}. Error: {e}")
            print("Printing final bounds only.")
            lb, ub = bounded_model.compute_bounds(x=(bounded_input,), method=method)
            print(f"Final Bounds for {method}:")
            print(f"  Lower: {lb.detach().numpy().flatten()} | Upper: {ub.detach().numpy().flatten()}")


if __name__ == '__main__':
    # You will need to install torch, auto_LiRPA, numpy, matplotlib, and scipy
    # pip install torch auto_LiRPA numpy matplotlib scipy
    
    # --- Generate visualization and analysis for ReLU Network ---
    relu_model = create_relu_network()
    visualize_propagation(relu_model, "ReLU", apply_relu_to_polytope, "polytope_propagation_relu.png")
    run_autolirpa_analysis(relu_model, "ReLU")

    # --- Generate visualization and analysis for GroupSort2 Network ---
    gs_model = create_gs_network()
    visualize_propagation(gs_model, "GroupSort2", apply_gs2_to_polytope, "polytope_propagation_gs2.png")
    run_autolirpa_analysis(gs_model, "GroupSort2")

