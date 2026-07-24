import torch
import torch.nn as nn
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
import copy
import warnings

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. MODEL DEFINITIONS
# ==============================================================================

class GroupSortReLU(nn.Module):
    def __init__(self):
        super(GroupSortReLU, self).__init__()
        self.relu = nn.ReLU() 
    def forward(self, x):
        bs, feats = x.shape
        x_reshaped = x.reshape(bs, -1, 2)
        a, b = x_reshaped[..., 0], x_reshaped[..., 1]
        diff = a - b
        relu_diff = self.relu(diff)
        y_min = a - relu_diff
        y_max = b + relu_diff
        out = torch.stack((y_min, y_max), dim=2)
        return out.reshape(bs, feats)

class GroupSortMax(nn.Module):
    def __init__(self):
        super(GroupSortMax, self).__init__()
    def forward(self, x):
        bs, feats = x.shape
        x_reshaped = x.reshape(bs, -1, 2)
        a, b = x_reshaped[..., 0], x_reshaped[..., 1]
        y_max = torch.max(a, b)
        y_min = torch.min(a, b)
        out = torch.stack((y_min, y_max), dim=2)
        return out.reshape(bs, feats)

# ==============================================================================
# 2. HELPER: Get Matrix (Robust Fix)
# ==============================================================================

def get_lA_tensor(lirpa_model, bounded_input):
    """ 
    Runs CROWN and returns the Lower Bound Slope Matrix (lA).
    Robustly handles auto_LiRPA version differences.
    """
    
    # 1. Robustly access nodes
    if callable(lirpa_model.nodes):
        # If nodes is a method (your error case)
        all_nodes = lirpa_model.nodes() 
        # If it returns a dict, take values; if list, take it as is
        if isinstance(all_nodes, dict):
            all_nodes = all_nodes.values()
    else:
        # If nodes is a property/dict (standard case)
        all_nodes = lirpa_model.nodes.values()

    # 2. Find the Input Node object dynamically
    input_node = None
    
    # First pass: Look for nodes with active perturbations
    for node in all_nodes:
        if hasattr(node, 'perturbation') and node.perturbation is not None:
            input_node = node
            break
    
    # Second pass: Look for BoundInput type if perturbation is not explicitly attached
    if input_node is None:
        # Re-iterate
        if callable(lirpa_model.nodes):
            all_nodes = lirpa_model.nodes() 
            if isinstance(all_nodes, dict): all_nodes = all_nodes.values()
        else:
            all_nodes = lirpa_model.nodes.values()

        for node in all_nodes:
            if type(node).__name__ == 'BoundInput':
                input_node = node
                break
                
    if input_node is None:
        raise ValueError("Could not find a BoundInput node in the model.")
    
    # 3. Define needed_A_dict
    # This tells LiRPA: "Please save the A matrix from Root w.r.t InputNode"
    needed_A_dict = { lirpa_model.roots: [input_node] }

    # 4. Compute Bounds
    lb, ub, A_dict = lirpa_model.compute_bounds(
        x=(bounded_input,), 
        method='crown', 
        return_A=True,
        needed_A_dict=needed_A_dict
    )
    
    # 5. Extract
    try:
        lA = A_dict[lirpa_model.roots][input_node]['lA']
        return lA.squeeze(0) # Remove batch dim
    except KeyError:
        print("ERROR: Could not extract A matrix.")
        return None

# ==============================================================================
# 3. THE EXPERIMENT
# ==============================================================================

def run_simple_decomposition(name, model_class):
    print(f"\n{'='*60}")
    print(f"TESTING: {name}")
    print(f"{'='*60}")

    # Setup
    FEATS = 2
    BATCH = 1
    EPSILON = 0.5
    
    # Correlated Weights: y is correlated with x
    W_val = torch.tensor([[1.0, 0.2], [0.2, 1.0]])
    
    x0 = torch.zeros(BATCH, FEATS)
    ptb = PerturbationLpNorm(norm=torch.inf, eps=EPSILON)
    bounded_input = BoundedTensor(x0, ptb)

    # --- PART A: GLOBAL SLOPE (Linear + Activation) ---
    linear_layer = nn.Linear(FEATS, FEATS, bias=False)
    linear_layer.weight = nn.Parameter(W_val)
    
    model_global = nn.Sequential(linear_layer, model_class())
    lirpa_global = BoundedModule(model_global, x0)
    
    print("1. Computing Global Slope...")
    A_global = get_lA_tensor(lirpa_global, bounded_input)

    # --- PART B: COMPOSED SLOPE (Box_Activation * Linear) ---
    print("2. Computing Composed Slope...")
    
    # 1. Get Concrete Output Bounds of Linear Layer
    model_lin = copy.deepcopy(linear_layer)
    lirpa_lin = BoundedModule(model_lin, x0)
    lb_lin, ub_lin = lirpa_lin.compute_bounds(x=(bounded_input,), method='crown')
    
    # 2. Create a "Box" Input matching those bounds
    center_sev = (lb_lin + ub_lin) / 2.0
    center_sev = center_sev.detach()
    lb_lin = lb_lin.detach()
    ub_lin = ub_lin.detach()
    
    ptb_sev = PerturbationLpNorm(norm=torch.inf, x_L=lb_lin, x_U=ub_lin)
    bounded_input_sev = BoundedTensor(center_sev, ptb_sev)
    
    # 3. Get Slope of Activation on that Box
    model_act = model_class()
    lirpa_act = BoundedModule(model_act, center_sev)
    
    S_box = get_lA_tensor(lirpa_act, bounded_input_sev)
    
    # 4. Multiply manually: A_composed = S_box * W
    A_composed = torch.matmul(S_box, W_val)

    # --- COMPARISON ---
    diff = (A_global - A_composed).abs().sum().item()
    
    slope_g = A_global[1,1].item()
    slope_c = A_composed[1,1].item()

    print(f"\nRESULTS for Dominant Slope (Index 1,1):")
    print(f"Global Slope   : {slope_g:.4f}")
    print(f"Composed Slope : {slope_c:.4f}")
    print(f"Diff (L1 Sum)  : {diff:.6f}")

    if diff < 1e-5:
        print(f"\n>>> VERDICT: EQUAL (Global == Composed)")
        print(f"    {name} ignores symbolic history.")
    else:
        print(f"\n>>> VERDICT: NOT EQUAL (Global != Composed)")
        print(f"    {name} uses symbolic history to sharpen the slope.")

if __name__ == "__main__":
    run_simple_decomposition("Method A (Max)", GroupSortMax)
    run_simple_decomposition("Method B (ReLU)", GroupSortReLU)