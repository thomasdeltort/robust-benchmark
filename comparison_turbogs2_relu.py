import torch
import torch.nn as nn
import numpy as np
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
import torch.optim as optim # We only need this for 'sgd'
import warnings

# Suppress auto_LiRPA warnings for cleaner output
warnings.filterwarnings("ignore")

# --- Model Definitions ---

class GroupSort(nn.Module):
    # This is VERIFIABLE because it only uses linear ops and nn.ReLU
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
        y1_min = x1s - relu_diff
        y2_max = x2s + relu_diff
        sorted_pairs = torch.stack((y1_min, y2_max), dim=2)
        sorted_flat = sorted_pairs.reshape(batch_size, -1)
        output = sorted_flat.reshape(original_shape)
        return output

class GroupSort2Optimized(nn.Module):
    # THIS IMPLEMENTATION IS NOT VERIFIABLE WITH auto_LiRPA
    # due to torch.max(a, b)
    def __init__(self):
        super(GroupSort2Optimized, self).__init__()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        batch_size = original_shape[0]
        num_features = np.prod(original_shape[1:])
        if num_features % 2 != 0: raise ValueError("Total features must be even.")
        x_flat = x.reshape(batch_size, -1)
        reshaped_x = x_flat.reshape(batch_size, -1, 2)
        x1s = reshaped_x[..., 0]
        x2s = reshaped_x[..., 1]
        y2_max = torch.max(x1s, x2s) # <--- THIS IS THE UNSUPPORTED OPERATION
        y1_min = x1s + x2s - y2_max
        sorted_pairs = torch.stack((y1_min, y2_max), dim=2)
        sorted_flat = sorted_pairs.reshape(batch_size, -1)
        output = sorted_flat.reshape(original_shape)
        return output

class MaxMin(nn.Module):
    # THIS IMPLEMENTATION IS ALSO NOT VERIFIABLE
    def forward(self, x):
        original_shape = x.shape
        batch_size = original_shape[0]
        num_features = np.prod(original_shape[1:])
        if num_features % 2 != 0: raise ValueError("Total features must be even.")
        x_flat = x.reshape(batch_size, -1)
        x_pairs = x_flat.reshape(batch_size, -1, 2)
        a_tensor, b_tensor = torch.split(x_pairs, 1, dim=-1)
        a = a_tensor.squeeze(-1)
        b = b_tensor.squeeze(-1)
        min_vals = -torch.max(-a, -b) # <--- UNSUPPORTED
        max_vals = torch.max(a, b)     # <--- UNSUPPORTED
        sorted_pairs = torch.stack((min_vals, max_vals), dim=-1)
        sorted_flat = sorted_pairs.reshape(batch_size, -1)
        return sorted_flat.reshape(original_shape)

# --- Helper function for printing ---
def print_comparison_table(title, model_names_map, results_dict, methods):
    """Prints a clean, formatted table of the results."""
    print("\n" + "="*80)
    print(f"--- {title} ---")
    print("="*80 + "\n")
    
    # Find max name length from the models that are *actually* in the map
    valid_names = [name for key, name in model_names_map.items() if key in archs]
    max_name_len = max(len(name) for name in valid_names)
    max_name_len = max(55, max_name_len + 2) 
    
    header = f"{'Model':<{max_name_len}} {'Method':<12} {'Sum of Widths':<15} {'Lower Bound':<40} {'Upper Bound':<40}"
    print(header)
    print("-" * (len(header) + 5))
    
    # Iterate over the archs dictionary to maintain order
    for model_key in archs: 
        model_name = model_names_map[model_key]
        for method in methods:
            if model_key not in results_dict[method]:
                lb_str = "N/A (SKIPPED/CRASHED)"
                ub_str = "N/A (SKIPPED/CRASHED)"
                sum_widths_str = "N/A"
            else:
                lb, ub = results_dict[method][model_key]
                sum_widths = torch.sum(ub - lb).item()
                sum_widths_str = f"{sum_widths:10.6f}"
                # Flatten bounds for printing, even if they are 4D
                lb_str = " ".join(f"{x:7.4f}" for x in lb.flatten())
                ub_str = " ".join(f"{x:7.4f}" for x in ub.flatten())
                lb_str = f"[{lb_str.strip()}]"
                ub_str = f"[{ub_str.strip()}]"
                
            print(f"{model_name:<{max_name_len}} {method:<12} {sum_widths_str:<15} {lb_str:<40} {ub_str:<40}")
        print("") 

# --- Helper function for running experiments ---
def run_one_shot_experiment(archs, x0, bounded_input, bound_opts_dict, method):
    """Runs a standard, one-shot experiment for all architectures."""
    results = {}
    
    # Get settings for printing
    opt_args = bound_opts_dict.get('optimize_bound_args', {})
    optimizer = opt_args.get('optimizer', 'adam') # 'adam' is default
    interm = bound_opts_dict.get('enable_opt_interm_bounds', False)
    
    print(f"  --- Running: optimizer={optimizer}, opt_interm_bounds={interm} ---")
    
    for key, model in archs.items():
        print(f"        Computing bounds for {model_names_map[key]}...")
        try:
            # We must set both general bound_opts AND optimize_bound_args
            lirpa_model = BoundedModule(model, x0, bound_opts=bound_opts_dict)
            lb, ub = lirpa_model.compute_bounds(x=(bounded_input,), method=method)
            results[key] = (lb.detach(), ub.detach())
        except Exception as e:
            # Catch the error and report it, but continue the experiment
            print(f"        FAILED for {model_names_map[key]}. Error: {e}")
            pass # Continue to the next model
    return {method: results}

# --- Main execution ---
if __name__ == '__main__':
    
    # --- 1. Setup ---
    IN_CHANNELS = 2
    MID_CHANNELS = 1 
    IMG_H = 2
    IMG_W = 2
    
    INPUT_SHAPE = (1, IN_CHANNELS, IMG_H, IMG_W) # (Batch, Channels, Height, Width)
    
    EPSILON = 0.1     
    METHODS_TO_TEST = ['alpha-crown']
    METHOD = METHODS_TO_TEST[0] 

    torch.manual_seed(123)
    
    # --- Define Shared Conv Layers ---
    shared_conv_layer1 = nn.Conv2d(
        in_channels=IN_CHANNELS, 
        out_channels=MID_CHANNELS, 
        kernel_size=1
    )
    shared_conv_layer2 = nn.Conv2d(
        in_channels=MID_CHANNELS, 
        out_channels=MID_CHANNELS, 
        kernel_size=1
    )
    shared_conv_layer3 = nn.Conv2d(
        in_channels=MID_CHANNELS, 
        out_channels=MID_CHANNELS, 
        kernel_size=1
    )
    
    # --- Define Model Architectures ONCE ---
    # ### MODIFICATION START ###
    # Re-enabling GroupSort2Optimized as requested.
    # This will fail during the BoundedModule creation.
    archs = {
        'gs_shared': nn.Sequential(shared_conv_layer1, GroupSort(), shared_conv_layer2, GroupSort(), shared_conv_layer3),
        'gs_opt': nn.Sequential(shared_conv_layer1, GroupSort2Optimized(), shared_conv_layer2, GroupSort2Optimized(), shared_conv_layer3),
        # 'torch_mm': nn.Sequential(MaxMin(), shared_conv_layer1, MaxMin(), shared_conv_layer2, MaxMin(), shared_conv_layer3),
    }

    # Model names for printing
    model_names_map = {
        'gs_shared': 'GroupSort (shared ReLU) - VERIFIABLE',
        'gs_opt': 'GroupSort2Optimized (torch.max) - NOT VERIFIABLE',
        'torch_mm': 'MaxMin (torch.min/max) - NOT VERIFIABLE',
    }
    # ### MODIFICATION END ###
    
    # --- Define Test Cases ---
    test_cases = {
        "TEST CASE 2: All Unstable ReLU States (Conv)": torch.tensor(
            [[0.0, 0.1, 0.0, 0.0, 0.1, 0.0, 0.5, 0.5]] # 8 features
        ).reshape(INPUT_SHAPE),
        
        "TEST CASE 4: Mostly Unstable States (Conv)": torch.tensor(
            [[0.05, -0.05, 0.0, 0.0, -0.1, 0.1, 0.0, 0.0]] # 8 features
        ).reshape(INPUT_SHAPE),
    }

    ptb = PerturbationLpNorm(norm=torch.inf, eps=EPSILON)
    
    # --- Define Experiment Optimization Settings ---
    opts_adam = {
        'enable_opt_interm_bounds': False, 
        'optimize_bound_args': {
            'iteration': 200, 
            'lr_alpha': 0.05, 
            'optimizer': 'adam'
        }
    }
    opts_sgd = {
        'enable_opt_interm_bounds': False,
        'optimize_bound_args': {
            'iteration': 200, 
            'lr_alpha': 0.05, 
            'optimizer': 'sgd'
        }
    }
    opts_adam_interm = {
        'enable_opt_interm_bounds': True,
        'optimize_bound_args': {
            'iteration': 200, 
            'lr_alpha': 0.05,
            'optimizer': 'adam'
        }
    }

    # --- 2. Run All Computations ---
    for case_name, x0 in test_cases.items():
        
        print("\n" + "#"*80)
        print(f"### {case_name}")
        print(f"### Input point (x0) shape: {x0.shape}")
        print(f"### Input point (x0) data (flattened): {x0.flatten()}")
        print("#"*80)

        bounded_input = BoundedTensor(x0, ptb)
        
        # --- Run and Print Experiment 1: Adam (Baseline) ---
        results_adam = run_one_shot_experiment(
            archs, x0, bounded_input, opts_adam, METHOD
        )
        print_comparison_table(
            title=f"COMPARISON 1: [Adam Baseline] ({case_name})",
            model_names_map=model_names_map,
            results_dict=results_adam,
            methods=METHODS_TO_TEST
        )
        
        # --- Run and Print Experiment 2: SGD ---
        results_sgd = run_one_shot_experiment(
            archs, x0, bounded_input, opts_sgd, METHOD
        )
        print_comparison_table(
            title=f"COMPARISON 2: [SGD Optimizer] ({case_name})",
            model_names_map=model_names_map,
            results_dict=results_sgd,
            methods=METHODS_TO_TEST
        )

        # --- Run and Print Experiment 3: Adam + Intermediate Bounds ---
        results_adam_interm = run_one_shot_experiment(
            archs, x0, bounded_input, opts_adam_interm, METHOD
        )
        print_comparison_table(
            title=f"COMPARISON 3: [Adam + Opt Interm Bounds] ({case_name})",
            model_names_map=model_names_map,
            results_dict=results_adam_interm,
            methods=METHODS_TO_TEST
        )

    print("\n--- End of Analysis ---")