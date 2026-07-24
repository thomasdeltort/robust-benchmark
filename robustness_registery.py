import os
import pickle
import numpy as np
import torch

class RobustnessRegistry:
    def __init__(self, model_name, norm, total_dataset_size, save_dir="./results/robust_points"):
        """
        Args:
            model_name (str): Name of the model (e.g., "ResNet18_CIFAR")
            norm (str): '2' or 'inf'
            total_dataset_size (int): Total N (e.g., 200, 10000)
            save_dir (str): Directory to save pkl files.
        """
        self.model_key = f"{model_name}_{norm}"
        self.total_size = total_dataset_size
        self.save_path = os.path.join(save_dir, f"{self.model_key}_vectors.pkl")
        self.data = {self.model_key: {}}
        
        # Ensure directory exists
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # Load existing data if verification is resumed/extended
        if os.path.exists(self.save_path):
            try:
                with open(self.save_path, 'rb') as f:
                    self.data = pickle.load(f)
                print(f"Loaded existing registry from {self.save_path}")
            except Exception as e:
                print(f"Could not load existing registry: {e}")

    def register(self, epsilon, method_name, robust_indices_tensor):
        """
        Converts list/tensor of robust indices into a binary vector and stores it.
        
        Args:
            epsilon (float): The perturbation radius.
            method_name (str): Name of the method (e.g., 'autoattack', 'sdp').
            robust_indices_tensor (torch.Tensor or list): Global indices of robust points.
        """
        # Ensure epsilon key exists
        if epsilon not in self.data[self.model_key]:
            self.data[self.model_key][epsilon] = {}
            
        # Create binary vector of zeros
        binary_vec = np.zeros(self.total_size, dtype=int)
        
        # Fill ones at robust indices
        if isinstance(robust_indices_tensor, torch.Tensor):
            indices = robust_indices_tensor.cpu().numpy().astype(int)
        else:
            indices = np.array(robust_indices_tensor).astype(int)
            
        # Handle empty cases safely
        if len(indices) > 0:
            binary_vec[indices] = 1
            
        # Store
        self.data[self.model_key][epsilon][method_name] = binary_vec

    def save(self):
        """Saves the current dictionary to the pickle file."""
        with open(self.save_path, 'wb') as f:
            pickle.dump(self.data, f)
        print(f"Robustness vectors saved to {self.save_path}")


import pickle
import os
import numpy as np

def load_robustness_data(filepath):
    """
    Loads the pickle file and returns the nested dictionary.
    Structure: { 'model_norm': { epsilon: { 'method': binary_vector } } }
    """
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return None

    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data

def print_robustness_stats(data, show_indices=True):
    """
    Iterates through the data structure and prints robustness statistics and indices.
    
    Args:
        data: The dictionary loaded from pickle.
        show_indices (bool): If True, prints the specific indices of robust points.
    """
    # 1. Loop through Models
    for model_key, eps_dict in data.items():
        print(f"\n{'='*60}")
        print(f"Model Key: {model_key}")
        print(f"{'='*60}")

        # 2. Loop through Epsilons (sorted)
        sorted_epsilons = sorted(eps_dict.keys())
        
        for eps in sorted_epsilons:
            methods_dict = eps_dict[eps]
            print(f"\n[Epsilon: {eps}]")
            print(f"-"*40)
            
            # Check for ground truth (Clean points)
            if 'clean' in methods_dict:
                clean_vec = methods_dict['clean']
                total_points = len(clean_vec)
                clean_count = np.sum(clean_vec)
                clean_indices = np.where(clean_vec == 1)[0]
                
                print(f"  > Dataset Size: {total_points}")
                print(f"  > Originally Clean: {clean_count} ({(clean_count/total_points)*100:.2f}%)")
                if show_indices:
                    print(f"    Indices: {clean_indices.tolist()}")
            else:
                print("  > Warning: 'clean' mask not found.")
                clean_count = 0

            # 3. Loop through Methods
            print(f"\n  {'Method':<20} | {'Robust':<8} | {'VRA (Total)':<12}")
            print(f"  {'-'*20} | {'-'*8} | {'-'*12}")
            
            # Store indices for overlap check later if needed
            method_indices = {}

            for method, vector in methods_dict.items():
                if method == 'clean': continue 
                
                n_robust = np.sum(vector)
                total_size = len(vector)
                vra = (n_robust / total_size) * 100.0
                
                # Extract indices where vector is 1
                indices = np.where(vector == 1)[0]
                method_indices[method] = indices

                print(f"  {method:<20} | {n_robust:<8} | {vra:.2f}%")
                
                if show_indices:
                    # Print indices nicely wrapped
                    idx_list = indices.tolist()
                    print(f"    [Indices]: {idx_list}")
                    print(f"  {'-'*44}")

            # 4. (Optional) Check Overlap specifically for Certificate vs SDP (or others)
            # You can customize which methods to compare here
            if 'certificate' in method_indices and 'sdp' in method_indices:
                cert_idx = set(method_indices['certificate'])
                sdp_idx = set(method_indices['sdp'])
                
                intersection = sorted(list(cert_idx.intersection(sdp_idx)))
                sdp_only = sorted(list(sdp_idx - cert_idx))
                
                print(f"\n  [Comparison] Certificate vs SDP:")
                print(f"    Both Robust ({len(intersection)}): {intersection}")
                print(f"    SDP Only    ({len(sdp_only)}): {sdp_only}")

# --- Usage ---
if __name__ == "__main__":
    # filename = "MLP_MNIST_1_LIP_Bjork_2_vectors.pkl"
    # Or search automatically:
    filename = None 
    results_dir = os.path.join("results", "robust_points")
    
    file_path = ""
    if filename:
        file_path = os.path.join(results_dir, filename)
    else:
        # Auto-find the first pickle file
        if os.path.exists(results_dir):
            files = [f for f in os.listdir(results_dir) if f.endswith(".pkl")]
            if files:
                file_path = os.path.join(results_dir, files[0])
                print(f"Auto-loading: {files[0]}...")
            else:
                print("No .pkl files found in results folder.")

    # Load and Print
    if file_path:
        results = load_robustness_data(file_path)
        if results:
            print_robustness_stats(results, show_indices=True)