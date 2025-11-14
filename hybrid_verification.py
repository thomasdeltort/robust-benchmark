# -*- coding: utf-8 -*-
"""
Hybrid Robustness Verification Script

This script evaluates certified robustness by splitting a model into two parts:
1.  A 1-Lipschitz (L2) prefix: f_1
2.  A standard suffix: f_2

It computes the "clean" intermediate activation z_k = f_1(x) and uses the
1-Lipschitz property to certify that the perturbed activation z_k' is within
an L2-ball of radius epsilon: ||z_k' - z_k||_2 <= epsilon.

It then soundly over-approximates this L2-ball with an L-infinity ball
(||z_k' - z_k||_inf <= epsilon) and feeds this as a new input to LIRPA,
which verifies the f_2 suffix.

This script requires an L2 perturbation (norm=2) and a 1-Lipschitz model.
"""
import torch
import torch.nn as nn
import numpy as np
from deel import torchlip
import argparse
import os
import sys
import time
import copy
import numpy as np

# --- LIRPA Imports ---
import auto_LiRPA
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLp

# --- Handle Local Module Imports ---
# Assuming 'models' is a module in the same directory or Python path
try:
    from models import *
except ImportError:
    print("Error: Could not import models. Make sure 'models.py' is accessible.")
    sys.exit(1)

# Assuming 'project_utils' contains these functions
# To make this script self-contained, we copy the necessary functions
# from your 'project_utils.py' file.

# --- 1. Copied Model Zoo ---
model_zoo = {
    # ... (Your full model_zoo dictionary) ...
    "ConvLarge_CIFAR10_1_LIP_GNP": ConvLarge_CIFAR10_1_LIP_GNP,
    # Add all other models you need
}

# --- 2. Copied Utility Functions from your project ---
from torch.nn.utils.parametrize import is_parametrized
from torchvision import datasets
from torchvision.transforms import v2

def preprocess_cifar(image, inception_preprocess=False, perturbation=False):
    """ Preprocessing used by the SDP paper. """
    MEANS = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
    STD = np.array([0.225, 0.225, 0.225], dtype=np.float32)
    if inception_preprocess:
        rescaled_devs = 0.5
        rescaled_means = 0.5
    else:
        rescaled_means = MEANS
        rescaled_devs = STD
    if perturbation:
        return image / rescaled_devs
    else:
        return (image - rescaled_means) / rescaled_devs

def load_dataset_benchmark(args):
    """ Loads the pre-processed benchmark data. """
    if "mnist" in args.dataset.lower():
        dataset = np.load('./prepared_data/mnist/X_sdp.npy')
        labels = np.load('./prepared_data/mnist/y_sdp.npy')
        dataset = torch.from_numpy(dataset).permute(0,3,1,2)
        labels = torch.from_numpy(labels)
        range = args.radius
        classes = 10
    elif "cifar10" in args.dataset.lower():
        dataset = np.load('./prepared_data/cifar/X_sdp.npy')
        labels = np.load('./prepared_data/cifar/y_sdp.npy')
        dataset = preprocess_cifar(dataset)
        dataset = torch.from_numpy(dataset).permute(0,3,1,2)
        labels = torch.from_numpy(labels)
        # Rescale the L2 epsilon by the normalization std
        range = args.radius / 0.225
        classes = 10
    else:
        raise ValueError(f"Unexpected model: {args.model}")
    return dataset, labels, range, classes

def vanilla_export(model1):
    """
    Converts a torchlip (parametrized) model to a standard nn.Module
    by copying the final weights.
    """
    model1.eval()
    model2 = copy.deepcopy(model1)
    model2.eval()
    dict_modified_layers = {}
    for (n1,p1), (n2,p2) in zip(model1.named_modules(), model2.named_modules()):
        assert n1 == n2
        if isinstance(p1, torch.nn.Conv2d) and is_parametrized(p1):
            new_conv = torch.nn.Conv2d(p1.in_channels, p1.out_channels, kernel_size=p1.kernel_size, stride=p1.stride, padding=p1.padding, padding_mode=p1.padding_mode,bias=(p1.bias is not None))
            new_conv.weight.data = p1.weight.data.clone()
            new_conv.bias.data = p1.bias.data.clone() if p1.bias is not None else None
            dict_modified_layers[n2] = new_conv
        if isinstance(p1, torch.nn.Linear) and is_parametrized(p1):
            new_lin = torch.nn.Linear(p1.in_features, p1.out_features, bias=(p1.bias is not None))
            new_lin.weight.data = p1.weight.data.clone()
            new_lin.bias.data = p1.bias.data.clone() if p1.bias is not None else None
            dict_modified_layers[n2] = new_lin

    for n2, new_layer in dict_modified_layers.items():
        split_hierarchy = n2.split('.')
        lay = model2
        for h in split_hierarchy[:-1]:
            lay = getattr(lay, h)
        setattr(lay, split_hierarchy[-1], new_layer)
    return model2

def build_C(label, classes):
    """ Builds the specification matrix for LIRPA. """
    device = label.device
    batch_size = label.size(0)
    C = torch.zeros((batch_size, classes-1, classes), device=device)
    all_cls = torch.arange(classes, device=device).unsqueeze(0).expand(batch_size, -1)
    mask = all_cls != label.unsqueeze(1)
    neg_cls = all_cls[mask].view(batch_size, -1)
    pos_idx = label.unsqueeze(1).expand(-1, classes-1).unsqueeze(-1)
    C.scatter_(dim=2, index=pos_idx, value=1.0)
    row_idx = torch.arange(classes-1, device=device).unsqueeze(0).expand(batch_size, -1)
    C[torch.arange(batch_size).unsqueeze(1), row_idx, neg_cls] = -1.0
    return C

# --- 3. New Model Loader for Hybrid Verification ---
def load_full_lip_model(args, model_zoo, device):
    """
    Loads the torchlip model AND its state_dict, but DOES NOT
    call vanilla_export.
    """
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Error: Model file not found at '{args.model_path}'")
    
    if args.model not in model_zoo:
        raise ValueError(f"Model '{args.model}' not found in zoo.")
    
    ModelClass = model_zoo[args.model]
    
    # Instantiate the 1-Lipschitz model structure
    model = ModelClass()
    
    # Load the trained weights
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# --- 4. Main Hybrid Verification Function ---
def main():
    """
    Main function to parse arguments, load data/model, run evaluations,
    and save results.
    """
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description='Perform HYBRID robustness verification (1-Lip Prefix + LIRPA Suffix).'
    )
    # --- Key arguments for this script ---
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the saved model .pth file.')
    parser.add_argument('--model', type=str, required=True, choices=model_zoo,
                        help='Name of the 1-Lipschitz model architecture.')
    parser.add_argument('--dataset', type=str, required=True, choices=['cifar10', 'mnist'],
                        help='Dataset to use for evaluation.')
    parser.add_argument('--epsilon', type=float, required=True,
                        help='Adversarial L2 perturbation radius (e.g., 0.5).')
    parser.add_argument('--split_index', type=int, required=True,
                        help='The index of the layer to split *before*. '
                             'e.g., 9 splits [0..8] (prefix) and [9..end] (suffix).')
    
    # --- Standard arguments ---
    parser.add_argument('--norm', type=int, default=2, choices=[2],
                        help="This script only supports L2 norms (default: 2).")
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Batch size for verification. Default: 100.')
    parser.add_argument('--start', default=0, type=int, help='start index for the dataset')
    parser.add_argument('--end', default=200, type=int, help='end index for the dataset')

    args = parser.parse_args()
    
    # For compatibility with `load_dataset_benchmark`
    args.radius = args.epsilon
    
    if args.norm != 2:
        print("Error: This hybrid verification script is designed for L2 1-Lipschitz models.")
        print("Please set --norm 2 (or omit it).")
        sys.exit(1)

    # --- Setup Device, Data, and Model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    images, targets, epsilon_rescaled, classes = load_dataset_benchmark(args)
    
    # Use only the specified subset of data
    images = images[args.start:args.end]
    targets = targets[args.start:args.end]
    
    print(f"Loading 1-Lipschitz model: {args.model}")
    full_model = load_full_lip_model(args, model_zoo, device)
    full_model.eval()

    # --- Split the Model ---
    all_layers = list(full_model.children())
    split_idx = args.split_index
    
    if split_idx <= 0 or split_idx >= len(all_layers):
        raise ValueError(f"--split_index {split_idx} is out of bounds for model with {len(all_layers)} layers.")

    # f1_prefix is the 1-Lipschitz part
    f1_prefix = torchlip.Sequential(*all_layers[:split_idx]).to(device).eval()
    
    # f2_suffix_lip is the original suffix, still with torchlip layers
    f2_suffix_lip = torchlip.Sequential(*all_layers[split_idx:]).to(device).eval()
    
    # f2_suffix_vanilla is converted to standard nn.Modules for LIRPA
    print("Converting model suffix for LIRPA using vanilla_export...")
    f2_suffix_vanilla = vanilla_export(f2_suffix_lip).to(device).eval()

    print(f"Model split complete. Prefix: {len(f1_prefix)} layers. Suffix: {len(f2_suffix_lip)} layers.")

    # --- Get Cleanly Classified Indices ---
    print("Calculating clean accuracy on the test subset...")
    with torch.no_grad():
        images = images.to(device)
        targets = targets.to(device)
        
        # We can use the full_model to get predictions
        output = full_model(images)
        predictions = output.argmax(dim=1)
        
        clean_indices = (predictions == targets).nonzero(as_tuple=True)[0]
        clean_accuracy = (len(clean_indices) / len(targets)) * 100
        
    print(f"Clean Accuracy: {clean_accuracy:.2f}% ({len(clean_indices)}/{len(targets)})")
    print(f"Verifying robustness for {len(clean_indices)} correctly classified samples.")
    print(f"Input L2 Epsilon: {args.epsilon} (rescaled to {epsilon_rescaled:.4f} for LIRPA)")

    # --- Run Hybrid Verification in Batches ---
    correct_images = images[clean_indices]
    correct_targets = targets[clean_indices]

    if len(correct_images) == 0:
        print("No correctly classified images to verify.")
        return

    num_robust_points = 0
    total_time = 0.0
    num_batches = (len(correct_images) + args.batch_size - 1) // args.batch_size

    print(f"Starting hybrid verification on {len(correct_images)} samples in {num_batches} batches...")

    for i in range(num_batches):
        start_idx = i * args.batch_size
        end_idx = min((i + 1) * args.batch_size, len(correct_images))
        
        batch_images = correct_images[start_idx:end_idx]
        batch_targets = correct_targets[start_idx:end_idx]
        
        if len(batch_images) == 0:
            continue

        if device.type == 'cuda':
            torch.cuda.synchronize()
        start_time_batch = time.time()

        # --- Step 1: Run 1-Lip Prefix ---
        # Get intermediate activations
        with torch.no_grad():
            z_k = f1_prefix(batch_images)

        # --- Step 2: Initialize LIRPA Suffix Model ---
        # We must re-create the BoundedModule for each batch if the
        # input shape (batch_size) changes, or just re-use if it's static.
        # For simplicity, we create it here.
        lirpa_model = BoundedModule(f2_suffix_vanilla, (z_k,), 
                                    bound_opts={"conv_mode": "patches"}, 
                                    device=device)

        # --- Step 3: Define LIRPA Input (The Hand-off) ---
        # We certify ||delta_z||_2 <= epsilon (from 1-Lip)
        # We over-approximate with ||delta_z||_inf <= epsilon
        # We use the *rescaled* epsilon, as this is what LIRPA expects
        ptb = PerturbationLp(norm=np.inf, eps=epsilon_rescaled)
        lirpa_input = BoundedTensor(z_k, ptb)
        
        # --- Step 4: Define LIRPA Specification ---
        c = build_C(batch_targets.to("cpu"), classes).to(device)

        # --- Step 5: Run LIRPA on Suffix ---
        # We can use a simple 'CROWN' or 'alpha-CROWN'
        lb_diff = lirpa_model.compute_bounds(x=(lirpa_input,), C=c, method='CROWN')[0]
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        total_time += (time.time() - start_time_batch)

        # --- Step 6: Check Robustness ---
        # If all logit differences are positive, the batch is robust
        is_robust = (lb_diff.view(len(batch_images), classes - 1) > 0).all(dim=1)
        num_robust_points += torch.sum(is_robust).item()
        
        print(f"  Batch {i+1}/{num_batches}: {torch.sum(is_robust).item()}/{len(batch_images)} robust.", end='\r')

    print("\nHybrid verification complete.")

    # --- Report Final Results ---
    vra = (num_robust_points / len(correct_images)) * 100.0
    avg_time = total_time / len(correct_images)

    print("\n--- 🚀 Final Results ---")
    print(f"Model:                {args.model}")
    print(f"Epsilon (L2):         {args.epsilon}")
    print(f"Split Index:          {args.split_index}")
    print(f"Total Verified:       {len(correct_images)}")
    print(f"Robustly Verified:    {num_robust_points}")
    print(f"Verified Robust Acc:  {vra:.2f}%")
    print(f"Total Time:           {total_time:.4f}s")
    print(f"Avg. Time per Image:  {avg_time:.4f}s")


if __name__ == '__main__':
    # --- Populate the model_zoo with your models ---
    # This is necessary for the script to find the model class
    model_zoo = {
        # --- Standard, non-Lipschitz constrained models ---
        "MNIST_MLP": MNIST_MLP,
        "MNIST_ConvSmall": MNIST_ConvSmall,
        "MNIST_ConvLarge": MNIST_ConvLarge,
        "CIFAR10_CNN_A": CIFAR10_CNN_A,
        "CIFAR10_CNN_B": CIFAR10_CNN_B,
        "CIFAR10_CNN_C": CIFAR10_CNN_C,
        "CIFAR10_ConvSmall": CIFAR10_ConvSmall,
        "CIFAR10_ConvDeep": CIFAR10_ConvDeep,
        "CIFAR10_ConvLarge": CIFAR10_ConvLarge,

        # --- 1-Lipschitz models (Spectral Normalization) ---
        "MLP_MNIST_1_LIP": MLP_MNIST_1_LIP,
        "ConvSmall_MNIST_1_LIP": ConvSmall_MNIST_1_LIP,
        "ConvLarge_MNIST_1_LIP": ConvLarge_MNIST_1_LIP,
        "CNNA_CIFAR10_1_LIP": CNNA_CIFAR10_1_LIP,
        "CNNB_CIFAR10_1_LIP": CNNB_CIFAR10_1_LIP,
        "CNNC_CIFAR10_1_LIP": CNNC_CIFAR10_1_LIP,
        "ConvSmall_CIFAR10_1_LIP": ConvSmall_CIFAR10_1_LIP,
        "ConvDeep_CIFAR10_1_LIP": ConvDeep_CIFAR10_1_LIP,
        "ConvLarge_CIFAR10_1_LIP": ConvLarge_CIFAR10_1_LIP,

        # --- 1-Lipschitz models (GNP technique) ---
        "MLP_MNIST_1_LIP_GNP": MLP_MNIST_1_LIP_GNP,
        "ConvSmall_MNIST_1_LIP_GNP": ConvSmall_MNIST_1_LIP_GNP,
        "ConvLarge_MNIST_1_LIP_GNP": ConvLarge_MNIST_1_LIP_GNP,
        "CNNA_CIFAR10_1_LIP_GNP": CNNA_CIFAR10_1_LIP_GNP,
        "CNNB_CIFAR10_1_LIP_GNP": CNNB_CIFAR10_1_LIP_GNP,
        "CNNC_CIFAR10_1_LIP_GNP": CNNC_CIFAR10_1_LIP_GNP,
        "ConvSmall_CIFAR10_1_LIP_GNP": ConvSmall_CIFAR10_1_LIP_GNP,
        "ConvDeep_CIFAR10_1_LIP_GNP": ConvDeep_CIFAR10_1_LIP_GNP,
        "ConvLarge_CIFAR10_1_LIP_GNP": ConvLarge_CIFAR10_1_LIP_GNP,
    }
    
    main()