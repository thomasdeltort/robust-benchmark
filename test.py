import torch
import torch.nn as nn
import sys
# sys.path.append('../SDP-CROWN-Share/auto_LiRPA')
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm

import inspect

# Use inspect.getfile() to reliably find the source file of the class
print(f"BoundedModule is from: {inspect.getfile(BoundedModule)}")
print(f"BoundedTensor is from: {inspect.getfile(BoundedTensor)}")

# Ensure reproducibility
torch.manual_seed(42)

## 1. Define a Simple Neural Network
# We'll use a small feed-forward network with ReLU activations.
# Input dimension: 2, Hidden layers: 10, 10, Output dimension: 1
model = nn.Sequential(
        nn.Conv2d(3, 16, 4, stride=2, padding=0),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=0),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(32*6*6,100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )

print("✅ Simple neural network created.")
print(model)

## 2. Define Input Data and Perturbation
# Create a dummy input tensor (batch size 1, 2 features).
# Replace the old dummy_input line with this:
dummy_input = torch.zeros(1, 3, 32, 32) # A tensor of all zeros

# Define the perturbation space around the input.
# Here, we use an L-infinity norm constraint with epsilon = 0.1.
# This means each input feature can be perturbed by +/- 0.1.
norm = 2
eps = 0.0001
ptb = PerturbationLpNorm(norm=norm, eps=eps)

# Create a BoundedTensor, which tracks the perturbation.
bounded_input = BoundedTensor(dummy_input, ptb)

print(f"\n✅ Input data defined with an L-infinity perturbation (epsilon={eps}).")

## 3. Wrap the Model for Bound Computation
# The BoundedModule from auto_LiRPA wraps our PyTorch model to enable bound computation.
bounded_model = BoundedModule(model, torch.empty_like(dummy_input))
print("✅ Model wrapped with auto_LiRPA's BoundedModule.")

## 4. Compute Bounds with CROWN
# CROWN (CROWN is a one-shot method, fast but often looser).
print("\n🔍 Computing bounds with CROWN...")
lb_crown, ub_crown = bounded_model.compute_bounds(x=(bounded_input,), method='CROWN')
print(f"   -> CROWN Lower Bound: {lb_crown}")

## 5. Compute Bounds with alpha-CROWN
# alpha-CROWN optimizes the bounding parameters (alphas) to get a tighter bound.
# It's slower but generally more precise.
print("\n🧠 Computing bounds with alpha-CROWN...")
lb_alpha_crown, ub_alpha_crown = bounded_model.compute_bounds(x=(bounded_input,), method='alpha-CROWN')
print(f"   -> alpha-CROWN Lower Bound: {lb_alpha_crown}")

## 6. Compare the Results
print("\n---" * 10)
print("📊 Comparison of Lower Bounds:")
print(f"   CROWN:       {lb_crown}")
print(f"   alpha-CROWN: {lb_alpha_crown}")
print("\nAs expected, the bound from alpha-CROWN is tighter (larger) than the one from CROWN.")
print("---" * 10)