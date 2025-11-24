import torch
import torch.nn as nn
import numpy as np
from deel import torchlip
import argparse
import os
import sys

from torchvision import datasets, transforms

# --- Handle module imports ---
sys.path.append('./..')
from models import *
from project_utils import *
sys.path.append('/home/aws_install/robustess_project/lip_notebooks/notebooks_creation_models')
from VGG_Arthur import HKRMultiLossLSE
from deel.torchlip import TauCrossEntropyLoss


def main(args):
    """
    Main function to train a model with specified hyperparameters.
    """
    
    print("--- Hyperparameters ---")
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.dataset}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning Rate: {args.lr}")
    print(f"Loss Temperature: {args.temperature}")
    print("-----------------------")

    # --- 1. Load Dataset ---
    train_loader, test_loader = load_dataset(args.dataset, args.batch_size)

    # --- 2. Initialize Model ---
    model_zoo = {
    # Lipschitz Models (Spectral Normalization)
    "MLP_MNIST_1_LIP": MLP_MNIST_1_LIP,
    "ConvSmall_MNIST_1_LIP": ConvSmall_MNIST_1_LIP,
    "ConvLarge_MNIST_1_LIP": ConvLarge_MNIST_1_LIP,
    "CNNA_CIFAR10_1_LIP": CNNA_CIFAR10_1_LIP,
    "CNNB_CIFAR10_1_LIP": CNNB_CIFAR10_1_LIP,
    "CNNC_CIFAR10_1_LIP": CNNC_CIFAR10_1_LIP,
    "ConvSmall_CIFAR10_1_LIP": ConvSmall_CIFAR10_1_LIP,
    "ConvDeep_CIFAR10_1_LIP": ConvDeep_CIFAR10_1_LIP,
    "ConvLarge_CIFAR10_1_LIP": ConvLarge_CIFAR10_1_LIP,

    # Lipschitz Gradient Norm Preserving Models (GNP)
    "MLP_MNIST_1_LIP_GNP": MLP_MNIST_1_LIP_GNP,
    "ConvSmall_MNIST_1_LIP_GNP": ConvSmall_MNIST_1_LIP_GNP,
    "ConvLarge_MNIST_1_LIP_GNP": ConvLarge_MNIST_1_LIP_GNP,
    "CNNA_CIFAR10_1_LIP_GNP": CNNA_CIFAR10_1_LIP_GNP,
    "CNNB_CIFAR10_1_LIP_GNP": CNNB_CIFAR10_1_LIP_GNP,
    "CNNC_CIFAR10_1_LIP_GNP": CNNC_CIFAR10_1_LIP_GNP,
    "ConvSmall_CIFAR10_1_LIP_GNP": ConvSmall_CIFAR10_1_LIP_GNP,
    "ConvDeep_CIFAR10_1_LIP_GNP": ConvDeep_CIFAR10_1_LIP_GNP,
    "ConvLarge_CIFAR10_1_LIP_GNP": ConvLarge_CIFAR10_1_LIP_GNP,
    "VGG13_1_LIP_GNP_CIFAR10" : VGG13_1_LIP_GNP_CIFAR10,
    "VGG19_1_LIP_GNP_CIFAR10" : VGG19_1_LIP_GNP_CIFAR10,
}

    if args.model_name not in model_zoo:
        raise ValueError(f"Model '{args.model_name}' not found. Available models are: {list(model_zoo.keys())}")
    
    ModelClass = model_zoo[args.model_name]
    model = ModelClass()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}\n")

    # --- 3. Setup Optimizer, Scheduler, and Loss ---
    optimizer = torch.optim.Adam(lr=args.lr, params=model.parameters())
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30, 40], gamma=0.5)
    
    # criterion = HKRMultiLossLSE(alpha=250, temperature=args.temperature, penalty=0.5, margin=1.0)
    criterion = TauCrossEntropyLoss(tau = args.temperature)
    
    kr_multiclass_loss = torchlip.KRMulticlassLoss()
    num_classes = 10 

    # --- 4. Training Loop ---
    # Initialize metrics that will be used in the filename
    train_acc, val_acc = 0.0, 0.0

    for epoch in range(args.epochs):
        model.train() 
        m_kr, m_acc, total_loss = 0, 0, 0

        for step, (data, target) in enumerate(train_loader):
            target = torch.nn.functional.one_hot(target, num_classes=num_classes)
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # Compute metrics on batch
            m_kr += kr_multiclass_loss(output, target)
            m_acc += (output.argmax(dim=1) == target.argmax(dim=1)).sum() / len(target)
            

        scheduler.step()

        metrics_log = [
            f"{k}: {v:.04f}"
            for k, v in {
                "loss": loss,
                "acc": m_acc / (step + 1),
                "KR": m_kr / (step + 1),
            }.items()
        ]
        
        if (epoch + 1) % 10 == 0 or (epoch + 1) == args.epochs:
            model.eval() 
            test_output, test_targets = [], []
            with torch.no_grad():
                for data, target in test_loader:
                    data = data.to(device)
                    test_output.append(model(data).cpu())
                    test_targets.append(torch.nn.functional.one_hot(target, num_classes=num_classes).cpu())
            
            test_output = torch.cat(test_output)
            test_targets = torch.cat(test_targets)
            
            val_acc = (test_output.argmax(dim=1) == test_targets.argmax(dim=1)).float().mean().item() # Updated here
            
            metrics_log += [
                        f"val_{k}: {v:.04f}"
                        for k, v in {
                            "loss": criterion(test_output, test_targets),
                            "acc": (test_output.argmax(dim=1) == test_targets.argmax(dim=1))
                            .float()
                            .mean(),
                            "KR": kr_multiclass_loss(test_output, test_targets),
                        }.items()
            ]
        print(f"Epoch {epoch + 1}/{args.epochs} | " + " - ".join(metrics_log))

    # --- 6. Save Model --- #
    # ##################### MODIFIED SECTION #####################
    # Create a descriptive filename including hyperparameters and final accuracies.
    # The train_acc and val_acc variables hold the values from the final epoch.
    # The formatting `:.1f` rounds the accuracy value to one decimal place.
    base_filename = (
        f"{args.dataset}_{args.model_name}_"
        f"temp{args.temperature}_bs{args.batch_size}_"
        f"trainacc{train_acc:.1f}_valacc{val_acc:.1f}"
    )
    
    # Ensure the save directory exists
    save_dir = "/home/aws_install/robustess_project/Robust_Benchmark/models"
    os.makedirs(save_dir, exist_ok=True)
    
    save_path_full = os.path.join(save_dir, f"{base_filename}.pth")
    save_path_vanilla = os.path.join(save_dir, f"vanilla_{base_filename}.pth")
    # ################# END OF MODIFIED SECTION #################

    print(f"\nSaving model to: {save_path_full}")
    torch.save(model.state_dict(), save_path_full)
    #FIXME wrong vanilla_export
    print(f"Saving vanilla model to: {save_path_vanilla}")
    torch.save(model.vanilla_export().state_dict(), save_path_vanilla)

choices = [
    # Standard Models
    "MNIST_MLP",
    "MNIST_ConvSmall",
    "MNIST_ConvLarge",
    "CIFAR10_CNN_A",
    "CIFAR10_CNN_B",
    "CIFAR10_CNN_C",
    "CIFAR10_ConvSmall",
    "CIFAR10_ConvDeep",
    "CIFAR10_ConvLarge",

    # Lipschitz Models (Spectral Normalization)
    "MLP_MNIST_1_LIP",
    "ConvSmall_MNIST_1_LIP",
    "ConvLarge_MNIST_1_LIP",
    "CNNA_CIFAR10_1_LIP",
    "CNNB_CIFAR10_1_LIP",
    "CNNC_CIFAR10_1_LIP",
    "ConvSmall_CIFAR10_1_LIP",
    "ConvDeep_CIFAR10_1_LIP",
    "ConvLarge_CIFAR10_1_LIP",

    # Lipschitz Gradient Norm Preserving Models (GNP)
    "MLP_MNIST_1_LIP_GNP",
    "ConvSmall_MNIST_1_LIP_GNP",
    "ConvLarge_MNIST_1_LIP_GNP",
    "CNNA_CIFAR10_1_LIP_GNP",
    "CNNB_CIFAR10_1_LIP_GNP",
    "CNNC_CIFAR10_1_LIP_GNP",
    "ConvSmall_CIFAR10_1_LIP_GNP",
    "ConvDeep_CIFAR10_1_LIP_GNP",
    "ConvLarge_CIFAR10_1_LIP_GNP",
    "VGG13_1_LIP_GNP_CIFAR10",
    "VGG19_1_LIP_GNP_CIFAR10",
]
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a PyTorch model with command-line arguments.')
    
    parser.add_argument('--model_name', type=str, default='CNNA_CIFAR10',
                        choices=choices,
                        help='Name of the model to train.')
    parser.add_argument('--temperature', type=float, default=200.0,
                        help='Temperature for the HKRMultiLossLSE loss function.')
                        
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Input batch size for training.')

    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'mnist'],
                        help='Dataset to use for training.')
                        
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train.')

    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate for the optimizer.')

    args = parser.parse_args()
    main(args)