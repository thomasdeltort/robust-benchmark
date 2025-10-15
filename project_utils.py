from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import os
from models import * # Make sure this import works from your utils.py location
try:
    sys.path.append('/home/aws_install/robustess_project/lip_notebooks/notebooks_creation_models')
    from VGG_Arthur import HKRMultiLossLSE
except ImportError:
    print("Warning: Could not import HKRMultiLossLSE. Using standard CrossEntropyLoss for evaluation.")
import pickle
import numpy as np
import torchattacks
import matplotlib.pyplot as plt
import time
import csv
import os
import copy
from torch.nn.utils.parametrize import is_parametrized

def load_cifar10(batch_size):
     # Initialize transforms
    train_transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        v2.RandomAffine(
            degrees=5,           # Rotate by a maximum of 5 degrees
            translate=(0.05, 0.05) # Shift by a maximum of 5%
        ),
        # v2.RandomCrop((32, 32), padding=8),
        # v2.RandomHorizontalFlip(),
        # v2.RandomApply([
            # v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
        # ], p=0.8),
        # shift et jitter
        # v2.RandAugment(),
        # v2.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3), value='random')
        v2.Normalize((0.49139968, 0.48215827, 0.44653124),
                     (0.225, 0.225, 0.225)),
    ])
    test_transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.49139968, 0.48215827, 0.44653124),
                     (0.225, 0.225, 0.225)),
    ])

    # Split dataset into train, calibration, and test sets
    train_dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        transform=train_transforms,
        download=True
    )
    
    test_dataset = datasets.CIFAR10(
        root='./data', 
        train=False, 
        transform=test_transforms,
        download=True
    )

    # Create data loaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    return train_loader,  test_loader

def load_mnist(batch_size):
    train_set = datasets.MNIST(
        root="./data",
        download=True,
        train=True,
        transform=v2.ToTensor(),
    )

    test_set = datasets.MNIST(
        root="./data",
        download=True,
        train=False,
        transform=v2.ToTensor(),
    )

    train_loader = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size)
    return train_loader, test_loader

def load_dataset(name, batch_size):
    if name=="mnist":
        train_loader, test_loader = load_mnist(batch_size)
    elif name=="cifar10":
        train_loader, test_loader = load_cifar10(batch_size)
    else :
        raise ValueError(f"Unexpected dataset: {name}")
    return train_loader, test_loader

def preprocess_cifar(image, inception_preprocess=False, perturbation=False):
    """
    Preprocess images and perturbations.Preprocessing used by the SDP paper.
    """
    MEANS = np.array([125.3, 123.0, 113.9], dtype=np.float32)/255
    STD = np.array([0.225, 0.225, 0.225], dtype=np.float32)
    if inception_preprocess:
        # Use 2x - 1 to get [-1, 1]-scaled images
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
        print(dataset.shape, "we need to verify channel first")
        labels = torch.from_numpy(labels)
        range = args.radius/0.225
        classes = 10
    else:
        raise ValueError(f"Unexpected model: {args.model}")
    return dataset, labels, range, classes




def vanilla_export(model1):
    model1.eval()
    model2 = copy.deepcopy(model1)
    model2.eval()
    dict_modified_layers = {}
    for (n1,p1), (n2,p2) in zip(model1.named_modules(), model2.named_modules()):
        #print(n1,type(p1), type(p2))
        assert n1 == n2
        if isinstance(p1, torch.nn.Conv2d) and is_parametrized(p1):
            new_conv = torch.nn.Conv2d(p1.in_channels, p1.out_channels, kernel_size=p1.kernel_size, stride=p1.stride, padding=p1.padding, padding_mode=p1.padding_mode,bias=(p1.bias is not None))
            new_conv.weight.data = p1.weight.data.clone()
            new_conv.bias.data = p1.bias.data.clone() if p1.bias is not None else None
            dict_modified_layers[n2] = new_conv
            #print("modified",n2,type(p1), type(new_conv),p1.in_channels, p1.out_channels, p1.kernel_size[0], p1.stride[0], p1.padding[0], p1.padding_mode,(p1.bias is not None))
            #setattr(model2, n2, new_conv)
            #print(n1,type(p1), type(getattr(model2, n2)))
        if isinstance(p1, torch.nn.Linear) and is_parametrized(p1):
            new_lin = torch.nn.Linear(p1.in_features, p1.out_features, bias=(p1.bias is not None))
            new_lin.weight.data = p1.weight.data.clone()
            new_lin.bias.data = p1.bias.data.clone() if p1.bias is not None else None
            #setattr(model2, n2, new_lin)
            dict_modified_layers[n2] = new_lin
            #print("modified",n2,type(p1), type(new_lin))
            #print(n1,type(p1), type(getattr(model2, n2)))
    for n2, new_layer in dict_modified_layers.items():
        split_hierarchy = n2.split('.')
        lay = model2
        for h in split_hierarchy[:-1]:
            lay = getattr(lay, h)
        # print("modified",n2, type(getattr(lay, split_hierarchy[-1])),type(new_layer))
        setattr(lay, split_hierarchy[-1], new_layer)
    return model2


def load_model(args, model_zoo, device):
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at '{args.model_path}'")
        return
    
    if args.model not in model_zoo:
        raise ValueError(f"Model '{args.model}' not found. Available models are: {list(model_zoo.keys())}")
    
    ModelClass = model_zoo[args.model]
    #TODO vanilla export only if the model is a torchlip.seq
    model = ModelClass()
    # .vanilla_export()
    
    # Load the saved state dictionary
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    # model.load_state_dict(torch.load("models/cifar10_convsmall.pth", map_location=device))
    model = vanilla_export(model)
    model.to(device)
    model.eval() # Set model to evaluation mode
    return model


def convert_lipschitz_constant(L_2, norm, input_dim):
    """
    Converts an L2 Lipschitz constant to another norm's constant.

    This function returns the L2 constant directly if the target norm is 2,
    or it calculates an upper bound for the L-infinity constant based on the
    L2 constant and the input dimension of the network.

    Args:
        L_2 (float): The L2 Lipschitz constant of the network.
        norm (str or int): The target norm. Accepts '2', 2, 'inf'.
        input_dim (int): The dimensionality of the network's input space (e.g., for
                         a flattened MNIST image, this would be 28*28 = 784).

    Returns:
        float: The Lipschitz constant for the specified norm.

    Raises:
        ValueError: If the input_dim is not a positive integer.
        ValueError: If the specified norm is not supported.
    """
    if not isinstance(input_dim, int) or input_dim <= 0:
        raise ValueError("input_dim must be a positive integer.")

    # Normalize the norm input for consistent checking
    norm_str = str(norm).lower()

    if norm_str == '2':
        # If the target norm is L2, no conversion is needed.
        return L_2
    elif norm_str == 'inf':
        # Convert L2 constant to an L-infinity constant upper bound.
        # The relationship is L_inf <= sqrt(input_dim) * L_2
        L_inf = L_2 * torch.sqrt(input_dim)
        return L_inf
    else:
        raise ValueError(f"Unsupported norm: '{norm}'. Please use '2' or 'inf'.")

def add_result_and_sort(result_dict, csv_filepath, round_digits=3, norm=2):
    """
    Adds a new result from a dictionary to a specified CSV file and rounds time values.

    The function reads the existing CSV, adds the new result, sorts all entries
    by the 'epsilon' value, and then overwrites the file with the sorted data.

    Args:
        result_dict (dict): A dictionary containing the new row of data.
        csv_filepath (str): The full path to the CSV file to write to.
        round_digits (int): The number of decimal places to round time values to.
    """
    #TODO adapt to linf norm
    # Get the directory part of the provided filepath
    directory = os.path.dirname(csv_filepath)

    # Define the header order based on your dictionary keys
    header = list(result_dict.keys())

    # --- 1. Read all existing data from the CSV ---
    all_data = []
    # Create the directory if it doesn't exist (and if a directory is part of the path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    
    if os.path.exists(csv_filepath) and os.path.getsize(csv_filepath) > 0:
        with open(csv_filepath, 'r', newline='') as file:
            reader = csv.reader(file)
            file_header = next(reader)
            try:
                # Ensure the new data has the same columns as the existing file
                if set(file_header) != set(header):
                    print("⚠️  Warning: CSV header mismatch. Overwriting the file with new headers.")
                else:
                    for row in reader:
                        all_data.append(row)
            except (ValueError, IndexError):
                print("⚠️  Could not parse existing file. Starting fresh.")
    
    # --- 2. Convert the new dictionary to a list and add it ---
    new_row = [
        round(value, round_digits) if isinstance(value, float) and key.startswith('time_') else value
        for key, value in result_dict.items()
    ]
    all_data.append(new_row)

    # --- 3. Sort the data by the 'epsilon' column ---
    epsilon_index = header.index('epsilon')
    all_data.sort(key=lambda row: float(row[epsilon_index]))

    # --- 4. Write the sorted data back to the file ---
    with open(csv_filepath, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(all_data)

    print(f"✅ Successfully added result to '{csv_filepath}' for epsilon={result_dict['epsilon']}.")


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def create_robustness_plot(filepath='results/results.csv', output_filename='robust_accuracy_vs_epsilon_colored.png'):
    """
    Generates a plot of robust accuracy vs. epsilon with a custom, distinct color palette.

    - 'certificate' is blue (distinct).
    - 'pgd' and 'aa' are reddish/orangish (similar to each other).
    - Other methods have distinct colors.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"❌ Error: The file '{filepath}' was not found.")
        return

    # Prepare data
    df = df.sort_values(by='epsilon').reset_index(drop=True)
    
    # Define the order of columns for plotting and color assignment
    # This ensures 'certificate' is handled first, then 'pgd'/'aa', then others
    plot_order = ['certificate', 'pgd', 'aa', 'lirpa_alphacrown', 'lirpa_betacrown', 'sdp']
    
    # Filter to only include columns that are in plot_order and are actually in the DataFrame
    method_columns = [col for col in plot_order if col in df.columns and col != 'epsilon' and not col.startswith('time_')]

    # --- ✨ NEW: Custom Color Mapping for precise control ---
    color_map = {}
    
    # 1. Assign a distinct color to 'certificate'
    color_map['certificate'] = '#1f77b4'  # A standard Matplotlib blue

    # 2. Assign similar reddish/orangish colors to 'pgd' and 'aa'
    color_map['pgd'] = '#d62728'  # Matplotlib red
    color_map['aa'] = '#ff7f0e'   # Matplotlib orange

    # 3. Assign distinct colors for the remaining methods using a palette
    #    We'll skip the colors already used by 'tab10' for the first three items.
    #    The 'tab10' palette has 10 distinct colors. We'll pick from the unused ones.
    default_palette = sns.color_palette('tab10', n_colors=10)
    
    # Get colors from the default palette that haven't been "claimed" by certificate, pgd, aa
    # Skip the first few default colors (blue, orange, green, red, purple) to ensure distinctness
    # and map them to the remaining methods.
    
    # Let's manually assign specific indices from tab10 or custom hex for better control
    # Or simply iterate through the palette for remaining methods, skipping those already assigned
    
    # Start filling other methods from a specific point in the palette
    # For instance, we'll pick from 'tab10' starting at index 4 (purple, brown, pink, grey, olive, cyan)
    
    other_methods_colors = [
        '#9467bd', # tab10 purple
        '#8c564b', # tab10 brown
        '#e377c2'  # tab10 pink
    ]
    other_idx = 0
    for method in method_columns:
        if method not in color_map:
            if other_idx < len(other_methods_colors):
                color_map[method] = other_methods_colors[other_idx]
                other_idx += 1
            else:
                # Fallback if there are more methods than predefined distinct colors
                # (unlikely with this problem but good practice)
                color_map[method] = default_palette[len(color_map) % len(default_palette)]


    # Create the plot
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 8))

    for method in method_columns:
        # Use the color assigned to each method in our new map
        plt.plot(df['epsilon'], df[method], 
                 marker='o', 
                 linestyle='-', 
                 label=method, 
                 color=color_map[method], # Use the defined color
                 linewidth=2)

    # Style the plot
    plt.title('Robust Accuracy vs. Epsilon for Various Methods', fontsize=16, weight='bold')
    plt.xlabel('Epsilon ($ε$)', fontsize=14)
    plt.ylabel('Robust Accuracy', fontsize=14)
    plt.legend(title='Verification Method', fontsize=10)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_filename)
    print(f"✅ Plot successfully saved as '{output_filename}' with refined colors.")
    plt.close()


def compute_certificates_CRA(images, model, epsilon, correct_indices, norm=2, L=1):
    """
    Computes certificates, CRA, and the time taken for the core computation. Adapting to the norm.

    Returns:
        A tuple containing:
        - certificates (torch.Tensor): Certificate values for each correctly classified image.
        - cra (float): The Certified Robust Accuracy percentage.
        - elapsed_time (float): The computation time in seconds.
    """
    correct_images = images[correct_indices]
    total_num_images = correct_images.shape[0]

    if len(correct_images) == 0:
        return torch.tensor([]), 0.0, 0.0

    # --- Step 1: Time the core computation ---
    device = next(model.parameters()).device # Get the model's device
    
    # Synchronize before starting the timer to ensure accurate measurement
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    start_time = time.time()

    # The main computation: a single forward pass through the model
    with torch.no_grad():
        values, _ = torch.topk(model(correct_images.to(device)), k=2)

    # Synchronize again to ensure the computation is complete before stopping the timer
    if device.type == 'cuda':
        torch.cuda.synchronize()

    end_time = time.time()
    elapsed_time = end_time - start_time
    # --- End of Timing ---

    norm_str = str(norm).lower()

    # We adapt the computation of the certificate to the given norm
    if norm_str == '2':
        scale_certificate = np.sqrt(2)
    elif norm_str == 'inf':
        scale_certificate = 2.0
    else:
        raise ValueError(f"Unsupported norm: '{norm}'. Please use '2' or 'inf'.")

    # --- Step 2: Calculate certificates and CRA (same as before) ---
    certificates = (values[:, 0] - values[:, 1]) / (scale_certificate * L)
    num_robust_points = torch.sum(certificates >= epsilon).item()
    cra = (num_robust_points / total_num_images) * 100.0
    return certificates.cpu(), cra, elapsed_time/len(correct_indices)

def compute_pgd_era_and_time(images, targets, model, epsilon, clean_indices, norm=2):
    """
    Computes Empirical Robust Accuracy (CRA) against a PGD L2 attack and
    measures the mean computation time per image.

    Args:
        images (torch.Tensor): The entire batch of input images.
        targets (torch.Tensor): The corresponding labels for all images.
        model (torch.nn.Module): The model to be attacked.
        epsilon (float): The PGD attack radius.
        clean_indices (torch.Tensor): A tensor of indices for images that were
                                     initially classified correctly.

    Returns:
        A tuple containing:
        - cra (float): The Empirical Robust Accuracy percentage.
        - mean_time_per_image (float): The average attack time per image in seconds.
    """
    device = next(model.parameters()).device

    # --- Step 1: Filter the dataset to only attack clean images ---
    # We only care about robustness for images the model gets right to begin with.
    # correct_images = images[clean_indices].to(device)
    correct_images = images[clean_indices].contiguous().to(device)
    correct_targets = targets[clean_indices].to(device)

    total_num_images = len(clean_indices) #FIXME

    if len(correct_images) == 0:
        # If no images were correct, robust accuracy is 0 and time is 0.
        return 0.0, 0.0

    # --- Step 2: Set up and time the BATCH attack ---
    norm_str = str(norm).lower()

    # We adapt the computation of the certificate to the given norm
    if norm_str == '2':
        atk = torchattacks.PGDL2(model, eps=epsilon, alpha=epsilon/5, steps=int(10 * epsilon), random_start=True)
    elif norm_str == 'inf':
        atk = torchattacks.PGD(model, eps=epsilon, alpha=epsilon/5, steps=int(10 * epsilon), random_start=True)
    else:
        raise ValueError(f"Unsupported norm: '{norm}'. Please use '2' or 'inf'.")
    
    # atk = torchattacks.PGDL2(model, eps=epsilon, alpha=epsilon, steps=1, random_start=True)
    atk.set_normalization_used(mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1).cuda(), std = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1).cuda())

    # Synchronize GPU before starting the timer for accurate measurement
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start_time = time.time()

    # Generate adversarial examples for the ENTIRE BATCH
    adv_images = atk(correct_images, correct_targets)
    
    # Synchronize GPU again before stopping the timer
    if device.type == 'cuda':
        torch.cuda.synchronize()
    end_time = time.time()

    total_time = end_time - start_time
    mean_time_per_image = total_time / len(correct_images)

    # --- Step 3: Calculate the Empirical Robust Accuracy (CRA) ---
    # See how the model performs on the adversarial examples
    with torch.no_grad():
        adv_outputs = model(adv_images)
        adv_predictions = adv_outputs.argmax(dim=1)

        # Count how many adversarial images were still classified correctly
        num_robust_points = torch.sum(adv_predictions == correct_targets).item()

    # CRA is the percentage of robust points relative to the TOTAL dataset size
    cra = (num_robust_points / total_num_images) * 100.0

    return cra, mean_time_per_image

def compute_autoattack_era_and_time(images, targets, model, epsilon, clean_indices, norm=2):
    """
    Computes Empirical Robust Accuracy (CRA) against AutoAttack (L2) and
    measures the mean computation time per image.

    Args:
        images (torch.Tensor): The entire batch of input images.
        targets (torch.Tensor): The corresponding labels for all images.
        model (torch.nn.Module): The model to be attacked.
        epsilon (float): The AutoAttack radius.
        clean_indices (torch.Tensor): A tensor of indices for images that were
                                     initially classified correctly.

    Returns:
        A tuple containing:
        - cra (float): The Empirical Robust Accuracy percentage.
        - mean_time_per_image (float): The average attack time per image in seconds.
    """
    device = next(model.parameters()).device
    total_num_images = len(clean_indices) #FIXME

    # --- Step 1: Filter the dataset ---
    correct_images = images[clean_indices].contiguous().to(device)
    correct_targets = targets[clean_indices].to(device)

    if len(correct_images) == 0:
        return 0.0, 0.0
    #TODO Implement linf attacks
    # --- Step 2: Set up and time the BATCH attack ---
    atk = torchattacks.AutoAttack(model, norm='L2', eps=epsilon)
    atk.set_normalization_used(mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1).cuda(), std = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1).cuda())

    # Synchronize GPU before starting the timer
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start_time = time.time()

    # Generate adversarial examples for the ENTIRE BATCH
    adv_images = atk(correct_images, correct_targets)
    # Synchronize GPU again before stopping the timer
    if device.type == 'cuda':
        torch.cuda.synchronize()
    end_time = time.time()

    total_time = end_time - start_time
    mean_time_per_image = total_time / len(correct_images)

    # --- Step 3: Calculate the Empirical Robust Accuracy (CRA) ---
    with torch.no_grad():
        adv_outputs = model(adv_images)
        adv_predictions = adv_outputs.argmax(dim=1)
        num_robust_points = torch.sum(adv_predictions == correct_targets).item()
    # CRA is relative to the TOTAL dataset size
    cra = (num_robust_points / total_num_images) * 100.0

    return cra, mean_time_per_image

# sys.path.append('../SDP-CROWN-Share/auto_LiRPA')
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm


def build_C(label, classes):
    """
    label: shape (B,). Each label[b] in [0..classes-1].
    Return:
        C: shape (B, classes-1, classes).
        For each sample b, each row is a “negative class” among [0..classes-1]\{label[b]}.
        Puts +1 at column=label[b], -1 at each negative class column.
    """
    device = label.device
    batch_size = label.size(0)
    
    # 1) Initialize
    C = torch.zeros((batch_size, classes-1, classes), device=device)
    
    # 2) All class indices
    # shape: (1, K) -> (B, K)
    all_cls = torch.arange(classes, device=device).unsqueeze(0).expand(batch_size, -1)
    
    # 3) Negative classes only, shape (B, K-1)
    # mask out the ground-truth
    mask = all_cls != label.unsqueeze(1)
    neg_cls = all_cls[mask].view(batch_size, -1)
    
    # 4) Scatter +1 at each sample’s ground-truth label
    #    shape needed: (B, K-1, 1)
    pos_idx = label.unsqueeze(1).expand(-1, classes-1).unsqueeze(-1)
    C.scatter_(dim=2, index=pos_idx, value=1.0)
    
    # 5) Scatter -1 at each row’s negative label
    #    We have (B, K-1) negative labels. For row j in each sample b, neg_cls[b, j] is that row’s negative label
    row_idx = torch.arange(classes-1, device=device).unsqueeze(0).expand(batch_size, -1)
    # shape: (B, K-1)
    
    # We can do advanced indexing:
    C[torch.arange(batch_size).unsqueeze(1), row_idx, neg_cls] = -1.0
    
    return C

def compute_alphacrown_vra_and_time(images, targets, model, epsilon, clean_indices, batch_size=1, norm=2):
    """
    Computes Certified Robust Accuracy (CRA) using Alpha-Crown in batches to manage memory,
    and measures the mean verification time per image.
    
    Args:
        images (Tensor): The entire dataset's images.
        targets (Tensor): The entire dataset's labels.
        model (nn.Module): The model to verify.
        epsilon (float): The perturbation radius.
        clean_indices (Tensor): Indices of correctly classified images.
        batch_size (int): The number of images to verify in each batch.
    """
    device = next(model.parameters()).device
    total_num_images = len(images)
    model.eval()
    
    # --- Step 1: Filter for correctly classified samples ---
    correct_images = images[clean_indices]
    correct_targets = targets[clean_indices]

    if len(correct_images) == 0:
        return 0.0, 0.0

    # --- Step 2: Initialize variables for batch processing ---
    num_robust_points = 0
    total_time = 0.0
    num_batches = (len(correct_images) + batch_size - 1) // batch_size

    # --- Step 3: Set up a reusable BoundedModule ---
    # We only need to create this once
    dummy_input = correct_images[0:1].to(device)
    bounded_model = BoundedModule(model, dummy_input, bound_opts={"conv_mode": "patches"}, verbose=False)
    bounded_model.eval()

    print(f"Verifying {len(correct_images)} samples in {num_batches} batches of size {batch_size}...")

    # --- Step 4: Loop through the data in batches ---
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(correct_images))
        
        batch_images = correct_images[start_idx:end_idx].to(device)
        batch_targets = correct_targets[start_idx:end_idx]

        # --- Set up BoundedTensor and specification for the current batch ---
        ptb = PerturbationLpNorm(norm=norm, eps=epsilon)
        bounded_input = BoundedTensor(batch_images, ptb)
        
        num_classes = model[-1].out_features # Assuming last layer is Linear
        c = build_C(batch_targets.to("cpu"), num_classes).to(device)

        # --- Time the verification for this batch ---
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start_time_batch = time.time()

        bounded_model.set_bound_opts({'optimize_bound_args': {'iteration': 100, 'early_stop_patience': 20, 'fix_interm_bounds': False, 'enable_opt_interm_bounds':True, 'verbosity':False}, 'verbosity':False})
        lb_diff = bounded_model.compute_bounds(x=(bounded_input,), C=c, method='alpha-crown')[0]

        if device.type == 'cuda':
            torch.cuda.synchronize()
        end_time_batch = time.time()
        
        total_time += (end_time_batch - start_time_batch)

        # --- Calculate robust points in the current batch ---
        is_robust = (lb_diff.view(len(batch_images), num_classes - 1) > 0).all(dim=1)
        num_robust_points += torch.sum(is_robust).item()

        # Optional: Print progress
        # print(f"  Batch {i+1}/{num_batches}: {torch.sum(is_robust).item()}/{len(batch_images)} robust.", end='\r')

    # print("\nBatch verification finished.") 
    
    # --- Step 5: Calculate final metrics ---
    # CRA is relative to the TOTAL dataset size
    cra = (num_robust_points / total_num_images) * 100.0
    mean_time_per_image = total_time / len(correct_images) if len(correct_images) > 0 else 0.0

    return cra, mean_time_per_image

#MAKE Imports



import time
import torch
import numpy as np
sys.path.append("/home/aws_install/robustess_project/alpha-beta-CROWN/complete_verifier")
from abcrown import ABCROWN # Import the main class from your script


def compute_alphabeta_vra_and_time(dataset_name, model_name, model_path, epsilon, CONFIG_FILE, norm='inf'):
    """
    Computes Certified Robust Accuracy (CRA) using α-β-CROWN and
    measures the mean verification time per image.
    """
    print(epsilon)

    params = {
            'model': model_name,
            'load_model': model_path,  # Use 'load_model' for the path
            'dataset': 'CIFAR_SDP',
            'epsilon': epsilon
            # TODO link with norm:
        }

    # Compute the lower bounds on the logit differences using α-β-CROWN
    verifier = ABCROWN(
            args=[],
            config=CONFIG_FILE,
            **params
        )
    summary = verifier.main()


    # Extract results.
    total = summary.get('total', 0)
    print(total)
    safe = summary.get('safe', 0)
    robust_accuracy = (safe / total) * 100 if total > 0 else 0
            
    instance_times = [res[2] for res in verifier.logger.results]
    avg_time = sum(instance_times) / len(instance_times) if instance_times else 0

    return robust_accuracy, avg_time



sys.path.append('/home/aws_install/robustess_project/SDP-CROWN-Share')
from sdp_crown import verified_sdp_crown

def compute_sdp_crown_vra(dataset, labels, model, radius, clean_output, device, classes, args):
    return verified_sdp_crown(dataset, labels, model, radius, clean_output, device, classes, args)

def evaluate(args):
    """
    Main function to evaluate a pre-trained model on a given dataset.
    """
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at '{args.model_path}'")
        return

    print("--- Evaluation Setup ---")
    print(f"Model Path: {args.model_path}")
    print(f"Model Name: {args.model_name}")
    print(f"Dataset: {args.dataset}")
    print(f"Batch Size: {args.batch_size}")
    print("------------------------")

    # --- 1. Setup Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # --- 2. Load Dataset ---
    # We only need the test_loader for evaluation.
    try:
        _, test_loader = load_dataset(args.dataset, args.batch_size)
    except NotImplementedError as e:
        print(f"Error: {e}")
        return
        
    num_classes = 10  # Assuming 10 classes for both CIFAR-10 and MNIST

    # --- 3. Initialize Model ---
    model_zoo = {
        "MLP_MNIST": MLP_MNIST,
        "ConvSmall_MNIST": ConvSmall_MNIST,
        "ConvLarge_MNIST": ConvLarge_MNIST,
        "CNNA_CIFAR10": CNNA_CIFAR10,
        "CNNB_CIFAR10": CNNB_CIFAR10,
        "CNNC_CIFAR10": CNNB_CIFAR10, # Note: CNNC uses CNNB class in training script
        "ConvSmall_CIFAR10": ConvSmall_CIFAR10,
        "ConvDeep_CIFAR10": ConvDeep_CIFAR10,
        "ConvLarge_CIFAR10": ConvLarge_CIFAR10,
    }

    if args.model_name not in model_zoo:
        raise ValueError(f"Model '{args.model_name}' not found. Available models are: {list(model_zoo.keys())}")
    
    ModelClass = model_zoo[args.model_name]
    model = ModelClass().vanilla_export()
    
    # Load the saved state dictionary
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval() # Set model to evaluation mode

    # --- 4. Setup Loss Functions for Metrics ---
    # Use dummy parameters for the custom loss, as they don't affect metric calculation.
    criterion = HKRMultiLossLSE(alpha=250, temperature=200.0, penalty=0.5, margin=1.0)
    kr_multiclass_loss = torchlip.KRMulticlassLoss()

    # --- 5. Evaluation Loop ---
    print("Starting evaluation...")
    test_output, test_targets = [], []
    with torch.no_grad(): # Disable gradient calculations
        for data, target in test_loader:
            data = data.to(device)
            # Store outputs and targets on CPU to save GPU memory
            test_output.append(model(data).cpu())
            test_targets.append(torch.nn.functional.one_hot(target, num_classes=num_classes).cpu())
    
    # Concatenate all batches
    test_output = torch.cat(test_output)
    test_targets = torch.cat(test_targets)

    # --- 6. Calculate and Print Final Metrics ---
    accuracy = (test_output.argmax(dim=1) == test_targets.argmax(dim=1)).float().mean().item()
    hkr_loss = criterion(test_output, test_targets).item()
    kr_loss = kr_multiclass_loss(test_output, test_targets).item()

    print("\n--- Evaluation Results ---")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"HKR Loss: {hkr_loss:.4f}")
    print(f"KR Loss:  {kr_loss:.4f}")
    print("--------------------------")

