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
    MEANS = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
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

    if norm == '2':
        # If the target norm is L2, no conversion is needed.
        return L_2
    elif norm == 'inf':
        # Convert L2 constant to an L-infinity constant upper bound.
        # The relationship is L_inf <= sqrt(input_dim) * L_2
        L_inf = L_2 * np.sqrt(input_dim)
        return L_inf
    else:
        raise ValueError(f"Unsupported norm: '{norm}'. Please use '2' or 'inf'.")
    
def add_result_and_sort(result_dict, base_csv_filepath, round_digits=3, norm='2'):
    """
    Adds a new result to a norm-specific CSV file and sorts it.

    The function constructs a filename based on the provided norm (e.g.,
    'results_2.csv', 'results_inf.csv'), reads the existing data from that
    specific file, adds the new result, sorts all entries by the 'epsilon'
    value, and then overwrites the file.

    Args:
        result_dict (dict): A dictionary containing the new row of data.
        base_csv_filepath (str): The base path for the CSV file. The norm will be
                                 appended to this filename.
        round_digits (int): The number of decimal places for rounding time values.
        norm (int or str): The norm used for the result (e.g., 2 or 'inf').
    """
    # --- 1. Construct the norm-specific filename ---
    name, ext = os.path.splitext(base_csv_filepath)
    csv_filepath = f"{name}_norm_{norm}{ext}"

    # --- 2. Prepare the data and header ---
    # The header is now just the keys of the result dictionary.
    header = list(result_dict.keys())

    # --- 3. Read all existing data into a list of dictionaries ---
    all_data_dicts = []
    
    directory = os.path.dirname(csv_filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    if os.path.exists(csv_filepath) and os.path.getsize(csv_filepath) > 0:
        try:
            with open(csv_filepath, 'r', newline='') as file:
                reader = csv.DictReader(file)
                # Check for header consistency
                if reader.fieldnames and set(reader.fieldnames) != set(header):
                     print(f"⚠️  Warning: Header mismatch in '{csv_filepath}'. Overwriting with new data.")
                else:
                    all_data_dicts.extend(list(reader))
        except Exception as e:
            print(f"⚠️  Could not parse existing file '{csv_filepath}'. Starting fresh. Error: {e}")

    # --- 4. Add the new result and apply rounding ---
    processed_result = {}
    for key, value in result_dict.items():
        if str(key).startswith('time_') and isinstance(value, (float, int)):
            processed_result[key] = round(value, round_digits)
        else:
            processed_result[key] = value
    all_data_dicts.append(processed_result)

    # --- 5. Sort the data by the 'epsilon' value ---
    try:
        all_data_dicts.sort(key=lambda row_dict: float(row_dict['epsilon']))
    except (KeyError, ValueError) as e:
        print(f"❌ Error: Could not sort. Ensure 'epsilon' exists and is a number. Error: {e}")
        return 

    # --- 6. Write the sorted data back to the file ---
    try:
        with open(csv_filepath, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=header)
            writer.writeheader()
            writer.writerows(all_data_dicts)
        
        print(f"✅ Successfully added result to '{csv_filepath}' for epsilon={result_dict['epsilon']}.")
    except Exception as e:
        print(f"❌ Error: Failed to write to CSV file '{csv_filepath}'. Error: {e}")


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os # Imported to check file name

def create_robustness_plot_v2(filepath='results/results.csv', output_filename='robust_accuracy_vs_epsilon_colored.png'):
    """
    Generates a plot of robust accuracy vs. epsilon with a custom, distinct color palette.

    - 'certificate' is blue (distinct).
    - 'aa' is orangish.
    - 'pgd' is NOT plotted.
    - 'sdp' is NOT plotted if 'Linf' is in the filepath.
    - 'lirpa_betacrown' is NOT plotted if 'L2' is in the filepath.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"❌ Error: The file '{filepath}' was not found.")
        return

    # Prepare data
    df = df.sort_values(by='epsilon').reset_index(drop=True)
    
    # --- ✨ MODIFIED: 'pgd' removed from the base plot order ---
    plot_order = ['certificate', 'aa', 'lirpa_alphacrown', 'lirpa_betacrown', 'sdp']
    
    # --- ✨ NEW: Conditional Filtering based on filename ---
    # Start with the base list of methods to consider
    methods_to_plot = list(plot_order) 

    # Conditionally remove methods based on the filepath
    # We use os.path.basename to just check the filename, not the whole path
    filename_only = os.path.basename(filepath)
    
    if 'Linf' in filename_only:
        if 'sdp' in methods_to_plot:
            methods_to_plot.remove('sdp')
            print("ℹ️ 'Linf' found in filename. Removing 'sdp' from the plot.")
    elif 'L2' in filename_only:
        if 'lirpa_betacrown' in methods_to_plot:
            methods_to_plot.remove('lirpa_betacrown')
            print("ℹ️ 'L2' found in filename. Removing 'lirpa_betacrown' from the plot.")
            
    # Filter to only include columns that are in our conditional list AND are actually in the DataFrame
    method_columns = [col for col in methods_to_plot if col in df.columns and col != 'epsilon' and not col.startswith('time_')]

    # --- ✨ MODIFIED: Custom Color Mapping (removed 'pgd') ---
    color_map = {}
    
    # 1. Assign a distinct color to 'certificate'
    if 'certificate' in method_columns:
        color_map['certificate'] = '#1f77b4'  # A standard Matplotlib blue

    # 2. Assign orangish color to 'aa'
    if 'aa' in method_columns:
        color_map['aa'] = '#ff7f0e'   # Matplotlib orange

    # 3. Assign distinct colors for the remaining methods
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
                # Fallback color if we run out
                color_map[method] = '#7f7f7f' # tab10 grey


    # Create the plot
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 8))

    for method in method_columns:
        # Use the color assigned to each method in our new map
        plt.plot(df['epsilon'], df[method], 
                 marker='o', 
                 linestyle='-', 
                 label=method, 
                 color=color_map.get(method, '#000000'), # Use .get for safety
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
    print(f"✅ Plot successfully saved as '{output_filename}' with specified methods.")
    plt.close()

# --- Example Usage ---
# create_robustness_plot(filepath='results/Linf_results.csv', output_filename='Linf_plot.png')
# create_robustness_plot(filepath='results/L2_data.csv', output_filename='L2_plot.png')
# create_robustness_plot(filepath='results/other_results.csv', output_filename='other_plot.png')

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


def compute_certificates_CRA(images, model, epsilon, correct_indices, norm='2', L=1):
    """
    Computes certificates, CRA, and the time taken for the core computation. Adapting to the norm.

    Returns:
        A tuple containing:
        - certificates (torch.Tensor): Certificate values for each correctly classified image.
        - cra (float): The Certified Robust Accuracy percentage.
        - elapsed_time (float): The computation time in seconds.
    """
    print(norm)
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

    # We adapt the computation of the certificate to the given norm
    if norm == '2':
        scale_certificate = np.sqrt(2)
    elif norm == 'inf':
        scale_certificate = 2.0
    else:
        raise ValueError(f"Unsupported norm: '{norm}'. Please use '2' or 'inf'.")

    # --- Step 2: Calculate certificates and CRA (same as before) ---
    certificates = (values[:, 0] - values[:, 1]) / (scale_certificate * L)
    num_robust_points = torch.sum(certificates >= epsilon).item()
    cra = (num_robust_points / total_num_images) * 100.0
    return certificates.cpu(), cra, elapsed_time/len(correct_indices)

def compute_pgd_era_and_time(images, targets, model, epsilon, clean_indices, norm='2', dataset_name='cifar10'):
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
    total_num_images = len(clean_indices) 

    # --- Step 1: Filter the dataset to only attack clean images ---
    # We only care about robustness for images the model gets right to begin with.
    # correct_images = images[clean_indices].to(device)
    correct_images = images[clean_indices].contiguous().to(device)
    correct_targets = targets[clean_indices].to(device)

    

    if len(correct_images) == 0:
        # If no images were correct, robust accuracy is 0 and time is 0.
        return 0.0, 0.0

    # --- Step 2: Set up and time the BATCH attack ---

    # We adapt the computation of the certificate to the given norm
    if norm == '2':
        atk = torchattacks.PGDL2(model, eps=epsilon, alpha=epsilon/5, steps=int(10 * epsilon))
    elif norm == 'inf':
        atk = torchattacks.PGD(model, eps=epsilon, alpha=epsilon/5, steps=int(10 * epsilon))
    else:
        raise ValueError(f"Unsupported norm: '{norm}'. Please use '2' or 'inf'.")
    
    # atk = torchattacks.PGDL2(model, eps=epsilon, alpha=epsilon, steps=1, random_start=True)
    if dataset_name=="cifar10":
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

def compute_autoattack_era_and_time(images, targets, model, epsilon, clean_indices, norm='2', dataset_name='cifar10'):
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
    total_num_images = len(clean_indices) 

    # --- Step 1: Filter the dataset ---
    correct_images = images[clean_indices].contiguous().to(device)
    correct_targets = targets[clean_indices].to(device)

    if len(correct_images) == 0:
        return 0.0, 0.0
    
    # --- Step 2: Set up and time the BATCH attack ---
    # We adapt the computation of the certificate to the given norm
    if norm == '2':
        atk = torchattacks.AutoAttack(model, norm='L2', eps=epsilon)
    elif norm == 'inf':
        atk = torchattacks.AutoAttack(model, norm='Linf', eps=epsilon)
    else:
        raise ValueError(f"Unsupported norm: '{norm}'. Please use '2' or 'inf'.")
    
    if dataset_name=="cifar10":
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
        robust_mask = (adv_predictions == correct_targets)
        num_robust_points = torch.sum(robust_mask).item()
        non_robust_mask = ~robust_mask  # or (adv_predictions != correct_targets)
        
        # Get the indices of the non-robust points within the current batch
        non_robust_indices = torch.nonzero(non_robust_mask, as_tuple=True)[0]
        
        # Print the list of indices
        if non_robust_indices.numel() > 0:
            print(f"Indices of non-robust images in this batch: {non_robust_indices.tolist()}")
        # CRA is relative to the TOTAL dataset size
        cra = (num_robust_points / total_num_images) * 100.0
#[7, 34, 39, 40, 47, 66, 67, 77, 80, 95, 114, 126]
    return cra, mean_time_per_image
# 7 34 39 40 66 80 95 114 126
#safe-incomplete (total 118), index: [0, 1, 2, 7, 8, 13, 14, 17, 18, 20, 21, 22, 24, 27, 28, 30, 31, 32, 33, 34, 37, 38, 39, 40, 41, 42, 47, 48, 49, 51, 52, 54, 55, 59, 60, 62, 64, 66, 68, 69, 70, 71, 73, 74, 79, 80, 81, 82, 83, 84, 85, 86, 87, 89, 91, 93, 95, 98, 99, 101, 103, 104, 108, 109, 110, 111, 112, 114, 115, 116, 117, 118, 121, 122, 126, 131, 133, 135, 136, 137, 139, 140, 142, 144, 145, 146, 148, 153, 154, 157, 158, 159, 160, 161, 164, 166, 167, 168, 171, 172, 174, 175, 177, 179, 180, 181, 182, 183, 186, 187, 188, 189, 190, 192, 194, 195, 197, 199]
# unsafe-pgd (total 81), index: [3, 4, 5, 6, 9, 10, 11, 12, 15, 16, 19, 23, 25, 26, 29, 35, 36, 43, 44, 45, 46, 50, 53, 56, 57, 61, 63, 65, 67, 72, 75, 76, 77, 78, 88, 90, 92, 94, 96, 97, 100, 102, 105, 106, 107, 113, 119, 120, 123, 124, 125, 127, 128, 129, 130, 132, 134, 138, 141, 143, 147, 149, 150, 151, 152, 155, 156, 162, 163, 165, 169, 170, 173, 176, 178, 184, 185, 191, 193, 196, 198]
# safe (total 1), index: [58]
import sys
sys.path.insert(0,'/home/aws_install/robustess_project/SDP-CROWN')
import auto_LiRPA
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm

print(auto_LiRPA.__file__)

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

def compute_alphacrown_vra_and_time(images, targets, model, epsilon, clean_indices, batch_size=2, norm=2):
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
        if norm=='inf':
            ptb = PerturbationLpNorm(norm=np.inf, eps=epsilon)
        else:
            ptb = PerturbationLpNorm(norm=norm, eps=epsilon)

        bounded_input = BoundedTensor(batch_images, ptb)
        
        num_classes = model[-1].out_features # Assuming last layer is Linear
        c = build_C(batch_targets.to("cpu"), num_classes).to(device)

        # --- Time the verification for this batch ---
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start_time_batch = time.time()
        bounded_model.set_bound_opts({'optimize_bound_args': {'iteration': 200, 'early_stop_patience': 30, 'fix_interm_bounds': False, 'enable_opt_interm_bounds':True, 'verbosity':False}, 'verbosity':False})
        lb_diff = bounded_model.compute_bounds(x=(bounded_input,), C=c, method='alpha-crown')[0]
        if device.type == 'cuda':
            torch.cuda.synchronize()
        end_time_batch = time.time()
        
        total_time += (end_time_batch - start_time_batch)

        # --- Calculate robust points in the current batch ---
        is_robust = (lb_diff.view(len(batch_images), num_classes - 1) > 0).all(dim=1)
        num_robust_points += torch.sum(is_robust).item()

        # Optional: Print progress
        print(f"  Batch {i+1}/{num_batches}: {torch.sum(is_robust).item()}/{len(batch_images)} robust.", end='\r')

    print("\nBatch verification finished.") 
    
    # --- Step 5: Calculate final metrics ---
    # CRA is relative to the TOTAL dataset size
    cra = (num_robust_points / total_num_images) * 100.0
    mean_time_per_image = total_time / len(correct_images) if len(correct_images) > 0 else 0.0

    return cra, mean_time_per_image


import time
import torch
import numpy as np
sys.path.append("/home/aws_install/robustess_project/alpha-beta-CROWN/complete_verifier")
from abcrown import ABCROWN # Import the main class from your script


import time  # Import the time module

import logging
logging.getLogger('auto_LiRPA').setLevel(logging.WARNING)

def compute_alphabeta_vra_and_time(dataset_name, model_name, model_path, epsilon, CONFIG_FILE, clean_indices, norm='inf'):
    """
    Computes Certified Robust Accuracy (CRA) using α-β-CROWN and
    measures the mean verification time per image.
    """
    #TODO apply to different dataset_names
    # model_path = model_path.replace('models/', 'models/vanilla_')
    # print(model_path)
    if dataset_name=='cifar10':
        dataset = 'CIFAR_SDP'
    elif dataset_name=='mnist':
        dataset = 'MNIST_SDP'
    else :
        raise ValueError(f"Unsupported dataset")
    
    params = {
            'model': model_name,
            'load_model': model_path,  # Use 'load_model' for the path
            'dataset': dataset,
            'epsilon': epsilon,
        }

    # Compute the lower bounds on the logit differences using α-β-CROWN
    verifier = ABCROWN(
            args=[],
            config=CONFIG_FILE,
            **params
        )
    
    # --- Start Timing ---
    start_time = time.time()
    
    summary = verifier.main()
    
    end_time = time.time()
    # --- End Timing ---


    # --- Calculate Total Samples Verified ---
    # This is more robust than hardcoding 200
    total_samples_verified = sum(len(v) for v in summary.values())

    # --- Calculate Average Time ---
    total_time = end_time - start_time
    avg_time = total_time / total_samples_verified if total_samples_verified > 0 else 0


    # --- Accuracy Calculations ---
    clean_indices_set = {t.item() for t in clean_indices}
    validated_indices_set = set()
    validated_keys = ['safe-incomplete', 'safe']
    
    # Note: 'total_validated' here means "total proven safe", not "total processed"
    total_proven_safe = 0 

    for key in validated_keys:
        validated_indices_set.update(summary.get(key, []))
        total_proven_safe += len(summary.get(key, [])) # .get() is safer if a key might be missing

    validated_clean_indices = validated_indices_set.intersection(clean_indices_set)
    count_validated_clean = len(validated_clean_indices)

    denominator = len(clean_indices)
    certified_robust_accuracy = (count_validated_clean / denominator) * 100 if denominator > 0 else 0

    print(f"🚀 Verification Complete!")
    print(f"   - Total samples verified: {total_samples_verified}")
    print(f"   - Correctly classified (clean): {denominator}")
    print(f"   - Correctly classified AND robust (clean & safe): {count_validated_clean}")
    print(f"   - Certified Robust Accuracy: {certified_robust_accuracy:.2f}%")
    print(f"   - Total verification time: {total_time:.2f} seconds")
    print(f"   - Average time per sample: {avg_time:.4f} seconds")
    # import pdb; pdb.set_trace()

    return certified_robust_accuracy, avg_time


sys.path.append('/home/aws_install/robustess_project/SDP-CROWN')
from sdp_crown import verified_sdp_crown


def compute_sdp_crown_vra(dataset, labels, model, radius, clean_output, device, classes, args):
    return verified_sdp_crown(dataset, labels, model, radius, clean_output, device, classes, args)
