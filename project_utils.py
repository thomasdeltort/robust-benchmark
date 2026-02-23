from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import os
from models import * # Make sure this import works from your utils.py location
import shutil
# try:
#     sys.path.append('/home/aws_install/robustess_project/lip_notebooks/notebooks_creation_models')
#     from VGG_Arthur import HKRMultiLossLSE
# except ImportError:
#     print("Warning: Could not import HKRMultiLossLSE. Using standard CrossEntropyLoss for evaluation.")
# import pickle
import numpy as np
import torchattacks
import matplotlib.pyplot as plt
import time
import csv
import os
import copy
from torch.nn.utils.parametrize import is_parametrized

import torch
from torchvision.transforms import v2
from torchvision import datasets
from torch.utils.data import DataLoader



def load_cifar10(batch_size, aug_level='medium'):
    """
    Args:
        batch_size (int): Size of the batch.
        aug_level (str): Level of augmentation ('none', 'light', 'medium', 'heavy').
    """
    
    # 1. Define Common Base and Normalization
    # These are applied to all levels and the test set
    base_transforms = [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ]
    
    norm_transform = v2.Normalize(
        mean=(0.49139968, 0.48215827, 0.44653124),
        std=(0.225, 0.225, 0.225)
    )

    # 2. Select Augmentation Strategy
    augmentations = []

    if aug_level == 'none':
        # No extra augmentations
        pass

    elif aug_level == 'light':
        # Standard CIFAR-10 augmentations (Crop + Flip)
        augmentations = [
            v2.RandomCrop((32, 32), padding=4),
            v2.RandomHorizontalFlip(p=0.5),
        ]

    elif aug_level == 'medium':
        # Your specific implementation (Affine + ColorJitter)
        augmentations = [
            v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            v2.RandomAffine(
                degrees=5,           # Rotate by a maximum of 5 degrees
                translate=(0.05, 0.05) # Shift by a maximum of 5%
            ),
        ]

    elif aug_level == 'heavy':
        # State-of-the-art style (RandAugment + Erasing)
        augmentations = [
            v2.RandAugment(num_ops=2, magnitude=9),
            v2.RandomHorizontalFlip(), 
            v2.RandomErasing(p=0.25, scale=(0.02, 0.1), ratio=(0.3, 3.3), value='random')
        ]
    
    else:
        raise ValueError(f"Invalid aug_level: {aug_level}. Choose 'none', 'light', 'medium', or 'heavy'.")

    # 3. Compose Transforms
    train_transforms = v2.Compose(base_transforms + augmentations + [norm_transform])
    
    test_transforms = v2.Compose(base_transforms + [norm_transform])

    # 4. Load Datasets
    train_dataset = datasets.CIFAR10(
        root='./data', train=True, transform=train_transforms, download=True
    )
    
    test_dataset = datasets.CIFAR10(
        root='./data', train=False, transform=test_transforms, download=True
    )

    # 5. Create Loaders
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    test_loader = DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return train_loader, test_loader

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

def load_dataset(name, batch_size, aug_level = 'medium'):
    if name=="mnist":
        train_loader, test_loader = load_mnist(batch_size)
    elif name=="cifar10":
        train_loader, test_loader = load_cifar10(batch_size, aug_level)
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

def load_dataset_benchmark_auto(args):
    if "mnist" in args.dataset.lower():
        dataset = np.load('./prepared_data/mnist/X_sdp.npy')
        labels = np.load('./prepared_data/mnist/y_sdp.npy')
        dataset = torch.from_numpy(dataset).permute(0,3,1,2)
        labels = torch.from_numpy(labels)
        classes = 10
    elif "cifar10" in args.dataset.lower():
        dataset = np.load('./prepared_data/cifar/X_sdp.npy')
        labels = np.load('./prepared_data/cifar/y_sdp.npy')
        dataset = preprocess_cifar(dataset)
        dataset = torch.from_numpy(dataset).permute(0,3,1,2)
        print(dataset.shape, "we need to verify channel first")
        labels = torch.from_numpy(labels)
        classes = 10
    else:
        raise ValueError(f"Unexpected model: {args.model}")
    return dataset, labels, classes




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
    model = vanilla_export(ModelClass())
    # .vanilla_export()
    # import pdb;pdb.set_trace()
    # Load the saved state dictionary
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    # model.load_state_dict(torch.load("models/cifar10_convsmall.pth", map_location=device))
    model.to(device)
    model.eval() # Set model to evaluation mode
    return model


def _find_output_padding(output_shape, input_shape, weight, stride, dilation, padding):
    """
    Calculates the output_padding needed for ConvTranspose2d to recover the original input shape.
    """
    output_padding0 = (
        int(input_shape[2]) - (int(output_shape[2]) - 1) * stride[0] + 2 *
        padding[0] - 1 - (int(weight.size()[2] - 1) * dilation[0]))
    output_padding1 = (
        int(input_shape[3]) - (int(output_shape[3]) - 1) * stride[1] + 2 *
        padding[1] - 1 - (int(weight.size()[3] - 1) * dilation[1]))
    return (output_padding0, output_padding1)

def power_iteration_conv(weight, input_shape, output_shape, stride, padding, dilation, groups, num_iterations=100):
    """
    Calculate the Lipschitz constant of a convolutional layer using Power Iteration.
    Uses the forward/backward pass method (Conv -> TransposeConv).
    """
    device = weight.device
    
    with torch.no_grad():
        # 1. Calculate the padding needed to restore the exact input shape
        output_padding = _find_output_padding(output_shape=output_shape, input_shape=input_shape,
                                              weight=weight, stride=stride, dilation=dilation, padding=padding)
        
        # 2. Initialize a random probe vector x_k with the Input Shape
        x_k = torch.randn(input_shape, device=device)
        
        # Normalize initialization
        norm = x_k.view(x_k.size(0), -1).norm(dim=1, keepdim=True).view(-1, 1, 1, 1)
        x_k = x_k / (norm + 1e-8)

        sigma = 0
        for _ in range(num_iterations):
            # A. Forward Pass (Conv2d)
            # x_{k+1} = Conv(x_k)
            x_k1 = F.conv2d(x_k, weight, bias=None, stride=stride, padding=padding, dilation=dilation, groups=groups)
            
            # Normalize the result (Block-Krylov style normalization)
            conv_norm = x_k1.reshape(x_k1.size(0), -1).norm(dim=1, keepdim=True).view(-1, 1, 1, 1)
            x_k1 = x_k1 / (conv_norm + 1e-8)
            
            # B. Backward Pass (ConvTranspose2d)
            # x_k = ConvTranspose(x_{k+1})
            x_k = F.conv_transpose2d(x_k1, weight, bias=None, stride=stride,
                                     padding=padding, dilation=dilation, 
                                     groups=groups, output_padding=output_padding)
            
            # Normalize again
            deconv_norm = x_k.reshape(x_k.size(0), -1).norm(dim=1, keepdim=True).view(-1, 1, 1, 1)
            x_k = x_k / (deconv_norm + 1e-8)
            
            # The singular value approximation is the norm before normalization in the forward pass
            # We take the max over the batch dimension if batch > 1
            sigma = conv_norm.view(conv_norm.size(0), -1).max(dim=1)[0]
            
    return sigma.item()

def compute_linear_spectral_norm(weight):
    """
    Compute spectral norm for Linear layers. 
    Using PyTorch's exact SVD for stability on standard layer sizes.
    """
    with torch.no_grad():
        return torch.linalg.norm(weight, ord=2).item()

# --- 2. The Dynamic Model Analyzer ---

def compute_model_lipschitz(model, input_shape=(1, 3, 32, 32), device='cuda'):
    """
    Propagates a dummy input through the network to capture shapes, 
    then computes the product of spectral norms (rho) for all learnable layers.
    """
    model = model.to(device)
    model.eval()
    
    # Dummy input to propagate shapes
    current_input = torch.randn(input_shape).to(device)
    
    L_global = 1.0
    print(f"\n{'Layer':<40} | {'Type':<15} | {'Spectral Norm (rho)':<10}")
    print("-" * 80)

    # We iterate over immediate children (assuming Sequential structure from your examples)
    for name, module in model.named_children():
        
        # 1. Capture Input Shape
        in_shape = current_input.shape
        
        # 2. Run Forward Pass to get Output Shape and Next Input
        with torch.no_grad():
            current_input = module(current_input)
        out_shape = current_input.shape
        
        layer_rho = 1.0 # Default (Activations, Pooling, etc.)
        
        # 3. Compute Spectral Norm based on Type
        if isinstance(module, nn.Conv2d): 
            # Extract parameters safely (handling tuples vs ints)
            s = module.stride if isinstance(module.stride, tuple) else (module.stride, module.stride)
            p = module.padding if isinstance(module.padding, tuple) else (module.padding, module.padding)
            d = module.dilation if isinstance(module.dilation, tuple) else (module.dilation, module.dilation)
            
            layer_rho = power_iteration_conv(
                weight=module.weight,
                input_shape=in_shape,
                output_shape=out_shape,
                stride=s,
                padding=p,
                dilation=d,
                groups=module.groups
            )
            
        elif isinstance(module, nn.Linear):
            layer_rho = compute_linear_spectral_norm(module.weight)
            
        # 4. Update Global
        L_global *= layer_rho
        
        # Log non-trivial layers (rho != 1.0) or specifically Conv/Linear
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            print(f"{name:<40} | {module.__class__.__name__:<15} | {layer_rho:.4f}")

    print("-" * 80)
    # print(f"Computed Global Lipschitz Constant: {L_global:.4f}")
    return L_global


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
    Safer version: Does not wipe data on header mismatch.
    """
    # --- 1. Construct the norm-specific filename ---
    name, ext = os.path.splitext(base_csv_filepath)
    csv_filepath = f"{name}_norm_{norm}{ext}"

    # --- 2. Prepare the data and header ---
    current_keys = list(result_dict.keys())
    
    # --- 3. Read all existing data ---
    all_data_dicts = []
    
    directory = os.path.dirname(csv_filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    existing_fieldnames = []

    if os.path.exists(csv_filepath) and os.path.getsize(csv_filepath) > 0:
        try:
            with open(csv_filepath, 'r', newline='') as file:
                reader = csv.DictReader(file)
                existing_fieldnames = reader.fieldnames if reader.fieldnames else []
                # Load data regardless of header mismatch
                all_data_dicts.extend(list(reader))
                
                # Check for mismatch just for notification
                if set(existing_fieldnames) != set(current_keys):
                    print(f"⚠️  Warning: Header mismatch in '{csv_filepath}'. Merging headers.")
                    
        except Exception as e:
            print(f"❌ CRITICAL ERROR: Could not read existing file '{csv_filepath}'.")
            print(f"❌ Error details: {e}")
            return 

    # --- 4. Merge Headers ---
    # Create a superset of all keys found in old file and new result
    final_header = list(dict.fromkeys(existing_fieldnames + current_keys))

    # --- 5. Add the new result and apply rounding ---
    processed_result = {}
    for key, value in result_dict.items():
        if str(key).startswith('time_') and isinstance(value, (float, int)):
            processed_result[key] = round(value, round_digits)
        else:
            processed_result[key] = value
    all_data_dicts.append(processed_result)

    # --- 6. Sort ---
    try:
        all_data_dicts.sort(key=lambda row_dict: float(row_dict.get('epsilon', 0)))
    except (KeyError, ValueError) as e:
        print(f"❌ Error: Could not sort. {e}")

    # --- 7. Write Safely ---
    try:
        # Use 'w' to overwrite, but we have all previous data in all_data_dicts
        with open(csv_filepath, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=final_header)
            writer.writeheader()
            writer.writerows(all_data_dicts)
        
        print(f"✅ Successfully added result to '{csv_filepath}' for epsilon={result_dict['epsilon']}.")
    except Exception as e:
        print(f"❌ Error: Failed to write to CSV file '{csv_filepath}'. Error: {e}")


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


# --- 1. Global Style Configuration ---
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "lines.linewidth": 2.5
})

def get_clean_title(filename):
    """
    Extracts Model, Dataset, and specific keywords (GNP, Lip, Bjork) from filename.
    Ignores 'vanilla'.
    """
    base = os.path.basename(filename)
    parts = base.split('_')
    
    keywords_to_keep = []
    
    # 1. Helper function to check keywords case-insensitively but preserve original casing
    def check_keyword(part, target):
        return part.lower() == target.lower()

    # Iterate through parts to find relevant info
    # We want to maintain the order they appear in the filename
    
    # Define all keywords we care about
    model_keywords = ['ConvSmall', 'ConvLarge', 'MLP', 'CNNA', 'ResNet', 'CNN', 'ViT', 'ResNet18']
    dataset_keywords = ['MNIST', 'CIFAR', 'CIFAR10', 'TinyImageNet']
    extra_keywords = ['GNP', 'Lip', 'Bjork'] # Added specific requests
    
    for part in parts:
        # Check Model
        if part in model_keywords:
            keywords_to_keep.append(part)
            continue
            
        # Check Dataset
        if part in dataset_keywords:
            keywords_to_keep.append(part)
            continue
            
        # Check Extra Keywords (Case insensitive check, append original)
        for extra in extra_keywords:
            if check_keyword(part, extra):
                keywords_to_keep.append(part)
                break
            
    if keywords_to_keep:
        return " ".join(keywords_to_keep)
    else:
        return "Robustness Evaluation"

def create_final_paper_plot(filepath, output_filename):
    """
    Generates the final plot with:
    - Expanded Title (Model + Dataset + GNP/Lip/Bjork).
    - Legend in Upper Right.
    - Distinct line styles.
    """
    # --- Load Data ---
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"❌ Error: File '{filepath}' not found.")
        return

    # --- Prepare Title & Norm ---
    title_text = get_clean_title(filepath)
    filename_lower = os.path.basename(filepath).lower()
    norm_label = r"$\ell_\infty$" if "norm_inf" in filename_lower else r"$\ell_2$"
    
    # --- Define Styles ---
    # Use empty tuple () for solid lines to fix TypeError
    styles = {
        'aa':               {'label': 'Upper Bound (Empirical)', 'color': '#8B0000', 'style': '--', 'dashes': (5, 3), 'zorder': 10},
        'certificate':      {'label': 'CRA',           'color': '#0072B2', 'style': '-',  'dashes': (),        'zorder': 5},
        # 'certificate_pi':   {'label': 'CRA (Pi)',      'color': "#850BF8", 'style': '--', 'dashes': (3, 1),    'zorder': 5}, # Teal/Greenish
        'lirpa_alphacrown': {'label': r'$\alpha$-CROWN', 'color': '#009E73', 'style': '-.', 'dashes': (3, 1, 1, 1), 'zorder': 6},
        'lirpa_betacrown':  {'label': r'$\beta$-CROWN',  'color': "#DDDA0E", 'style': '--', 'dashes': (5, 5),    'zorder': 7},
        'sdp':              {'label': 'SDP',           'color': '#CC79A7', 'style': ':',  'dashes': (1, 1),     'zorder': 4},
    }

    plot_methods = [m for m in styles.keys() if m in df.columns]
    print(plot_methods)
    # Exclusion Logic
    if 'norm_inf' in filename_lower and 'sdp' in plot_methods: 
        plot_methods.remove('sdp')
    else:
        # Assuming beta-crown is excluded for L2
        plot_methods.remove('lirpa_betacrown')
    print(plot_methods)
    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(10, 7))

    # 1. Dynamic Shading (Gap)
    cert_cols = [m for m in plot_methods if m != 'aa']
    if 'aa' in df.columns and cert_cols:
        virtual_best = df[cert_cols].max(axis=1)
        ax.fill_between(
            df['epsilon'], 
            virtual_best, 
            df['aa'], 
            color='#D55E00', 
            alpha=0.10,      
            label='Verification Gap'
        )

    # 2. Draw Lines
    for method in plot_methods:
        s = styles[method]
        kwargs = {
            'label': s['label'],
            'color': s['color'],
            'linestyle': s['style'],
            'linewidth': 2.5,
            'alpha': 0.85,
            'zorder': s['zorder']
        }
        if s['dashes']: kwargs['dashes'] = s['dashes']
        
        ax.plot(df['epsilon'], df[method], **kwargs)

    # --- Final Polish ---
    full_title = f"{title_text} ({norm_label})"
    ax.set_title(full_title, pad=20, weight='bold')
    ax.set_xlabel(f"Perturbation Radius ({norm_label})", labelpad=10)
    ax.set_ylabel("Robust Accuracy (%)", labelpad=10)
    
    ax.set_ylim(-5, 105)
    ax.set_xlim(left=0, right=df['epsilon'].max())
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # Legend - UPPER RIGHT
    ax.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='black', framealpha=1.0)

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    print(f"✅ Saved updated plot to {output_filename}")

# Example Usage
# filename = "results_relu/new_experiment_vanilla_ConvSmall_MNIST_1_LIP_Bjork_mnist_tau_a250.0_T0.2_bs64_lr0.001_eps0.5_medium_1765204811_acc0.87_norm_2.csv"
# create_distinct_paper_plot(filename, "final_plot_full_title.png")

def compute_certificates_CRA(images, model, epsilon, correct_indices, norm='2', L=1, return_robust_points=False):
    """
    Computes certificates and True CRA (Certified Robust Accuracy) relative to the ENTIRE dataset.

    Args:
        images (torch.Tensor): The FULL dataset images (used to determine total count).
        model (torch.nn.Module): The neural network model.
        epsilon (float): The radius for robustness certification.
        correct_indices (torch.Tensor or list): Indices of correctly classified images in the original set.
        norm (str): The norm to use ('2' or 'inf').
        L (float): Lipschitz constant.
        return_robust_points (bool): If True, returns the indices of robust images.

    Returns:
        If return_robust_points is False:
            (certificates, cra, time_per_image)
        If return_robust_points is True:
            (certificates, cra, time_per_image, robust_indices)
    """
    
    # Ensure correct_indices is a tensor
    if not isinstance(correct_indices, torch.Tensor):
        correct_indices = torch.tensor(correct_indices)

    # 1. Total Dataset Size (The denominator for True CRA)
    total_dataset_size = images.shape[0] 
        
    # 2. Subset to Compute On (We only verify the Cleanly Classified points)
    correct_images = images[correct_indices]
    
    # Handle empty case (if model got 0% accuracy)
    if len(correct_images) == 0:
        empty_certs = torch.tensor([])
        if return_robust_points:
            return empty_certs, 0.0, 0.0, torch.tensor([])
        return empty_certs, 0.0, 0.0

    # --- Step 1: Time the core computation ---
    device = next(model.parameters()).device 
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    start_time = time.time()

    with torch.no_grad():
        # We need the top 2 logits to calculate the margin
        values, _ = torch.topk(model(correct_images.to(device)), k=2)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    end_time = time.time()
    elapsed_time = end_time - start_time
    # --- End of Timing ---

    # Adapt certificate computation to the norm
    scale_certificate = np.sqrt(2)

    # --- Step 2: Calculate certificates ---
    # Calculate the margin divided by the Lipschitz constant and scale
    margin = values[:, 0] - values[:, 1]
    # Numerical stability
    margin = torch.clamp(margin, min=0.0)
    certificates = margin / (scale_certificate * L)
    
    # Check which certificates exceed the epsilon threshold
    is_robust_mask = (certificates >= epsilon).cpu()
    
    num_robust_points = torch.sum(is_robust_mask).item()
    
    # --- Step 3: Compute True CRA ---
    # True CRA = (Robust Count / TOTAL Dataset Size) * 100
    cra = (num_robust_points / total_dataset_size) * 100.0

    # Calculate average time per image (only counting the ones we actually processed)
    time_per_img = elapsed_time / len(correct_indices)

    # --- Step 4: Return results ---
    if return_robust_points:
        # Apply the mask to the original indices to get the ID of robust images
        robust_indices = correct_indices[is_robust_mask]
        return certificates.cpu(), cra, time_per_img, robust_indices
    
    return certificates.cpu(), cra, time_per_img
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
    total_num_images = images.shape[0] 

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


def compute_autoattack_era_and_time(images, targets, model, epsilon, clean_indices, norm='2', dataset_name='cifar10', return_robust_points=False):
    """
    Computes Empirical Robust Accuracy (CRA) against AutoAttack (L2/Linf).

    Args:
        images (torch.Tensor): The entire batch of input images.
        targets (torch.Tensor): The corresponding labels for all images.
        model (torch.nn.Module): The model to be attacked.
        epsilon (float): The AutoAttack radius.
        clean_indices (torch.Tensor): Indices of images that were initially classified correctly.
        norm (str): '2' or 'inf'.
        return_robust_points (bool): If True, returns the global indices of robust points.

    Returns:
        (cra, mean_time_per_image) OR (cra, mean_time_per_image, robust_indices)
    """
    device = next(model.parameters()).device
    total_num_images = images.shape[0] 

    # --- Step 1: Filter the dataset ---
    # We only attack images that were originally correct
    correct_images = images[clean_indices].contiguous().to(device)
    correct_targets = targets[clean_indices].to(device)

    # Handle edge case: No correct images to begin with
    if len(correct_images) == 0:
        if return_robust_points:
            return 0.0, 0.0, torch.tensor([], dtype=torch.long, device=device)
        return 0.0, 0.0
    
    # --- Step 2: Set up Attack ---
    if norm == '2':
        atk = torchattacks.AutoAttack(model, norm='L2', eps=epsilon)
    elif norm == 'inf':
        atk = torchattacks.AutoAttack(model, norm='Linf', eps=epsilon)
    else:
        raise ValueError(f"Unsupported norm: '{norm}'. Please use '2' or 'inf'.")
    
    # FIX: Correct Normalization for CIFAR-10 (Standard Deviations were incorrect)
    if dataset_name == "cifar10":
        atk.set_normalization_used(
            mean=torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1).to(device), 
            std=torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1).to(device)
        )

    # --- Step 3: Run and Time Attack ---
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start_time = time.time()

    # Generate adversarial examples
    adv_images = atk(correct_images, correct_targets)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    end_time = time.time()

    total_time = end_time - start_time
    mean_time_per_image = total_time / len(correct_images)

    # --- Step 4: Calculate ERA (True Robust Accuracy) ---
    with torch.no_grad():
        adv_outputs = model(adv_images)
        adv_predictions = adv_outputs.argmax(dim=1)
        
        # Boolean mask: True where the model resisted the attack
        robust_mask = (adv_predictions == correct_targets)
        
        num_robust_points = torch.sum(robust_mask).item()
        
        # Calculate ERA relative to the TOTAL original dataset
        cra = (num_robust_points / total_num_images) * 100.0

    # --- Step 5: Return Results ---
    if return_robust_points:
        # We need to map the boolean mask back to the original indices.
        # clean_indices contains the original IDs of the images we attacked.
        # robust_mask tells us which of those survived.
        robust_indices = clean_indices[robust_mask.cpu()]
        
        return cra, mean_time_per_image, robust_indices

    return cra, mean_time_per_image


import sys
sys.path.insert(0,'SDP-CROWN')
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


def compute_alphacrown_vra_and_time(images, targets, model, epsilon, clean_indices, args, batch_size=2, norm=2, return_robust_points=False, x_U=None, x_L=None):
    """
    Computes Certified Robust Accuracy (CRA) using Alpha-Crown.
    
    CRITICAL NOTE: 
    x_L and x_U here should represent the GLOBAL valid data range (e.g. 0 and 1), 
    NOT the local epsilon bounds. The function handles the epsilon intersection internally.
    """

    device = next(model.parameters()).device
    total_num_images = images.shape[0]
    model.eval()
    
    if not isinstance(clean_indices, torch.Tensor):
        clean_indices = torch.tensor(clean_indices)

    # --- Step 1: Filter for correctly classified samples ---
    correct_images = images[clean_indices]
    correct_targets = targets[clean_indices]

    if len(correct_images) == 0:
        if return_robust_points:
            return 0.0, 0.0, torch.tensor([])
        return 0.0, 0.0

    # --- Step 2: Initialize variables ---
    num_robust_points = 0
    total_time = 0.0
    num_batches = (len(correct_images) + batch_size - 1) // batch_size
    robust_indices_list = []

    # --- Step 3: Setup BoundedModule ---
    # We disable "patches" mode to ensure stability with explicit bounds
    dummy_input = correct_images[0:1].to(device)
    bounded_model = BoundedModule(model, dummy_input, bound_opts={"conv_mode": "patches"}, verbose=False)
    # bounded_model = BoundedModule(model, dummy_input, verbose=False)
    bounded_model.eval()

    print(f"Verifying {len(correct_images)} samples in {num_batches} batches...")

    # --- Step 4: Batch Loop ---
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(correct_images))
        
        # Clone to avoid modifying original dataset
        batch_images = correct_images[start_idx:end_idx].clone().to(device)
        batch_targets = correct_targets[start_idx:end_idx]
        current_bs = batch_images.shape[0]

        # --- A. Prepare Global Domain Bounds (0 to 1) ---
        # Expand global limits (x_L/x_U) to match current batch shape and ensure contiguity
        if x_L is not None:
            batch_global_L = x_L.expand(current_bs, *x_L.shape[1:]).contiguous()
        else:
            batch_global_L = None
            
        if x_U is not None:
            batch_global_U = x_U.expand(current_bs, *x_U.shape[1:]).contiguous()
        else:
            batch_global_U = None

        # --- B. CLAMP IMAGES (Crucial for Stability) ---
        # Ensure the center point 'x' is mathematically inside the global domain [0, 1]
        # This prevents the "Invalid Center" crash.
        if batch_global_L is not None and batch_global_U is not None:
            batch_images = torch.max(torch.min(batch_images, batch_global_U), batch_global_L)

        # --- C. Define Perturbation Constraints ---
        
        if norm == 'inf' or norm == float('inf'):
            # STRATEGY: TIGHT BOX INTERSECTION
            # ptb_L = max(Global_Min, x - epsilon)
            # ptb_U = min(Global_Max, x + epsilon)
            
            # We calculate this manually to give the verifier the easiest job possible.
            if batch_global_L is not None and batch_global_U is not None:
                ptb_L = torch.max(batch_global_L, batch_images - epsilon)
                ptb_U = torch.min(batch_global_U, batch_images + epsilon)
                
                ptb = PerturbationLpNorm(norm=np.inf, eps=epsilon, x_L=ptb_L, x_U=ptb_U)
            else:
                # Fallback if no global bounds provided
                ptb = PerturbationLpNorm(norm=np.inf, eps=epsilon)
                
        else:
            # STRATEGY: GLOBAL BOUNDS + EPSILON SPHERE (Best for L2)
            # For L2, we don't intersect with a box (which would imply Linf). 
            # We just say "Don't go past 0 or 1" using x_L/x_U.
            ptb = PerturbationLpNorm(norm=norm, eps=epsilon, x_L=batch_global_L, x_U=batch_global_U)

        bounded_input = BoundedTensor(batch_images, ptb)
        
        num_classes = model[-1].out_features 
        c = build_C(batch_targets.to("cpu"), num_classes).to(device)

        # --- Time the verification ---
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start_time_batch = time.time()
        
        # Optimize bounds (Alpha-CROWN settings)
        bounded_model.set_bound_opts({
            'optimize_bound_args': {
                'iteration': 300, 
                'lr_alpha': args.lr_alpha,
                'early_stop_patience': 20, 
                'enable_opt_interm_bounds': True, 
                'verbosity': False
            }, 
            'verbosity': False
        })
        
        lb_diff = bounded_model.compute_bounds(x=(bounded_input,), C=c, method='alpha-crown')[0]
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        end_time_batch = time.time()
        total_time += (end_time_batch - start_time_batch)

        # --- Check Robustness ---
        is_robust = (lb_diff.view(current_bs, num_classes - 1) > 0).all(dim=1)
        num_robust_points += torch.sum(is_robust).item()
        
        if return_robust_points:
            batch_global_indices = clean_indices[start_idx:end_idx]
            robust_indices_list.append(batch_global_indices[is_robust.cpu()])

        print(f"  Batch {i+1}/{num_batches}: {torch.sum(is_robust).item()}/{current_bs} robust.", end='\r')

    print("\nBatch verification finished.") 
    
    cra = (num_robust_points / total_num_images) * 100.0
    mean_time_per_image = total_time / len(correct_images) if len(correct_images) > 0 else 0.0

    if return_robust_points:
        all_robust_indices = torch.cat(robust_indices_list) if robust_indices_list else torch.tensor([])
        return cra, mean_time_per_image, all_robust_indices

    return cra, mean_time_per_image

import time
import torch
import numpy as np

sys.path.append("alpha-beta-CROWN/complete_verifier")
from abcrown import ABCROWN # Import the main class from your script


import time  # Import the time module

def compute_alphabeta_vra_and_time(dataset_name, model_name, model_path, epsilon, CONFIG_FILE, clean_indices, total_samples, norm='inf', return_robust_points=False):
    """
    Computes True VRA (Verified Robust Accuracy) relative to the TOTAL dataset size.
    
    Args:
        total_samples (int): The total number of images in the chunk being verified 
                             (including misclassified ones).
        return_robust_points (bool): If True, returns a tensor of indices that are robust.
    """
    if dataset_name=='cifar10':
        dataset = 'CIFAR_SDP'
    elif dataset_name=='mnist':
        dataset = 'MNIST_SDP'
    else :
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    params = {
            'model': model_name,
            'load_model': model_path,
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

    # --- Metrics ---
    # Convert clean_indices to a set of Python integers for intersection logic
    if isinstance(clean_indices, torch.Tensor):
        clean_indices_set = {t.item() for t in clean_indices}
    else:
        clean_indices_set = set(clean_indices)
    
    # 1. Collect all "Safe" indices from the verifier
    validated_indices_set = set()
    # "safe" = verified robust, "safe-incomplete" = verified robust (often bound propagation)
    for key in ['safe-incomplete', 'safe']: 
        validated_indices_set.update(summary.get(key, []))

    # 2. Filter: We only count them if they were ORIGINALLY correct (Clean AND Robust)
    validated_clean_indices_set = validated_indices_set.intersection(clean_indices_set)
    num_robust_points = len(validated_clean_indices_set)

    # 3. True VRA Calculation
    # We divide by the TOTAL samples, not just the clean ones
    true_vra = (num_robust_points / total_samples) * 100.0
    
    # Optional: Conditional VRA (for debugging)
    conditional_vra = (num_robust_points / len(clean_indices)) * 100.0 if len(clean_indices) > 0 else 0

    # 4. Timing
    total_time = end_time - start_time
    avg_time = total_time / total_samples if total_samples > 0 else 0

    print(f"🚀 Verification Complete!")
    print(f"   - Total Dataset Size: {total_samples}")
    print(f"   - Clean Samples: {len(clean_indices)}")
    print(f"   - Verified Robust (Clean & Safe): {num_robust_points}")
    print(f"   - True VRA: {true_vra:.2f}%")
    print(f"   - (Conditional Robustness: {conditional_vra:.2f}%)")
    print(f"   - Avg Time/Sample: {avg_time:.4f}s")

    # 5. Return Results
    if return_robust_points:
        # Convert the set back to a sorted Torch Tensor
        robust_indices_tensor = torch.tensor(sorted(list(validated_clean_indices_set)), dtype=torch.long)
        return true_vra, avg_time, robust_indices_tensor

    return true_vra, avg_time


sys.path.append('SDP-CROWN')
from sdp_crown import verified_sdp_crown


# def compute_sdp_crown_vra(dataset, labels, model, radius, clean_output, device, classes, args, batch_size=1, return_robust_points=False, x_U=None, x_L=None, groupsort=False):
#     return verified_sdp_crown(dataset, labels, model, radius, clean_output, device, classes, args, batch_size=1, return_robust_points=return_robust_points, x_U=x_U, x_L=x_L, groupsort=groupsort)
# def compute_sdp_crown_fixed_bs(dataset, labels, model, radius, clean_output, device, classes, args, batch_size=1, return_robust_points=False, x_U=None, x_L=None, groupsort=False):
#     return verified_sdp_crown_fixed_bs(dataset, labels, model, radius, clean_output, device, classes, args, batch_size=batch_size, return_robust_points=return_robust_points, x_U=x_U, x_L=x_L, groupsort=groupsort)

def compute_sdp_crown_vra(dataset, labels, model, radius, clean_output, device, classes, args, batch_size=1, return_robust_points=False, x_U=None, x_L=None, groupsort=False):
    return verified_sdp_crown(dataset, labels, model, radius, clean_output, device, classes, args, batch_size=1, return_robust_points=return_robust_points, x_U=x_U, x_L=x_L, groupsort=groupsort)
# import boto3
# import os
# from botocore.exceptions import NoCredentialsError, ClientError

# def download_s3_folder(bucket_name="tdrobustbucket", s3_folder="lip_models", local_dir="./models"):
#     """
#     Downloads an S3 folder to a local directory.
#     """
#     s3 = boto3.client('s3')

#     # Ensure the local directory exists
#     if not os.path.exists(local_dir):
#         os.makedirs(local_dir)

#     # Handle pagination in case there are many models
#     paginator = s3.get_paginator('list_objects_v2')
#     pages = paginator.paginate(Bucket=bucket_name, Prefix=s3_folder)

#     print(f"--- Starting sync from s3://{bucket_name}/{s3_folder} to {local_dir} ---")

#     download_count = 0

#     try:
#         for page in pages:
#             # Check if the folder is empty
#             if 'Contents' not in page:
#                 continue

#             for obj in page['Contents']:
#                 s3_key = obj['Key']
                
#                 # Skip directory markers (keys ending in /)
#                 if s3_key.endswith('/'):
#                     continue

#                 # Remove the 'lip_models/' prefix to map correctly to './models/'
#                 # e.g., 'lip_models/model.pth' becomes 'model.pth'
#                 relative_path = os.path.relpath(s3_key, s3_folder)
#                 local_file_path = os.path.join(local_dir, relative_path)

#                 # Create local subdirectories if they exist in S3 structure
#                 local_file_dir = os.path.dirname(local_file_path)
#                 if not os.path.exists(local_file_dir):
#                     os.makedirs(local_file_dir)

#                 # Download
#                 print(f"Downloading: {s3_key} -> {local_file_path}")
#                 s3.download_file(bucket_name, s3_key, local_file_path)
#                 download_count += 1

#         if download_count == 0:
#             print("No files found in the specified S3 folder.")
#         else:
#             print(f"\nSuccess! {download_count} files downloaded.")

#     except NoCredentialsError:
#         print("Error: AWS credentials not found. Please run 'aws configure'.")
#     except ClientError as e:
#         print(f"AWS Client Error: {e}")
#     except Exception as e:
#         print(f"Unexpected Error: {e}")

# if __name__ == "__main__":
#     # --- Configuration matches your training script ---
#     BUCKET_NAME = "tdrobustbucket"
#     S3_FOLDER = "lip_models" 
#     LOCAL_DIRECTORY = "./models" 

#     download_s3_folder(BUCKET_NAME, S3_FOLDER, LOCAL_DIRECTORY)