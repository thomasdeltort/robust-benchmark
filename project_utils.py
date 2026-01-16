from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import os
from models import * # Make sure this import works from your utils.py location
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

# import boto3
import os
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

if __name__ == "__main__":
    # --- Configuration matches your training script ---
    BUCKET_NAME = "tdrobustbucket"
    S3_FOLDER = "lip_models" 
    LOCAL_DIRECTORY = "./models" 

    # download_s3_folder(BUCKET_NAME, S3_FOLDER, LOCAL_DIRECTORY)

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

# from models import GroupSort_AutoLirpa

# def swap_to_reformulated_gs2(model):
#     """
#     Recursively replaces decomposed GroupSort_General modules with 
#     the custom operator version (GroupSort_AutoLirpa).
#     """
#     for name, module in model.named_children():
#         # Check if the module is the standard/decomposed GroupSort
#         if isinstance(module, GroupSort_General):
#             # Create the reformulated module, preserving the axis
#             new_module = GroupSort_AutoLirpa(axis=module.axis)
#             setattr(model, name, new_module)
#             print(f"  [✓] Swapped {name} to reformulated GroupSort")
#         else:
#             # Recurse into sub-modules (Sequential, etc.)
#             swap_to_reformulated_gs2(module)
#     return model

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

def replace_groupsort(model, dummy_input):
    """
    Replaces GroupSort layers with GroupSort_SparseConv (or user defined GroupSort).
    """
    # Dictionary to store (layer_name -> input_channels)
    layer_configs = {}
    hooks = []

    # 1. Define Hook to capture shapes
    def get_shape_hook(name):
        def hook(module, input, output):
            shape = input[0].shape
            channels = shape[1] 
            layer_configs[name] = channels
        return hook

    # 2. Register hooks on all GroupSort layers
    for name, module in model.named_modules():
        if isinstance(module, GroupSort_General) or "GroupSort" in str(type(module)):
            hooks.append(module.register_forward_hook(get_shape_hook(name)))

    # 3. Run Dummy Pass
    model.eval()
    with torch.no_grad():
        model(dummy_input)

    # 4. Remove hooks
    for h in hooks:
        h.remove()

    # 5. Perform Replacement
    print(f"   [Auto-Infer] Detected GroupSort channels: {layer_configs}")
    
    for name, module in model.named_modules():
        for child_name, _ in module.named_children():
            full_child_name = f"{name}.{child_name}" if name else child_name
            
            if full_child_name in layer_configs:
                channels = layer_configs[full_child_name]
                axis = 1 
                
                print(f"   -> Replacing {full_child_name} (Channels={channels})")
                new_layer = GroupSort(channels=channels, axis=axis)
                setattr(module, child_name, new_layer)

    return model

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def create_robustness_plot_v3(filepath, output_filename):
    """
    Generates a plot of robust accuracy vs. epsilon.
    - Extracts model name from filename.
    - Cleans title by removing trailing 'inf' or '2'.
    - Renames 'aa' to 'AutoAttack'.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"❌ Error: The file '{filepath}' was not found.")
        return

    # --- 1. Extract and Clean Model Name ---
    filename_with_ext = os.path.basename(filepath)
    filename_no_ext = os.path.splitext(filename_with_ext)[0]
    
    # Split the filename into parts
    parts = filename_no_ext.split('_')
    
    # Check the last part and remove it if it is 'inf' or '2'
    if parts and parts[-1].lower() in ['inf', '2']:
        parts.pop()
        
    # Join back with spaces for the plot title
    model_title_text = " ".join(parts)

    # --- 2. Prepare Data ---
    df = df.sort_values(by='epsilon').reset_index(drop=True)
    
    # Base order
    plot_order = ['certificate', 'aa', 'lirpa_alphacrown', 'lirpa_betacrown', 'sdp']
    
    methods_to_plot = list(plot_order) 
    
    # Conditional logic based on the FULL filename string (so logic remains robust)
    if 'Linf' in filename_no_ext:
        if 'sdp' in methods_to_plot:
            methods_to_plot.remove('sdp')
            print("ℹ️ 'Linf' found. Removing 'sdp'.")
    elif 'L2' in filename_no_ext:
        if 'lirpa_betacrown' in methods_to_plot:
            methods_to_plot.remove('lirpa_betacrown')
            print("ℹ️ 'L2' found. Removing 'lirpa_betacrown'.")
            
    # Filter columns
    method_columns = [col for col in methods_to_plot if col in df.columns and col != 'epsilon' and not col.startswith('time_')]

    # --- 3. Color Mapping ---
    color_map = {}
    if 'certificate' in method_columns:
        color_map['certificate'] = '#1f77b4'  # Blue
    if 'aa' in method_columns:
        color_map['aa'] = '#ff7f0e'   # Orange

    other_methods_colors = ['#9467bd', '#8c564b', '#e377c2']
    other_idx = 0
    for method in method_columns:
        if method not in color_map:
            if other_idx < len(other_methods_colors):
                color_map[method] = other_methods_colors[other_idx]
                other_idx += 1
            else:
                color_map[method] = '#7f7f7f'

    # --- 4. Plotting ---
    sns.set_theme(style="whitegrid", context="talk")
    plt.figure(figsize=(14, 9))

    for method in method_columns:
        # Rename 'aa' to 'AutoAttack' for the label
        label_name = 'AutoAttack' if method == 'aa' else method
        
        plt.plot(df['epsilon'], df[method], 
                 marker='o', 
                 linestyle='-', 
                 label=label_name, 
                 color=color_map.get(method, '#000000'), 
                 linewidth=3,
                 markersize=10)

    plt.title(f"Robust Accuracy vs. Epsilon\nModel: {model_title_text}", fontsize=22, weight='bold', pad=20)
    plt.xlabel('Epsilon ($ε$)', fontsize=18, labelpad=10)
    plt.ylabel('Robust Accuracy', fontsize=18, labelpad=10)
    plt.legend(title='Verification Method', title_fontsize=16, fontsize=14, loc='best', frameon=True, shadow=True)
    plt.grid(True, which='both', linestyle='--', linewidth=1, alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(output_filename)
    print(f"✅ Plot saved as '{output_filename}' (Title: {model_title_text})")
    plt.close()
# Example Usage:
# create_robustness_plot_v3(filepath='results/results_Linf.csv', model_name="ResNet-18 (CIFAR-10)")

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
    print(f"Norm: {norm}")
    
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
    if norm == '2':
        scale_certificate = np.sqrt(2)
    elif norm == 'inf':
        scale_certificate = 2.0
    else:
        raise ValueError(f"Unsupported norm: '{norm}'. Please use '2' or 'inf'.")

    # --- Step 2: Calculate certificates ---
    # Calculate the margin divided by the Lipschitz constant and scale
    certificates = (values[:, 0] - values[:, 1]) / (scale_certificate * L)
    
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

# def compute_certificates_CRA(images, model, epsilon, correct_indices, norm='2', L=1, return_robust_points=False):
#     """
#     Computes certificates, CRA, and the time taken for the core computation. Adapting to the norm.

#     Args:
#         images (torch.Tensor): The dataset images.
#         model (torch.nn.Module): The neural network model.
#         epsilon (float): The radius for robustness certification.
#         correct_indices (torch.Tensor or list): Indices of correctly classified images in the original set.
#         norm (str): The norm to use ('2' or 'inf').
#         L (float): Lipschitz constant.
#         return_robust_points (bool): If True, returns the indices of robust images.

#     Returns:
#         If return_robust_points is False:
#             (certificates, cra, time_per_image)
#         If return_robust_points is True:
#             (certificates, cra, time_per_image, robust_indices)
#     """
#     print(f"Norm: {norm}")
    
#     # Ensure correct_indices is a tensor for boolean masking later
#     if not isinstance(correct_indices, torch.Tensor):
#         correct_indices = torch.tensor(correct_indices)
        
#     correct_images = images[correct_indices]
#     total_num_images = correct_images.shape[0]

#     # Handle empty case
#     if len(correct_images) == 0:
#         empty_certs = torch.tensor([])
#         if return_robust_points:
#             return empty_certs, 0.0, 0.0, torch.tensor([])
#         return empty_certs, 0.0, 0.0

#     # --- Step 1: Time the core computation ---
#     device = next(model.parameters()).device 
    
#     if device.type == 'cuda':
#         torch.cuda.synchronize()
    
#     start_time = time.time()

#     with torch.no_grad():
#         # We need the top 2 logits to calculate the margin
#         values, _ = torch.topk(model(correct_images.to(device)), k=2)

#     if device.type == 'cuda':
#         torch.cuda.synchronize()

#     end_time = time.time()
#     elapsed_time = end_time - start_time
#     # --- End of Timing ---

#     # Adapt certificate computation to the norm
#     if norm == '2':
#         scale_certificate = np.sqrt(2)
#     elif norm == 'inf':
#         scale_certificate = 2.0
#     else:
#         raise ValueError(f"Unsupported norm: '{norm}'. Please use '2' or 'inf'.")

#     # --- Step 2: Calculate certificates and CRA ---
#     # Calculate the margin divided by the Lipschitz constant and scale
#     certificates = (values[:, 0] - values[:, 1]) / (scale_certificate * L)
    
#     # Create a boolean mask on CPU (to match correct_indices device usually)
#     # Check which certificates exceed the epsilon threshold
#     is_robust_mask = (certificates >= epsilon).cpu()
    
#     num_robust_points = torch.sum(is_robust_mask).item()
#     cra = (num_robust_points / total_num_images) * 100.0

#     # Calculate average time per image
#     time_per_img = elapsed_time / len(correct_indices)

#     # --- Step 3: Return results ---
#     if return_robust_points:
#         # Apply the mask to the original indices to get the ID of robust images
#         robust_indices = correct_indices[is_robust_mask]
#         return certificates.cpu(), cra, time_per_img, robust_indices
    
#     return certificates.cpu(), cra, time_per_img

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

# def compute_autoattack_era_and_time(images, targets, model, epsilon, clean_indices, norm='2', dataset_name='cifar10'):
#     """
#     Computes Empirical Robust Accuracy (CRA) against AutoAttack (L2) and
#     measures the mean computation time per image.

#     Args:
#         images (torch.Tensor): The entire batch of input images.
#         targets (torch.Tensor): The corresponding labels for all images.
#         model (torch.nn.Module): The model to be attacked.
#         epsilon (float): The AutoAttack radius.
#         clean_indices (torch.Tensor): A tensor of indices for images that were
#                                      initially classified correctly.

#     Returns:
#         A tuple containing:
#         - cra (float): The Empirical Robust Accuracy percentage.
#         - mean_time_per_image (float): The average attack time per image in seconds.
#     """
#     device = next(model.parameters()).device
#     total_num_images = images.shape[0] 

#     # --- Step 1: Filter the dataset ---
#     correct_images = images[clean_indices].contiguous().to(device)
#     correct_targets = targets[clean_indices].to(device)

#     if len(correct_images) == 0:
#         return 0.0, 0.0
    
#     # --- Step 2: Set up and time the BATCH attack ---
#     # We adapt the computation of the certificate to the given norm
#     if norm == '2':
#         atk = torchattacks.AutoAttack(model, norm='L2', eps=epsilon)
#     elif norm == 'inf':
#         atk = torchattacks.AutoAttack(model, norm='Linf', eps=epsilon)
#     else:
#         raise ValueError(f"Unsupported norm: '{norm}'. Please use '2' or 'inf'.")
    
#     if dataset_name=="cifar10":
#         atk.set_normalization_used(mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1).cuda(), std = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1).cuda())

#     # Synchronize GPU before starting the timer
#     if device.type == 'cuda':
#         torch.cuda.synchronize()
#     start_time = time.time()

#     # Generate adversarial examples for the ENTIRE BATCH
#     adv_images = atk(correct_images, correct_targets)
#     # Synchronize GPU again before stopping the timer
#     if device.type == 'cuda':
#         torch.cuda.synchronize()
#     end_time = time.time()

#     total_time = end_time - start_time
#     mean_time_per_image = total_time / len(correct_images)

#     # --- Step 3: Calculate the Empirical Robust Accuracy (CRA) ---
#     with torch.no_grad():
#         adv_outputs = model(adv_images)
#         adv_predictions = adv_outputs.argmax(dim=1)
#         robust_mask = (adv_predictions == correct_targets)
#         num_robust_points = torch.sum(robust_mask).item()
#         non_robust_mask = ~robust_mask  # or (adv_predictions != correct_targets)
        
#         # Get the indices of the non-robust points within the current batch
#         non_robust_indices = torch.nonzero(non_robust_mask, as_tuple=True)[0]
        
#         # Print the list of indices
#         if non_robust_indices.numel() > 0:
#             print(f"Indices of non-robust images in this batch: {non_robust_indices.tolist()}")
#         # CRA is relative to the TOTAL dataset size
#         cra = (num_robust_points / total_num_images) * 100.0
# #[7, 34, 39, 40, 47, 66, 67, 77, 80, 95, 114, 126]
#     return cra, mean_time_per_image

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
            std=torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1).to(device)
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

# def compute_alphacrown_vra_and_time(images, targets, model, epsilon, clean_indices, batch_size=1, norm=2, return_robust_points=False, x_U=None, x_L=None):
#     """
#     Computes Certified Robust Accuracy (CRA) using Alpha-Crown in batches to manage memory,
#     and measures the mean verification time per image.
    
#     Args:
#         images (Tensor): The entire dataset's images.
#         targets (Tensor): The entire dataset's labels.
#         model (nn.Module): The model to verify.
#         epsilon (float): The perturbation radius.
#         clean_indices (Tensor): Indices of correctly classified images.
#         batch_size (int): The number of images to verify in each batch.
#         norm (int or float): The norm (e.g., 2 or np.inf).
#         return_robust_points (bool): If True, returns the list of global indices of robust images.

#     Returns:
#         If return_robust_points is False:
#             (cra, mean_time_per_image)
#         If return_robust_points is True:
#             (cra, mean_time_per_image, robust_indices)
#     """

#     device = next(model.parameters()).device
#     total_num_images = len(images)
#     model.eval()
    
#     # Ensure clean_indices is a tensor for indexing
#     if not isinstance(clean_indices, torch.Tensor):
#         clean_indices = torch.tensor(clean_indices)

#     # --- Step 1: Filter for correctly classified samples ---
#     correct_images = images[clean_indices]
#     correct_targets = targets[clean_indices]

#     if len(correct_images) == 0:
#         if return_robust_points:
#             return 0.0, 0.0, torch.tensor([])
#         return 0.0, 0.0

#     # --- Step 2: Initialize variables for batch processing ---
#     num_robust_points = 0
#     total_time = 0.0
#     num_batches = (len(correct_images) + batch_size - 1) // batch_size
    
#     # List to accumulate robust indices across batches
#     robust_indices_list = []

#     # --- Step 3: Set up a reusable BoundedModule ---
#     # We only need to create this once
#     dummy_input = correct_images[0:1].to(device)
#     # Note: Ensure auto_LiRPA is installed for BoundedModule
#     # bounded_model = BoundedModule(model, dummy_input, bound_opts={"conv_mode": "patches"}, verbose=False)
#     bounded_model = BoundedModule(model, dummy_input, verbose=False)
#     bounded_model.eval()

#     print(f"Verifying {len(correct_images)} samples in {num_batches} batches of size {batch_size}...")

#     #Sanity Check 
#     print("Verifying data bounds...")
#     # Allow a tiny margin for floating point errors
#     margin = 1e-5 
#     assert (images[clean_indices] >= x_L - margin).all(), "Error: Some inputs are smaller than x_L!"
#     assert (images[clean_indices] <= x_U + margin).all(), "Error: Some inputs are larger than x_U!"
#     print("Data is valid within bounds.")



#     # --- Step 4: Loop through the data in batches ---
#     for i in range(num_batches):
#         start_idx = i * batch_size
#         end_idx = min((i + 1) * batch_size, len(correct_images))
        
#         batch_images = correct_images[start_idx:end_idx].to(device)
#         batch_targets = correct_targets[start_idx:end_idx]
        
#         # --- Set up BoundedTensor and specification for the current batch ---
        
#         if norm == 'inf':
#             # ptb = PerturbationLpNorm(norm=np.inf, eps=epsilon, x_U=x_U.expand(batch_images.shape[0], -1, -1, -1), x_L=x_L.expand(batch_images.shape[0], -1, -1, -1))
#             ptb = PerturbationLpNorm(norm=np.inf, eps=epsilon, x_U=x_U.expand(batch_images.shape[0], -1, -1, -1).contiguous(), x_L=x_L.expand(batch_images.shape[0], -1, -1, -1).contiguous())
#         else:
#             ptb = PerturbationLpNorm(norm=norm, eps=epsilon, x_U=x_U.expand(batch_images.shape[0], -1, -1, -1), x_L=x_L.expand(batch_images.shape[0], -1, -1, -1))
#             # ptb = PerturbationLpNorm(norm=norm, eps=epsilon, x_U=x_U, x_L=x_L)

#         bounded_input = BoundedTensor(batch_images, ptb)
        
#         num_classes = model[-1].out_features # Assuming last layer is Linear
#         c = build_C(batch_targets.to("cpu"), num_classes).to(device)

#         # --- Time the verification for this batch ---
#         if device.type == 'cuda':
#             torch.cuda.synchronize()
#         start_time_batch = time.time()
        
#         bounded_model.set_bound_opts({'optimize_bound_args': {'iteration': 200, 'early_stop_patience': 30, 'fix_interm_bounds': False, 'enable_opt_interm_bounds':True, 'verbosity':False}, 'verbosity':False})
        
#         # Compute bounds
#         lb_diff = bounded_model.compute_bounds(x=(bounded_input,), C=c, method='alpha-crown')[0]
#         # lb_diff = bounded_model.compute_bounds(x=(bounded_input,), C=c, method='IBP')[0]
        
#         if device.type == 'cuda':
#             torch.cuda.synchronize()
#         end_time_batch = time.time()
        
#         total_time += (end_time_batch - start_time_batch)

#         # --- Calculate robust points in the current batch ---
#         # Check if lower bound > 0 for all classes (except target)
#         is_robust = (lb_diff.view(len(batch_images), num_classes - 1) > 0).all(dim=1)
#         num_robust_points += torch.sum(is_robust).item()
        
#         # --- Collect Indices if requested ---
#         if return_robust_points:
#             # 1. Get the global indices corresponding to this batch
#             batch_global_indices = clean_indices[start_idx:end_idx]
#             # 2. Use the boolean mask (moved to cpu) to filter these indices
#             # We move is_robust to CPU to match clean_indices location usually
#             batch_robust_indices = batch_global_indices[is_robust.cpu()]
#             robust_indices_list.append(batch_robust_indices)

#         # Optional: Print progress
#         print(f"  Batch {i+1}/{num_batches}: {torch.sum(is_robust).item()}/{len(batch_images)} robust.", end='\r')

#     print("\nBatch verification finished.") 
    
#     # --- Step 5: Calculate final metrics ---
#     # CRA is relative to the TOTAL dataset size (including originally incorrect ones)
#     cra = (num_robust_points / total_num_images) * 100.0
#     mean_time_per_image = total_time / len(correct_images) if len(correct_images) > 0 else 0.0

#     if return_robust_points:
#         if len(robust_indices_list) > 0:
#             all_robust_indices = torch.cat(robust_indices_list)
#         else:
#             all_robust_indices = torch.tensor([])
#         return cra, mean_time_per_image, all_robust_indices

#     return cra, mean_time_per_image

def compute_alphacrown_vra_and_time(images, targets, model, epsilon, clean_indices, batch_size=2, norm=2, return_robust_points=False, x_U=None, x_L=None):
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
                'iteration': 200, 
                'early_stop_patience': 30, 
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

sys.path.append("/home/aws_install/robustess_project/alpha-beta-CROWN/complete_verifier")
from abcrown import ABCROWN # Import the main class from your script


import time  # Import the time module

import logging
logging.getLogger('auto_LiRPA').setLevel(logging.WARNING)

# def compute_alphabeta_vra_and_time(dataset_name, model_name, model_path, epsilon, CONFIG_FILE, clean_indices, norm='inf'):
#     """
#     Computes Certified Robust Accuracy (CRA) using α-β-CROWN and
#     measures the mean verification time per image.
#     """
#     if dataset_name=='cifar10':
#         dataset = 'CIFAR_SDP'
#     elif dataset_name=='mnist':
#         dataset = 'MNIST_SDP'
#     else :
#         raise ValueError(f"Unsupported dataset")
    
#     params = {
#             'model': model_name,
#             'load_model': model_path,  # Use 'load_model' for the path
#             'dataset': dataset,
#             'epsilon': epsilon,
#         }

#     # Compute the lower bounds on the logit differences using α-β-CROWN
#     verifier = ABCROWN(
#             args=[],
#             config=CONFIG_FILE,
#             **params
#         )
    
#     # --- Start Timing ---
#     start_time = time.time()
    
#     summary = verifier.main()
    
#     end_time = time.time()
#     # --- End Timing ---


#     # --- Calculate Total Samples Verified ---
#     # This is more robust than hardcoding 200
#     total_samples_verified = sum(len(v) for v in summary.values())

#     # --- Calculate Average Time ---
#     total_time = end_time - start_time
#     avg_time = total_time / total_samples_verified if total_samples_verified > 0 else 0


#     # --- Accuracy Calculations ---
#     clean_indices_set = {t.item() for t in clean_indices}
#     validated_indices_set = set()
#     validated_keys = ['safe-incomplete', 'safe']
    
#     # Note: 'total_validated' here means "total proven safe", not "total processed"
#     total_proven_safe = 0 

#     for key in validated_keys:
#         validated_indices_set.update(summary.get(key, []))
#         total_proven_safe += len(summary.get(key, [])) # .get() is safer if a key might be missing

#     validated_clean_indices = validated_indices_set.intersection(clean_indices_set)
#     count_validated_clean = len(validated_clean_indices)

#     denominator = len(clean_indices)
#     certified_robust_accuracy = (count_validated_clean / denominator) * 100 if denominator > 0 else 0

#     print(f"🚀 Verification Complete!")
#     print(f"   - Total samples verified: {total_samples_verified}")
#     print(f"   - Correctly classified (clean): {denominator}")
#     print(f"   - Correctly classified AND robust (clean & safe): {count_validated_clean}")
#     print(f"   - Certified Robust Accuracy: {certified_robust_accuracy:.2f}%")
#     print(f"   - Total verification time: {total_time:.2f} seconds")
#     print(f"   - Average time per sample: {avg_time:.4f} seconds")
#     # import pdb; pdb.set_trace()

#     return certified_robust_accuracy, avg_time


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


sys.path.append('/home/aws_install/robustess_project/SDP-CROWN')
from sdp_crown import verified_sdp_crown


def compute_sdp_crown_vra(dataset, labels, model, radius, clean_output, device, classes, args, batch_size=1, return_robust_points=False, x_U=None, x_L=None):
    return verified_sdp_crown(dataset, labels, model, radius, clean_output, device, classes, args, batch_size, return_robust_points=return_robust_points, x_U=x_U, x_L=x_L)

if __name__ == '__main__':
    download_s3_folder()