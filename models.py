import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from deel import torchlip
# from project_utils import GroupSort_General
import numpy as np

# Note: This python file present every model we use for benchmarking. 
# These models correspond to all the model architectures from Wang et al. (2021) & Leino et al. (2021)
# This version uses ReLU as the activation function.
import torch
import torch.nn as nn
import sys

## Update path to your local SDP-CROWN repository
#SDP_CROWN_PATH = 'SDP-CROWN'
#sys.path.insert(0, SDP_CROWN_PATH)
#
#from auto_LiRPA import BoundedModule, BoundedTensor, register_custom_op
#from auto_LiRPA.operators import BoundTwoPieceLinear
#from auto_LiRPA.perturbations import PerturbationLpNorm
#from auto_LiRPA.patches import Patches

class GroupSort_General(nn.Module):
    """
    Applies GroupSort specifically on the channel dimension.
    
    It permutes the input from (N, C, ...) to (N, ..., C), applies the 
    sort logic so that pairs (c_2k, c_2k+1) are sorted, and then restores
    the original layout.
    """
    def __init__(self, axis=1):
        super(GroupSort_General, self).__init__()
        self.axis = axis
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Permute to Channel Last
        # We assume the channel is at self.axis (usually 1).
        # We move that axis to the very end (-1).
        dims = list(range(x.dim()))
        # Remove the channel axis from its current spot and append to end
        channel_dim = dims.pop(self.axis) 
        dims.append(channel_dim)
        
        # permute returns a view, but we usually need contiguous memory for reshaping
        x_permuted = x.permute(dims).contiguous()
        
        # Capture the shape after permutation: (N, D1, D2, ..., C)
        permuted_shape = x_permuted.shape
        batch_size = permuted_shape[0]
        num_channels = permuted_shape[-1]
        
        # if num_channels % 2 != 0:
        #      raise ValueError(
        #         f"The number of channels must be even, but got {num_channels} "
        #         f"for input shape {x.shape}."
        #     )

        # 2. Flatten for the sorting logic
        # We flatten everything except batch. Since Channel is now last,
        # adjacent elements in this flattened view correspond to adjacent channels.
        x_flat = x_permuted.reshape(batch_size, -1)

        # --- Sort Logic (Verifiable / Auto_Lirpa compatible) ---
        
        # Group into pairs. 
        # Because we are Channel Last, the last dim is C.
        # This reshaping groups (c0, c1), (c2, c3), etc.
        reshaped_x = x_flat.reshape(batch_size, -1, 2)
        
        x1s = reshaped_x[..., 0]
        x2s = reshaped_x[..., 1]
        
        # Calculate diff and apply ReLU to determine min/max without conditional branching
        # min(a,b) = x2 - ReLU(x2 - x1)
        # max(a,b) = x1 + ReLU(x2 - x1)
        diff = x2s + (-1*x1s)
        relu_diff = self.relu(diff)
        
        y1 = x2s + (-1*relu_diff) # The smaller value
        y2 = x1s + relu_diff # The larger value
        
        sorted_pairs = torch.stack((y1, y2), dim=2)
        sorted_flat = sorted_pairs.reshape(batch_size, -1)
        
        # --- End Logic ---

        # 3. Restore Shape
        
        # First reshape back to the permuted shape (N, ..., C)
        output_permuted = sorted_flat.reshape(permuted_shape)
        
        # Finally, permute back to Channel First (N, C, ...)
        # We need to calculate the inverse permutation indices
        inv_dims = list(range(x.dim()))
        # Move the last dim (which is now channels) back to self.axis
        last_dim = inv_dims.pop(-1)
        inv_dims.insert(self.axis, last_dim)
        
        output = output_permuted.permute(inv_dims)
        
        return output
    
    
class GroupSort2Optimized(nn.Module):
    # THIS IMPLEMENTATION IS NOT VERIFIABLE WITH auto_LiRPA
    # due to torch.max(a, b)
    def __init__(self):
        super(GroupSort2Optimized, self).__init__()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.shape
        x_reshaped = x.view(N, C // 2, 2, H, W)
        x1s = x_reshaped[:, :, 0, :, :]
        x2s = x_reshaped[:, :, 1, :, :]
        # Max + Sum Preservation Logic
        y2_max = torch.max(x1s, x2s)
        y1_min = (x1s + x2s) - y2_max
        sorted_pairs = torch.stack((y1_min, y2_max), dim=2)
        return sorted_pairs.view(N, C, H, W)

    
class GroupSort2Conventional(nn.Module):
    """
    Applies GroupSort specifically on the channel dimension using classic max.
    
    It permutes the input from (N, C, ...) to (N, ..., C), applies the 
    sort logic so that pairs (c_2k, c_2k+1) are sorted using torch.max and 
    the identity min(a,b) = -max(-a,-b) to avoid alpha-CROWN compilation bugs, 
    and then restores the original layout.
    """
    def __init__(self, axis=1):
        super(GroupSort2Conventional, self).__init__()
        self.axis = axis

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Permute to Channel Last
        # We assume the channel is at self.axis (usually 1).
        # We move that axis to the very end (-1).
        dims = list(range(x.dim()))
        # Remove the channel axis from its current spot and append to end
        channel_dim = dims.pop(self.axis) 
        dims.append(channel_dim)
        
        # permute returns a view, but we usually need contiguous memory for reshaping
        x_permuted = x.permute(dims).contiguous()
        
        # Capture the shape after permutation: (N, D1, D2, ..., C)
        permuted_shape = x_permuted.shape
        batch_size = permuted_shape[0]
        num_channels = permuted_shape[-1]
        
        if num_channels % 2 != 0:
             raise ValueError(
                f"The number of channels must be even, but got {num_channels} "
                f"for input shape {x.shape}."
            )

        # 2. Flatten for the sorting logic
        # We flatten everything except batch. Since Channel is now last,
        # adjacent elements in this flattened view correspond to adjacent channels.
        x_flat = x_permuted.reshape(batch_size, -1)

        # --- Sort Logic (Max-Only Formulation) ---
        
        # Group into pairs. 
        # Because we are Channel Last, the last dim is C.
        # This reshaping groups (c0, c1), (c2, c3), etc.
        reshaped_x = x_flat.reshape(batch_size, -1, 2)
        
        # Slicing separates the pairs
        x1s = reshaped_x[..., 0]
        x2s = reshaped_x[..., 1]
        
        # 1. Compute Max directly
        y_max = torch.max(x1s, x2s)
        
        # 2. Compute Min using the identity: min(a, b) = -max(-a, -b)
        # This bypasses the alpha-CROWN bug with torch.min
        y_min = -torch.max(-x1s, -x2s)
        
        # Stack back together: [min, max]
        sorted_pairs = torch.stack((y_min, y_max), dim=2)
        sorted_flat = sorted_pairs.reshape(batch_size, -1)
        
        # --- End Logic ---

        # 3. Restore Shape
        
        # First reshape back to the permuted shape (N, ..., C)
        output_permuted = sorted_flat.reshape(permuted_shape)
        
        # Finally, permute back to Channel First (N, C, ...)
        # We need to calculate the inverse permutation indices
        inv_dims = list(range(x.dim()))
        # Move the last dim (which is now channels) back to self.axis
        last_dim = inv_dims.pop(-1)
        inv_dims.insert(self.axis, last_dim)
        
        output = output_permuted.permute(inv_dims)
        
        return output

def MNIST_MLP():
	model = nn.Sequential(
		nn.Flatten(),
		nn.Linear(784, 100),
		nn.ReLU(),
        # GroupSort_General(),

		nn.Linear(100, 100),
		nn.ReLU(),
        # GroupSort_General(),

		nn.Linear(100, 10)
	)
	return model

class MNIST_ConvSmall(nn.Module):
    def __init__(self):
        super(MNIST_ConvSmall, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=32*7*7, out_features=100)
        self.fc2 = nn.Linear(in_features=100, out_features=10)
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.flatten(x)
        x_p = torch.relu(self.fc1(x))
        x = self.fc2(x_p)
        return x

class MNIST_ConvLarge(nn.Module):
    def __init__(self):
        super(MNIST_ConvLarge, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=64*7*7, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
def CIFAR10_CNN_A():
    model = nn.Sequential(
        nn.Conv2d(3, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(32*8*8,100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model

def CIFAR10_CNN_B():
    return nn.Sequential(
        nn.ZeroPad2d((1,2,1,2)),
        nn.Conv2d(3, 32, (5,5), stride=2, padding=0),
        nn.ReLU(),
        nn.Conv2d(32, 128, (4,4), stride=2, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(8192, 250),
        nn.ReLU(),
        nn.Linear(250, 10),
    )

def CIFAR10_CNN_C():
    model = nn.Sequential(
        nn.Conv2d(3, 8, 4, stride=2, padding=0),
        nn.ReLU(),
        nn.Conv2d(8, 16, 4, stride=2, padding=0),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(576, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )
    return model

def CIFAR10_ConvSmall():
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
    return model

def CIFAR10_ConvDeep():
    model = nn.Sequential(
        nn.Conv2d(3, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(8*8*8, 100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model

# class CIFAR10_ConvLarge(nn.Module):
#     def __init__(self):
#         super(CIFAR10_ConvLarge, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1)
#         self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
#         self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
#         self.flatten = nn.Flatten()
#         self.fc1 = nn.Linear(in_features=64*8*8, out_features=512)
#         self.fc2 = nn.Linear(in_features=512, out_features=512)
#         self.fc3 = nn.Linear(in_features=512, out_features=10)

#     def forward(self, x):
#         x = torch.relu(self.conv1(x))
#         x = torch.relu(self.conv2(x))
#         x = torch.relu(self.conv3(x))
#         x = torch.relu(self.conv4(x))
#         x = self.flatten(x)
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
def CIFAR10_ConvLarge():
    """
    Creates the CIFAR10_ConvLarge model using nn.Sequential.
    """
    model = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1),
        nn.ReLU(),
        
        nn.Flatten(),
        
        nn.Linear(in_features=64*8*8, out_features=512),
        nn.ReLU(),
        nn.Linear(in_features=512, out_features=512),
        nn.ReLU(),
        nn.Linear(in_features=512, out_features=10)
    )
    return model

    
# Note: This python file present every model we use for benchmarking. 
# These models correspond to the lipschitz version of all the model architectures from Wang et al. (2021) & Leino et al. (2021)
# This version uses ReLU as the activation function.

def MLP_MNIST_1_LIP():
    """
    Model: MLP_1_LIP (MNIST)
    Structure: Linear(784, 100) -> ReLU -> Linear(100, 100) -> ReLU -> Linear(100, 10)
    """
    model = torchlip.Sequential(
        nn.Flatten(),
        torchlip.SpectralLinear(784, 100),
        nn.ReLU(),
        torchlip.SpectralLinear(100, 100),
        nn.ReLU(),
        torchlip.SpectralLinear(100, 10)
    )
    return model

def ConvSmall_MNIST_1_LIP():
    """
    Model: ConvSmall_1_LIP (MNIST)
    Structure: Conv(1, 16, 4, 2, 1) -> ReLU -> Conv(16, 32, 4, 2, 1) -> ReLU -> Linear(1568, 100) -> ReLU -> Linear(100, 10)
    """
    model = torchlip.Sequential(
        torchlip.SpectralConv2d(in_channels=1, out_channels=16, kernel_size=4, stride=2, padding=1, eps_bjorck=None),
        nn.ReLU(),
        torchlip.SpectralConv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1, eps_bjorck=None),
        nn.ReLU(),
        nn.Flatten(),
        torchlip.SpectralLinear(32 * 7 * 7, 100), # 1568 input features
        nn.ReLU(),
        torchlip.SpectralLinear(100, 10)
    )
    return model

def ConvLarge_MNIST_1_LIP():
    """
    Model: ConvLarge_1_LIP (MNIST)
    Structure: Conv(1, 32, 3, 1, 1) -> ReLU -> Conv(32, 32, 4, 2, 1) -> ReLU -> Conv(32, 64, 3, 1, 1) -> ReLU ->
               Conv(64, 64, 4, 2, 1) -> ReLU -> Linear(3136, 512) -> ReLU -> Linear(512, 512) -> ReLU -> Linear(512, 10)
    """
    model = torchlip.Sequential(
        torchlip.SpectralConv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1, eps_bjorck=None),
        nn.ReLU(),
        torchlip.SpectralConv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1, eps_bjorck=None),
        nn.ReLU(),
        torchlip.SpectralConv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, eps_bjorck=None),
        nn.ReLU(),
        torchlip.SpectralConv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1, eps_bjorck=None),
        nn.ReLU(),
        nn.Flatten(),
        torchlip.SpectralLinear(64 * 7 * 7, 512), # 3136 input features
        nn.ReLU(),
        torchlip.SpectralLinear(512, 512),
        nn.ReLU(),
        torchlip.SpectralLinear(512, 10)
    )
    return model

def CNNA_CIFAR10_1_LIP():
    """
    Model: CNN-A_1_LIP (CIFAR-10)
    Structure: Conv(3, 16, 4, 2, 1) -> ReLU -> Conv(16, 32, 4, 2, 1) -> ReLU -> Linear(2048, 100) -> ReLU -> Linear(100, 10)
    """
    model = torchlip.Sequential(
        torchlip.SpectralConv2d(in_channels=3, out_channels=16, kernel_size=4, stride=2, padding=1, eps_bjorck=None),
        nn.ReLU(),
        # GroupSort_General(),
        torchlip.SpectralConv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1, eps_bjorck=None),
        nn.ReLU(),
        # GroupSort_General(),
        nn.Flatten(),
        torchlip.SpectralLinear(32 * 8 * 8, 100), # 2048 input features
        nn.ReLU(),
        # GroupSort_General(),
        torchlip.SpectralLinear(100, 10)
    )
    return model

def CNNB_CIFAR10_1_LIP():
    """
    Model: CNN-B_1_LIP (CIFAR-10)
    Structure: Conv(3, 32, 5, 2, 0) -> ReLU -> Conv(32, 128, 4, 2, 1) -> ReLU -> Linear(6272, 250) -> ReLU -> Linear(250, 10)
    Note: The paper specifies Linear(8192, 250), which implies an 8x8 feature map before flattening (128*8*8=8192).
          However, the specified convolutional layers produce a 7x7 feature map (128*7*7=6272).
          This implementation follows the specified conv layers, resulting in 6272 input features.
    """
    model = torchlip.Sequential(
        torchlip.SpectralConv2d(in_channels=3, out_channels=32, kernel_size=5, stride=2, padding=0, eps_bjorck=None),
        nn.ReLU(),
        torchlip.SpectralConv2d(in_channels=32, out_channels=128, kernel_size=4, stride=2, padding=1, eps_bjorck=None),
        nn.ReLU(),
        nn.Flatten(),
        torchlip.SpectralLinear(128 * 7 * 7, 250), # 6272 input features
        nn.ReLU(),
        torchlip.SpectralLinear(250, 10)
    )
    return model

def CNNC_CIFAR10_1_LIP():
    """
    Model: CNN-C_1_LIP (CIFAR-10)
    Structure: Conv(3, 8, 4, 2, 0) -> ReLU -> Conv(8, 16, 4, 2, 0) -> ReLU -> Linear(576, 128) -> ReLU ->
               Linear(128, 64) -> ReLU -> Linear(64, 10)
    """
    model = torchlip.Sequential(
        torchlip.SpectralConv2d(in_channels=3, out_channels=8, kernel_size=4, stride=2, padding=0, eps_bjorck=None),
        nn.ReLU(),
        torchlip.SpectralConv2d(in_channels=8, out_channels=16, kernel_size=4, stride=2, padding=0, eps_bjorck=None),
        nn.ReLU(),
        nn.Flatten(),
        torchlip.SpectralLinear(16 * 6 * 6, 128), # 576 input features
        nn.ReLU(),
        torchlip.SpectralLinear(128, 64),
        nn.ReLU(),
        torchlip.SpectralLinear(64, 10)
    )
    return model

def ConvSmall_CIFAR10_1_LIP():
    """
    Model: ConvSmall_1_LIP (CIFAR-10)
    Structure: Conv(3, 16, 4, 2, 0) -> ReLU -> Conv(16, 32, 4, 2, 0) -> ReLU -> Linear(1152, 100) -> ReLU -> Linear(100, 10)
    """
    model = torchlip.Sequential(
        torchlip.SpectralConv2d(in_channels=3, out_channels=16, kernel_size=4, stride=2, padding=0, eps_bjorck=None),
        nn.ReLU(),
        torchlip.SpectralConv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=0, eps_bjorck=None),
        nn.ReLU(),
        nn.Flatten(),
        torchlip.SpectralLinear(32 * 6 * 6, 100), # 1152 input features
        nn.ReLU(),
        torchlip.SpectralLinear(100, 10)
    )
    return model

def ConvDeep_CIFAR10_1_LIP():
    """
    Model: ConvDeep_1_LIP (CIFAR-10)
    Structure: Conv(3, 8, 4, 2, 1) -> ReLU -> Conv(8, 8, 3, 1, 1) -> ReLU -> Conv(8, 8, 3, 1, 1) -> ReLU ->
               Conv(8, 8, 4, 2, 1) -> ReLU -> Linear(512, 100) -> ReLU -> Linear(100, 10)
    """
    model = torchlip.Sequential(
        torchlip.SpectralConv2d(in_channels=3, out_channels=8, kernel_size=4, stride=2, padding=1, eps_bjorck=None),
        nn.ReLU(),
        torchlip.SpectralConv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1, eps_bjorck=None),
        nn.ReLU(),
        torchlip.SpectralConv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1, eps_bjorck=None),
        nn.ReLU(),
        torchlip.SpectralConv2d(in_channels=8, out_channels=8, kernel_size=4, stride=2, padding=1, eps_bjorck=None),
        nn.ReLU(),
        nn.Flatten(),
        torchlip.SpectralLinear(8 * 8 * 8, 100), # 512 input features
        nn.ReLU(),
        torchlip.SpectralLinear(100, 10)
    )
    return model

def ConvLarge_CIFAR10_1_LIP():
    """
    Model: ConvLarge_1_LIP (CIFAR-10)
    Structure: Conv(3, 32, 3, 1, 1) -> ReLU -> Conv(32, 32, 4, 2, 1) -> ReLU -> Conv(32, 64, 3, 1, 1) -> ReLU ->
               Conv(64, 64, 4, 2, 1) -> ReLU -> Linear(4096, 512) -> ReLU -> Linear(512, 512) -> ReLU -> Linear(512, 10)
    """
    model = torchlip.Sequential(
        torchlip.SpectralConv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, eps_bjorck=None),
        nn.ReLU(),
        torchlip.SpectralConv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1, eps_bjorck=None),
        nn.ReLU(),
        torchlip.SpectralConv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, eps_bjorck=None),
        nn.ReLU(),
        torchlip.SpectralConv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1, eps_bjorck=None),
        nn.ReLU(),
        nn.Flatten(),
        torchlip.SpectralLinear(64 * 8 * 8, 512), # 4096 input features
        nn.ReLU(),
        torchlip.SpectralLinear(512, 512),
        nn.ReLU(),
        torchlip.SpectralLinear(512, 10)
    )
    return model

def VGG13_1_LIP_CIFAR10():
    """
    Model: VGG13-like_1_LIP_Bjork (CIFAR-10)
    Structure matched to GNP version but with SpectralConv2d
    """
    model = torchlip.Sequential(
        # Block 1
        torchlip.SpectralConv2d(3, 64, 3, 1, 1, eps_bjorck=None),
        nn.ReLU(),
        torchlip.SpectralConv2d(64, 64, 3, 1, 1, eps_bjorck=None),
        nn.ReLU(),
        # Downsample
        torchlip.SpectralConv2d(64, 64, 3, 2, 1, eps_bjorck=None),
        nn.ReLU(),

        # Block 2
        torchlip.SpectralConv2d(64, 128, 3, 1, 1, eps_bjorck=None),
        nn.ReLU(),
        torchlip.SpectralConv2d(128, 128, 3, 1, 1, eps_bjorck=None),
        nn.ReLU(),
        # Downsample
        torchlip.SpectralConv2d(128, 128, 3, 2, 1, eps_bjorck=None),
        nn.ReLU(),

        # Block 3
        torchlip.SpectralConv2d(128, 256, 3, 1, 1, eps_bjorck=None),
        nn.ReLU(),
        torchlip.SpectralConv2d(256, 256, 3, 1, 1, eps_bjorck=None),
        nn.ReLU(),
        # Downsample
        torchlip.SpectralConv2d(256, 256, 3, 2, 1, eps_bjorck=None),
        nn.ReLU(),

        # Classifier
        nn.Flatten(),
        torchlip.SpectralLinear(256 * 4 * 4, 512),
        nn.ReLU(),
        torchlip.SpectralLinear(512, 512),
        nn.ReLU(),
        torchlip.SpectralLinear(512, 10)
    )
    return model



# Note: This python file present every model we use for benchmarking. 
# These models correspond to the lipschitz Gradient Norm Preserving version of all the model architectures from Wang et al. (2021) & Leino et al. (2021)
# This version uses Group Sort 2 as the activation function.

from orthogonium.layers.conv.AOC import AdaptiveOrthoConv2d
from orthogonium.reparametrizers import DEFAULT_ORTHO_PARAMS

def MLP_MNIST_1_LIP_GNP():
    """
    Model: MLP_1_LIP_GNP (MNIST)
    Structure: Linear(784, 100) -> ReLU -> Linear(100, 100) -> ReLU -> Linear(100, 10)
    """
    model = torchlip.Sequential(
        nn.Flatten(),
        torchlip.SpectralLinear(784, 100),
        GroupSort_General(),
        torchlip.SpectralLinear(100, 100),
        GroupSort_General(),
        torchlip.SpectralLinear(100, 10)
    )
    return model

def ConvSmall_MNIST_1_LIP_GNP():
    """
    Model: ConvSmall_1_LIP_GNP (MNIST)
    Structure: Conv(1, 16, 4, 2, 1) -> ReLU -> Conv(16, 32, 4, 2, 1) -> ReLU -> Linear(1568, 100) -> ReLU -> Linear(100, 10)
    """
    model = torchlip.Sequential(
        AdaptiveOrthoConv2d(in_channels=1, out_channels=16, kernel_size=4, stride=2, padding=1, padding_mode='zeros',ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        AdaptiveOrthoConv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1, padding_mode='zeros',ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        nn.Flatten(),
        torchlip.SpectralLinear(32 * 7 * 7, 100), # 1568 input features
        GroupSort_General(),
        torchlip.SpectralLinear(100, 10)
    )
    return model

def ConvLarge_MNIST_1_LIP_GNP():
    """
    Model: ConvLarge_1_LIP_GNP (MNIST)
    Structure: Conv(1, 32, 3, 1, 1) -> ReLU -> Conv(32, 32, 4, 2, 1) -> ReLU -> Conv(32, 64, 3, 1, 1) -> ReLU ->
               Conv(64, 64, 4, 2, 1) -> ReLU -> Linear(3136, 512) -> ReLU -> Linear(512, 512) -> ReLU -> Linear(512, 10)
    """
    model = torchlip.Sequential(
        AdaptiveOrthoConv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1, padding_mode='zeros',ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        AdaptiveOrthoConv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1, padding_mode='zeros',ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        AdaptiveOrthoConv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, padding_mode='zeros',ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        AdaptiveOrthoConv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1, padding_mode='zeros',ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        nn.Flatten(),
        torchlip.SpectralLinear(64 * 7 * 7, 512), # 3136 input features
        GroupSort_General(),
        torchlip.SpectralLinear(512, 512),
        GroupSort_General(),
        torchlip.SpectralLinear(512, 10)
    )
    return model

def CNNA_CIFAR10_1_LIP_GNP():
    """
    Model: CNN-A_1_LIP_GNP (CIFAR-10)
    Structure: Conv(3, 16, 4, 2, 1) -> ReLU -> Conv(16, 32, 4, 2, 1) -> ReLU -> Linear(2048, 100) -> ReLU -> Linear(100, 10)
    """
    model = torchlip.Sequential(
        AdaptiveOrthoConv2d(in_channels=3, out_channels=16, kernel_size=4, stride=2, padding=1, padding_mode='zeros', ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        # nn.ReLU(),
        AdaptiveOrthoConv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1, padding_mode='zeros',ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        # nn.ReLU(),
        nn.Flatten(),
        torchlip.SpectralLinear(32 * 8 * 8, 100), # 2048 input features
        GroupSort_General(),
        # nn.ReLU(),
        torchlip.SpectralLinear(100, 10)
    )
    return model



class GroupSort(nn.Module):
    """
    1-Lipschitz GroupSort operator (Universal).
    
    - Implemented via Sparse Convolutions (1x1).
    - Agnostic to input shape: works automatically for (B, C), (B, C, H, W), etc.
    - Verified Auto_LiRPA sound.
    """
    def __init__(self, channels, axis=1):
        super().__init__()
        self.axis = axis
        self.channels = channels
        
        # We use Conv2d(1x1) as the universal computing engine
        self.diff = nn.Conv2d(channels, channels // 2, kernel_size=1, bias=False)
        self.expand = nn.Conv2d(channels // 2, channels, kernel_size=1, bias=False)
        
        # Freeze and initialize weights
        self.diff.weight.requires_grad = False
        self.expand.weight.requires_grad = False
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            self.diff.weight.fill_(0)
            for k in range(self.channels // 2):
                self.diff.weight[k, 2*k, 0, 0] = 1.0
                self.diff.weight[k, 2*k + 1, 0, 0] = -1.0
            
            self.expand.weight.fill_(0)
            for k in range(self.channels // 2):
                self.expand.weight[2*k, k, 0, 0] = -1.0 
                self.expand.weight[2*k + 1, k, 0, 0] = 1.0

    def forward(self, x):
        # 1. Standardize the sorting axis to index 1 (Channel dimension)
        if self.axis != 1:
            x_raw = x.transpose(1, self.axis)
        else:
            x_raw = x

        # 2. Capture original shape to restore later
        original_shape = x_raw.shape
        
        # 3. Agnostic Reshape: (Batch, Channel, ...) -> (Batch, Channel, 1, Flat)
        #    This forces the tensor into a 4D shape compatible with Conv2d
        #    regardless of whether the input was 2D, 3D, or 4D.
        x_view = x_raw.reshape(original_shape[0], original_shape[1], 1, -1)

        # 4. Apply Sorting Logic
        #    Note: The 1x1 kernel treats the flattened dim independently
        v = self.diff(x_view)
        z = torch.relu(v)
        correction = self.expand(z)
        out_view = x_view + correction

        # 5. Restore original shape
        out = out_view.view(original_shape)
        
        # 6. Restore original axis
        if self.axis != 1:
            out = out.transpose(1, self.axis)
            
        return out

# def CNNA_CIFAR10_1_LIP_GNP():
#     """
#     Model: CNN-A_1_LIP_GNP (CIFAR-10)
#     Structure: Conv(3, 16, 4, 2, 1) -> ReLU -> Conv(16, 32, 4, 2, 1) -> ReLU -> Linear(2048, 100) -> ReLU -> Linear(100, 10)
#     """
#     model = torchlip.Sequential(
#         AdaptiveOrthoConv2d(in_channels=3, out_channels=16, kernel_size=4, stride=2, padding=1, padding_mode='zeros', ortho_params=DEFAULT_ORTHO_PARAMS),
#         GroupSort(channels=16, axis=1),
#         # nn.ReLU(),
#         AdaptiveOrthoConv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1, padding_mode='zeros',ortho_params=DEFAULT_ORTHO_PARAMS),
#         GroupSort(channels=32, axis=1),
#         # nn.ReLU(),
#         nn.Flatten(),
#         torchlip.SpectralLinear(32 * 8 * 8, 100), # 2048 input features
#         GroupSort(channels=100, axis=1),
#         # nn.ReLU(),
#         torchlip.SpectralLinear(100, 10)
#     )
#     return model

def CNNA_CIFAR10_1_LIP_GNP_torchlip():
    """
    Model: CNN-A_1_LIP_GNP (CIFAR-10)
    Structure: Conv(3, 16, 4, 2, 1) -> ReLU -> Conv(16, 32, 4, 2, 1) -> ReLU -> Linear(2048, 100) -> ReLU -> Linear(100, 10)
    """
    model = torchlip.Sequential(
        AdaptiveOrthoConv2d(in_channels=3, out_channels=16, kernel_size=4, stride=2, padding=1, padding_mode='zeros', ortho_params=DEFAULT_ORTHO_PARAMS),
        torchlip.GroupSort2(),
        # nn.ReLU(),
        AdaptiveOrthoConv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1, padding_mode='zeros',ortho_params=DEFAULT_ORTHO_PARAMS),
        torchlip.GroupSort2(),
        # nn.ReLU(),
        nn.Flatten(),
        torchlip.SpectralLinear(32 * 8 * 8, 100), # 2048 input features
        torchlip.GroupSort2(),
        # nn.ReLU(),
        torchlip.SpectralLinear(100, 10)
    )
    return model

def CNNA_CIFAR10_1_LIP_GNP_circular():
    """
    Model: CNN-A_1_LIP_GNP (CIFAR-10)
    Structure: Conv(3, 16, 4, 2, 1) -> ReLU -> Conv(16, 32, 4, 2, 1) -> ReLU -> Linear(2048, 100) -> ReLU -> Linear(100, 10)
    """
    model = torchlip.Sequential(
        AdaptiveOrthoConv2d(in_channels=3, out_channels=16, kernel_size=4, stride=2, padding=1, ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        # nn.ReLU(),
        AdaptiveOrthoConv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1,ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        # nn.ReLU(),
        nn.Flatten(),
        torchlip.SpectralLinear(32 * 8 * 8, 100), # 2048 input features
        GroupSort_General(),
        # nn.ReLU(),
        torchlip.SpectralLinear(100, 10)
    )
    return model

def CNNB_CIFAR10_1_LIP_GNP():
    """
    Model: CNN-B_1_LIP_GNP (CIFAR-10)
    Structure: Conv(3, 32, 5, 2, 0) -> ReLU -> Conv(32, 128, 4, 2, 1) -> ReLU -> Linear(6272, 250) -> ReLU -> Linear(250, 10)
    """
    model = torchlip.Sequential(
        AdaptiveOrthoConv2d(in_channels=3, out_channels=32, kernel_size=5, stride=2, padding=0, padding_mode='zeros',ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        AdaptiveOrthoConv2d(in_channels=32, out_channels=128, kernel_size=4, stride=2, padding=1, padding_mode='zeros',ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        nn.Flatten(),
        torchlip.SpectralLinear(128 * 7 * 7, 250), # 6272 input features
        GroupSort_General(),
        torchlip.SpectralLinear(250, 10)
    )
    return model

def CNNC_CIFAR10_1_LIP_GNP():
    """
    Model: CNN-C_1_LIP_GNP (CIFAR-10)
    Structure: Conv(3, 8, 4, 2, 0) -> ReLU -> Conv(8, 16, 4, 2, 0) -> ReLU -> Linear(576, 128) -> ReLU ->
               Linear(128, 64) -> ReLU -> Linear(64, 10)
    """
    model = torchlip.Sequential(
        AdaptiveOrthoConv2d(in_channels=3, out_channels=8, kernel_size=4, stride=2, padding=0, padding_mode='zeros',ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        AdaptiveOrthoConv2d(in_channels=8, out_channels=16, kernel_size=4, stride=2, padding=0, padding_mode='zeros',ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        nn.Flatten(),
        torchlip.SpectralLinear(16 * 6 * 6, 128), # 576 input features
        GroupSort_General(),
        torchlip.SpectralLinear(128, 64),
        GroupSort_General(),
        torchlip.SpectralLinear(64, 10)
    )
    return model

def ConvSmall_CIFAR10_1_LIP_GNP():
    """
    Model: ConvSmall_1_LIP_GNP (CIFAR-10)
    Structure: Conv(3, 16, 4, 2, 0) -> ReLU -> Conv(16, 32, 4, 2, 0) -> ReLU -> Linear(1152, 100) -> ReLU -> Linear(100, 10)
    """
    model = torchlip.Sequential(
        AdaptiveOrthoConv2d(in_channels=3, out_channels=16, kernel_size=4, stride=2, padding=0, padding_mode='zeros',ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        # nn.ReLU(),
        AdaptiveOrthoConv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=0, padding_mode='zeros',ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        # nn.ReLU(),
        nn.Flatten(),
        torchlip.SpectralLinear(32 * 6 * 6, 100), # 1152 input features
        GroupSort_General(),
        # nn.ReLU(),
        torchlip.SpectralLinear(100, 10)
    )
    return model

def ConvDeep_CIFAR10_1_LIP_GNP():
    """
    Model: ConvDeep_1_LIP_GNP (CIFAR-10)
    Structure: Conv(3, 8, 4, 2, 1) -> ReLU -> Conv(8, 8, 3, 1, 1) -> ReLU -> Conv(8, 8, 3, 1, 1) -> ReLU ->
               Conv(8, 8, 4, 2, 1) -> ReLU -> Linear(512, 100) -> ReLU -> Linear(100, 10)
    """
    model = torchlip.Sequential(
        AdaptiveOrthoConv2d(in_channels=3, out_channels=8, kernel_size=4, stride=2, padding=1, padding_mode='zeros',ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        AdaptiveOrthoConv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1, padding_mode='zeros',ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        AdaptiveOrthoConv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1, padding_mode='zeros',ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        AdaptiveOrthoConv2d(in_channels=8, out_channels=8, kernel_size=4, stride=2, padding=1, padding_mode='zeros',ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        nn.Flatten(),
        torchlip.SpectralLinear(8 * 8 * 8, 100), # 512 input features
        GroupSort_General(),
        torchlip.SpectralLinear(100, 10)
    )
    return model

def ConvLarge_CIFAR10_1_LIP_GNP():
    """
    Model: ConvLarge_1_LIP_GNP (CIFAR-10)
    Structure: Conv(3, 32, 3, 1, 1) -> ReLU -> Conv(32, 32, 4, 2, 1) -> ReLU -> Conv(32, 64, 3, 1, 1) -> ReLU ->
               Conv(64, 64, 4, 2, 1) -> ReLU -> Linear(4096, 512) -> ReLU -> Linear(512, 512) -> ReLU -> Linear(512, 10)
    """
    model = torchlip.Sequential(
        AdaptiveOrthoConv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, padding_mode='zeros',ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        # nn.ReLU(),
        AdaptiveOrthoConv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1, padding_mode='zeros',ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        # nn.ReLU(),
        AdaptiveOrthoConv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, padding_mode='zeros',ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        # nn.ReLU(),
        AdaptiveOrthoConv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1, padding_mode='zeros',ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        # nn.ReLU(),
        nn.Flatten(),
        torchlip.SpectralLinear(64 * 8 * 8, 512), # 4096 input features
        GroupSort_General(),
        # nn.ReLU(),
        torchlip.SpectralLinear(512, 512),
        GroupSort_General(),
        # nn.ReLU(),
        torchlip.SpectralLinear(512, 10)
    )
    return model

def ConvLarge_Bottleneck_1_LIP_GNP():
    """
    Model: ConvLarge_Bottleneck_1_LIP_GNP (CIFAR-10)
    Structure: Conv(3, 32, 3, 1, 1) -> GroupSort -> Conv(32, 8, 4, 2, 1) -> GroupSort -> 
               Conv(8, 64, 3, 1, 1) -> GroupSort -> Conv(64, 64, 4, 2, 1) -> GroupSort -> 
               Linear(4096, 1024) -> GroupSort -> Linear(1024, 512) -> GroupSort -> Linear(512, 10)
    """
    model = torchlip.Sequential(
        AdaptiveOrthoConv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, padding_mode='zeros', ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        # nn.ReLU(),
        AdaptiveOrthoConv2d(in_channels=32, out_channels=8, kernel_size=4, stride=2, padding=1, padding_mode='zeros', ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        # nn.ReLU(),
        AdaptiveOrthoConv2d(in_channels=8, out_channels=64, kernel_size=3, stride=1, padding=1, padding_mode='zeros', ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        # nn.ReLU(),
        AdaptiveOrthoConv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1, padding_mode='zeros', ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        # nn.ReLU(),
        nn.Flatten(),
        torchlip.SpectralLinear(64 * 8 * 8, 1024), # 4096 input features
        GroupSort_General(),
        # nn.ReLU(),
        torchlip.SpectralLinear(1024, 512),
        GroupSort_General(),
        # nn.ReLU(),
        torchlip.SpectralLinear(512, 10)
    )
    return model

def VGG13_1_LIP_GNP_CIFAR10():
    """
    Model: VGG13-like_1_LIP_GNP (CIFAR-10)
    Structure: [Conv(64) x 2] -> StridedConv -> [Conv(128) x 2] -> StridedConv -> 
               [Conv(256) x 2] -> StridedConv -> [Linear(512) x 2] -> Linear(10)
    Input: 3x32x32
    """
    model = torchlip.Sequential(
        # Block 1: 3x32x32 -> 64x32x32
        AdaptiveOrthoConv2d(3, 64, 3, 1, 1, padding_mode='zeros',ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        AdaptiveOrthoConv2d(64, 64, 3, 1, 1, padding_mode='zeros',ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        # Downsample: 64x32x32 -> 64x16x16
        AdaptiveOrthoConv2d(64, 64, 3, 2, 1, padding_mode='zeros',ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),

        # Block 2: 64x16x16 -> 128x16x16
        AdaptiveOrthoConv2d(64, 128, 3, 1, 1, padding_mode='zeros', ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        AdaptiveOrthoConv2d(128, 128, 3, 1, 1, padding_mode='zeros', ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        # Downsample: 128x16x16 -> 128x8x8
        AdaptiveOrthoConv2d(128, 128, 3, 2, 1, padding_mode='zeros', ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),

        # Block 3: 128x8x8 -> 256x8x8
        AdaptiveOrthoConv2d(128, 256, 3, 1, 1,padding_mode='zeros',  ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        AdaptiveOrthoConv2d(256, 256, 3, 1, 1, padding_mode='zeros', ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        # Downsample: 256x8x8 -> 256x4x4
        AdaptiveOrthoConv2d(256, 256, 3, 2, 1, padding_mode='zeros', ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),

        # Classifier
        nn.Flatten(),
        # Input features: 256 * 4 * 4 = 4096
        torchlip.SpectralLinear(256 * 4 * 4, 512),
        GroupSort_General(),
        torchlip.SpectralLinear(512, 512),
        GroupSort_General(),
        torchlip.SpectralLinear(512, 10)
    )
    return model


def VGG16_1_LIP_GNP_CIFAR10():
    """
    Model: VGG16-like_1_LIP_GNP (CIFAR-10)
    Structure: [Conv(64) x 2] -> StridedConv -> [Conv(128) x 2] -> StridedConv -> 
               [Conv(256) x 3] -> StridedConv -> [Linear(512) x 2] -> Linear(10)
    Input: 3x32x32
    """
    model = torchlip.Sequential(
        # Block 1: 3x32x32 -> 64x32x32
        AdaptiveOrthoConv2d(3, 64, 3, 1, 1, padding_mode='zeros', ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        AdaptiveOrthoConv2d(64, 64, 3, 1, 1, padding_mode='zeros', ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        # Downsample: 64x32x32 -> 64x16x16
        AdaptiveOrthoConv2d(64, 64, 3, 2, 1, padding_mode='zeros', ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),

        # Block 2: 64x16x16 -> 128x16x16
        AdaptiveOrthoConv2d(64, 128, 3, 1, 1, padding_mode='zeros', ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        AdaptiveOrthoConv2d(128, 128, 3, 1, 1, padding_mode='zeros', ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        # Downsample: 128x16x16 -> 128x8x8
        AdaptiveOrthoConv2d(128, 128, 3, 2, 1, padding_mode='zeros', ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),

        # Block 3: 128x8x8 -> 256x8x8
        AdaptiveOrthoConv2d(128, 256, 3, 1, 1, padding_mode='zeros', ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        AdaptiveOrthoConv2d(256, 256, 3, 1, 1, padding_mode='zeros', ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        AdaptiveOrthoConv2d(256, 256, 3, 1, 1, padding_mode='zeros', ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        # Downsample: 256x8x8 -> 256x4x4
        AdaptiveOrthoConv2d(256, 256, 3, 2, 1, padding_mode='zeros', ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),

        # Classifier
        nn.Flatten(),
        # Input features: 256 * 4 * 4 = 4096
        torchlip.SpectralLinear(256 * 4 * 4, 512),
        GroupSort_General(),
        torchlip.SpectralLinear(512, 512),
        GroupSort_General(),
        torchlip.SpectralLinear(512, 10)
    )
    return model

def VGG19_1_LIP_GNP_CIFAR10():
    """
    Model: VGG19-like_1_LIP_GNP (CIFAR-10)
    Structure: [Conv(64) x 2] -> StridedConv -> [Conv(128) x 2] -> StridedConv -> 
               [Conv(256) x 4] -> StridedConv -> [Conv(512) x 4] -> StridedConv ->
               [Linear(512) x 2] -> Linear(10)
    Input: 3x32x32
    """
    model = torchlip.Sequential(
        # Block 1: 3x32x32 -> 64x32x32
        AdaptiveOrthoConv2d(3, 64, 3, 1, 1, padding_mode='zeros', ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        AdaptiveOrthoConv2d(64, 64, 3, 1, 1, padding_mode='zeros', ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        # Downsample: 64x32x32 -> 64x16x16
        AdaptiveOrthoConv2d(64, 64, 3, 2, 1, padding_mode='zeros', ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),

        # Block 2: 64x16x16 -> 128x16x16
        AdaptiveOrthoConv2d(64, 128, 3, 1, 1, padding_mode='zeros', ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        AdaptiveOrthoConv2d(128, 128, 3, 1, 1, padding_mode='zeros', ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        # Downsample: 128x16x16 -> 128x8x8
        AdaptiveOrthoConv2d(128, 128, 3, 2, 1, padding_mode='zeros', ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),

        # Block 3: 128x8x8 -> 256x8x8
        AdaptiveOrthoConv2d(128, 256, 3, 1, 1, padding_mode='zeros', ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        AdaptiveOrthoConv2d(256, 256, 3, 1, 1, padding_mode='zeros', ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        AdaptiveOrthoConv2d(256, 256, 3, 1, 1, padding_mode='zeros', ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        AdaptiveOrthoConv2d(256, 256, 3, 1, 1, padding_mode='zeros', ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        # Downsample: 256x8x8 -> 256x4x4
        AdaptiveOrthoConv2d(256, 256, 3, 2, 1, padding_mode='zeros', ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),

        # Block 4: 256x4x4 -> 512x4x4
        AdaptiveOrthoConv2d(256, 512, 3, 1, 1, padding_mode='zeros', ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        AdaptiveOrthoConv2d(512, 512, 3, 1, 1, padding_mode='zeros', ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        AdaptiveOrthoConv2d(512, 512, 3, 1, 1, padding_mode='zeros', ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        AdaptiveOrthoConv2d(512, 512, 3, 1, 1, padding_mode='zeros', ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        # Downsample: 512x4x4 -> 512x2x2
        AdaptiveOrthoConv2d(512, 512, 3, 2, 1, padding_mode='zeros', ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),

        # Classifier
        nn.Flatten(),
        # Input features: 512 * 2 * 2 = 2048
        torchlip.SpectralLinear(512 * 2 * 2, 512),
        GroupSort_General(),
        torchlip.SpectralLinear(512, 512),
        GroupSort_General(),
        torchlip.SpectralLinear(512, 10)
    )
    return model


# Note: This section defines the 1_LIP_Bjork family.
# Characteristics: SpectralConv2d (default Bjorck), SpectralLinear, GroupSort 2 activation.

def MLP_MNIST_1_LIP_Bjork():
    """
    Model: MLP_1_LIP_Bjork (MNIST)
    """
    model = torchlip.Sequential(
        nn.Flatten(),
        torchlip.SpectralLinear(784, 100),
        GroupSort_General(),
        torchlip.SpectralLinear(100, 100),
        GroupSort_General(),
        torchlip.SpectralLinear(100, 10)
    )
    return model

def ConvSmall_MNIST_1_LIP_Bjork():
    """
    Model: ConvSmall_1_LIP_Bjork (MNIST)
    """
    model = torchlip.Sequential(
        torchlip.SpectralConv2d(in_channels=1, out_channels=16, kernel_size=4, stride=2, padding=1),
        GroupSort_General(),
        torchlip.SpectralConv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1),
        GroupSort_General(),
        nn.Flatten(),
        torchlip.SpectralLinear(32 * 7 * 7, 100), # 1568 input features
        GroupSort_General(),
        torchlip.SpectralLinear(100, 10)
    )
    return model

def ConvLarge_MNIST_1_LIP_Bjork():
    """
    Model: ConvLarge_1_LIP_Bjork (MNIST)
    """
    model = torchlip.Sequential(
        torchlip.SpectralConv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
        GroupSort_General(),
        torchlip.SpectralConv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1),
        GroupSort_General(),
        torchlip.SpectralConv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
        GroupSort_General(),
        torchlip.SpectralConv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1),
        GroupSort_General(),
        nn.Flatten(),
        torchlip.SpectralLinear(64 * 7 * 7, 512), # 3136 input features
        GroupSort_General(),
        torchlip.SpectralLinear(512, 512),
        GroupSort_General(),
        torchlip.SpectralLinear(512, 10)
    )
    return model

def CNNA_CIFAR10_1_LIP_Bjork():
    """
    Model: CNN-A_1_LIP_Bjork (CIFAR-10)
    """
    model = torchlip.Sequential(
        torchlip.SpectralConv2d(in_channels=3, out_channels=16, kernel_size=4, stride=2, padding=1),
        GroupSort_General(),
        torchlip.SpectralConv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1),
        GroupSort_General(),
        nn.Flatten(),
        torchlip.SpectralLinear(32 * 8 * 8, 100), # 2048 input features
        GroupSort_General(),
        torchlip.SpectralLinear(100, 10)
    )
    return model

def CNNB_CIFAR10_1_LIP_Bjork():
    """
    Model: CNN-B_1_LIP_Bjork (CIFAR-10)
    """
    model = torchlip.Sequential(
        torchlip.SpectralConv2d(in_channels=3, out_channels=32, kernel_size=5, stride=2, padding=0),
        GroupSort_General(),
        torchlip.SpectralConv2d(in_channels=32, out_channels=128, kernel_size=4, stride=2, padding=1),
        GroupSort_General(),
        nn.Flatten(),
        torchlip.SpectralLinear(128 * 7 * 7, 250), # 6272 input features
        GroupSort_General(),
        torchlip.SpectralLinear(250, 10)
    )
    return model

def CNNC_CIFAR10_1_LIP_Bjork():
    """
    Model: CNN-C_1_LIP_Bjork (CIFAR-10)
    """
    model = torchlip.Sequential(
        torchlip.SpectralConv2d(in_channels=3, out_channels=8, kernel_size=4, stride=2, padding=0),
        GroupSort_General(),
        torchlip.SpectralConv2d(in_channels=8, out_channels=16, kernel_size=4, stride=2, padding=0),
        GroupSort_General(),
        nn.Flatten(),
        torchlip.SpectralLinear(16 * 6 * 6, 128), # 576 input features
        GroupSort_General(),
        torchlip.SpectralLinear(128, 64),
        GroupSort_General(),
        torchlip.SpectralLinear(64, 10)
    )
    return model

def ConvSmall_CIFAR10_1_LIP_Bjork():
    """
    Model: ConvSmall_1_LIP_Bjork (CIFAR-10)
    """
    model = torchlip.Sequential(
        torchlip.SpectralConv2d(in_channels=3, out_channels=16, kernel_size=4, stride=2, padding=0),
        GroupSort_General(),
        torchlip.SpectralConv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=0),
        GroupSort_General(),
        nn.Flatten(),
        torchlip.SpectralLinear(32 * 6 * 6, 100), # 1152 input features
        GroupSort_General(),
        torchlip.SpectralLinear(100, 10)
    )
    return model

def ConvDeep_CIFAR10_1_LIP_Bjork():
    """
    Model: ConvDeep_1_LIP_Bjork (CIFAR-10)
    """
    model = torchlip.Sequential(
        torchlip.SpectralConv2d(in_channels=3, out_channels=8, kernel_size=4, stride=2, padding=1),
        GroupSort_General(),
        torchlip.SpectralConv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1),
        GroupSort_General(),
        torchlip.SpectralConv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1),
        GroupSort_General(),
        torchlip.SpectralConv2d(in_channels=8, out_channels=8, kernel_size=4, stride=2, padding=1),
        GroupSort_General(),
        nn.Flatten(),
        torchlip.SpectralLinear(8 * 8 * 8, 100), # 512 input features
        GroupSort_General(),
        torchlip.SpectralLinear(100, 10)
    )
    return model

def ConvLarge_CIFAR10_1_LIP_Bjork():
    """
    Model: ConvLarge_1_LIP_Bjork (CIFAR-10)
    """
    model = torchlip.Sequential(
        torchlip.SpectralConv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
        GroupSort_General(),
        torchlip.SpectralConv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1),
        GroupSort_General(),
        torchlip.SpectralConv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
        GroupSort_General(),
        torchlip.SpectralConv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1),
        GroupSort_General(),
        nn.Flatten(),
        torchlip.SpectralLinear(64 * 8 * 8, 512), # 4096 input features
        GroupSort_General(),
        torchlip.SpectralLinear(512, 512),
        GroupSort_General(),
        torchlip.SpectralLinear(512, 10)
    )
    return model

def VGG13_1_LIP_Bjork_CIFAR10():
    """
    Model: VGG13-like_1_LIP_Bjork (CIFAR-10)
    Structure matched to GNP version but with SpectralConv2d
    """
    model = torchlip.Sequential(
        # Block 1
        torchlip.SpectralConv2d(3, 64, 3, 1, 1),
        GroupSort_General(),
        torchlip.SpectralConv2d(64, 64, 3, 1, 1),
        GroupSort_General(),
        # Downsample
        torchlip.SpectralConv2d(64, 64, 3, 2, 1),
        GroupSort_General(),

        # Block 2
        torchlip.SpectralConv2d(64, 128, 3, 1, 1),
        GroupSort_General(),
        torchlip.SpectralConv2d(128, 128, 3, 1, 1),
        GroupSort_General(),
        # Downsample
        torchlip.SpectralConv2d(128, 128, 3, 2, 1),
        GroupSort_General(),

        # Block 3
        torchlip.SpectralConv2d(128, 256, 3, 1, 1),
        GroupSort_General(),
        torchlip.SpectralConv2d(256, 256, 3, 1, 1),
        GroupSort_General(),
        # Downsample
        torchlip.SpectralConv2d(256, 256, 3, 2, 1),
        GroupSort_General(),

        # Classifier
        nn.Flatten(),
        torchlip.SpectralLinear(256 * 4 * 4, 512),
        GroupSort_General(),
        torchlip.SpectralLinear(512, 512),
        GroupSort_General(),
        torchlip.SpectralLinear(512, 10)
    )
    return model

def VGG16_1_LIP_Bjork_CIFAR10():
    """
    Model: VGG16-like_1_LIP_Bjork (CIFAR-10)
    """
    model = torchlip.Sequential(
        # Block 1
        torchlip.SpectralConv2d(3, 64, 3, 1, 1),
        GroupSort_General(),
        torchlip.SpectralConv2d(64, 64, 3, 1, 1),
        GroupSort_General(),
        # Downsample
        torchlip.SpectralConv2d(64, 64, 3, 2, 1),
        GroupSort_General(),

        # Block 2
        torchlip.SpectralConv2d(64, 128, 3, 1, 1),
        GroupSort_General(),
        torchlip.SpectralConv2d(128, 128, 3, 1, 1),
        GroupSort_General(),
        # Downsample
        torchlip.SpectralConv2d(128, 128, 3, 2, 1),
        GroupSort_General(),

        # Block 3
        torchlip.SpectralConv2d(128, 256, 3, 1, 1),
        GroupSort_General(),
        torchlip.SpectralConv2d(256, 256, 3, 1, 1),
        GroupSort_General(),
        torchlip.SpectralConv2d(256, 256, 3, 1, 1),
        GroupSort_General(),
        # Downsample
        torchlip.SpectralConv2d(256, 256, 3, 2, 1),
        GroupSort_General(),

        # Classifier
        nn.Flatten(),
        torchlip.SpectralLinear(256 * 4 * 4, 512),
        GroupSort_General(),
        torchlip.SpectralLinear(512, 512),
        GroupSort_General(),
        torchlip.SpectralLinear(512, 10)
    )
    return model

def VGG19_1_LIP_Bjork_CIFAR10():
    """
    Model: VGG19-like_1_LIP_Bjork (CIFAR-10)
    """
    model = torchlip.Sequential(
        # Block 1
        torchlip.SpectralConv2d(3, 64, 3, 1, 1),
        GroupSort_General(),
        torchlip.SpectralConv2d(64, 64, 3, 1, 1),
        GroupSort_General(),
        # Downsample
        torchlip.SpectralConv2d(64, 64, 3, 2, 1),
        GroupSort_General(),

        # Block 2
        torchlip.SpectralConv2d(64, 128, 3, 1, 1),
        GroupSort_General(),
        torchlip.SpectralConv2d(128, 128, 3, 1, 1),
        GroupSort_General(),
        # Downsample
        torchlip.SpectralConv2d(128, 128, 3, 2, 1),
        GroupSort_General(),

        # Block 3
        torchlip.SpectralConv2d(128, 256, 3, 1, 1),
        GroupSort_General(),
        torchlip.SpectralConv2d(256, 256, 3, 1, 1),
        GroupSort_General(),
        torchlip.SpectralConv2d(256, 256, 3, 1, 1),
        GroupSort_General(),
        torchlip.SpectralConv2d(256, 256, 3, 1, 1),
        GroupSort_General(),
        # Downsample
        torchlip.SpectralConv2d(256, 256, 3, 2, 1),
        GroupSort_General(),

        # Block 4
        torchlip.SpectralConv2d(256, 512, 3, 1, 1),
        GroupSort_General(),
        torchlip.SpectralConv2d(512, 512, 3, 1, 1),
        GroupSort_General(),
        torchlip.SpectralConv2d(512, 512, 3, 1, 1),
        GroupSort_General(),
        torchlip.SpectralConv2d(512, 512, 3, 1, 1),
        GroupSort_General(),
        # Downsample
        torchlip.SpectralConv2d(512, 512, 3, 2, 1),
        GroupSort_General(),

        # Classifier
        nn.Flatten(),
        torchlip.SpectralLinear(512 * 2 * 2, 512),
        GroupSort_General(),
        torchlip.SpectralLinear(512, 512),
        GroupSort_General(),
        torchlip.SpectralLinear(512, 10)
    )
    return model

from typing import List
from deel import torchlip
from orthogonium.layers import AdaptiveOrthoConv2d, OrthoLinear
import torch
import torch.nn as nn
import math
import torch
import torch.nn as nn

class LirpaFriendlyL2Pool2d(nn.Module):
    def __init__(self, kernel_size, stride, k_coef_lip=1.0):
        super().__init__()
        if isinstance(kernel_size, tuple):
            self.n_pixels = kernel_size[0] * kernel_size[1]
        else:
            self.n_pixels = kernel_size ** 2
        
        self.scaling_factor = math.sqrt(self.n_pixels) * k_coef_lip
        self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        # Remove epsilon for the sanity check match
        # auto_LiRPA can handle sqrt(x) as long as x > 0
        return torch.sqrt(self.pool(x.pow(2))) * self.scaling_factor

class LirpaFriendlyAdaptiveL2Pool2d(nn.Module):
    def __init__(self, output_size=(1, 1), k_coef_lip=1.0):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(output_size)
        self.k_coef_lip = k_coef_lip

    def forward(self, x):
        n_pixels = x.shape[-2] * x.shape[-1]
        # Match the DEEL math exactly: sqrt(sum(x^2))
        return torch.sqrt(self.pool(x.pow(2)) * n_pixels) * self.k_coef_lip
    
class LirpaBatchCentering2D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # We create a buffer so vanilla_export can copy the trained running_mean into it
        self.register_buffer("running_mean", torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        # Subtract the fixed mean stored during training
        return x - self.running_mean

class LirpaPixelUnshuffle(nn.Module):
    def __init__(self, upscale_factor):
        super().__init__()
        self.r = upscale_factor

    def forward(self, x):
        # Explicitly extract dimensions as integers
        # This prevents auto_LiRPA from creating "BoundGather" nodes for shapes
        b, c, h, w = x.shape
        r = self.r
        
        # Manually perform the unshuffle using explicit views
        # This bypasses the problematic nn.PixelUnshuffle trace
        out = x.view(b, c, h // r, r, w // r, r)
        out = out.permute(0, 1, 3, 5, 2, 4).contiguous()
        return out.view(b, c * (r ** 2), h // r, w // r)
    
class BasicBlockLipschitz(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, orthogonal: bool = False):
        super().__init__()

        conv = AdaptiveOrthoConv2d if orthogonal else torchlip.SpectralConv2d

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.alpha = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        
        self.conv1 = conv(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, padding_mode="zeros", bias=False)
        self.conv2 = conv(out_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode="zeros", bias=False)
        
        self.bc1 = LirpaBatchCentering2D(out_channels)
        self.bc2 = LirpaBatchCentering2D(out_channels)
        self.act = GroupSort_General()
        
        if in_channels != out_channels:
            if stride != 1:
                self.skipconv = nn.Sequential(
                    LirpaPixelUnshuffle(stride),
                    conv(in_channels * stride ** 2, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
                )
            else:
                self.skipconv = conv(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        else:
            self.skipconv = nn.Identity()
            
        self.skipbc = LirpaBatchCentering2D(out_channels)

    def forward(self, x):
        residual = x
        
        if self.in_channels != self.out_channels:
            residual = self.skipconv(residual)
        residual = self.skipbc(residual)

        x = self.conv1(x)
        x = self.bc1(x)
        x = self.act(x)
        
        x = self.conv2(x)
        x = self.bc2(x)

        alpha = torch.sigmoid(self.alpha).view(1, 1, 1, 1)
        x = alpha * x + (1 - alpha) * residual
        return x


class BottleneckBlockLipschitz(nn.Module):
    expansion = 4

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, orthogonal: bool = False):
        super().__init__()

        conv = AdaptiveOrthoConv2d if orthogonal else torchlip.SpectralConv2d

        self.alpha = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        
        self.conv1 = conv(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = conv(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv3 = conv(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0, bias=False)
        
        self.bc1 = LirpaBatchCentering2D(out_channels)
        self.bc2 = LirpaBatchCentering2D(out_channels)
        self.bc3 = LirpaBatchCentering2D(out_channels * self.expansion)
        self.act = GroupSort_General()
        
        # FIXED: Added PixelUnshuffle logic to handle stride > 1 safely for Orthogonal Convolutions
        if stride != 1 or in_channels != out_channels * self.expansion:
            if stride != 1:
                self.shortcut = nn.Sequential(
                    LirpaPixelUnshuffle(stride),
                    conv(in_channels * (stride ** 2), out_channels * self.expansion, kernel_size=1, stride=1, padding=0, bias=False),
                    LirpaBatchCentering2D(out_channels * self.expansion)
                )
            else:
                self.shortcut = nn.Sequential(
                    conv(in_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0, bias=False),
                    LirpaBatchCentering2D(out_channels * self.expansion)
                )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = x
        alpha = alpha = torch.sigmoid(self.alpha).view(1, 1, 1, 1)
        
        x = self.conv1(x)
        x = self.bc1(x)
        x = self.act(x)
        
        x = self.conv2(x)
        x = self.bc2(x)
        x = self.act(x)
        
        x = self.conv3(x)
        x = self.bc3(x)
        
        x = alpha * x + (1 - alpha) * self.shortcut(residual)
        x = self.act(x)
        return x
    
class ResNetLipschitz(nn.Module):
    def __init__(
            self, 
            in_channels: int, 
            out_channels: int, 
            block: BasicBlockLipschitz | BottleneckBlockLipschitz, 
            layers: List[int], 
            num_classes: int, 
            orthogonal: bool = False,
            input_size: int = 32) -> None:
        super().__init__()

        conv = AdaptiveOrthoConv2d if orthogonal else torchlip.SpectralConv2d
        
        # We now use SpectralLinear in all cases, removing the 'linear' toggle.

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_size = input_size

        # 1. Initial Pooling (Only for large inputs)
        if input_size == 224:
            self.conv1 = conv(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False)
            self.pool1 = LirpaFriendlyL2Pool2d(kernel_size=2, stride=2)
        else:
            self.conv1 = conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.pool1 = nn.Identity()

        self.bc1 = LirpaBatchCentering2D(out_channels)
        self.act = GroupSort_General()
        
        self.layers = nn.ModuleList([
            self._make_layer(block, out_channels * (2 ** i), layers[i], stride=1 if i == 0 else 2, orthogonal=orthogonal) 
            for i in range(len(layers))
        ])
        
        # 2. Global Pooling (Before the classifier)
        self.pool = LirpaFriendlyAdaptiveL2Pool2d((1, 1))
        
        # Always use SpectralLinear
        self.fc = torchlip.SpectralLinear(int(out_channels * (2 ** (len(layers) - 1)) * block.expansion), num_classes)

    def _make_layer(self, block: nn.Module, out_channels: int, num_blocks: int, stride: int, orthogonal: bool) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.out_channels, out_channels, s, orthogonal))
            self.out_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bc1(x)
        x = self.act(x)
        x = self.pool1(x)
        
        for layer in self.layers:
            for block in layer:
                x = block(x)
                
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
def ResNet18_1_LIP_GNP():
    """
    Wrapper for a 1-Lipschitz ResNet-18 on CIFAR-10.
    
    Note: A standard ResNet-18 uses `BasicBlockLipschitz` with a [2, 2, 2, 2] layout. 
    If your strict requirement is to use Bottleneck blocks based on the naming convention, 
    you can swap `BasicBlockLipschitz` for `BottleneckBlockLipschitz`, but it will no 
    longer be a standard ResNet-18.
    """
    return ResNetLipschitz(
        in_channels=3,             # RGB images for CIFAR-10
        out_channels=64,           # Starting number of filters
        block=BasicBlockLipschitz, # Standard ResNet-18 uses Basic Blocks
        layers=[2, 2, 2, 2],       # The block distribution for ResNet-18
        num_classes=10,            # 10 classes in CIFAR-10
        orthogonal=True,          # Set to True if you want AdaptiveOrthoConv2d instead of SpectralConv
        input_size=32              # 32x32 resolution for CIFAR-10
    )

    
def ResNet18_1_LIP_Bjork():
    """
    Wrapper for a 1-Lipschitz ResNet-18 on CIFAR-10.
    
    Note: A standard ResNet-18 uses `BasicBlockLipschitz` with a [2, 2, 2, 2] layout. 
    If your strict requirement is to use Bottleneck blocks based on the naming convention, 
    you can swap `BasicBlockLipschitz` for `BottleneckBlockLipschitz`, but it will no 
    longer be a standard ResNet-18.
    """
    return ResNetLipschitz(
        in_channels=3,             # RGB images for CIFAR-10
        out_channels=64,           # Starting number of filters
        block=BasicBlockLipschitz, # Standard ResNet-18 uses Basic Blocks
        layers=[2, 2, 2, 2],       # The block distribution for ResNet-18
        num_classes=10,            # 10 classes in CIFAR-10
        orthogonal=False,          # Set to True if you want AdaptiveOrthoConv2d instead of SpectralConv
        input_size=32              # 32x32 resolution for CIFAR-10
    )

def ResNet18_1_LIP_GNP_Imagenette():
    """
    Wrapper for a 1-Lipschitz ResNet-18 on Imagenette.
    Uses Orthogonal convolutions (GNP).
    """
    return ResNetLipschitz(
        in_channels=3,             # RGB images
        out_channels=64,           # Starting number of filters
        block=BasicBlockLipschitz, # Standard ResNet-18 uses Basic Blocks
        layers=[2, 2, 2, 2],       # The block distribution for ResNet-18
        num_classes=10,            # 10 classes in Imagenette
        orthogonal=True,           # Set to True for AdaptiveOrthoConv2d (GNP)
        input_size=224             # 224x224 resolution for Imagenette/ImageNet
    )


def ResNet18_1_LIP_Bjork_Imagenette():
    """
    Wrapper for a 1-Lipschitz ResNet-18 on Imagenette.
    Uses Spectral Normalization convolutions (Bjork).
    """
    return ResNetLipschitz(
        in_channels=3,             # RGB images
        out_channels=64,           # Starting number of filters
        block=BasicBlockLipschitz, # Standard ResNet-18 uses Basic Blocks
        layers=[2, 2, 2, 2],       # The block distribution for ResNet-18
        num_classes=10,            # 10 classes in Imagenette
        orthogonal=False,          # Set to False for SpectralConv2d (Bjork)
        input_size=224             # 224x224 resolution for Imagenette/ImageNet
    )
