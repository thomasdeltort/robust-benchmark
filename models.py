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
        
        if num_channels % 2 != 0:
             raise ValueError(
                f"The number of channels must be even, but got {num_channels} "
                f"for input shape {x.shape}."
            )

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



# class GroupSort(nn.Module):
#     """
#     1-Lipschitz GroupSort operator (Universal).
    
#     - Implemented via Sparse Convolutions (1x1).
#     - Agnostic to input shape: works automatically for (B, C), (B, C, H, W), etc.
#     - Verified Auto_LiRPA sound.
#     """
#     def __init__(self, channels, axis=1):
#         super().__init__()
#         self.axis = axis
#         self.channels = channels
        
#         # We use Conv2d(1x1) as the universal computing engine
#         self.diff = nn.Conv2d(channels, channels // 2, kernel_size=1, bias=False)
#         self.expand = nn.Conv2d(channels // 2, channels, kernel_size=1, bias=False)
        
#         # Freeze and initialize weights
#         self.diff.weight.requires_grad = False
#         self.expand.weight.requires_grad = False
#         self._init_weights()

#     def _init_weights(self):
#         with torch.no_grad():
#             self.diff.weight.fill_(0)
#             for k in range(self.channels // 2):
#                 self.diff.weight[k, 2*k, 0, 0] = 1.0
#                 self.diff.weight[k, 2*k + 1, 0, 0] = -1.0
            
#             self.expand.weight.fill_(0)
#             for k in range(self.channels // 2):
#                 self.expand.weight[2*k, k, 0, 0] = -1.0 
#                 self.expand.weight[2*k + 1, k, 0, 0] = 1.0

#     def forward(self, x):
#         # 1. Standardize the sorting axis to index 1 (Channel dimension)
#         if self.axis != 1:
#             x_raw = x.transpose(1, self.axis)
#         else:
#             x_raw = x

#         # 2. Capture original shape to restore later
#         original_shape = x_raw.shape
        
#         # 3. Agnostic Reshape: (Batch, Channel, ...) -> (Batch, Channel, 1, Flat)
#         #    This forces the tensor into a 4D shape compatible with Conv2d
#         #    regardless of whether the input was 2D, 3D, or 4D.
#         x_view = x_raw.reshape(original_shape[0], original_shape[1], 1, -1)

#         # 4. Apply Sorting Logic
#         #    Note: The 1x1 kernel treats the flattened dim independently
#         v = self.diff(x_view)
#         z = torch.relu(v)
#         correction = self.expand(z)
#         out_view = x_view + correction

#         # 5. Restore original shape
#         out = out_view.view(original_shape)
        
#         # 6. Restore original axis
#         if self.axis != 1:
#             out = out.transpose(1, self.axis)
            
#         return out


# class GroupSort(nn.Module):
#     """
#     Universal GroupSort (1-Lipschitz).
    
#     - If input is 4D (N, C, H, W): Acts as a 1x1 Convolution (Spatial).
#     - If input is 2D (N, C): Acts as a Linear layer (Dense).
    
#     Implemented via Conv2d weights to ensure maximum compatibility with 
#     Auto_LiRPA's bound propagation graph.
#     """
#     def __init__(self, channels, axis=1):
#         super().__init__()
#         # Auto_LiRPA verification usually targets the channel axis (1).
#         if axis != 1:
#             raise ValueError("GroupSort axis must be 1 (Channel).")
        
#         if channels % 2 != 0:
#             raise ValueError(f"Channels must be even, got {channels}")
            
#         self.channels = channels
        
#         # We use Conv2d(1x1) as the core engine. 
#         # For 2D inputs, we simply unsqueeze dimensions to fit this engine.
#         self.conv_diff = nn.Conv2d(channels, channels // 2, kernel_size=1, bias=False)
#         self.conv_expand = nn.Conv2d(channels // 2, channels, kernel_size=1, bias=False)
        
#         # Freeze weights
#         self.conv_diff.weight.requires_grad = False
#         self.conv_expand.weight.requires_grad = False
#         self._init_weights()

#     def _init_weights(self):
#         with torch.no_grad():
#             # 1. Diff Conv: Calculate (Even - Odd)
#             self.conv_diff.weight.fill_(0)
#             for k in range(self.channels // 2):
#                 self.conv_diff.weight[k, 2*k, 0, 0] = 1.0     # Even
#                 self.conv_diff.weight[k, 2*k + 1, 0, 0] = -1.0 # Odd
            
#             # 2. Expand Conv: Apply corrections
#             self.conv_expand.weight.fill_(0)
#             for k in range(self.channels // 2):
#                 # If (Even > Odd), subtract difference from Even (swap)
#                 self.conv_expand.weight[2*k, k, 0, 0] = -1.0
#                 # If (Even > Odd), add difference to Odd (swap)
#                 self.conv_expand.weight[2*k + 1, k, 0, 0] = 1.0

#     def forward(self, x):
#         # 1. Detect Input Type
#         is_2d = (x.dim() == 2)
        
#         if is_2d:
#             # Case: Linear Layer Output (N, C)
#             # We reshape to (N, C, 1, 1) so it looks like an image pixel.
#             # This is safe for verification because there is no spatial structure to lose.
#             x_reshaped = x.view(*x.shape, 1, 1)
#         else:
#             # Case: Conv Layer Output (N, C, H, W)
#             # Use as is.
#             x_reshaped = x

#         # 2. Sort Logic (Identical for both)
#         diff = self.conv_diff(x_reshaped)
#         activation = torch.relu(diff)
#         correction = self.conv_expand(activation)
#         out_reshaped = x_reshaped + correction
        
#         # 3. Restore Output Shape
#         if is_2d:
#             # Reshape (N, C, 1, 1) back to (N, C)
#             return out_reshaped.view(x.shape)
#         else:
#             return out_reshaped

import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupSort(nn.Module):
    """
    SDP-CROWN Safe GroupSort.
    
    Crucial for Stability:
    - 4D Inputs (Conv): Uses nn.Conv2d.
    - 2D Inputs (Linear): Uses F.linear (Explicit Matrix Multiply).
    
    This prevents the "Size mismatch at dimension 3" error in SDP-CROWN
    by ensuring 2D inputs never have a 'ghost' spatial dimension.
    """
    def __init__(self, channels, axis=1):
        super().__init__()
        if axis != 1:
            raise ValueError("GroupSort axis must be 1 (Channel).")
        
        if channels % 2 != 0:
            raise ValueError(f"Channels must be even, got {channels}")
            
        self.channels = channels
        self.axis = axis
        
        # Store weights in Conv2d to allow easy loading of state_dicts
        self.conv_diff = nn.Conv2d(channels, channels // 2, kernel_size=1, bias=False)
        self.conv_expand = nn.Conv2d(channels // 2, channels, kernel_size=1, bias=False)
        
        # Freeze and Initialize
        self.conv_diff.weight.requires_grad = False
        self.conv_expand.weight.requires_grad = False
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            self.conv_diff.weight.fill_(0)
            for k in range(self.channels // 2):
                self.conv_diff.weight[k, 2*k, 0, 0] = 1.0
                self.conv_diff.weight[k, 2*k + 1, 0, 0] = -1.0
            
            self.conv_expand.weight.fill_(0)
            for k in range(self.channels // 2):
                self.conv_expand.weight[2*k, k, 0, 0] = -1.0
                self.conv_expand.weight[2*k + 1, k, 0, 0] = 1.0

    def forward(self, x):
        # Path A: Convolutional (N, C, H, W)
        if x.dim() == 4:
            diff = self.conv_diff(x)
            activation = torch.relu(diff)
            correction = self.conv_expand(activation)
            return x + correction

        # Path B: Linear (N, C) - The Fix for SDP-CROWN
        elif x.dim() == 2:
            # Dynamically reshape the 1x1 conv weights to Linear weights (Out, In)
            w_diff = self.conv_diff.weight.view(self.channels // 2, self.channels)
            w_expand = self.conv_expand.weight.view(self.channels, self.channels // 2)
            
            # Use functional linear instead of fake reshaping
            diff = F.linear(x, w_diff)
            activation = torch.relu(diff)
            correction = F.linear(activation, w_expand)
            return x + correction
            
        else:
            raise ValueError(f"GroupSort input must be 2D or 4D, got {x.dim()}D")
        
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