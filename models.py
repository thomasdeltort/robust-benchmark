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
class GroupSort_General(nn.Module):
    """
    A universal, auto_lirpa-compatible PyTorch module that sorts pairs of features.

    This module can handle inputs of any shape (e.g., 2D, 4D). It works by 
    temporarily flattening the feature dimensions, applying the sort logic in a 
    verifier-friendly way (with ReLU on a 2D tensor), and then reshaping the 
    output back to the original input shape.

    The total number of features (product of dimensions after the batch dim) must be even.
    """
    def __init__(self):
        super(GroupSort_General, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        batch_size = original_shape[0]
        
        num_features = np.prod(original_shape[1:])
        
        if num_features % 2 != 0:
            raise ValueError(
                f"The total number of features must be even, but got {num_features} "
                f"for shape {original_shape}."
            )

        # Utiliser .reshape() pour gérer les tenseurs non-contigus
        x_flat = x.reshape(batch_size, -1)

        # --- Logique de tri ---
        
        # .reshape() est aussi plus sûr ici, par précaution
        reshaped_x = x_flat.reshape(batch_size, -1, 2)
        
        x1s = reshaped_x[..., 0]
        x2s = reshaped_x[..., 1]
        
        diff = x2s - x1s
        relu_diff = self.relu(diff)
        
        y1 = x2s - relu_diff
        y2 = x1s + relu_diff
        
        sorted_pairs = torch.stack((y1, y2), dim=2)
        
        # .reshape() est aussi plus sûr ici
        sorted_flat = sorted_pairs.reshape(batch_size, -1)

        # --- Fin de la logique ---

        # Restaurer la forme originale en utilisant .reshape()
        output = sorted_flat.reshape(original_shape)
        
        return output
    
class MaxMin(nn.Module):
    """
    Computes min/max using torch.split.
    
    WARNING: This will still cause the broadcasting RuntimeError
    in auto_LiRPA when batch size > 1.
    """
    def forward(self, x):
        original_shape = x.shape
        batch_size = original_shape[0]
        num_features = np.prod(original_shape[1:])
        
        if num_features % 2 != 0: 
            raise ValueError("Total features must be even.")
            
        x_flat = x.reshape(batch_size, -1)
        x_pairs = x_flat.reshape(batch_size, -1, 2)
        
        # --- Using torch.split ---
        # x_pairs has shape [batch_size, num_pairs, 2]
        # This splits along dim=-1 into two tensors of size 1
        a_tensor, b_tensor = torch.split(x_pairs, 1, dim=-1)
        
        # Squeeze the last dim to get shape [batch_size, num_pairs]
        a = a_tensor.squeeze(-1)
        b = b_tensor.squeeze(-1)
        
        # --- This is the part that causes the error ---
        # These operations trigger the buggy handler in auto_LiRPA
        min_vals = -torch.max(-a, -b)
        max_vals = torch.max(a, b)
        
        # --- End of problematic part ---
        
        sorted_pairs = torch.stack((min_vals, max_vals), dim=-1)
        sorted_flat = sorted_pairs.reshape(batch_size, -1)
        return sorted_flat.reshape(original_shape)
    
class GroupSort2Optimized(nn.Module):
    # THIS IMPLEMENTATION IS NOT VERIFIABLE WITH auto_LiRPA
    # due to torch.max(a, b)
    def __init__(self):
        super(GroupSort2Optimized, self).__init__()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        batch_size = original_shape[0]
        num_features = np.prod(original_shape[1:])
        if num_features % 2 != 0: raise ValueError("Total features must be even.")
        x_flat = x.reshape(batch_size, -1)
        reshaped_x = x_flat.reshape(batch_size, -1, 2)
        x1s = reshaped_x[..., 0]
        x2s = reshaped_x[..., 1]
        y2_max = torch.max(x1s, x2s) # <--- THIS IS THE UNSUPPORTED OPERATION
        y1_min = x1s + x2s - y2_max
        sorted_pairs = torch.stack((y1_min, y2_max), dim=2)
        sorted_flat = sorted_pairs.reshape(batch_size, -1)
        output = sorted_flat.reshape(original_shape)
        return output
    
def ConvLarge_MNIST_1_LIP_GNP_MaxMin():
    """
    Model: ConvLarge_1_LIP_GNP (MNIST)
    Structure: Conv(1, 32, 3, 1, 1) -> ReLU -> Conv(32, 32, 4, 2, 1) -> ReLU -> Conv(32, 64, 3, 1, 1) -> ReLU ->
               Conv(64, 64, 4, 2, 1) -> ReLU -> Linear(3136, 512) -> ReLU -> Linear(512, 512) -> ReLU -> Linear(512, 10)
    """
    model = torchlip.Sequential(
        AdaptiveOrthoConv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1, padding_mode='zeros',ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort2Optimized(),
        AdaptiveOrthoConv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1, padding_mode='zeros',ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort2Optimized(),
        AdaptiveOrthoConv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, padding_mode='zeros',ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort2Optimized(),
        AdaptiveOrthoConv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1, padding_mode='zeros',ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort2Optimized(),
        nn.Flatten(),
        torchlip.SpectralLinear(64 * 7 * 7, 512), # 3136 input features
        GroupSort2Optimized(),
        torchlip.SpectralLinear(512, 512),
        GroupSort2Optimized(),
        torchlip.SpectralLinear(512, 10)
    )
    return model

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
        #FIXME Convert to zero padding
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
        AdaptiveOrthoConv2d(3, 64, 3, 1, 1, ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        AdaptiveOrthoConv2d(64, 64, 3, 1, 1, ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        # Downsample: 64x32x32 -> 64x16x16
        AdaptiveOrthoConv2d(64, 64, 3, 2, 1, ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),

        # Block 2: 64x16x16 -> 128x16x16
        AdaptiveOrthoConv2d(64, 128, 3, 1, 1, ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        AdaptiveOrthoConv2d(128, 128, 3, 1, 1, ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        # Downsample: 128x16x16 -> 128x8x8
        AdaptiveOrthoConv2d(128, 128, 3, 2, 1, ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),

        # Block 3: 128x8x8 -> 256x8x8
        AdaptiveOrthoConv2d(128, 256, 3, 1, 1, ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        AdaptiveOrthoConv2d(256, 256, 3, 1, 1, ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        # Downsample: 256x8x8 -> 256x4x4
        AdaptiveOrthoConv2d(256, 256, 3, 2, 1, ortho_params=DEFAULT_ORTHO_PARAMS),
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
        AdaptiveOrthoConv2d(3, 64, 3, 1, 1, ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        AdaptiveOrthoConv2d(64, 64, 3, 1, 1, ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        # Downsample: 64x32x32 -> 64x16x16
        AdaptiveOrthoConv2d(64, 64, 3, 2, 1, ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),

        # Block 2: 64x16x16 -> 128x16x16
        AdaptiveOrthoConv2d(64, 128, 3, 1, 1, ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        AdaptiveOrthoConv2d(128, 128, 3, 1, 1, ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        # Downsample: 128x16x16 -> 128x8x8
        AdaptiveOrthoConv2d(128, 128, 3, 2, 1, ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),

        # Block 3: 128x8x8 -> 256x8x8
        AdaptiveOrthoConv2d(128, 256, 3, 1, 1, ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        AdaptiveOrthoConv2d(256, 256, 3, 1, 1, ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        AdaptiveOrthoConv2d(256, 256, 3, 1, 1, ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        # Downsample: 256x8x8 -> 256x4x4
        AdaptiveOrthoConv2d(256, 256, 3, 2, 1, ortho_params=DEFAULT_ORTHO_PARAMS),
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
        AdaptiveOrthoConv2d(3, 64, 3, 1, 1, ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        AdaptiveOrthoConv2d(64, 64, 3, 1, 1, ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        # Downsample: 64x32x32 -> 64x16x16
        AdaptiveOrthoConv2d(64, 64, 3, 2, 1, ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),

        # Block 2: 64x16x16 -> 128x16x16
        AdaptiveOrthoConv2d(64, 128, 3, 1, 1, ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        AdaptiveOrthoConv2d(128, 128, 3, 1, 1, ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        # Downsample: 128x16x16 -> 128x8x8
        AdaptiveOrthoConv2d(128, 128, 3, 2, 1, ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),

        # Block 3: 128x8x8 -> 256x8x8
        AdaptiveOrthoConv2d(128, 256, 3, 1, 1, ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        AdaptiveOrthoConv2d(256, 256, 3, 1, 1, ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        AdaptiveOrthoConv2d(256, 256, 3, 1, 1, ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        AdaptiveOrthoConv2d(256, 256, 3, 1, 1, ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        # Downsample: 256x8x8 -> 256x4x4
        AdaptiveOrthoConv2d(256, 256, 3, 2, 1, ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),

        # Block 4: 256x4x4 -> 512x4x4
        AdaptiveOrthoConv2d(256, 512, 3, 1, 1, ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        AdaptiveOrthoConv2d(512, 512, 3, 1, 1, ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        AdaptiveOrthoConv2d(512, 512, 3, 1, 1, ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        AdaptiveOrthoConv2d(512, 512, 3, 1, 1, ortho_params=DEFAULT_ORTHO_PARAMS),
        GroupSort_General(),
        # Downsample: 512x4x4 -> 512x2x2
        AdaptiveOrthoConv2d(512, 512, 3, 2, 1, ortho_params=DEFAULT_ORTHO_PARAMS),
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