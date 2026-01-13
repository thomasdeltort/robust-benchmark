import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import sys
import numpy as np

# -------------------------------------------------------------------------
# SETUP
# -------------------------------------------------------------------------
SDP_CROWN_PATH = '/home/aws_install/robustess_project/SDP-CROWN'
sys.path.insert(0, SDP_CROWN_PATH)

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm

try:
    from deel import torchlip
except ImportError:
    print("[-] deel-torchlip not found.")
    sys.exit(1)

# =========================================================================
# 1. MODEL A: Sparse Conv (Sound) - Ascending Sort
# =========================================================================
class GroupSort_SparseConv(nn.Module):
    def __init__(self, channels, axis=1):
        super().__init__()
        self.axis = axis
        self.channels = channels
        self.diff = nn.Conv2d(channels, channels // 2, kernel_size=1, bias=False)
        self.expand = nn.Conv2d(channels // 2, channels, kernel_size=1, bias=False)
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
        if self.axis != 1: x = x.transpose(1, self.axis)
        v = self.diff(x)
        z = torch.relu(v)
        correction = self.expand(z)
        out = x + correction
        if self.axis != 1: out = out.transpose(1, self.axis)
        return out

# =========================================================================
# 2. MODEL B: General (Unsound/Loose) - Ascending Sort
# =========================================================================
class GroupSort_General(nn.Module):
    def __init__(self, channels, axis=1):
        super(GroupSort_General, self).__init__()
        self.axis = axis
        self.relu = nn.ReLU()
        self.channels = channels 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dims = list(range(x.dim()))
        channel_dim = dims.pop(self.axis) 
        dims.append(channel_dim)
        x_permuted = x.permute(dims).contiguous()
        permuted_shape = x_permuted.shape
        batch_size = permuted_shape[0]
        x_flat = x_permuted.reshape(batch_size, -1, 2)
        
        x_even = x_flat[..., 0] 
        x_odd  = x_flat[..., 1] 
        
        diff = x_even - x_odd
        relu_diff = self.relu(diff)
        
        y_min = x_even - relu_diff
        y_max = x_odd  + relu_diff
        
        sorted_pairs = torch.stack((y_min, y_max), dim=-1)
        sorted_flat = sorted_pairs.reshape(permuted_shape)
        
        inv_dims = list(range(x.dim()))
        last_dim = inv_dims.pop(-1)
        inv_dims.insert(self.axis, last_dim)
        output = sorted_flat.permute(inv_dims)
        return output

# =========================================================================
# 3. BUILDER & TRAINER
# =========================================================================
def make_mnist_model(gs_class):
    C = 16 
    model = nn.Sequential(
        torchlip.SpectralConv2d(1, C, kernel_size=3, padding=1),
        gs_class(channels=C, axis=1),
        torchlip.SpectralConv2d(C, C*2, kernel_size=3, padding=1, stride=2), 
        gs_class(channels=C*2, axis=1),
        torchlip.SpectralConv2d(C*2, C*4, kernel_size=3, padding=1, stride=2),
        gs_class(channels=C*4, axis=1),
        nn.Flatten(),
        torchlip.SpectralLinear(C*4 * 7 * 7, 10)
    )
    return model

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 200 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}]\tLoss: {loss.item():.6f}')

# =========================================================================
# 4. EXPERIMENT RUNNER
# =========================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- STEP 1: TRAIN MODEL A (The Sound One) ---
    print("\n" + "="*60)
    print("STEP 1: Training Model A (Sparse Conv)...")
    print("="*60)
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = DataLoader(datasets.MNIST('../data', train=True, download=True, transform=transform), batch_size=64, shuffle=True)
    test_loader = DataLoader(datasets.MNIST('../data', train=False, transform=transform), batch_size=1000, shuffle=False)
    
    model_a = make_mnist_model(GroupSort_SparseConv).to(device)
    # Init spectral layers
    _ = model_a(torch.randn(1, 1, 28, 28).to(device))
    optimizer = optim.Adam(model_a.parameters(), lr=0.001)

    for epoch in range(1, 4): # Train 3 Epochs
        train(model_a, device, train_loader, optimizer, epoch)
    
    # --- STEP 2: CREATE MODEL B & COPY WEIGHTS ---
    print("\n" + "="*60)
    print("STEP 2: Syncing to Model B (General)...")
    print("="*60)
    
    model_b = make_mnist_model(GroupSort_General).to(device)
    _ = model_b(torch.randn(1, 1, 28, 28).to(device)) # Init hooks
    model_b.load_state_dict(model_a.state_dict(), strict=False)
    
    # Verify equivalence
    dummy = torch.randn(1, 1, 28, 28).to(device)
    diff = (model_a(dummy) - model_b(dummy)).abs().max().item()
    print(f"Model Equivalence Check (Output Diff): {diff:.9f}")

    # --- STEP 3: VERIFY BOTH ---
    print("\n" + "="*60)
    print("STEP 3: Robustness Verification (Eps=0.1)")
    print("="*60)
    
    data, target = next(iter(test_loader))
    data, target = data[0:1].to(device), target[0:1].to(device)
    c_matrix = torch.zeros((1, 1, 10), device=device)
    c_matrix[0, 0, target.item()] = 1.0
    
    ptb = PerturbationLpNorm(norm=2, eps=0.1)
    x_in = BoundedTensor(data, ptb)
    
    import warnings
    warnings.filterwarnings("ignore")

    results = []

    for name, model in [("Model A (Sparse)", model_a), ("Model B (General)", model_b)]:
        print(f"\nVerifying {name}...")
        model.eval()
        lirpa = BoundedModule(model, data)
        
        # Alpha-CROWN
        lirpa.set_bound_opts({'optimize_bound_args': {'iteration': 50, 'lr_alpha': 0.1, 'enable_SDP_crown': False, 'verbosity': 0}})
        lb_alpha, ub_alpha = lirpa.compute_bounds(x=(x_in,), C=c_matrix, method='CROWN-Optimized')
        
        # SDP-CROWN
        lirpa.set_bound_opts({'optimize_bound_args': {'iteration': 50, 'lr_alpha': 0.05, 'lr_lambda': 0.05, 'enable_SDP_crown': True, 'verbosity': 1}})
        lb_sdp, ub_sdp = lirpa.compute_bounds(x=(x_in,), C=c_matrix, method='CROWN-Optimized')
        
        results.append({
            "name": name,
            "alpha": (lb_alpha.item(), ub_alpha.item()),
            "sdp": (lb_sdp.item(), ub_sdp.item())
        })

    # --- FINAL REPORT ---
    print("\n" + "="*95)
    print(f"{'Model':<20} | {'Method':<12} | {'Lower Bound':<15} | {'Upper Bound':<15} | {'Width'}")
    print("="*95)
    for res in results:
        lb_a, ub_a = res['alpha']
        lb_s, ub_s = res['sdp']
        print(f"{res['name']:<20} | Alpha-CROWN  | {lb_a:<15.5f} | {ub_a:<15.5f} | {ub_a-lb_a:.5f}")
        print(f"{'':<20} | SDP-CROWN    | {lb_s:<15.5f} | {ub_s:<15.5f} | {ub_s-lb_s:.5f}")
        print("-" * 95)