import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from deel import torchlip
import argparse
import os
import sys
import boto3
import io
import time
import copy
from torch.utils.tensorboard import SummaryWriter 
from torch.nn.utils.parametrize import is_parametrized

from torchvision import datasets, transforms

# --- Handle module imports ---
sys.path.append('./..')
try:
    from models import *
    from project_utils import *
except ImportError:
    print("Warning: 'models' or 'project_utils' not found. Ensure you are in the correct directory.")

from deel.torchlip import TauCrossEntropyLoss

# ==========================================
# 1. Helper Function: Sync Logs to AWS S3
# ==========================================
def upload_folder_to_s3(local_folder, bucket_name, s3_prefix):
    """
    Recursively uploads the TensorBoard log folder to S3.
    """
    s3 = boto3.client('s3')
    # Walk through the local directory
    for root, dirs, files in os.walk(local_folder):
        for filename in files:
            local_path = os.path.join(root, filename)
            
            # Create the relative path for S3 key (preserve folder structure)
            relative_path = os.path.relpath(local_path, local_folder)
            s3_key = os.path.join(s3_prefix, relative_path).replace("\\", "/") 
            
            try:
                s3.upload_file(local_path, bucket_name, s3_key)
            except Exception as e:
                # Catch upload errors (e.g., network blip) so training doesn't crash
                print(f"Warning: Failed to upload {filename} to S3: {e}")

# ==========================================
# 2. Helper Function: CRA Computation
# ==========================================
def compute_certificates_CRA(images, model, epsilon, correct_indices, norm='2', L=1, return_robust_points=False):
    """
    Computes certificates, CRA, and the time taken for the core computation.
    """
    if not isinstance(correct_indices, torch.Tensor):
        correct_indices = torch.tensor(correct_indices)
        
    correct_images = images[correct_indices]
    total_num_images = correct_images.shape[0]

    if len(correct_images) == 0:
        empty_certs = torch.tensor([])
        if return_robust_points:
            return empty_certs, 0.0, 0.0, torch.tensor([])
        return empty_certs, 0.0, 0.0

    device = next(model.parameters()).device 
    
    # --- Timing ---
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start_time = time.time()

    with torch.no_grad():
        values, _ = torch.topk(model(correct_images.to(device)), k=2)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    end_time = time.time()
    elapsed_time = end_time - start_time
    # --- End Timing ---

    if norm == '2':
        scale_certificate = np.sqrt(2)
    elif norm == 'inf':
        scale_certificate = 2.0
    else:
        raise ValueError(f"Unsupported norm: '{norm}'. Please use '2' or 'inf'.")

    # Margin / (Scale * Lipschitz)
    certificates = (values[:, 0] - values[:, 1]) / (scale_certificate * L)
    
    is_robust_mask = (certificates >= epsilon).cpu()
    num_robust_points = torch.sum(is_robust_mask).item()
    cra = (num_robust_points / total_num_images) * 100.0
    time_per_img = elapsed_time / len(correct_indices)

    if return_robust_points:
        robust_indices = correct_indices[is_robust_mask]
        return certificates.cpu(), cra, time_per_img, robust_indices
    
    return certificates.cpu(), cra, time_per_img

# ==========================================
# 3. Helper Class: HKR Loss
# ==========================================
class HKRMultiLossLSE(torch.nn.Module):
    def __init__(self, alpha: float = 1., temperature: float = 1., penalty = 1., margin = 1):
        super().__init__()
        self.penalty = penalty
        self.alpha = alpha
        self.temperature = temperature
        self.margin = margin

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_pred = y_pred * self.temperature
        
        # Positive examples
        pos = y_pred[y_true == 1]
        hinge_pos = torch.mean(F.relu(self.margin -  pos))
        kr_pos = torch.mean(pos)

        # Negative examples
        neg = torch.where(y_true == 1, -float('inf'),  y_pred)
        nb_bins = y_pred.new_tensor(y_pred.size(1) - 1)
        nb_bins = torch.log(nb_bins)
        t = nb_bins/ (self.margin*self.penalty)
        neg_soft = 1/t*torch.logsumexp(t*neg, dim = 1)
        
        hinge_neg = torch.mean(F.relu(self.margin +  neg_soft))
        kr_neg = torch.mean(neg_soft)
      
        hinge_loss = hinge_pos + hinge_neg
        kr = kr_pos  - kr_neg
        loss_val = (1-1./self.alpha)*hinge_loss -1./self.alpha* kr
        return loss_val

# ==========================================
# 4. Helper Function: Vanilla Export
# ==========================================
def vanilla_export(model1):
    """
    Takes a parametrized model (e.g. Deel-Lip), strips parametrizations,
    and returns a standard PyTorch model with the computed weights.
    """
    model1.eval()
    model2 = copy.deepcopy(model1)
    model2.eval()
    dict_modified_layers = {}
    
    for (n1,p1), (n2,p2) in zip(model1.named_modules(), model2.named_modules()):
        assert n1 == n2
        # Handle Conv2d (e.g. SpectralConv2d)
        if isinstance(p1, torch.nn.Conv2d) and is_parametrized(p1):
            new_conv = torch.nn.Conv2d(
                p1.in_channels, p1.out_channels, 
                kernel_size=p1.kernel_size, stride=p1.stride, 
                padding=p1.padding, padding_mode=p1.padding_mode,
                bias=(p1.bias is not None)
            )
            new_conv.weight.data = p1.weight.data.clone()
            new_conv.bias.data = p1.bias.data.clone() if p1.bias is not None else None
            dict_modified_layers[n2] = new_conv
            
        # Handle Linear (e.g. SpectralLinear)
        if isinstance(p1, torch.nn.Linear) and is_parametrized(p1):
            new_lin = torch.nn.Linear(
                p1.in_features, p1.out_features, 
                bias=(p1.bias is not None)
            )
            new_lin.weight.data = p1.weight.data.clone()
            new_lin.bias.data = p1.bias.data.clone() if p1.bias is not None else None
            dict_modified_layers[n2] = new_lin
            
    for n2, new_layer in dict_modified_layers.items():
        split_hierarchy = n2.split('.')
        lay = model2
        for h in split_hierarchy[:-1]:
            lay = getattr(lay, h)
        setattr(lay, split_hierarchy[-1], new_layer)
        
    return model2

   
# ==========================================
# 5. Main Training Function (Updated)
# ==========================================
def main(args):
    # --- A. Setup Paths & Logging ---
    
    run_id = (
        f"{args.dataset}_"
        f"{args.criterion}_a{args.alpha}_"
        f"T{args.temperature}_bs{args.batch_size}_"
        f"lr{args.lr}_eps{args.epsilon}_"
        f"{args.aug_level}_"
        f"{int(time.time())}"
    )
    
    local_log_dir = os.path.join("runs", args.model_name, run_id)
    
    # AWS S3 Configuration
    BUCKET_NAME = "tdrobustbucket"
    S3_LOG_PREFIX = f"tensorboard_logs/{args.model_name}/{run_id}"
    
    print(f"--- Experiment ID: {run_id} ---")
    print(f"--- Augmentation Level: {args.aug_level} ---")
    print(f"--- Local Logs: {local_log_dir} ---")
    print(f"--- S3 Logs: s3://{BUCKET_NAME}/{S3_LOG_PREFIX} ---")

    writer = SummaryWriter(log_dir=local_log_dir)

    # --- B. Load Dataset & Model ---
    train_loader, test_loader = load_dataset(
        args.dataset, 
        args.batch_size, 
        aug_level=args.aug_level
    )

    model_zoo = {
        "MLP_MNIST_1_LIP": MLP_MNIST_1_LIP if 'MLP_MNIST_1_LIP' in globals() else None,
        "CNNA_CIFAR10_1_LIP": CNNA_CIFAR10_1_LIP if 'CNNA_CIFAR10_1_LIP' in globals() else None,
    }
    
    if args.model_name not in model_zoo:
        try:
             ModelClass = globals()[args.model_name]
        except KeyError:
             raise ValueError(f"Model '{args.model_name}' not found.")
    else:
        ModelClass = model_zoo[args.model_name]

    model = ModelClass()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    # --- C. Optimizer & Loss ---
    optimizer = torch.optim.Adam(lr=args.lr, params=model.parameters())
    
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #     optimizer, milestones=[args.epochs//2, 2*args.epochs//3], gamma=0.2
    # )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    if args.criterion == 'HKR':
        criterion = HKRMultiLossLSE(alpha=args.alpha, temperature=args.temperature, penalty=0.5, margin=1.0)
    elif args.criterion == 'tau':
        criterion = TauCrossEntropyLoss(tau=args.temperature)
    else:
        raise ValueError(f"Criterion {args.criterion} not recognized.")
        
    kr_multiclass_loss = torchlip.KRMulticlassLoss()
    num_classes = 10 

    # --- D. Training Loop ---
    train_acc, val_acc = 0.0, 0.0
    global_cra = 0.0

    for epoch in range(args.epochs):
        model.train() 
        m_kr, m_acc, total_loss = 0, 0, 0

        for step, (data, target) in enumerate(train_loader):
            target_oh = torch.nn.functional.one_hot(target, num_classes=num_classes)
            data, target_oh = data.to(device), target_oh.to(device)

            optimizer.zero_grad()
            # import pdb;pdb.set_trace()
            output = model(data)
            loss = criterion(output, target_oh)
            loss.backward()
            optimizer.step()

            m_kr += kr_multiclass_loss(output, target_oh).item()
            m_acc += (output.argmax(dim=1) == target_oh.argmax(dim=1)).float().mean().item()
            total_loss += loss.item()

        # Epoch Metrics
        train_acc = m_acc / len(train_loader)
        avg_loss = total_loss / len(train_loader)
        avg_kr = m_kr / len(train_loader)
        current_lr = scheduler.get_last_lr()[0]
        
        # Log Training to TensorBoard
        writer.add_scalar("Train/Loss", avg_loss, epoch)
        writer.add_scalar("Train/Accuracy", train_acc, epoch)
        writer.add_scalar("Train/KR_Term", avg_kr, epoch)
        writer.add_scalar("Train/LR", current_lr, epoch)
        
        scheduler.step()

        # --- E. Validation & CRA ---
        val_metrics_str = ""
        
        if (epoch + 1) % 10 == 0 or (epoch + 1) == args.epochs:
            model.eval() 
            test_output, test_targets = [], []
            test_images = [] 

            with torch.no_grad():
                for data, target in test_loader:
                    data_device = data.to(device)
                    test_images.append(data.cpu()) 
                    
                    out = model(data_device)
                    test_output.append(out.cpu())
                    test_targets.append(torch.nn.functional.one_hot(target, num_classes=num_classes).cpu())
            
            all_test_images = torch.cat(test_images)
            all_test_output = torch.cat(test_output)
            all_test_targets = torch.cat(test_targets)
            
            preds = all_test_output.argmax(dim=1)
            truth = all_test_targets.argmax(dim=1)
            val_acc = (preds == truth).float().mean().item()
            val_loss = criterion(all_test_output, all_test_targets).item()
            
            # Calculate CRA
            correct_indices = (preds == truth).nonzero(as_tuple=False).squeeze()
            _, cra_percentage, cra_time = compute_certificates_CRA(
                images=all_test_images, model=model, epsilon=args.epsilon,
                correct_indices=correct_indices, norm=args.cra_norm, L=1.0 
            )
            global_cra = (len(correct_indices) / len(all_test_targets)) * (cra_percentage / 100.0)

            # Log Validation to TensorBoard
            writer.add_scalar("Val/Loss", val_loss, epoch)
            writer.add_scalar("Val/Accuracy", val_acc, epoch)
            writer.add_scalar("Val/CRA", global_cra * 100, epoch)
            
            val_metrics_str = f"| Val Acc {val_acc:.3f} | CRA {global_cra*100:.2f}%"

            # --- SYNC: Upload Logs to S3 ---
            upload_folder_to_s3(local_log_dir, BUCKET_NAME, S3_LOG_PREFIX)

        print(f"Epoch {epoch+1}/{args.epochs}: Train Acc {train_acc:.3f} | Loss {avg_loss:.4f} | LR {current_lr:.2e} {val_metrics_str}")

    # --- F. Final Experiment Logging ---
    writer.add_hparams(
        hparam_dict={
            "model": args.model_name,
            "lr": args.lr,
            "bs": args.batch_size,
            "alpha": args.alpha,
            "temp": args.temperature,
            "epsilon": args.epsilon,
            "criterion": args.criterion,
            "aug_level": args.aug_level 
        },
        metric_dict={
            "hparam/accuracy": val_acc,
            "hparam/cra": global_cra * 100,
            "hparam/train_acc": train_acc
        }
    )
    writer.close()

    # =========================================================================
    # --- G. Save Models Locally (./models) AND S3 ---
    # =========================================================================
    
    # 1. Setup Local Directory
    LOCAL_MODEL_DIR = "./models"
    os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
    
    S3_MODEL_FOLDER = "lip_models"
    base_filename = f"{args.model_name}_{run_id}_acc{val_acc:.2f}"
    
    s3 = boto3.client('s3')

    # -----------------------------------------------------------------
    # 2. SAVE TORCHLIP MODEL (Local -> S3)
    # -----------------------------------------------------------------
    full_model_local_path = os.path.join(LOCAL_MODEL_DIR, f"{base_filename}.pth")
    full_model_s3_key = f"{S3_MODEL_FOLDER}/{base_filename}.pth"

    print(f"\n[1/2] Saving Original (Lip) model locally: {full_model_local_path}")
    torch.save(model.state_dict(), full_model_local_path)
    
    print(f"      Uploading to s3://{BUCKET_NAME}/{full_model_s3_key}...")
    try:
        s3.upload_file(full_model_local_path, BUCKET_NAME, full_model_s3_key)
    except Exception as e:
        print(f"      Warning: S3 upload failed: {e}")
    
    # -----------------------------------------------------------------
    # 3. SAVE VANILLA MODEL (Local -> S3)
    # -----------------------------------------------------------------
    print(f"[2/2] Converting to Vanilla model...")
    model.cpu() 
    vanilla_model = vanilla_export(model)
    
    vanilla_filename = f"vanilla_{base_filename}.pth"
    vanilla_local_path = os.path.join(LOCAL_MODEL_DIR, vanilla_filename)
    vanilla_s3_key = f"{S3_MODEL_FOLDER}/{vanilla_filename}"
    
    print(f"      Saving Vanilla model locally: {vanilla_local_path}")
    torch.save(vanilla_model.state_dict(), vanilla_local_path)
    
    print(f"      Uploading to s3://{BUCKET_NAME}/{vanilla_s3_key}...")
    try:
        s3.upload_file(vanilla_local_path, BUCKET_NAME, vanilla_s3_key)
    except Exception as e:
        print(f"      Warning: S3 upload failed: {e}")
    
    # Final Log Sync
    print("Final log sync to S3...")
    upload_folder_to_s3(local_log_dir, BUCKET_NAME, S3_LOG_PREFIX)
    print("Experiment Complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='CNNA_CIFAR10_1_LIP')
    parser.add_argument('--temperature', type=float, default=200.0)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--criterion', type=str, default='HKR')
    parser.add_argument('--alpha', type=float, default=250.0)
    parser.add_argument('--epsilon', type=float, default=0.01)
    parser.add_argument('--cra_norm', type=str, default='2')
    
    parser.add_argument('--aug_level', type=str, default='medium', 
                        help='Augmentation strength: none, light, medium, or heavy')

    args = parser.parse_args()
    main(args)