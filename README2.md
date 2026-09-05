It is almost perfect! You just accidentally included my conversational filler text in the middle, and there are a couple of nested code block artifacts at the very end. I also updated the script names to match your exact filenames (`train_hybrid_complete.py`).

Here is the fully cleaned, combined version. You can copy this entire block and paste it directly into your `README.md`:

```markdown
# Robust Hybrid Lipschitz Networks

This repository contains the training pipeline for **Hybrid 1-Lipschitz Neural Networks**, designed for certifiable adversarial robustness. It features a modular architecture builder, specialized robust loss functions, and a dual-optimizer Schedule-Free learning setup.

## Key Features

* **Modular Bipartite Architecture:** Split any standard architecture (VGG13, VGG16, ConvLarge) into a robust **Prefix** (Block 1) and an accuracy-focused **Suffix** (Block 2).
* **Multiple Constraint Types:** Support for `aoc` (Adaptive Orthogonal Convolutions), `torchlip` (Spectral Normalization), and `unconstrained` layers.
* **Advanced Training Modes:** Train jointly (`conventional`) or use alternating gradient updates (`biphase`).
* **Cross-Constraint Weight Loading:** Seamlessly load unconstrained (vanilla) pretrained weights into strictly constrained layers (AOC/Torchlip) without breaking mathematical bounds.
* **Automatic CRA Tracking:** Automatically computes Certified Robust Accuracy (CRA) during validation if the network is fully Lipschitz-bounded.

---

## 🚀 Quick Start Examples

### 1. Conventional Training (Fully Constrained)

Train a fully robust VGG13 network where the first 4 layers are AOC and the rest are Torchlip, both strictly 1-Lipschitz.

```bash
python train_hybrid_complete.py \
    --arch VGG13 \
    --split_idx 4 \
    --train_mode conventional \
    --b1_type aoc --b1_lip 1.0 --b1_act GroupSort \
    --b2_type torchlip --b2_lip 1.0 --b2_act GroupSort \
    --criterion tau --temperature 1.0 \
    --epochs 50 --lr 3e-4

```

### 2. Biphase Training (Hybrid Network)

Train a network with a robust 1-Lipschitz AOC backbone and an *unconstrained* head. The script uses two alternating optimizers to prevent momentum leakage, applying robust loss (Tau) to the backbone and standard Cross-Entropy (CE) to the head.

```bash
python train_hybrid_complete.py \
    --arch ConvLarge \
    --split_idx 4 \
    --train_mode biphase \
    --b1_type aoc --b1_lip 1.0 \
    --b2_type unconstrained \
    --criterion tau --temperature 0.5 \
    --b2_criterion CE \
    --wd 1e-2

```

### 3. Loading Pretrained Weights (Cross-Constraint)

You can load weights from a `vanilla_*.pth` export and map them into a new constraint geometry. For example, loading an unconstrained pretrained backbone into a new AOC architecture:

```bash
python train_hybrid_complete.py \
    --arch ConvLarge \
    --b1_type aoc \
    --pretrained_weights ./saved_models/vanilla_pretrained_model.pth

```

> **Note:** The script will automatically detect the vanilla weights and securely project them into the new constraint latent parameters.

---

## ⚙️ Core Arguments

### Architecture & Constraints

| Argument | Description | Default |
| --- | --- | --- |
| `--arch` | Architecture template (`VGG13`, `VGG16`, `ConvLarge`) | `VGG13` |
| `--split_idx` | Layer index where Block 1 ends and Block 2 begins | `4` |
| `--b1_type`, `--b2_type` | Constraint for Block 1 / Block 2 (`aoc`, `torchlip`, `unconstrained`) | `aoc` / `unconstrained` |
| `--b1_lip`, `--b2_lip` | Lipschitz constant for the respective block | `1.0` / `1.0` |
| `--b1_act`, `--b2_act` | Activation functions (`GroupSort`, `ReLU`) | `GroupSort` / `ReLU` |

### Training Logic

| Argument | Description | Default |
| --- | --- | --- |
| `--train_mode` | `conventional` (joint updates) or `biphase` (alternating) | `conventional` |
| `--criterion` | Main loss function (`tau`, `HKR`, `CE`) | `tau` |
| `--b2_criterion` | Head loss function (Only used in `biphase` mode) | `CE` |
| `--temperature` | Temperature scale for the main criterion | `1.0` |
| `--wd` | Weight decay (Applied **only** to Block 2. Block 1 is always 0.0) | `0.0` |

### I/O & Pretraining

| Argument | Description |
| --- | --- |
| `--pretrained_weights` | Path to a `.pth` file to initialize the network. |
| `--model_class` | Override the dynamic builder to use a hardcoded class from `models.py`. |
| `--save_dir` | Output directory for the trained checkpoints. |

---

## 💾 Saved Outputs

At the end of training, the script automatically saves **two** files to your `--save_dir`:

1. **`Hyb_[Config]_[Timestamp].pth`**:
The fully parameterized model containing the mathematical constraint variables (e.g., Björck orthonormalization parameters). It also contains a `config` dictionary with all your CLI arguments. **Pass this file to your evaluation/verification scripts.**
2. **`vanilla_Hyb_[Config]_[Timestamp].pth`**:
A lightweight, stripped-down export where the mathematical bounds are baked into standard PyTorch `nn.Conv2d` and `nn.Linear` layers. Use this for extremely fast inference deployment or cross-constraint transfer learning.

---

## 🛡️ Verification & Evaluation (`main_auto_complete.py`)

The evaluation script is designed to be completely **plug-and-play**. Because `train_hybrid_complete.py` saves the entire training configuration inside the `.pth` file, you only need to provide the path to the model.

The script will automatically:

1. Rebuild the exact architecture and constraint types.
2. Compute **Clean Accuracy**.
3. Compute analytical **Certified Robust Accuracy (CRA)** (Global or LLN) if the bounds allow it.
4. Run empirical **AutoAttack** (ERA) to filter out vulnerable points.
5. Run **Hybrid Formal Verification (VRA)** on the surviving points using SDP-CROWN or Alpha-CROWN.
6. Append a formatted row with all metrics to `verification_results_updated.csv`.

### 1. Standard Automated Run (Easiest)

Simply point the script to your trained model. It works with both the standard and `vanilla_` exports.

```bash
python main_auto_complete.py --model_path ./saved_models/vanilla_Hyb_conventional_tauT1.0_aocL1.0_torchlipL5.0_split4_1725443513.pth

```

### 2. Fast Verification (Skip AutoAttack)

AutoAttack can take a long time to run. If you only care about the formal verification metrics (VRA) and want to skip the empirical attack, use the `--no_aa` flag:

```bash
python main_auto_complete.py --model_path ./saved_models/your_model.pth --no_aa

```

### 3. Customizing the Threat Model

By default, the script verifies against an $L_2$ perturbation of $\epsilon = 0.141$. You can easily change the radius, the norm, and the verification backend.

```bash
python main_auto_complete.py \
    --model_path ./saved_models/your_model.pth \
    --eps 0.03137 \
    --norm inf \
    --hybrid_backend alphacrown

```

### 📊 Verification Arguments

| Argument | Description | Default |
| --- | --- | --- |
| `--model_path` | **(Required)** Path to the `.pth` or `vanilla_*.pth` checkpoint. | `None` |
| `--eps` | The adversarial perturbation radius to verify against. | `0.141` |
| `--norm` | The threat model norm space (`2` or `inf`). | `2` |
| `--hybrid_backend` | The verification backend for the unconstrained suffix (`sdp`, `alphacrown`). | `sdp` |
| `--no_aa` | Skips AutoAttack and verifies all cleanly correctly classified images. | `False` |
| `--run_full_sdp` | Ignores the hybrid split and attempts to run SDP-CROWN on the entire network at once (Use only for `ConvLarge`). | `False` |

### 📁 Output format

All results are dynamically appended to **`verification_results_updated.csv`** in your root directory. The CSV tracks:

`Model` | `Dataset` | `Split_Idx` | `B1/B2 Configs` | `Epsilon` | `Clean_Acc` | `ERA` | `Global_CRA` | `LLN_CRA` | `VRA` | `Time_s`

```

```