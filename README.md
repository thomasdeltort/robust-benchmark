This repository provides an intelligent robustness evaluation framework for neural networks, supporting various certification methods including AutoAttack, CRA (Certified Robust Accuracy), Alpha-CROWN, SDP-CROWN, and Hybrid verification.

## ⚙️ 1. Installation and Environment Setup

To run this code, you need to clone the main repository, fetch the required verification sub-repositories, and set up a dedicated Conda environment.

Run the following commands in your terminal:

```bash
# 1. Clone the main benchmark repository
wget [https://anonymous.4open.science/api/repo/robust-benchmark-DAF2/zip](https://anonymous.4open.science/api/repo/robust-benchmark-DAF2/zip) -O robust-benchmark.zip && unzip robust-benchmark.zip -d robust-benchmark && rm robust-benchmark.zip
cd robust-benchmark
# 2. Download and extract required sub-repositories anonymously
wget [https://anonymous.4open.science/api/repo/SDP-CROWN-9D8C/zip](https://anonymous.4open.science/api/repo/SDP-CROWN-9D8C/zip) -O SDP-CROWN.zip && unzip SDP-CROWN.zip -d SDP-CROWN && rm SDP-CROWN.zip
wget [https://anonymous.4open.science/api/repo/alpha-beta-CROWN-4EBF/zip](https://anonymous.4open.science/api/repo/alpha-beta-CROWN-4EBF/zip) -O alpha-beta-CROWN.zip && unzip alpha-beta-CROWN.zip -d alpha-beta-CROWN && rm alpha-beta-CROWN.zip

# 3. Create and activate the conda environment
cd alpha-beta-CROWN
conda create -n lirpa_env python=3.10.12 -y
conda activate lirpa_env

# 4. Install dependencies for alpha-beta-crown, then for the main benchmark
pip install -r complete_verifier/requirements.txt
cd ..
pip install -r requirements.txt

```

> **Note on Imagenette Experiments:** To run the Imagenette experiments, you must add the datasets and pretrained models provided in the `.zip` archive from the supplementary materials. Please extract the archive into your repository and ensure the Imagenette model weights (`.pth`) are placed directly inside the `models/` directory before running the evaluation scripts.

## 🚀 2. Tutorial: Evaluating a Single Epsilon

You can evaluate your models for a specific, single epsilon using either the command line or a Python script. By default, the script performs systematic "paving" across multiple points. To force the script to evaluate **only a single epsilon value**, set `--num_points 1` and define your target epsilon using `--epsilon_max`.

The pipeline supports multiple robustness verification methods. By default, it runs all of them to provide a comprehensive evaluation:

```json
{
  "aa": True, 
  "cra": True, 
  "cra_pi": True, 
  "alphacrown": True, 
  "heavy_certified": True,
  "hybrid": False 
}

```

Run the evaluation command on a single line:

```bash
python main_auto.py --dataset cifar10 --model CNNA_CIFAR10_1_LIP_Bjork --model_path ./models/vanilla_CNNA_CIFAR10_1_LIP_Bjork_cifar10_tau_a250.0_T1.0_bs256_lr0.0003_1776931197_acc0.64.pth --num_points 1 --epsilon_max 0.03 --output_csv results/single_eps_study.csv --solvers_config "{'aa': False, 'cra': False, 'cra_pi': False, 'heavy_certified': False}"

```

**Key Evaluation Flags:**

* **Groupsort Reformulation:** To compare the conventional method with our groupsort2 reformulation, append `--use_conventional_groupsort` to your command, or run the provided shell script: `./compare_reformulations_paper.sh`
* **SDP-CROWN Auto-Tuning:** The script now automatically runs SDP verification for both `high_tau=False` and `high_tau=True` internally and registers the best certified accuracy.

---

## 🏋️ 3. Tutorial: Training Hybrid Models

The repository includes a highly modular training script (`train_hybrid_complete.py`) to build and train hybrid Lipschitz architectures (e.g., a constrained prefix with an unconstrained suffix).

### Setting the Split Index

As described in the paper, the optimal hybrid configuration leaves the last 2 linear layers completely unconstrained (`--b2_type unconstrained`). Because the training script calculates the split based on absolute layer depth, the `--split_idx` varies by architecture:

* **ConvLarge:** `--split_idx 5` (7 total layers)
* **VGG13 (CIFAR-10):** `--split_idx 10` (12 total layers)
* **VGG13 (Imagenette):** `--split_idx 11` (13 total layers)
* **VGG16:** `--split_idx 14` (16 total layers)

### Example: Biphase Hybrid Training

To train a VGG13 model on CIFAR-10 where the prefix is constrained (AOC + GroupSort) and the suffix (the last 2 layers) is unconstrained (Standard Linear + ReLU), run the following single-line command:

```bash
python train_hybrid_complete.py --dataset cifar10 --arch VGG13 --split_idx 10 --b1_type aoc --b1_lip 1.0 --b1_act GroupSort --b2_type unconstrained --b2_act ReLU --train_mode conventional --use_bn --eval_benchmark

```

**Key Training Flags:**

* `--train_mode`: Choose between `conventional` (joint update), `biphase` (alternating head/backbone updates), or `head_only` (frozen prefix).
* `--split_idx`: The absolute layer index where the model splits between Block 1 (prefix) and Block 2 (suffix).
* `--b1_type` / `--b2_type`: Choose the layer constraints (`aoc`, `torchlip`, `spectral`, or `unconstrained`).
* `--use_bn`: Safely integrates `BatchNorm1d` into the unconstrained linear layers (automatically fused during verification).
* `--eval_benchmark`: Automatically runs Formal Verification (CRA and SDP-CROWN VRA) on a 200-point benchmark set during the validation step to track robustness during training.
EOF

```

```
