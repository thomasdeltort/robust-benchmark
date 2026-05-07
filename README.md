# Robust Benchmark: Setup & Quickstart

This repository provides an intelligent robustness evaluation framework for neural networks, supporting various certification methods including AutoAttack, CRA (Certified Robust Accuracy), Alpha-CROWN, SDP-CROWN, and Hybrid verification.

## ⚙️ 1. Installation and Environment Setup

To run this code, you need to clone the main repository, fetch the required verification sub-repositories, and set up a dedicated Conda environment. 

Run the following commands in your terminal:

```bash
# 1. Clone the main benchmark repository
git clone https://github.com/thomasdeltort/robust-benchmark.git
cd robust-benchmark

# 2. Clone the required verification solvers into the project root
git clone https://github.com/thomasdeltort/SDP-CROWN.git
git clone https://github.com/thomasdeltort/alpha-beta-CROWN.git

# 3. Create and activate the conda environment
cd alpha-beta-crown
conda create -n lirpa_env python=3.10.12 -y
conda activate lirpa_env

# 4. Install dependencies for alpha-beta-crown, then for the main benchmark
pip install -r complete_verifier/requirements.txt
cd ..
pip install -r requirements.txt
```

## 🚀 Tutorial: Evaluating a Single Epsilon

You can evaluate your models for a specific, single epsilon using either the command line or a Python script.

### Option 1: Command Line Interface (CLI)

By default, the script performs systematic "paving" across multiple points. To force the script to evaluate **only a single epsilon value**, set `--num_points 1` and define your target epsilon using `--epsilon_max`.

The pipeline supports multiple robustness verification methods. By default, the script runs almost all of them to provide a comprehensive evaluation. The default configuration looks like this:

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
Run the command like this : 
```bash
python main.py \
  --dataset cifar10 \
  --model CNNA_CIFAR10_1_LIP_Bjork \
  --model_path ./models/models/vanilla_CNNA_CIFAR10_1_LIP_Bjork_cifar10_tau_a250.0_T1.0_bs256_lr0.0003_1776931197_acc0.64.pth \
  --norm 2 \
  --num_points 1 \
  --epsilon_max 0.03 \
  --output_csv results/single_eps_study.csv
  --solvers_config "{'aa': False, 'cra': False, 'cra_pi': False, 'heavy_certified': False}"
```


