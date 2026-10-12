# LAPLEX: The FFT of Learnable Laplace Kernels

🎉 **Accepted at NeurIPS 2026**

Łukasz Struski, Hanna Blazhko, Piotr Kubaty, Jacek Tabor

[![arXiv](https://img.shields.io/badge/arXiv-2605.24584-b31b1b.svg)](https://arxiv.org/abs/2605.24584)

This repository contains the official implementation of **"LAPLEX: The FFT of Learnable Laplace Kernels"**, published at the *Conference on Neural Information Processing Systems (NeurIPS) 2026*.

## 📁 Repository Structure

```
.
├── comparison_experiments/          # Kernel-operator benchmarks: runtime, memory, accuracy, RFF and FFT comparisons
│   ├── laplex.py                    # CUDA-accelerated LAPLEX operators (fp32/fp64)
│   ├── laplex_phased.py             # Phased (single-tau) variant built on top of laplex.py
│   ├── laplex_noCUDA.py             # Portable pure-PyTorch reference implementation (CPU / any backend)
│   ├── prefix_suffix.py             # Custom CUDA prefix/suffix scan primitive (compiled on first import)
│   ├── _common.py                   # Shared benchmarking, memory tracking and plotting utilities
│   ├── exp*.py                      # Individual experiments (see comparison_experiments/README.md)
│   └── run.sh                       # Runs all experiments
│
└── CountSketch_LogisticRegression/  # ImageNet-10 dimensionality reduction / classification experiments
    ├── logistic_regression.py       # Frozen projections (PCA, Gaussian RP, CountSketch, LAPLEX) + logistic regression
    ├── laplex.py                    # LAPLEX projection layer + linear classifier, trained end-to-end
    ├── dense.py                     # Fully learnable dense projection baseline, trained end-to-end
    ├── imagenet10_dataset.py        # ImageNet-10 dataset and data loaders
    ├── utils.py                     # Logging and early-stopping utilities
    └── run_experiments.sh           # Runs 5 independent trials of every method
```

### `comparison_experiments/`

Benchmarks of the LAPLEX kernel operators themselves. The scripts compare LAPLEX against an explicit dense kernel matrix (GPU time, peak memory, batch-size scaling, asymmetric kernels, weighted Gram matrices), against Random Fourier Features (value and gradient accuracy), and against FFT / Toeplitz-FFT (runtime and numerical accuracy on uniform grids). Each script prints a results table and saves figures to `comparison_experiments/figures/`. Details for every script are given in [comparison_experiments/README.md](comparison_experiments/README.md).

### `CountSketch_LogisticRegression/`

Experiments using LAPLEX as a compact projection layer on ImageNet-10. They cover two settings: (i) frozen projections followed by logistic regression, comparing LAPLEX with PCA, Gaussian Random Projection and CountSketch, and (ii) end-to-end training of a LAPLEX projection against a fully learnable dense projection. Results are written to CSV files. Details are given in [CountSketch_LogisticRegression/README.md](CountSketch_LogisticRegression/README.md).

---

## ⚙️ Installation

Create and activate a conda environment:

```bash
ENV_NAME=laplex

conda create -n $ENV_NAME python=3.11 -y
conda activate $ENV_NAME
```

Install PyTorch (choose the wheel matching your CUDA driver, see [pytorch.org](https://pytorch.org/get-started/locally/)) and the remaining dependencies:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install numpy pandas scikit-learn matplotlib tqdm pillow ninja
```

The GPU experiments in `comparison_experiments/` compile custom CUDA kernels with `torch.utils.cpp_extension.load_inline` on first import. This requires `nvcc` and a host C++ compiler. If they are not available system-wide, install them into the environment, using the CUDA version that matches your PyTorch build:

```bash
conda install -c nvidia cuda-toolkit=12.8 -y
conda install -c conda-forge gxx_linux-64 -y
```

The first compilation takes around 30 seconds and is cached afterwards. CPU-only scripts (`exp6_cpu.py`, `exp8_cpu_fft.py`) and everything in `CountSketch_LogisticRegression/` do not need the CUDA toolchain.

---

## 🚀 Running Experiments

**Kernel benchmarks:**

```bash
cd comparison_experiments
bash run.sh
```

Before running, set `TORCH_CUDA_ARCH_LIST` in `run.sh` to match your GPU (e.g. `"8.9"` for RTX 4090, `"9.0"` for H100). Individual experiments can also be run directly, e.g. `python exp1_gpu_time_memory.py`.

**ImageNet-10 experiments:**

Prepare the ImageNet-10 subset with the `train/` and `val/` layout described in [imagenet10_dataset.py](CountSketch_LogisticRegression/imagenet10_dataset.py), replace `path_to_ImageNet` in `run_experiments.sh` with its location, and run:

```bash
cd CountSketch_LogisticRegression
bash run_experiments.sh
```

---

## 📚 Citation

If you find this work useful, please cite our NeurIPS 2026 paper:

```bibtex
@inproceedings{struski2026laplex,
  title     = {LAPLEX: The FFT of Learnable Laplace Kernels},
  author    = {Struski, {\L}ukasz and Blazhko, Hanna and Kubaty, Piotr and Tabor, Jacek},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```
