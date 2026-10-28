# LAPLEX: The FFT of Learnable Laplace Kernels

🎉 **Accepted at NeurIPS 2026**

Łukasz Struski, Hanna Blazhko, Piotr Kubaty, Jacek Tabor

[![arXiv](https://img.shields.io/badge/arXiv-2605.24584-b31b1b.svg)](https://arxiv.org/abs/2605.24584)

This repository contains the official implementation of **"LAPLEX: The FFT of Learnable Laplace Kernels"**, published at the *Conference on Neural Information Processing Systems (NeurIPS) 2026*.

## 📁 Repository Structure

The experiments are grouped by what LAPLEX is used for. Each directory has its own README.

```
.
├── benchmarks/                          # Kernel operators: runtime, memory, accuracy; FFT and RFF comparisons
│   ├── comparison_experiments/          # Released LAPLEX operators and the benchmark scripts of the submitted version
│   │   ├── laplex.py                    # CUDA-accelerated LAPLEX operators (fp32/fp64)
│   │   ├── laplex_phased.py             # Phased (single-tau) variant built on top of laplex.py
│   │   ├── laplex_noCUDA.py             # Portable pure-PyTorch reference implementation (CPU / any backend)
│   │   ├── prefix_suffix.py             # Custom CUDA prefix/suffix scan primitive (compiled on first import)
│   │   ├── _common.py, exp*.py, run.sh  # Original benchmark scripts
│   ├── figNN_*.py, table03_*.py         # One script per figure / table of the camera-ready version
│   ├── common.py, impls.py, plotting.py # Measurement protocol, implementations, figure style
│   ├── second_order/                    # App. K: Fisher scoring with the exact LAPLEX curvature (Tabs. 10–13, Fig. 22)
│   ├── impl_compare.py                  # All LAPLEX implementations in the Fig. 1 setting
│   ├── last_interval_test.py            # Anchors beyond the ends of the sorted anchors (Thm. 1/4, Thm. 6)
│   ├── vendor/                          # Further LAPLEX implementations (incl. the fused kernel of Fig. 1)
│   ├── run_all.sh, submit_max_n.sh      # Run every figure locally / as Slurm jobs
│   └── sif.sh, sbatch/                  # Singularity wrapper, Slurm jobs
│
├── nn_training/                         # LAPLEX as a trainable layer
│   ├── CountSketch_LogisticRegression/  # ImageNet-10: frozen projections + logistic regression, end-to-end LAPLEX vs dense
│   ├── classification_heads/            # ImageNet-1k: LAPLEX vs low-rank classification heads on frozen Swin-T / MaxViT-T (Sec. 7)
│   ├── crossing_trace.py, plot_crossing.py  # Anchor-crossing diagnostic with per-step traces (App. M, Fig. 23)
│   ├── parity_stats.py                  # Statistics of a fresh 5-seed LAPLEX vs dense run (Table 1, App. G)
│   ├── table15_anchor_ablations.py      # Anchor initialization and number of components (App. M, Tab. 15)
│   ├── appJ_attention*.py               # LAPLEX as an attention operator (App. J)
│   ├── run_appJ_table15.sh              # The sweeps of App. J and Tab. 15
│   ├── make_imagenet10.py               # ImageNet-10 (or ImageNet-1k) as symlinks into a full ImageNet
│   └── sif.sh, sbatch/                  # Singularity wrapper, Slurm jobs
│
├── log_likelihood/                      # Density models: structured covariance baselines (App. I.2 Tab. 9, App. L Tab. 14)
│
└── environment/                         # Singularity definition, conda environment, pinned package versions
```

### `benchmarks/`

Benchmarks of the LAPLEX kernel operators themselves: LAPLEX against an explicit dense kernel matrix (GPU time, peak memory, batch-size scaling, asymmetric kernels, weighted Gram matrices, CPU), against Random Fourier Features (value and gradient accuracy), and against FFT / Toeplitz-FFT (runtime and numerical accuracy on uniform grids). The `figNN_*.py` scripts produce the figures of the camera-ready version (Figs. 1, 3, 5, 6, 8–18, Tab. 3); every method runs until it no longer fits in device memory. `second_order/` uses the exact weighted Gram as the curvature of a second-order optimizer (App. K). Details in [benchmarks/README.md](benchmarks/README.md); the original scripts are described in [benchmarks/comparison_experiments/README.md](benchmarks/comparison_experiments/README.md).

### `nn_training/`

LAPLEX as a compact, trainable layer. On ImageNet-10 it covers (i) frozen projections followed by logistic regression, comparing LAPLEX with PCA, Gaussian Random Projection and CountSketch, (ii) end-to-end training of a LAPLEX projection against a fully learnable dense projection, and (iii) the anchor-crossing diagnostic and a fresh-seed re-run of (ii). On ImageNet-1k, (iv) LAPLEX replaces the classification head of a frozen pretrained Swin-T / MaxViT-T and is compared with a low-rank head of the same size. It also contains (v) LAPLEX as a drop-in attention operator on MNIST / SVHN / CIFAR-10 (App. J) and (vi) the anchor-initialization and component ablations (App. M). Details in [nn_training/README.md](nn_training/README.md), [nn_training/CountSketch_LogisticRegression/README.md](nn_training/CountSketch_LogisticRegression/README.md) and [nn_training/classification_heads/README.md](nn_training/classification_heads/README.md).

### `log_likelihood/`

Density modelling on ImageNette-320 and DIV2K-1024: LAPLEX factor analysis against dense low-rank, PPCA, Kronecker, FFT-circulant and budget-matched circulant covariances, in raster and permuted pixel order, with MAP denoising (App. I.2, App. L). The code of the main density experiments (Sec. 4.2, App. I.1) is not in this repository yet. Details in [log_likelihood/README.md](log_likelihood/README.md).

---

## ⚙️ Installation

[`environment/`](environment/) describes the software of the paper's runs in two forms. Both install the same Python packages at the same versions ([requirements.txt](environment/requirements.txt): Python 3.14.6, PyTorch 2.13.0, NumPy 2.4.4, SciPy 1.18.0, ...); they differ in the CUDA version.

**Singularity / Apptainer: the paper's environment.** [laplex.def](environment/laplex.def) rebuilds, with the same versions, the part of the image used for the paper's runs that this repository needs: Ubuntu 24.04, CUDA 13.2 with `nvcc` and gcc 13, TeX Live, and the packages above with PyTorch for CUDA 13.2. CUDA 13 needs an NVIDIA driver ≥ 580.

```bash
cd environment
apptainer build laplex.sif laplex.def      # or: singularity build --fakeroot laplex.sif laplex.def
export LAPLEX_SIF=$PWD/laplex.sif
```

Building needs root, `--fakeroot` (an entry for your user in `/etc/subuid`), or Apptainer with unprivileged user namespaces; on a cluster without these, build on a workstation and copy the `.sif`. The `sif.sh` wrappers in `benchmarks/` and `nn_training/` run every command in the image named by `LAPLEX_SIF`, e.g. `bash sif.sh python fig03_08_square.py --quick`.

**Conda.** [environment.yml](environment/environment.yml) installs the same packages with PyTorch built for CUDA 12.6 (NVIDIA driver ≥ 525), plus `nvcc` and gcc from conda-forge for the CUDA kernels:

```bash
conda env create -f environment/environment.yml
conda activate laplex
```

For RTX 50xx GPUs, which need CUDA 13, replace `cu126` by `cu130` and `cuda-nvcc=12.6`, `cuda-cudart-dev=12.6` by `13.0` in `environment.yml` (driver ≥ 580). The `sif.sh` wrappers run commands in the active environment when no image is found.

The GPU benchmarks compile custom CUDA kernels with `torch.utils.cpp_extension.load_inline` on first import, from 30 seconds to a few minutes per implementation; the result is cached in `benchmarks/.cache/`. The CPU benchmarks and everything in `nn_training/` do not need the CUDA toolchain; `benchmarks/second_order/` and `log_likelihood/` use the CUDA operators of `benchmarks/comparison_experiments/`.

**LaTeX (optional, included in the image).** With `pdflatex`, `pgf` and the Times/Helvetica fonts (e.g. Ubuntu: `apt install texlive-latex-extra texlive-fonts-recommended cm-super`), the figures of `benchmarks/` are typeset with the paper's fonts (`.pgf` / `.pdf`); without it, an approximate `.pdf` and a `.png` are written.

---

## 🚀 Running Experiments

**Kernel benchmarks (camera-ready figures):**

```bash
cd benchmarks
bash sif.sh python fig03_08_square.py --quick   # one figure, development preset
bash run_all.sh                                 # every figure, one after another
```

**Original benchmark scripts:**

```bash
cd benchmarks/comparison_experiments
bash run.sh
```

Before running, set `TORCH_CUDA_ARCH_LIST` in `run.sh` to match your GPU (e.g. `"8.9"` for RTX 4090, `"9.0"` for H100). Individual experiments can also be run directly, e.g. `python exp1_gpu_time_memory.py`.

**ImageNet-10 experiments:**

Prepare the ImageNet-10 subset with the `train/` and `val/` layout described in [imagenet10_dataset.py](nn_training/CountSketch_LogisticRegression/imagenet10_dataset.py) (or build it from a full ImageNet with [make_imagenet10.py](nn_training/make_imagenet10.py)), replace `path_to_ImageNet` in `run_experiments.sh` with its location, and run:

```bash
cd nn_training/CountSketch_LogisticRegression
bash run_experiments.sh
```

The anchor-crossing diagnostic and the fresh-seed re-run are described in [nn_training/README.md](nn_training/README.md).

**ImageNet-1k classification heads:**

```bash
cd nn_training
python3 make_imagenet10.py --all-classes --src <ImageNet>/CLS-LOC --val-csv <ImageNet>/LOC_val_solution.csv
export DATA_ROOT=$PWD/data/imagenet VAL_ROOT=$PWD/data/imagenet/val
python classification_heads/train/train_laplex_frozen.py   # and train_lora_frozen.py
```

Backbone, rank and seed are set in `classification_heads/configs/`; see [nn_training/classification_heads/README.md](nn_training/classification_heads/README.md).

**Experiments added in the revision (App. J–M):**

```bash
cd benchmarks && bash second_order/run.sh                                  # App. K: Tabs. 10–13, Fig. 22
cd nn_training && bash run_appJ_table15.sh anchor_init anchor_components   # App. M: Tab. 15
cd nn_training && DOWNLOAD=1 bash run_appJ_table15.sh attention            # App. J
cd log_likelihood && bash run.sh                                           # App. I.2 / L: Tabs. 9, 14
```

See the README of each directory for data and caveats.

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
