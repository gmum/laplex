# LAPLEX: The FFT of Learnable Laplace Kernels

🎉 **Accepted at NeurIPS 2026**

Łukasz Struski, Hanna Blazhko, Piotr Kubaty, Jacek Tabor

[![arXiv](https://img.shields.io/badge/arXiv-2605.24584-b31b1b.svg)](https://arxiv.org/abs/2605.24584)
[![NeurIPS 2026 poster](https://img.shields.io/badge/NeurIPS_2026-poster_page-68448c.svg)](https://neurips.cc/virtual/2026/loc/sydney/poster/155528)

Official implementation of **"LAPLEX: The FFT of Learnable Laplace Kernels"** (NeurIPS 2026).
LAPLEX applies a dense matrix with entries exp(−|aᵢ − bⱼ|), defined by learnable anchors on a
line, exactly in O(n log n) time and O(n) memory, and differentiates through the anchors.

## Repository structure

The experiments are grouped by what LAPLEX is used for. Each directory has a README with the
details, the data it needs and the commands for every figure and table.

| directory | content |
|---|---|
| [benchmarks/](benchmarks/README.md) | the LAPLEX operators: runtime, memory and accuracy against dense kernels, Random Fourier Features and FFT / Toeplitz-FFT (Figs. 1, 3, 5, 6, 8–18, Tab. 3); second-order optimization with the exact curvature (App. K); the released operators in [comparison_experiments/](benchmarks/comparison_experiments/README.md) |
| [nn_training/](nn_training/README.md) | LAPLEX as a trainable layer: ImageNet-10 projections ([CountSketch_LogisticRegression/](nn_training/CountSketch_LogisticRegression/README.md), Tab. 1), classification heads on frozen Swin-T / MaxViT-T ([classification_heads/](nn_training/classification_heads/README.md), Sec. 7), attention (App. J), anchor ablations and the anchor-crossing diagnostic (App. M) |
| [log_likelihood/](log_likelihood/README.md) | Gaussian density models on DIV2K-1024 and ImageNette-320 (Sec. 4.2, App. I) and structured covariance baselines (App. L) |
| [environment/](environment/) | container definition, conda environment and pinned package versions |
| [.env.example](.env.example) | template of the machine-specific settings (container image, dataset paths, Slurm partition) |

## Installation

[`environment/`](environment/) describes the software of the paper's runs in two forms with
the same Python packages ([requirements.txt](environment/requirements.txt): Python 3.14.6,
PyTorch 2.13.0, NumPy 2.4.4, SciPy 1.18.0, ...); they differ in the CUDA version.

**Apptainer / Singularity (the paper's environment).** [laplex.def](environment/laplex.def)
rebuilds Ubuntu 24.04, CUDA 13.2 with `nvcc` and gcc 13, TeX Live and the packages above
(NVIDIA driver ≥ 580):

```bash
cd environment
apptainer build laplex.sif laplex.def      # or: singularity build --fakeroot laplex.sif laplex.def
```

Building needs root, `--fakeroot` or unprivileged user namespaces; otherwise build on a
workstation and copy the `.sif`.

**Conda.** [environment.yml](environment/environment.yml) installs the same packages with
PyTorch for CUDA 12.6 (driver ≥ 525), plus `nvcc` and gcc for the CUDA kernels. For RTX 50xx
GPUs replace `cu126` by `cu130` and the `12.6` CUDA packages by `13.0`.

```bash
conda env create -f environment/environment.yml
conda activate laplex
```

The custom CUDA kernels are compiled on first import (30 s to a few minutes per
implementation) and cached in `.cache/`. With `pdflatex`, the benchmark figures are typeset
with the paper's fonts; without it, an approximate PDF and a PNG are written.

## Configuration

Machine-specific settings live in `.env` (not tracked): copy the template and fill in what
applies to your machine.

```bash
cp .env.example .env
```

- `LAPLEX_SIF`: the image built above. The `sif.sh` wrappers in `benchmarks/`, `nn_training/`
  and `log_likelihood/` run every command in it (`bash sif.sh python ...`), with the GPU and
  the dataset directories of `.env` visible inside. Without it, they use the current Python
  environment (e.g. the conda environment).
- Dataset paths: `IMAGENET_ROOT` and `IMAGENET_VAL_CSV` (to build ImageNet-10),
  `IMNET`, `DATA_ROOT` / `VAL_ROOT`, `TV_DATA`, `DIV2K_ROOT`, `IMAGENETTE_ROOT`. Which
  experiment needs which is given in the directory READMEs.
- Slurm: the sbatch files request only resources (GPU, CPUs, memory, time). Set
  `SBATCH_PARTITION`, `SBATCH_QOS` and `SBATCH_ACCOUNT` for your cluster and export them
  before submitting (`set -a; source .env; set +a`), or pass `sbatch -p <partition> ...`.

## Quick start

```bash
cd benchmarks && bash sif.sh python fig03_08_square.py --quick    # one benchmark figure, development preset
cd benchmarks && bash run_all.sh                                   # every benchmark figure
cd nn_training && python3 make_imagenet10.py                       # ImageNet-10 from IMAGENET_ROOT
cd nn_training/CountSketch_LogisticRegression && bash run_experiments.sh   # Tab. 1
cd log_likelihood && bash table15_downstream/run.sh                # App. L, Tab. 15
```

## Citation

If you find this work useful, please cite our NeurIPS 2026 paper:

```bibtex
@inproceedings{struski2026laplex,
  title     = {LAPLEX: The FFT of Learnable Laplace Kernels},
  author    = {Struski, {\L}ukasz and Blazhko, Hanna and Kubaty, Piotr and Tabor, Jacek},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```
