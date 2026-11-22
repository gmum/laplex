# LAPLEX Density Model — DIV2K-1024

Sec. 4.2. Self-contained: it holds its own copies of the LAPLEX operators.

Gaussian density model with a complex Laplace kernel (LAPLEX) trained on DIV2K high-resolution images at 1024×1024 resolution.

## Model

**Dataset:** DIV2K (Agustsson & Timofte, CVPR 2017) — 800 train / 100 val images, each cropped to 1024×1024 RGB, giving n = 3,145,728 dimensions per sample.

**Results:** test log-likelihood = **1,266,679** after 20 epochs (complex kernel, k=1000, single τ).

**Architecture:** The model is a Gaussian with structured covariance Σ = diag(s²) + BB^T, where B = diag(w)·A·L^T. The low-rank factor is built from a complex Laplace kernel: K_c(α,β,γ,δ)[l,i] = exp(−|α_l − β_i|/τ + i(γ_l − δ_i)), with the effective real projection matrix A[l,i] = Re(K_c[l,i]). The Gram matrix G = A^T diag(D) A is computed efficiently in fp64 via a fused CUDA decoupled-lookback scan (O(n) per batch). The Woodbury matrix identity reduces the log-likelihood to a k×k Cholesky solve.

## Requirements

- Python >= 3.10
- PyTorch >= 2.6 with CUDA (see note below)
- torchvision >= 0.21
- numpy
- Pillow
- matplotlib

Install:
```
pip install -r requirements.txt
```

> **Note: CUDA is required.** The scan kernels (`prefix_suffix.py`, `prefix_suffix_dl_fp64.py`) are JIT-compiled at first import via `torch.utils.cpp_extension.load_inline`. This takes approximately 30 seconds on first run; subsequent runs use the cached build.

## Data

Download DIV2K from http://data.vision.ee.ethz.ch/cvl/DIV2K/

Expected directory structure:
```
/path/to/div2k/
    DIV2K_train_HR/   (0001.png ... 0800.png)
    DIV2K_valid_HR/   (0801.png ... 0900.png)
```

## Reproduce Training

```bash
python _run_div2k.py --complex --k 1000 --batch-size 8 --epochs 20 --div2k-root /path/to/div2k
```

> **Memory note:** `--batch-size 4` fits 8 GB VRAM; `--batch-size 8` requires approximately 14 GB.

## Demo (Pretrained)

The checkpoint (`CKPT_NAME` in `demo.py`) is not in this repository; `_run_div2k.py` writes
one to `results/`, and `--ckpt` selects it.

Evaluates test log-likelihood on the val set and generates 8 samples:

```bash
python demo.py --div2k-root /path/to/div2k
```

Outputs:
- `demo_samples.png` — 8 generated 1024×1024 RGB images
- `demo_results.txt` — test LL and timing
