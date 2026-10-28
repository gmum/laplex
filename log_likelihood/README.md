# Log-likelihood (density models)

Density-modelling experiments: Sec. 4.2 and Fig. 4 (DIV2K), App. I (ImageNette), App. L.
This directory holds the code of the revision's density comparisons, App. I.2 (Tab. 9)
and App. L (Tab. 14). The main experiments of Sec. 4.2 and App. I.1 (Fig. 4, Tab. 8,
Figs. 20–21) are not in this repository yet.

| Script | Paper | What it does |
|---|---|---|
| `density_models.py` | Tabs. 9, 14 | the Gaussian covariance models; `python density_models.py` checks every model's log-likelihood (and MAP where it exists) against a dense Gaussian on a tiny problem |
| `table09_14_density.py` | Tab. 9, Tab. 14 | fits the models on ImageNette-320 or DIV2K-1024 in raster and/or permuted pixel order; test log-likelihood (nats, nats/dim, bits/dim) and, on ImageNette, MAP denoising (PSNR, SSIM) |
| `run.sh` | Tabs. 9, 14 | self-test, then the rows of Tabs. 9 and 14 and the raster-order circulant values of App. I.2 |

Models (`density_models.py`), each a single Gaussian N(m, Σ):

- **LAPLEX**: factor analysis Σ = F Fᵀ + diag(σ²), F = diag(w) A L with A = LAPLEX(a, b) ∈ R^{n×k}
  and L ∈ R^{k×k} lower triangular, trained by maximum likelihood (Woodbury, Adam).
- **dense-LR**: the same template with a free F ∈ R^{n×r}, r matched to the LAPLEX budget.
- **PPCA**: closed form.
- **Kronecker**: tensor-normal Σ = Σ_C ⊗ Σ_H ⊗ Σ_W, flip-flop MLE.
- **FFT-circulant**: stationary covariance, closed form.
- **circulant-K1, circulant-mix**: Σ⁻¹ = LᵀL with L = D_K C_K ⋯ D_1 C_1 (per-pixel diagonal times
  per-channel FFT circulant). The mix uses K stages, chosen to match the LAPLEX parameter count.
- **diagonal**: the floor.

## Paper mapping and status

- **Tab. 14:** the ImageNette rows in raster order (PPCA, dense-LR, LAPLEX, FFT-circulant,
  Kronecker), with `--max_per_class 800` (8,000 training images) and 20 epochs.
- **Tab. 9:**
  - The circulant (K = 5 / K = 3 and K = 1) and diagonal rows are the permuted-order rows of
    `circulant-mix`, `circulant-K1` and `diagonal`.
  - The LAPLEX rows of Tab. 9 are the runs of App. I / Sec. 4.2. They are not produced here.
- **App. I.2 text:** the raster-order circulant bits/dim (−3.51 ImageNette, −3.46 DIV2K) come
  from `circulant-K1` / `circulant-mix` in raster order. The paper does not say which of the two.
- **Reconstructed code.** The original scripts of the permuted-order comparison were not
  recovered, so the circulant-mix model is a reconstruction. It matches the published
  parameter counts exactly (K = 1: 0.46M / 4.72M, K = 5 ImageNette: 2.31M, K = 3 DIV2K:
  14.2M), but new runs do not reproduce the published log-likelihoods digit for digit.
- **Rank.** The dense-LR and PPCA rank is matched to the full LAPLEX count 4n + k + k² (rank 5
  on ImageNette).
- **MAP denoising** is available only for models with a closed-form posterior mean. The
  circulant-K rows have log-likelihood only.

The CUDA LAPLEX operators are imported from
[../benchmarks/comparison_experiments/](../benchmarks/comparison_experiments/) (`laplex.py`,
`prefix_suffix.py`); they are compiled on first import, so a CUDA GPU and `nvcc` are needed.

## Data

- **"ImageNette-320"** means the ten ImageNet synsets of ImageNet-10, resized to 320 × 320
  (n = 307,200), not the fast.ai Imagenette. `--imagenet_root` takes the ImageNet-10 tree
  built by [../nn_training/make_imagenet10.py](../nn_training/make_imagenet10.py) (default
  `../nn_training/data/imagenet10`) or any root with `train/<synset>/`, `val/<synset>/`.
- **DIV2K-1024:** shorter side resized to 1024 (bicubic), centre crop; n = 3,145,728.
  `--div2k_root` must contain `DIV2K_train_HR/` and `DIV2K_valid_HR/`; the official 800 / 100
  split is used.
- `--ood_root <folder of images>` adds an OOD-detection AUROC on ImageNette (e.g. with
  CelebA); it is not part of the paper.

## Running

```bash
cd log_likelihood
bash sif.sh python density_models.py            # self-test
bash run.sh                                     # Tabs. 9 and 14 (IMNET, DIV2K, OUT)
bash sif.sh python table09_14_density.py --dataset imagenette --orders raster \
    --max_per_class 800 --models LAPLEX dense-LR PPCA Kronecker FFT-circulant   # Tab. 14 only
```

`sif.sh` works as in [../nn_training/](../nn_training/README.md): it uses the Singularity
image if it exists and binds `/shared/sets/datasets`; otherwise it runs in the current
environment. Results: `results/density/<dataset>_<timestamp>/{results.json, tables.md}`.
