# Log-likelihood (density models)

Gaussian density models on ImageNette-320 and DIV2K-1024. Test log-likelihood (LL) is in
nats per image; bits/dim = −LL / (n ln 2), so lower bits/dim is better.

| Directory | Paper | What it does |
|---|---|---|
| [fig04_div2k/](fig04_div2k/) | Sec. 4.2, Fig. 4 | LAPLEX (width 1000) and dense low-rank (rank 4) Gaussians on DIV2K-1024; has its own [README](fig04_div2k/README.md) |
| [table08_imagenette/](table08_imagenette/) | App. I.1, Tab. 8 | LAPLEX (I = 1) and dense low-rank Gaussians on ImageNette-320 |
| [table09_10_circulant/](table09_10_circulant/) | App. I.2, Tabs. 9, 10 | block-circulant Gaussian (K = 1), mixtures (K = 5, K = 3) and the diagonal floor, in permuted and raster coordinates |
| [table15_downstream/](table15_downstream/) | App. L, Tab. 15 | PPCA, dense low-rank, LAPLEX, FFT-circulant and Kronecker on raster-order ImageNette, with MAP denoising |

`bash sif.sh <command>` runs a command in the Singularity image of the paper
(`LAPLEX_SIF` selects another one) from this directory. All scripts need one CUDA GPU; the
LAPLEX scan kernels are compiled with `nvcc` on first import.

## Data

- **`imagenette2-320`** (fast.ai Imagenette, 9,469 train / 3,925 val, resized and centre-cropped
  to 320 × 320, n = 307,200) for Tabs. 8–10. Set `IMAGENETTE_ROOT`.
- **DIV2K** with `DIV2K_train_HR/` and `DIV2K_valid_HR/` (800 / 100, resized and cropped to
  1024 × 1024, n = 3,145,728) for Fig. 4 and Tabs. 9, 10. Set `DIV2K_ROOT`.
- **ImageNet-10** for Tab. 15: the ten ImageNet synsets, built by
  [../nn_training/make_imagenet10.py](../nn_training/make_imagenet10.py), capped at 800 images
  per class (8,000 training images), `--imagenet_root`.

## Sec. 4.2 and App. I.1

```bash
cd fig04_div2k
python _run_div2k.py --complex --k 1000 --batch-size 8 --epochs 20 --div2k-root $DIV2K_ROOT
python _run_div2k.py --dense --k-dense 4 --batch-size 8 --div2k-root $DIV2K_ROOT

cd table08_imagenette
python _run_imagenette320.py --complex --k 1000 --epochs 25      # I = 1, +198,549 nats
python _run_imagenette320.py --dense --k-dense 6 --epochs 25     # dense low-rank
```

Results (progress log, samples, checkpoint) go to `results/` of each directory.

Not in this repository: the I = 2, 3 LAPLEX variants, PPCA and the empirical-Gaussian
reference of Tab. 8, MAP RMSE, Figs. 20–21, and the MAP panels of Fig. 4.

## App. I.2

```bash
bash table09_10_circulant/run.sh                 # every row, about 7 h on an RTX 4090
python table09_10_circulant/print_tables.py      # the rows of Tabs. 9 and 10
python table09_10_circulant/test_bccb.py         # LL and sampler against a dense Gaussian
```

- **Model.** `_model_bccb.py`: one 3×3 Hermitian block per spatial frequency,
  B = L Lᴴ + 10⁻⁴ I, 9 H (⌊W/2⌋ + 1) parameters, frozen empirical mean. A mixture has K such
  covariances, one shared mean and learned weights, and starts from the trained K = 1 model of
  the same coordinate order (`--warm-start`).
- **Permutation.** `_perm_dataset.py`: one `torch.randperm(n)` with seed 0 over all n coordinates.
  It is cached in the directory as `perm_n*_seed0.pt` and rebuilt from the seed when absent.
- **Parameters.** The diagonal row of the tables counts the n variances; `_run_bccb.py` also
  counts the frozen means in its result files (2n).
- The LAPLEX rows of both tables are the raster-order runs of `table08_imagenette/` and
  `fig04_div2k/`.

## App. L

```bash
bash table15_downstream/run.sh
```

`density_models.py` holds the models, and `python density_models.py` checks every LL and MAP against
a dense Gaussian on a tiny problem. `table15_density_downstream.py` fits them on raster-order
ImageNet-10 and reports nats/dim and bits/dim, PSNR and SSIM. The CUDA LAPLEX operators come from
[../benchmarks/comparison_experiments/](../benchmarks/comparison_experiments/). `--ood_root`
adds an optional OOD-detection AUROC, which is not part of the paper.
