# Classification heads (Sec. 7, App. H)

The experiment of Sec. 7 ("Learned anchor geometry in neural heads"). The
classification head of an ImageNet-1k-pretrained torchvision backbone (Swin-T, MaxViT-T)
is replaced by a structured head, and only the head is trained (frozen backbone):

| Head | Paper | Weight |
|---|---|---|
| `LaplexLinear` | LAPLEX^(r) | `W[i,j] = Σ_r c_r · exp(−\|a_r[i] − b_r[j]\| / τ)`, trained without forming `W` (`X_Laplex`) |
| `LoRALinear` | LowRank^(r) | `W = B @ A` |

For MaxViT-T both linear layers of the classifier MLP are replaced. The parameter
counts of the code match Tab. 2/7 for every backbone and rank (e.g. Swin-T,
r = 4: 8,077 and 8,072 = 8.1 K).

| Paper (camera-ready) | Code |
|---|---|
| Tab. 2 (Sec. 7), Tab. 7 (App. H): mean top-1 over 5 runs | `train/train_laplex_frozen.py`, `train/train_lora_frozen.py` with `MODEL` ∈ {`swin_t`, `maxvit_t`}, `RANK` ∈ {4, 8, 16}, `SEED` ∈ {0, …, 4} |
| Tab. 6 (App. H): training configuration | `configs/laplex_frozen.py`, `configs/lora_frozen.py` |
| Fig. 19 (App. H): training and validation curves | the per-epoch losses and accuracies the trainer prints; no plotting script |
| rows "Pretrained" and "Random FullRank" | not in this directory: "Pretrained" is torchvision's accuracy, and no code for the FullRank head was found |

`*_unfrozen` (backbone trained too) is not reported in the paper.

## Origin and changes

The code of the paper's runs (Hanna Blazhko, `heads_experiment/`, May–June 2026) and
the LAPLEX library it imports (`laplex_lib/`, the version that was next to it). Changes:

- `laplex_lib/cuda_kernel.py` is the original `cuda.py`, and `laplex_lib/laplex.py`
  the library's own later version that imports it under this name (otherwise
  identical): recent PyTorch installs `cuda-bindings`, which provides a `cuda` package.
- `merged_events.py` is not copied: it does not parse, so `cuda_kernel.py` always fell
  back to its own scan (which it does when the module is missing). With batch 128 it is
  not used in any case (`NAIVE_SCAN_THRESHOLD = 4096`).
- `models/laplex_linear.py`, `train/setup.py`: path to `laplex_lib/`.
- `utils/misc.py`: `get_worker_init_fn` returns a `functools.partial` instead of a
  closure, with the same seeding. Python 3.14 starts DataLoader workers with
  forkserver, which cannot pickle a closure.

`laplex_lib/prefix_suffix.py` is the version of the paper's runs. It differs from
`../../benchmarks/vendor/lib2026/prefix_suffix.py` only in the CUDA grid layout.

Differences between the code and Tab. 6, left as they are:

- Tab. 6 gives lr / weight decay 1e-5 / 0.05 for "FullRank & bias". In
  `configs/*_frozen.py` the bias (and the mixing weights `c` of `LaplexLinear`) are in
  the base group with `LR_BASE = 1e-4`, `WD_BASE = 0`; 1e-5 / 0.05 are the values of the
  `*_unfrozen` configs.
- Tab. 6 gives a cosine schedule to η_min = 1e-6. `run.py` uses the `warmup_cosine`
  scheduler, which anneals to 0; `ETA_MIN` is only read by the `cosine` scheduler.

## Running

Environment: the one of the main [README](../../README.md) (`torch`, `torchvision`,
`numpy`, `tqdm`; `nvcc` for the CUDA backend of LAPLEX). The pretrained weights are
downloaded by torchvision into `$TORCH_HOME` (default `~/.cache/torch`).

Data: ImageNet-1k in ImageFolder layout, `DATA_ROOT/train/<synset>/` and
`VAL_ROOT/<synset>/`. From an ImageNet with a flat `val/` (ILSVRC CLS-LOC layout) and
`LOC_val_solution.csv`, `../make_imagenet10.py --all-classes` builds such a tree of
symlinks:

```bash
cd nn_training
python3 make_imagenet10.py --all-classes --src <ImageNet>/CLS-LOC --val-csv <ImageNet>/LOC_val_solution.csv
export DATA_ROOT=$PWD/data/imagenet VAL_ROOT=$PWD/data/imagenet/val
bash sif.sh python classification_heads/train/train_laplex_frozen.py
bash sif.sh python classification_heads/train/train_lora_frozen.py
```

(the defaults of `--src` and `--val-csv` are the copies on the cluster used for the paper).
For the grid of Tab. 2/7, set `MODEL`, `RANK` and `SEED` in the two `*_frozen.py`
configs: 2 backbones × 3 ranks × 5 seeds = 30 runs of 30 epochs per head. Each run
prints per-epoch train/val loss and top-1 accuracy, and saves the weights with the best
validation accuracy to `best_model.pth` in the working directory (`nn_training/` under
`sif.sh`), which the next run overwrites.

Multi-GPU (DDP), with `BATCH_SIZE` per GPU:

```bash
torchrun --nproc_per_node=4 classification_heads/train/train_laplex_frozen.py
```

## Files

```
classification_heads/
├── configs/      laplex_{frozen,unfrozen}.py, lora_{frozen,unfrozen}.py: all hyperparameters
├── models/       laplex_linear.py (LaplexLinear), lora_linear.py (LoRALinear)
├── train/        setup.py (backbone, head replacement, freezing), run.py (pipeline),
│                 train_{laplex,lora}_{frozen,unfrozen}.py
├── utils/        data.py (ImageFolder loaders), training.py (AMP/DDP trainer, AdamW
│                 groups, scheduler), misc.py (metrics, seeding)
└── laplex_lib/   LAPLEX library: laplex.py (backend selection), cuda_kernel.py,
                  prefix_suffix.py (CUDA), pure.py (PyTorch), LICENSE
```

## Configuration

All hyperparameters are in `configs/`. Each config is a plain Python module — edit it directly or patch it programmatically before calling `run()`. Data paths can be overridden with `DATA_ROOT` and `VAL_ROOT` environment variables.

### Model and head

| Key | Type | Description |
|-----|------|-------------|
| `MODEL` | `str` | Backbone: `"swin_t"` or `"maxvit_t"` |
| `FROZEN` | `bool` | `True` — train head only; `False` — train backbone and head |
| `RANK` | `int` | Number of rank components in the head decomposition |
| `TAU_INIT` | `float` | Initial kernel temperature τ — *LaplexLinear only* |
| `AB_SCALE` | `float \| "auto"` | Init std for `a`, `b`; `"auto"` sets it to `σ / (in · out)^0.25` — *LaplexLinear only* |
| `C_SCALE` | `float \| "auto"` | Init std for coefficients `c`; `"auto"` sets it to `σ` — *LaplexLinear only* |
| `SIGMA_INIT` | `float` | Base init scale σ; for `LoRALinear`, directly used as `A`, `B` init std |

### Data

| Key | Type | Description |
|-----|------|-------------|
| `DATA_ROOT` | `str` | Parent directory whose `train/` subdirectory is in ImageFolder layout |
| `VAL_ROOT` | `str` | Validation set root — must be in ImageFolder layout (one subfolder per class) |
| `BATCH_SIZE` | `int` | Per-GPU batch size |
| `NUM_WORKERS` | `int` | DataLoader worker processes per GPU |

### Optimizer

AdamW with two parameter groups:

| Group | Parameters |
|-------|-----------|
| base | backbone weights, `c` coefficients, `bias` |
| exponential | `a`, `b`, `log_tau` (LAPLEX) or `A`, `B` (LoRA) |

| Key | Type | Description |
|-----|------|-------------|
| `LR_BASE` | `float` | Learning rate for the base group |
| `LR_EXP` | `float` | Learning rate for the exponential / LoRA group |
| `WD_BASE` | `float` | Weight decay for the base group |
| `WD_EXP` | `float` | Weight decay for the exponential / LoRA group |
| `WEIGHT_DECAY` | `float` | Global fallback weight decay |
| `BETAS` | `tuple[float, float]` | AdamW momentum coefficients `(β1, β2)` |
| `EPS` | `float` | AdamW ε for numerical stability |
| `GRAD_CLIP` | `float \| None` | Max gradient norm; `None` disables clipping |

### Scheduler

Linear warmup followed by cosine annealing.

| Key | Type | Description |
|-----|------|-------------|
| `EPOCHS` | `int` | Total training epochs |
| `WARMUP_EPOCHS` | `int` | Epochs for linear LR warmup; 0 = cosine from epoch 1 |
| `ETA_MIN` | `float` | Minimum LR at the end of cosine annealing |

### Loss

| Key | Type | Description |
|-----|------|-------------|
| `LABEL_SMOOTHING` | `float` | Label smoothing for `CrossEntropyLoss`; 0 = disabled |

### Reproducibility

| Key | Type | Description |
|-----|------|-------------|
| `SEED` | `int` | Global random seed (Python, NumPy, PyTorch, CUDA) |
| `CUDNN_BENCHMARK` | `bool` | Enable cuDNN auto-tuner; set `False` for reproducibility |
| `CUDNN_DETERMINISTIC` | `bool` | Force deterministic cuDNN algorithms |
| `CUBLAS_WORKSPACE_CONFIG` | `str \| None` | cuBLAS workspace config required by `torch.use_deterministic_algorithms` |

---

## LAPLEX in brief

`LaplexLinear` implicitly defines its weight matrix as:

```
W[i, j] = Σ_r  c_r · exp(−|a_r[i] − b_r[j]| / τ)
```

During training, `x @ W` is evaluated without forming `W`, using an O(n log n) prefix-scan algorithm from the LAPLEX library. In eval mode, `W` is built once and cached so inference uses a standard `F.linear` call.

The LAPLEX library is in `laplex_lib/`. `laplex.py` uses the CUDA backend (`cuda_kernel.py`, compiled on first import) when a GPU is visible and falls back to pure PyTorch (`pure.py`) otherwise. `LAPLEX_BACKEND=cuda|pure` forces one of them.
