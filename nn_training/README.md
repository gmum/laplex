# LAPLEX as a trainable layer

LAPLEX as a learnable projection d → m in front of a linear classifier on ImageNet-10,
compared with PCA, Gaussian random projection, CountSketch and a fully learnable dense
projection (Sec. 5), and as the classification head of a pretrained ImageNet-1k
backbone, compared with a low-rank head (Sec. 7). Also LAPLEX as an attention operator
(App. J) and the anchor-initialization and component ablations (App. M).

| Directory / script | Paper | What it does |
|---|---|---|
| [CountSketch_LogisticRegression/](CountSketch_LogisticRegression/) | Table 1 (Sec. 5), App. G | the released experiments: frozen projections + logistic regression (`logistic_regression.py`), LAPLEX and dense projections trained end to end (`laplex.py`, `dense.py`); see its own README |
| `crossing_trace.py`, `plot_crossing.py` | App. M, Fig. 23 | anchor-crossing diagnostic of the trainable projection (m = 500) with per-step loss, anchor-gradient norms and crossing rates |
| `table15_anchor_ablations.py` | App. M, Tab. 15 | the trainable projection (m = 500) on cached, non-augmented features: seven anchor initializations (`--mode init_sweep`) and I = 1…4 kernel components (`--mode components`) |
| `appJ_attention.py`, `appJ_attention_table.py` | App. J (table) | pixel-token Transformer on MNIST / SVHN / CIFAR-10 with only the attention sub-layer swapped (softmax, Performer RFF, RFF + τ, LAPLEX); the table script picks the best configuration per model |
| `run_appJ_table15.sh` | App. J, Tab. 15 | the sweeps of App. J and Tab. 15 with the paper's hyper-parameters |
| `parity_stats.py` with `sbatch/parity.sbatch` | Table 1, App. G | a fresh 5-seed run of `laplex.py` and `dense.py`: are LAPLEX and dense still statistically indistinguishable (Welch, Mann–Whitney)? |
| [classification_heads/](classification_heads/) | Tab. 2 (Sec. 7), App. H | LAPLEX vs low-rank (LoRA-style) heads on frozen Swin-T and MaxViT-T, ImageNet-1k; see its own README |
| `make_imagenet10.py` | data | ImageNet-10 (or, with `--all-classes`, ImageNet-1k for `classification_heads/`) as a tree of symlinks into a full ImageNet |

`crossing_trace.py` imports the model, optimiser settings and loader of
`CountSketch_LogisticRegression/laplex.py` and `imagenet10_dataset.py`;
`sbatch/parity.sbatch` runs `laplex.py` and `dense.py` unchanged.
`table15_anchor_ablations.py` uses the model, optimiser and early stopping of
`CountSketch_LogisticRegression/laplex.py` and its `imagenet10_dataset.py`.

## Environment

The environment of the main [README](../README.md) (an image built from
[../environment/laplex.def](../environment/laplex.def) or the conda environment;
`torch`, `torchvision`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, and `scipy` for
`parity_stats.py`). No CUDA toolchain is needed: the trainable layer materialises K
densely. `sif.sh` runs a command in the Singularity image of `../benchmarks/sif.sh`
(`LAPLEX_SIF` selects another one) and binds `/shared/sets/datasets` when it exists;
without an image it runs the command in the current environment.

## Data: ImageNet-10

`imagenet10_dataset.py` expects `root/{train,val}/<synset>/` and keeps the ten synsets
of `IMAGENET10_SYNSETS`. The scripts here read `data/imagenet10`; either link an existing
ImageNet-10 there, or build it from a full ImageNet (ILSVRC CLS-LOC layout, flat `val/`)
with

```bash
python3 make_imagenet10.py --src <ImageNet>/CLS-LOC --val-csv <ImageNet>/LOC_val_solution.csv
```

- train: links to the per-synset folders of `CLS-LOC/train`;
- val: links to the matching files of the flat `CLS-LOC/val`, using the labels in
  `LOC_val_solution.csv`.

The result is 12,711 training and 500 validation images at full resolution. The
defaults of `--src` and `--val-csv` are the copies on the cluster used for the paper
(`/shared/sets/datasets/...`), and the sbatch files run the script with them if
`data/imagenet10` is missing. The docstring of `imagenet10_dataset.py` gives
n03994562 for "pinwheel", but the list the code uses has n03944341.

## Running

```bash
cd nn_training
# crossing diagnostic: augmented data, 6 epochs (Fig. 23 shows 0-5); then figures and summary
bash sif.sh python crossing_trace.py
bash sif.sh python crossing_trace.py --no-augment --seed 1
bash sif.sh python plot_crossing.py
# quick check without data or GPU
bash sif.sh python crossing_trace.py --fake-data 64 --fake-dim 3072 --batch-size 16 --epochs 1 --device cpu
# parity re-run: one seed and method by hand (the sbatch array runs all 10), then statistics
bash sif.sh python CountSketch_LogisticRegression/laplex.py --root_dataset data/imagenet10 \
    --dims 25 100 500 1000 --batch_size 512 --lr 1e-3 --epochs 100 --patience 20 \
    --save_model 0 --seed 1001 --outputs results/parity/laplex/seed1001
bash sif.sh python parity_stats.py
```

### Attention (App. J) and anchor ablations (Tab. 15)

```bash
bash run_appJ_table15.sh                                  # list the sections
bash run_appJ_table15.sh anchor_init anchor_components    # Tab. 15, ImageNet-10 from data/imagenet10
DOWNLOAD=1 bash run_appJ_table15.sh attention             # App. J: 216 runs, then the table
```

- **Attention data.** `appJ_attention.py` reads the torchvision datasets from `--data_path`
  (`TV_DATA`, default `data/torchvision`): MNIST in `MNIST/`, CIFAR-10 in
  `cifar10/cifar-10-batches-py/`, SVHN in `SVHN/`. `--download` (`DOWNLOAD=1`) fetches
  them. On the paper's cluster, link them instead:
  ```bash
  mkdir -p data/torchvision && cd data/torchvision
  ln -s /shared/sets/datasets/MNIST MNIST
  ln -s /shared/sets/datasets/cifar10 cifar10
  ln -s /shared/sets/datasets/svhn SVHN
  ```
- **Batch size.** In the paper's sweep, the six largest runs (Structured, att_dim 256,
  depth 8, SVHN and CIFAR-10) were completed with batch 32 instead of 128 on 10 GB GPUs.
  They are not batch-comparable with the rest of the grid. `run_appJ_table15.sh` runs them with
  batch 128.
- **Tab. 15** uses cached, non-augmented features (`--features_on_gpu`). Its absolute
  accuracies (~40%) are therefore below the augmented ~43.8% of Table 1.

### Slurm

Submit from this directory, because the log paths in the sbatch files are relative.
The sbatch files name the partitions of the cluster the paper's runs used (dgxa100,
cpu); on another cluster pass your own, e.g. `-p <partition> -q <qos>`.

```bash
bash sbatch/submit_all.sh                     # everything, statistics chained after the parity array
bash sbatch/submit_all.sh -p rtx4090_batch -q batch --exclude=c22   # GPU jobs on another partition
# or one at a time
sbatch sbatch/crossing.sbatch                 # 2 runs x 6 epochs; EPOCHS, SEEDS, VARIANTS configurable
jid=$(sbatch --parsable sbatch/parity.sbatch) # array of 10 (laplex/dense x 5 seeds), up to ~7 h per task
sbatch --dependency=afterany:$jid sbatch/parity_stats.sbatch   # cpu partition, after every task
```

`submit_all.sh` submits the two GPU jobs (default partition dgxa100, QOS quick; its
arguments override that) and `parity_stats.sbatch` on the `cpu` partition with
`--dependency=afterany:<array>`. `afterany` lets the statistics run even if some tasks
fail; `parity_stats.py` lists the missing seeds.

## Outputs

Not tracked by git (`results/`, `logs/`, `data/`).

- `results/crossing/<variant>_m500_e<epochs>_seed<s>/`:
  - `steps.csv` and `epochs.csv`;
  - `fig_crossing.{pdf,png}` with four panels: per-step loss, per-step gradient norms,
    per-step crossing rate, and per-epoch rates with the loss as in Fig. 23;
  - `summary.json`.

  `results/crossing/summary.md` compares every run with the paper's numbers.
  - `aug` uses the released training transform.
  - `noaug` uses the validation transform for training. App. M says its ablations used
    "cached, non-augmented features"; the protocol of the original diagnostic is not
    known.
- `results/parity/<method>/seed<S>/<timestamp>/` holds the released scripts' own
  `results.csv` and `log.txt`. `parity_stats.py` writes `per_seed.csv`, `summary.md` and
  `summary.json` next to them.
- `results/attention/<dataset>/<timestamp>/{params.json, results.csv}`, one directory per
  run; `results/attention/table.txt` is the App. J table.
- `results/anchor_dynamics/<mode>_<timestamp>/`: `init_sweep_{raw,summary}.csv`,
  `components_{raw,summary}.csv` (Tab. 15).
