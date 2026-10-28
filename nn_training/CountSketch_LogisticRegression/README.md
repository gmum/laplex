

This repository contains the experimental pipeline for comparing **LAPLEX** (exact exponential kernel projection) against classical dimensionality reduction techniques (PCA, Random Projection, CountSketch) and a fully trainable dense baseline. The suite evaluates both **frozen-projection + Logistic Regression** and **end-to-end differentiable training** on the ImageNet-10 dataset.

All experiments are designed for statistical robustness (5 independent runs) and automatically log metrics to CSV files.

---



| File | Purpose |
|:---|:---|
| `logistic_regression.py` | Scikit-learn/NumPy pipeline: compares PCA, Gaussian RP, CountSketch, and `ExpKernelProjection` + Logistic Regression. |
| `laplex.py` | Proposed method: `ExpKernelLayer` (differentiable kernel projection) + linear classifier, trained end-to-end in PyTorch. |
| `dense.py` | Dense baseline: fully learnable linear projection matrix + linear classifier, trained identically to LAPLEX. *(Note: referred to as `dense.py` in the loop script)* |
| `run_experiments.sh` | Bash wrapper that executes 5 independent trials for each method. |
| `utils.py` | Shared utilities: `TeedLog` (stdout/stderr redirection to file), `EarlyStopping`. |
| `imagenet10_dataset.py` | Custom PyTorch DataLoader for the ImageNet-10 subset. |

---



- **Python** 3.10+
- **PyTorch** 2.x (CUDA optional; automatically falls back to CPU)
- **Scikit-learn**, **NumPy**, **Pandas**, **TQDM**
- **Dataset:** ImageNet-10 (or compatible structure) at the path specified by `--root_dataset`
- **Hardware:** 
  - `logistic_regression.py` runs on CPU
  - `laplex.py` & `dense.py` automatically use GPU if `torch.cuda.is_available()` is `True`

---




Save the loop script as `run_experiments.sh`, make it executable, and run:
```bash
chmod +x run_experiments.sh
bash run_experiments.sh
```
> ⚠️ **Important:** Replace `path_to_ImageNet` inside the script with the actual path to your dataset before running.


Each script supports full CLI argument parsing and can be run independently:

```bash

python logistic_regression.py \
  --root_dataset /path/to/imagenet10 \
  --dims 25 50 100 250 500 750 1000 \
  --batch_size 512 \
  --iters 1000 \
  --num_workers 8 \
  --outputs results/exp_logistic


python laplex.py \
  --root_dataset /path/to/imagenet10 \
  --dims 25 50 100 250 500 750 1000 \
  --batch_size 512 \
  --lr 1e-3 \
  --epochs 100 \
  --patience 20 \
  --save_model 1 \
  --num_workers 8 \
  --outputs results/exp_laplex


python dense.py \
  --root_dataset /path/to/imagenet10 \
  --dims 25 50 100 250 500 750 1000 \
  --batch_size 512 \
  --lr 1e-3 \
  --epochs 100 \
  --patience 20 \
  --save_model 1 \
  --num_workers 8 \
  --outputs results/exp_dense
```

---



| Script | Methodology | Key Features |
|:---|:---|:---|
| `logistic_regression.py` | **Frozen Projections + Logistic Regression**. Projects data once using PCA, Gaussian RP, CountSketch, or `ExpKernelProjection`, then fits `sklearn.linear_model.LogisticRegression`. | CPU-only, fast baseline, outputs accuracy per target dimension `m`. Handles large `d` via torch-based PCA fallback. |
| `laplex.py` | **End-to-End LAPLEX**. Uses `ExpKernelLayer` with learnable anchors (`alpha`, `beta`), weights (`w`), and temperature (`log_tau`). Optimized with AdamW + `ReduceLROnPlateau`. | GPU-accelerated, differentiable w.r.t. projection parameters, logs learned `tau`, supports early stopping & checkpointing. |
| `dense.py` | **End-to-End Dense Baseline**. Replaces the kernel layer with a fully learnable `nn.Parameter` matrix `W` of shape `(d, m)`. Shares the exact training loop as LAPLEX. | GPU-accelerated, standard unstructured linear projection baseline, enables fair architectural comparison. |

---



- **Console Logging:** `sys.stdout` is redirected to a timestamped `.txt` log via `TeedLog`.
- **Metrics CSV:** Each run saves a `results.csv` in the output directory:
  - `logistic_regression.py`: `m`, `method`, `accuracy`
  - `laplex.py` / `dense.py`: `m`, `epoch`, `train_loss`, `test_loss`, `acc`, `lr`, `tau` (LAPLEX only)
- **Directory Layout:**
  ```
  results/
  ├── experiments_I/   
  │   └── 20231025_143000/
  │       ├── log.txt
  │       └── results.csv
  ├── experiments_II/  
  └── experiments_III/ 
  ```

---



| Parameter | Recommendation |
|:---|:---|
| `--dims` | Sweep values: `25 50 100 250 500 750 1000`. Controls target projection dimension `m`. |
| `--batch_size` | `512` is optimal for modern GPUs with fast storage. Reduce if VRAM is limited. |
| `--patience` | `20` epochs. Halts training if validation loss doesn't improve. |
| `--save_model` | Set to `1` (or any `N`) to save `.pth` checkpoints every `N` epochs. |
| `--seed` | Omitted for random seed generation. Pass an integer for strict reproducibility. |
| `dense.py` vs `dense.py` | The loop script calls `dense.py`. Ensure the file is named `dense.py` or update the loop to call `dense.py`. |


- `logistic_regression.py` completes in **~2–5 minutes** on a modern CPU. Accuracy typically improves monotonically with `m`, revealing the compression-performance trade-off.
- `laplex.py` & `dense.py` run on **GPU**. Expect **10–40 min** per full sweep (5 dims × 5 runs × ≤100 epochs). `laplex.py` learns structured, interpretable projections (via `tau` and anchor spacing), while `dense.py` learns dense, unstructured weights. Both converge reliably under the provided scheduler & early-stopping configuration.

