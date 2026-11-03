# Kernel-operator benchmarks

Runtime, peak memory and accuracy of the LAPLEX operators X K(a, b) and
K(a, b) diag(D) K(a, b)^T against an explicit dense kernel, FFT / Toeplitz-FFT and
Random Fourier Features (Figs. 1, 3, 5, 6, 8–18 and Tab. 3 of the paper).

* The scripts in this directory produce the figures of the camera-ready version.
  They run every benchmark under **one stopping rule**, the one of Figs. 1 and 15:

  > A method is measured at every size of the sweep until it no longer fits in
  > device memory (a real `OutOfMemoryError`). The first size that does not fit
  > is recorded as `OOM`, and the method is not attempted at larger sizes.
  > There is no wall-clock or VRAM budget.

* [comparison_experiments/](comparison_experiments/) holds the released LAPLEX
  operators (`laplex.py`, `laplex_noCUDA.py`, `prefix_suffix.py`, `laplex_phased.py`)
  and the scripts behind the App. D–F figures of the submitted version. Those
  scripts stop the dense baseline at a 500 ms / 1.5 GB budget (`DenseGate` in
  `_common.py`) and at hard-coded sizes; see its own README.
* [second_order/](second_order/) is App. K. It runs Fisher scoring for a logistic GLM
  with the exact LAPLEX weighted Gram as curvature, using the operators of
  `comparison_experiments/`.
* `last_interval_test.py` checks that the implementations handle anchors beyond
  both ends of the sorted a (Thm. 1/4 and Thm. 6).

## Layout

| File | Role |
|---|---|
| `common.py` | measurement protocol: timing, peak memory, the stopping rule (`Sweep`), the generic sweeps `run_timing_sweep` / `run_accuracy_sweep`, CLI, results I/O |
| `impls.py` | operators: LAPLEX implementations (`--impl`), dense baselines, Toeplitz-FFT, RFF |
| `plotting.py` | shared figure style and the uniform OOM marking |
| `manuscript.py`, `figure_geometry.json` | size, axes rectangles, y ranges and font sizes of the published figures, which the reproduced figures reuse |
| `figNN_*.py`, `table03_anchor_cost.py` | one independent script per published figure or table (run + plot) |
| `impl_compare.py` | all 10 LAPLEX implementations in the Fig. 1 setting (one process each): correctness check, timing, "which one is fastest at which n" (`dominance.md`) |
| `last_interval_test.py` | anchors beyond the ends of the sorted a, against a float64 dense reference |
| `comparison_experiments/` | the released operators and the original benchmark scripts |
| `second_order/` | App. K (Tabs. 11–14, Fig. 22); `run.sh` runs all of it, results in `second_order/results/` |
| `vendor/` | the other LAPLEX implementations, copied here; see [vendor/README.md](vendor/README.md) |
| `run_all.sh` | runs the figure scripts one after another (never two benchmarks at once) |
| `submit_max_n.sh` | one Slurm job per experiment up to the largest n (the camera-ready runs) |
| `sif.sh` | runs a command inside the Singularity image, or in the current environment without it |
| `sbatch/*.sbatch` | Slurm jobs for H100, RTX 4090, RTX 2080 Ti, CPU, and `last_interval_test.py` |
| `results/<tag>/<script>/` | output (not tracked): `raw.csv` (every repeat), `meta.json` (protocol, environment, stop points), `summary.json` (numbers quoted in the text), figures |

## Figures and scripts

Numbers and sections as in the camera-ready version.

| Figure / table | Section | Script | What it measures |
|---|---|---|---|
| Fig. 1, Fig. 15 | Sec. 1, App. E.1 | `fig01_15_runtime_memory.py` | runtime and peak memory vs n, B = 1 (LAPLEX, Dense, cuFFT) |
| Fig. 3 = Fig. 8 | Sec. 4.1, App. D.2 | `fig03_08_square.py` | X K(a, b) for a square kernel, B = 8 |
| Fig. 5 = Fig. 14(a), Fig. 14(b) | Sec. 6, App. E.1 | `fig05_14_fft.py` | LAPLEX vs Toeplitz-FFT vs cuFFT: runtime (a), accuracy (b) |
| Fig. 6 | Sec. 6 | `fig06_rff_value.py` | Random Fourier Features vs exact LAPLEX, n = k = 2^16, B = 1 |
| Fig. 9 | App. D.3 | `fig09_accuracy.py` | relative error of fp32 X K(a, b) against an fp64 dense reference |
| Fig. 10 | App. D.4 | `fig10_batch.py` | batch-size scaling at n = k = 2^14 |
| Fig. 11 | App. D.5 | `fig11_asymmetric.py` | asymmetric kernels, one axis fixed at 1024, B = 8 |
| Fig. 12 | App. D.6 | `fig12_gram.py` | weighted Gram matrix K(a, b) diag(D) K(a, b)^T |
| Fig. 13 | App. D.7 | `fig13_cpu_dense.py` (CPU) | pure-PyTorch LAPLEX (`laplex_noCUDA`) vs explicit dense |
| Fig. 16 | App. E.2 | `fig16_toeplitz_accuracy.py` | accuracy on a uniform-grid (Toeplitz) problem |
| Fig. 17 | App. E.3 | `fig17_cpu_fft.py` (CPU) | CPU runtime of LAPLEX (no CUDA), Toeplitz-FFT and one FFT |
| Fig. 18 | App. F.2 | `fig18_rff_gradient.py` | RFF gradient accuracy, n = k = 2^16 |
| Tab. 3 | App. D.1 | `table03_anchor_cost.py` | relative cost of anchor gradients, sorting and permutation |
| Tabs. 11, 12, 13 | App. K | `second_order/table11_13_second_order.py` | Fisher scoring with the exact LAPLEX curvature (n = 512, k = 50,000, fp64): convergence trace, iterations and time to tolerance vs Adam / GD, rank-r Nyström curvature |
| Fig. 22, Tab. 14 | App. K | `second_order/fig22_table14_second_order.py` | `--mode convergence`: Fisher scoring vs Adam (Fig. 22 left); `--mode scaling`: time and peak memory of one curvature build vs k, LAPLEX vs dense (Fig. 22 right, Tab. 14); `--mode correctness`: LAPLEX vs dense Fisher step |

The App. K scripts are standalone: they write `.png`/`.json`/`.md` with their own
plotting, not the paper's figure style, and they do not use the stopping rule above.
The dense curvature is attempted at every k and recorded as OOM where it does not fit.
Tab. 14 was measured on a 10 GB GPU.

The figure files are named after the paper: `fig03_08_square.py` writes
`fig03.*` and `fig08.*`, `fig05_14_fft.py` writes `fig05.*` and `fig14.*`.

## Environment

The environment of the main [README](../README.md) (an image built from
[../environment/laplex.def](../environment/laplex.def) or the conda environment) is
enough: Python, PyTorch, `numpy`, `pandas`, `matplotlib`, `ninja`, and `nvcc` with a
host compiler for the GPU scripts. The CUDA extensions are compiled by `load_inline` on first import
(~100 s) into `.cache/torch_extensions/sm<arch>-torch<ver>-cu<ver>/`, one directory
per GPU architecture. With `pdflatex` (and the `times`/`helvet` fonts) the figures
are typeset like the paper's (`.pgf` and `.pdf`); without it only an approximate
`.pdf` and the `.png` preview are written.

`sif.sh` runs a command in the Singularity image the paper's runs used,
`/shared/sets/singularity/python_image_2026-08-04.sif` (Python 3.14, torch 2.13 +
CUDA 13.2, nvcc, gcc 13, LaTeX; kernels for sm_75 … sm_120). `LAPLEX_SIF` selects
another image, e.g. one built from `../environment/laplex.def`. If the image or
`singularity`/`apptainer` is missing, `sif.sh` runs the command
in the current environment, so every command below also works without it
(`LAPLEX_SIF= bash sif.sh ...` forces that).

## Running

```bash
cd benchmarks
# development: warmup 2, iters 5, 1 repeat, 2 s dev time limit, results tagged <GPU>-quick
bash sif.sh python fig03_08_square.py --quick
# final protocol of the paper (App. D): 3 repeats x 30 timed calls after 10 warm-up calls
bash sif.sh python fig03_08_square.py
# all figures, one after another (GROUP=gpu|cpu, FIGS="fig03_08_square fig12_gram")
bash run_all.sh
# re-plot without a GPU, or overlay devices (Fig. 1 style: solid / dashed)
bash sif.sh python fig01_15_runtime_memory.py --stage plot --tags H100 RTX2080Ti
```

Common options of every script: `--stage run|plot|all`, `--min-log/--max-log`
(sweep 2^min … 2^max; defaults are the published ranges), `--repeats`,
`--warmup`, `--iters`, `--quick`, `--time-limit-ms` (development only),
`--tag`, `--tags`, `--seed`, and `--impl` for GPU scripts.

### Slurm

The sbatch files name the partitions of the cluster the paper's runs used
(dgxh100, rtx4090_batch, rtx2080, dgxa100); on another cluster pass your own,
e.g. `sbatch -p <partition> -q <qos> sbatch/h100.sbatch`. Submit from `benchmarks/`.

```bash
sbatch sbatch/h100.sbatch                     # all figures on one GPU, extra args go to every script
sbatch sbatch/rtx4090.sbatch                  # rtx4090_batch, QOS batch, excludes c22 (RTX 5090)
sbatch sbatch/rtx2080ti.sbatch                # szarancza
FIGS="fig12_gram" sbatch sbatch/h100.sbatch --max-log 31
FIGS=fig13_cpu_dense sbatch sbatch/cpu.sbatch --tag CPU-max --max-log 32
sbatch sbatch/last_interval.sbatch            # ~15 min, IMPLS="lib2026 pure" for a subset
```

### Up to the largest n (camera-ready runs)

```bash
bash submit_max_n.sh          # DRY=1 to print the sbatch commands, ONLY="fig03 fig12" for a subset
```

One Slurm job per experiment whose published sweep did not reach a limit
(16 jobs: 10 GPU figures/tables, 4 FFT/RFF figures again with `--impl fused`,
2 CPU figures), each on its own GPU (`--mem=64G --cpus-per-task=4`). Results go
to `results/H100-max`, `results/H100-max-fused` and `results/CPU-max`; Fig. 6
(RFF already stops at OOM) and the RTX 2080 Ti curves of Figs. 1/15 come from
runs of `sbatch/h100.sbatch` and `sbatch/rtx2080ti.sbatch` (tags `H100` and
`RTX2080Ti`).

## Protocol details

* **Timing.** Each point: `repeats` independent runs (repeat r uses seed r
  for the random inputs) of `iters` calls after `warmup` calls, with
  `cuda.synchronize()` around the timed region (App. D). Forward runs under
  `no_grad`; forward+backward builds fresh leaves for X, a, b, computes
  `L = sum(output)` and back-propagates (App. D). App. D figures plot the
  best repeat, Figs. 1/15 the mean ± std. CPU scripts keep the original CPU
  timing (3 warm-up, 10 calls, best of 3); Fig. 6 keeps the original RFF timing
  (one warm-up and one timed call per feature draw, mean over 10 draws).
* **Memory.** Peak allocated memory of one call = inputs + everything the
  call allocates on top of what was allocated before it. Persistent
  allocations from other calls (cuBLAS workspaces) are excluded; the raw
  `max_memory_allocated` and the baseline are also stored in `raw.csv`.
* **Stopping.** A point is kept only if all repeats succeed. The first failing
  size is stored with its status and the allocator message: `OOM`
  (allocator failure), `ERROR` (any other failure, e.g. a launch-configuration
  limit), `TIMEOUT` (dev time limit only), `LIMIT` (the next size exceeds the
  int32 element indexing of the custom kernels). If the inputs shared by all
  methods do not fit, every remaining method stops with `OOM`. On CPU the
  cgroup OOM killer would end the process, so the dense footprint is
  predicted against the job's memory limit (`common.cpu_guard`).
* **Busy GPU warning.** Every run logs `nvidia-smi` and warns if less than 95 %
  of the GPU memory is free at start: OOM points and timings are only
  meaningful on an otherwise idle GPU.

## LAPLEX implementation (`--impl`)

| value | code | used by default in |
|---|---|---|
| `fused` (**default**) | `vendor/fused/` (`dl-fused`, the code of the published Figs. 1/15 and Tab. 3) with one launch-layout change: scan grid `(tiles, B)` when there are more than 65535 tiles. Outputs equal the original up to its own run-to-run atomicAdd noise; reproduces the Fig. 1 H100 runtimes for n ≥ 2^21 to ~1 % | Figs. 1/15, 3/8, 9, 10, 11, 12, Tab. 3 |
| `public-fix` | `public-raw` semantics (dense when n·k ≤ 2^20, scan otherwise) on the `vendor/lib2026/` build, i.e. the released `laplex.py` without the n < 2^26 cap | Figs. 5/14, 6, 16, 18 (comparisons with FFT/RFF that the paper made with the released code) |
| `fused-orig` | the unmodified `vendor/fused_orig/kerlap_fused.py` | – (fails for n ≥ 2^26) |
| `lib2026` | `vendor/lib2026/cuda_kernel.py`, scan path forced, grid `(tiles, B)` | – |
| `public` | the released `comparison_experiments/laplex.py`, scan path forced | – (fails for n ≥ 2^26) |
| `public-raw` | as `public`, called exactly as the original scripts (hidden dense shortcut for n·k ≤ 2^20) | – |

CPU figures (13, 17) use `comparison_experiments/laplex_noCUDA.py`, as in the paper.
`impl_compare.py` also covers `lib2026-auto`, `merged`, `dl`, `dl-opt`, `mix-fused`,
`nocuda` (see the docstring of `impls.py`).

## Plot conventions

* Each reproduced figure takes the size, axes rectangles and font sizes of the
  published figure (`figure_geometry.json`), the seaborn "colorblind"
  palette in the same series order, the same markers, labels and legend
  placement, and LaTeX-typeset sans-serif text (matplotlib pgf backend with
  the paper's fonts, `\sfdefault = phv`). Files: `<stem>.pgf`, `<stem>.pdf`, and
  an Agg `<stem>.png` preview; `--paper` writes `<stem>_paper.{pgf,pdf}` at the
  published size with a y range fitted to the data.
* x axis: always base 2 (`2^k` labels). y axis: base 10.
* y axis: every panel keeps the published inches per decade; where the
  reproduction (or another version, `--ylim-json`) goes beyond the published
  range, all panels grow by the same height above or below it
  (`<stem>.layout.json` records the shift).
* Legends: as in the paper, with GPU names (`H100`, `RTX2080`), not results
  tags; added entries only for stop markers.
* Stop markers, on the last size measured (no vertical lines): red dot = the
  next size does not fit (OOM, "GPU Memory Limit"); open square = the next
  size exceeds the int32 element indexing of the custom kernels (`LIMIT`,
  "Implementation limit") or failed; open circle = accuracy figures, the
  fp64 reference does not fit at the next size, so no error can be computed
  (`REF_OOM`, "Reference OOM").

## Anchors beyond the ends of the sorted a (`last_interval_test.py`)

Thm. 1/4 (matvec) and Thm. 6 (weighted Gram) are stated for the intervals
P_1 … P_n; the test checks that the code also covers b_j ≥ a_n (P_{n+1}) and
b_j < a_1. Each implementation is compared with a float64 dense reference on
anchor sets with every b above max(a), below min(a), on both sides, exactly on
the end anchors, or inside [min(a), max(a)) as a control; values and the
gradients of the matvec are checked. Implementations: `lib2026`, `public`,
`fused`, `nocuda` from `impls.py` (scan path forced) and `pure` =
`vendor/lib2026/pure.py` (dense dispatch disabled), one per process.

```bash
bash sif.sh python last_interval_test.py --impl pure --device cpu
bash sif.sh python last_interval_test.py --impl lib2026          # needs a GPU
bash sif.sh python last_interval_test.py --summarize             # results/last_interval/summary.md
```

Output: `results/last_interval/<impl>_<device>.json` and `summary.md`.

* The verdict is `ok` when the error is at rounding level, `WRONG` when it points to
  missing contributions (> 1e-2 in fp32, > 1e-6 in fp64), and `imprecise` in between.
* Anchor gradients at exact ties a_i = b_j are reported but not judged: the kernel has
  a kink there. This covers ties by construction (case `end_ties`) and accidental ties
  after rounding to fp32.
* `fused` casts its inputs to float32 internally (`kerlap_fused`: `a32 = alpha.float()`),
  so its float64 rows are judged, and their ties counted, at float32 precision.

## Allocator setting

`sif.sh` and `common.init()` enable `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
Without it, dense fwd+bwd at n = 2^16 (B = 1, 64 GiB peak) fails on an 80 GB H100
because ~16 GiB stay cached in an unusable block; the stopping rule is about
fitting in device memory, not about the caching allocator.
