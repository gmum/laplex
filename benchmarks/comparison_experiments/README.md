

This repository contains everything needed to reproduce the benchmark, accuracy, and comparison experiments reported in the LAPLEX paper. It covers **kernel-algebra performance**, **Random Fourier Features (RFF) convergence**, and **FFT/Toeplitz-FFT** runtime and numerical accuracy analyses.

All scripts are self-contained, print benchmark tables to `stdout`, and automatically generate publication-ready `PDF`/`PNG` figures.

---



The following files are shared across all experiments and should be described only once:

| File | Purpose |
|:---|:---|
| `laplex.py` | CUDA-optimized kernel operations (fp32/fp64 auto-dispatch) |
| `laplex_phased.py` | Phased (single-tau) variant built on top of `laplex.py` |
| `laplex_noCUDA.py` | Portable pure-PyTorch reference (CPU / any backend) |
| `prefix_suffix.py` | Underlying decoupled-lookback CUDA scan primitive |
| `_common.py` | Shared benchmarking utilities: timing loops, memory tracking, plotting helpers, dense-baseline gating |
| `figures/` | Auto-created directory where all output plots are saved |

> ⚠️ **First Run Note:** On GPU experiments, importing `laplex` triggers `torch.utils.cpp_extension.load_inline`, which compiles two CUDA extensions. This takes ~30 seconds on the first run and is cached thereafter.

---



- **Python** 3.10+
- **PyTorch** 2.x (CUDA 12.x recommended for GPU experiments)
- **C++ Compiler** reachable by PyTorch (`MSVC` on Windows, `gcc` on Linux)
- **Packages:** `matplotlib`, `pandas`, `ninja`
- **Hardware:** 
  - GPU experiments require a CUDA-capable device (tested on RTX 2080/4090, scales to H100)
  - `exp6_cpu.py`, `exp8_cpu_fft.py`, and `exp_fft_combined.py` (CPU mode) run without a GPU

---



Run any experiment as a standard Python script. Execution time on an RTX 2080, 4090, H100.

```bash

python exp_fft_combined.py


python exp_rff_value.py
python exp_rff_gradient.py


python exp1_gpu_time_memory.py
python exp2_accuracy.py
python exp3_batch.py
python exp4_asymm.py
python exp5_gram.py


python exp6_cpu.py
python exp8_cpu_fft.py
python exp7_toeplitz_accuracy.py
```



| Script | Description |
|:---|:---|
| `exp_fft_combined.py` | Generates a **2-panel main-paper figure** for the FFT comparison. **Left panel:** GPU runtime (`LAPLEX`, `Toeplitz-FFT`, `cuFFT`) vs $n$ on irregular anchors ($B=1$). **Right panel:** Relative $\ell_2$ error on a uniform grid vs $n$, measured against an fp64 dense reference. Outputs to `figures/fft_combined/`. |
| `exp_rff_value.py` | Value accuracy & runtime: Compares LAPLEX (exact) vs RFF (stochastic) across $D \in \{1, 2, \dots, 2048\}$. Measures $\ell_2$ relative error and wall-clock time. Generates the Sec. 4.3 figure. |
| `exp_rff_gradient.py` | Gradient accuracy: Back-propagates through the RFF estimator and reports relative errors for $\nabla_a$ and $\nabla_b$ vs $D$. Demonstrates the unbounded variance of Cauchy-driven RFF gradients. Generates the App. C figure. |
| `exp1_gpu_time_memory.py` | Main GPU benchmark. Sweeps $n=k \in [2^{10}, 2^{20}]$ at $B=8$. Reports forward time, forward+backward time, and peak VRAM for LAPLEX vs explicit dense. |
| `exp2_accuracy.py` | Accuracy sweep. Compares LAPLEX fp32 and dense fp32 against an fp64 dense reference for relative $\ell_2$ and $\ell_\infty$ errors. |
| `exp3_batch.py` | Batch-size scaling. Fixes $n=k=2^{14}$, sweeps $B \in \{1, 2, \dots, 256\}$. Reports fwd/fwd+bwd times. Dense backward is skipped when graph retention exceeds VRAM. |
| `exp4_asymm.py` | Asymmetric kernels. Two sweeps with one axis pinned to 1024 while the other varies up to $2^{20}$. Tests both scan branches (sort+gather vs scatter+scan). |
| `exp5_gram.py` | Weighted Gram matrix. Fixes $n=1024$, sweeps inner dimension $k$. Measures time & memory for $K \operatorname{diag}(D) K^\top$. |
| `exp6_cpu.py` | Pure-PyTorch CPU reference. Runs `laplex_noCUDA` vs explicit dense on CPU ($B=1$, $n \in [2^{10}, 2^{14}]$). No CUDA toolchain required. |
| `exp7_toeplitz_accuracy.py` | Toeplitz accuracy (GPU). Compares LAPLEX, Toeplitz-FFT, and dense fp32 against fp64 ground truth on a uniform grid. Shows LAPLEX maintains flat $\mathcal{O}(\varepsilon)$ rounding. |
| `exp8_cpu_fft.py` | CPU runtime comparison. Runs `laplex_noCUDA`, Toeplitz-FFT, and a single `torch.fft.fft` call ($B=1$, $n \in [2^{10}, 2^{18}]$). Demonstrates hardware-agnostic $\mathcal{O}(n \log n)$ scaling. |

---



After execution, all plots are saved in `figures/` (or subdirectories like `figures/fft_combined/`):

```
figures/
├── exp1_gpu.{pdf,png}
├── exp2_accuracy.{pdf,png}
├── exp3_batch.{pdf,png}
├── exp4_asymm.{pdf,png}
├── exp5_gram.{pdf,png}
├── exp6_cpu.{pdf,png}
├── exp7_toeplitz_accuracy.{pdf,png}
├── exp8_cpu_fft.{pdf,png}
├── exp_rff_value.{pdf,png}
├── exp_rff_gradient.{pdf,png}
└── fft_combined/
    └── fft_combined.{pdf,png}
```

---



- **Dense Baseline Gating:** In `_common.py`, the dense baseline is automatically halted if it exceeds `500 ms` or `1.5 GB` VRAM to prevent OOM during long sweeps. On stronger GPUs (e.g., H100, 80 GB), you can increase these limits:
  ```python
  
  class DenseGate:
      def __init__(self, time_limit_ms=2000.0, mem_limit_gb=10.0):
  ```
- **Extending Size Ranges:** Edit the `LOG_NS_OURS` or `range(...)` lists at the top of each script to push sweeps beyond $2^{20}$ (e.g., to $2^{22}$ on H100).
- **Constant-Factor Expectations:** On CPU, pure-PyTorch LAPLEX is ~10× slower than highly optimized cuFFT due to single-threaded `torch.logcumsumexp`. On GPU, the gap shrinks to ~4× while maintaining exact gradient support and irregular anchor compatibility.

---



| Comparison | Result |
|:---|:---|
| **LAPLEX vs RFF (Value)** | RFF reaches `<10%` error only at $D \geq 1024$, where it becomes ~75× slower than LAPLEX. |
| **LAPLEX vs RFF (Gradient)** | RFF gradient error w.r.t. anchors does **not** concentrate as $D \to \infty$ (Cauchy frequency factor → infinite variance). Errors stay at 200–460%. |
| **LAPLEX vs FFT** | Same $\mathcal{O}(n \log n)$ complexity class. LAPLEX is slightly slower in raw runtime but offers **exact differentiability w.r.t. irregular anchors** and superior fp32 numerical stability on uniform grids. |
| **Accuracy** | LAPLEX maintains near machine-epsilon rounding independent of $n$, outperforming dense fp32 and slightly beating Toeplitz-FFT on uniform grids. |


