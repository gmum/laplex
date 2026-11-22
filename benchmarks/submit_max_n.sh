#!/bin/bash
# Re-run, up to the largest n that fits, every benchmark whose published sweep ended before any
# limit, one Slurm job per experiment so that they run in parallel (each job has its own GPU).
# These runs produced the reproduced figures of the camera-ready version.
#
#   bash submit_max_n.sh               # everything below
#   DRY=1 bash submit_max_n.sh         # print the sbatch commands only
#   ONLY="fig01 fig05" bash submit_max_n.sh     # these jobs only (exact names below, e.g. fig05-fused)
#
# Results: results/H100-max (implementation of each script: fused, or public-fix for the FFT/RFF
# comparisons), results/H100-max-fused (FFT/RFF comparisons again with the default fused),
# results/CPU-max (CPU figures).  Fig. 6 (RFF already stops at OOM) and
# the RTX 2080 Ti curves of Figs. 1/15 (all series stop at OOM) are reused from results/H100 and
# results/RTX2080Ti.  Sizes are capped only by OOM, the int32 indexing of the custom kernels
# (status LIMIT) or the --max-log values below, which lie beyond both.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
# Partition / QOS / account for every job: SBATCH_* in ../.env (see ../.env.example).
if [[ -f ../.env ]]; then set -a; source ../.env; set +a; fi
mkdir -p logs
unset SLURM_JOB_ID SLURM_JOBID SLURM_NODELIST SLURM_JOB_NODELIST
GPU_RES=(--mem=64G --cpus-per-task=4)     # smaller than sbatch/h100.sbatch, so that several jobs share a node

# name | script | sbatch file | arguments
JOBS=(
  "fig01|fig01_15_runtime_memory|h100|--tag H100-max --max-log 34"
  "fig03|fig03_08_square|h100|--tag H100-max --max-log 32"
  "fig05|fig05_14_fft|h100|--tag H100-max --max-log 34 --acc-max-log 24"
  "fig09|fig09_accuracy|h100|--tag H100-max --max-log 24"
  "fig10|fig10_batch|h100|--tag H100-max --max-log 24"
  "fig11|fig11_asymmetric|h100|--tag H100-max --max-log 34"
  "fig12|fig12_gram|h100|--tag H100-max --max-log 34"
  "fig16|fig16_toeplitz_accuracy|h100|--tag H100-max --max-log 24"
  "fig18|fig18_rff_gradient|h100|--tag H100-max --max-log 24"
  "table03|table03_anchor_cost|h100|--tag H100-max --max-log 32"
  "fig05-fused|fig05_14_fft|h100|--tag H100-max-fused --impl fused --max-log 34 --acc-max-log 24"
  "fig06-fused|fig06_rff_value|h100|--tag H100-max-fused --impl fused --max-log 30"
  "fig16-fused|fig16_toeplitz_accuracy|h100|--tag H100-max-fused --impl fused --max-log 24"
  "fig18-fused|fig18_rff_gradient|h100|--tag H100-max-fused --impl fused --max-log 24"
  "fig13|fig13_cpu_dense|cpu|--tag CPU-max --max-log 32"
  "fig17|fig17_cpu_fft|cpu|--tag CPU-max --max-log 34"
)

submit() {
    if [[ -n "${DRY:-}" ]]; then echo "sbatch $*" >&2; echo "DRY$RANDOM"; else sbatch --parsable "$@"; fi
}

for j in "${JOBS[@]}"; do
    IFS='|' read -r name script kind args <<<"$j"
    if [[ -n "${ONLY:-}" && " $ONLY " != *" $name "* ]]; then continue; fi
    if [[ $kind == h100 ]]; then
        id=$(FIGS=$script submit "${GPU_RES[@]}" --job-name="max-$name" sbatch/h100.sbatch $args)
    else
        id=$(FIGS=$script submit --job-name="max-$name" sbatch/cpu.sbatch $args)
    fi
    echo "$name: $id"
done
echo "logs: $(pwd)/logs/   results: $(pwd)/results/"
