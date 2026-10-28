#!/bin/bash
#SBATCH --job-name=laplex-repro-cpu
#SBATCH --partition=rtx4090_batch
#SBATCH --qos=batch
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.out
#
# CPU benchmark figures (13, 17), CPU only (no GPU requested) on the rtx4090_batch nodes: the cpu
# partition (one node, trzmiel) is usually full.  All CPU figures run here, so they share one CPU
# model.  qos batch: not preemptible, at most 1 day.  Submit from benchmarks/:
#   FIGS=fig13_cpu_dense sbatch sbatch/cpu.sbatch --tag CPU-max --max-log 32
# The memory guard (common.cpu_guard) stops a series before it would exceed --mem.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[[ -f run_all.sh ]] || cd benchmarks
export GROUP=cpu
exec bash run_all.sh "$@"
