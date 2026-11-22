#!/bin/bash
#SBATCH --job-name=laplex-repro-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.out
#
# CPU benchmark figures (13, 17), CPU only (no GPU requested).  All CPU figures should run on
# the same CPU model.  The memory guard (common.cpu_guard) stops a series before it would
# exceed --mem.
# Partition / QOS / account: set SBATCH_PARTITION, SBATCH_QOS, SBATCH_ACCOUNT in ../.env and export
# them before submitting (set -a; source ../.env; set +a), or pass sbatch -p <partition> -q <qos>.
# Submit from benchmarks/:
#   FIGS=fig13_cpu_dense sbatch sbatch/cpu.sbatch --tag CPU-max --max-log 32
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[[ -f run_all.sh ]] || cd benchmarks
export GROUP=cpu
exec bash run_all.sh "$@"
