#!/bin/bash
#SBATCH --job-name=laplex-repro-2080ti
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.out
#
# All benchmark figures on one RTX 2080 Ti (11 GB, sm_75).  The paper's Figs. 1/15 used an
# RTX 2080 (8 GB); results are tagged RTX2080Ti, not RTX2080.
# Partition / QOS / account: set SBATCH_PARTITION, SBATCH_QOS, SBATCH_ACCOUNT in ../.env and export
# them before submitting (set -a; source ../.env; set +a), or pass sbatch -p <partition> -q <qos>.
# Submit from benchmarks/:   sbatch sbatch/rtx2080ti.sbatch [args for every script]
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[[ -f run_all.sh ]] || cd benchmarks
exec bash run_all.sh "$@"
