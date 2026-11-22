#!/bin/bash
#SBATCH --job-name=laplex-repro-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.out
#
# All benchmark figures on one H100 (80 GB), final protocol, one script at a time.
# Partition / QOS / account: set SBATCH_PARTITION, SBATCH_QOS, SBATCH_ACCOUNT in ../.env and export
# them before submitting (set -a; source ../.env; set +a), or pass sbatch -p <partition> -q <qos>.
# Request an H100 the way your cluster does (its partition, or e.g. --gres=gpu:h100:1).
# Submit from benchmarks/:
#   sbatch sbatch/h100.sbatch                                   # all figures
#   FIGS="fig03_08_square fig12_gram" sbatch sbatch/h100.sbatch # a subset
#   GROUP=gpu sbatch sbatch/h100.sbatch --max-log 21            # extra args go to every script
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[[ -f run_all.sh ]] || cd benchmarks
exec bash run_all.sh "$@"
