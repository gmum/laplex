#!/bin/bash
#SBATCH --job-name=laplex-patch-parity-stats
#SBATCH --partition=cpu
#SBATCH --qos=quick
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.out
#
# Statistics of the parity re-run (results/parity/summary.md). submit_all.sh submits it
# with --dependency=afterany:<parity array>, so it runs once every task has ended, failed
# ones included (parity_stats.py reports missing seeds). No GPU needed.
# For an array that is already queued, submit from nn_training/:
#   sbatch --dependency=afterany:<ARRAY_JOB_ID> sbatch/parity_stats.sbatch
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[[ -f sif.sh ]] || cd nn_training
bash sif.sh python parity_stats.py
