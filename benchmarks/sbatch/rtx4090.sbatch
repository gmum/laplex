#!/bin/bash
#SBATCH --job-name=laplex-repro-4090
#SBATCH --partition=rtx4090_batch
#SBATCH --qos=batch
#SBATCH --exclude=c22
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.out
#
# All benchmark figures on one RTX 4090 (24 GB).  c22 is excluded because it has
# RTX 5090 cards.  rtx4090_batch accepts only sbatch jobs with QOS batch (<= 1 day).
# Submit from benchmarks/:   sbatch sbatch/rtx4090.sbatch [args for every script]
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[[ -f run_all.sh ]] || cd benchmarks
exec bash run_all.sh "$@"
