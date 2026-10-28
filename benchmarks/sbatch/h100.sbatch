#!/bin/bash
#SBATCH --job-name=laplex-repro-h100
#SBATCH --partition=dgxh100
#SBATCH --qos=quick
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.out
#
# All benchmark figures on one H100 (80 GB), final protocol, one script at a time.
# Submit from benchmarks/:
#   sbatch sbatch/h100.sbatch                                   # all figures
#   FIGS="fig03_08_square fig12_gram" sbatch sbatch/h100.sbatch # a subset
#   GROUP=gpu sbatch sbatch/h100.sbatch --max-log 21            # extra args go to every script
# QOS quick: <= 1 day, <= 10 CPUs, 1 GPU (dgxh100 also allows test/normal/big).
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[[ -f run_all.sh ]] || cd benchmarks
exec bash run_all.sh "$@"
