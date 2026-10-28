#!/bin/bash
#SBATCH --job-name=laplex-patch-parity
#SBATCH --partition=dgxa100
#SBATCH --qos=quick
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-9%5
#SBATCH --output=logs/slurm-%x-%A_%a.out
#SBATCH --error=logs/slurm-%x-%A_%a.out
#
# From-scratch re-run of the trainable columns of Table 1 / App. G (App. G:
# "An independent from-scratch re-run (5 fresh seeds) gives the same picture").
# Task i runs the released, unchanged script laplex.py (even i) or dense.py (odd i) with
# seed SEEDS[i / 2] and the settings of run_experiments.sh (batch 512, lr 1e-3, <= 100
# epochs, early-stopping patience 20) for m in DIMS. Up to ~7 h per task.
# Accuracy, not timing: any GPU works (default dgxa100; override with sbatch -p ... -q ...).
# Submit from nn_training/:
#   sbatch sbatch/parity.sbatch
#   SEEDS="2001 2002 2003 2004 2005" sbatch sbatch/parity.sbatch
# Statistics (results/parity/summary.md): submit_all.sh chains sbatch/parity_stats.sbatch
# after the array; for a standalone submission
#   jid=$(sbatch --parsable sbatch/parity.sbatch)
#   sbatch --dependency=afterany:$jid sbatch/parity_stats.sbatch
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[[ -f sif.sh ]] || cd nn_training
[[ -d data/imagenet10 ]] || python3 make_imagenet10.py
nvidia-smi
read -r -a SEED_LIST <<< "${SEEDS:-1001 1002 1003 1004 1005}"
METHODS=(laplex dense)
i="${SLURM_ARRAY_TASK_ID:-0}"
method="${METHODS[$((i % 2))]}"
seed="${SEED_LIST[$((i / 2))]}"
DIMS="${DIMS:-25 100 500 1000}"
echo "=== task $i: $method seed $seed, m in $DIMS"
bash sif.sh python CountSketch_LogisticRegression/"$method".py \
    --root_dataset data/imagenet10 --dims $DIMS --batch_size 512 --lr 1e-3 \
    --epochs 100 --patience 20 --num_workers "${SLURM_CPUS_PER_TASK:-8}" --save_model 0 \
    --seed "$seed" --outputs "results/parity/$method/seed$seed"
echo "done; when all tasks have finished: bash sif.sh python parity_stats.py"
