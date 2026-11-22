#!/bin/bash
#SBATCH --job-name=laplex-patch-crossing
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.out
#
# Anchor-crossing diagnostic with per-step traces (App. M, Fig. 23):
# trainable projection m = 500, released model and optimiser, with the released
# (augmented) training transform and without augmentation, then figures and summary.
# Data loading dominates (~1 min per epoch with 8 workers).  Accuracy, not timing: any GPU works.
# Partition / QOS / account: set SBATCH_PARTITION, SBATCH_QOS, SBATCH_ACCOUNT in ../.env and export
# them before submitting (set -a; source ../.env; set +a), or pass sbatch -p <partition> -q <qos>.
# Submit from nn_training/:
#   sbatch sbatch/crossing.sbatch
#   EPOCHS=30 SEEDS="0 1 2" VARIANTS=aug sbatch sbatch/crossing.sbatch
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[[ -f sif.sh ]] || cd nn_training
if [[ -f ../.env ]]; then set -a; source ../.env; set +a; fi
[[ -d "${IMNET:-data/imagenet10}" ]] || python3 make_imagenet10.py
nvidia-smi
VARIANTS="${VARIANTS:-aug noaug}"
SEEDS="${SEEDS:-0}"
EPOCHS="${EPOCHS:-6}"
for variant in $VARIANTS; do
    flag=""
    [[ "$variant" == noaug ]] && flag="--no-augment"
    for seed in $SEEDS; do
        echo "=== $variant seed $seed, $EPOCHS epochs"
        bash sif.sh python crossing_trace.py $flag --seed "$seed" --epochs "$EPOCHS" \
            --num-workers "${SLURM_CPUS_PER_TASK:-8}" || echo "[warn] $variant/$seed exited with $?"
    done
done
bash sif.sh python plot_crossing.py
