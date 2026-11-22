#!/bin/bash
#SBATCH --job-name=laplex-patch-lastint
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.out
#
# Thm. 1/4 and Thm. 6: do the implementations handle anchors b beyond the ends of the
# sorted a? One process per implementation (they import different prefix_suffix modules).
# ~10 min plus the first CUDA builds (~100 s each, cached in .cache/).
# Accuracy, not timing: any GPU works.
# Partition / QOS / account: set SBATCH_PARTITION, SBATCH_QOS, SBATCH_ACCOUNT in ../.env and export
# them before submitting (set -a; source ../.env; set +a), or pass sbatch -p <partition> -q <qos>.
# Submit from benchmarks/:
#   sbatch sbatch/last_interval.sbatch
#   IMPLS="lib2026 pure" sbatch sbatch/last_interval.sbatch
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[[ -f sif.sh ]] || cd benchmarks
nvidia-smi
IMPLS="${IMPLS:-lib2026 public fused nocuda pure}"
for impl in $IMPLS; do
    echo "=== $impl (cuda)"
    bash sif.sh python last_interval_test.py --impl "$impl" --device cuda || echo "[warn] $impl exited with $?"
done
echo "=== pure (cpu)"
bash sif.sh python last_interval_test.py --impl pure --device cpu || echo "[warn] pure/cpu exited with $?"
bash sif.sh python last_interval_test.py --summarize
