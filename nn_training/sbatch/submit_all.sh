#!/bin/bash
# Build the ImageNet10 links if needed and submit all jobs: crossing, the parity array
# and its statistics (afterany dependency on the array). Extra arguments go to the two
# GPU jobs and override their #SBATCH defaults; the statistics job always runs on the
# cpu partition. Examples:
#   bash sbatch/submit_all.sh
#   bash sbatch/submit_all.sh -p rtx4090_batch -q batch --exclude=c22
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs
[[ -d data/imagenet10 ]] || python3 make_imagenet10.py
j_cross=$(sbatch --parsable "$@" sbatch/crossing.sbatch)
j_parity=$(sbatch --parsable "$@" sbatch/parity.sbatch)
j_stats=$(sbatch --parsable --dependency=afterany:"$j_parity" sbatch/parity_stats.sbatch)
cat <<EOF
submitted from $(pwd):
  crossing       $j_cross    -> results/crossing/summary.md, results/crossing/*/fig_crossing.pdf
  parity (array) $j_parity    -> results/parity/<method>/seed<S>/
  parity stats   $j_stats    -> results/parity/summary.md (starts after every task of $j_parity)
EOF
