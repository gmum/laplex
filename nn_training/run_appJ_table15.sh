#!/bin/bash
# App. J (attention) and App. M, Tab. 15 (anchor initialization, number of components),
# with the hyper-parameters of the paper.  Runs inside the Singularity image (sif.sh).
#
#   bash run_appJ_table15.sh                  list the sections
#   bash run_appJ_table15.sh <section> ...    run the given sections in order
#
# Environment (defaults in brackets):
#   TV_DATA   torchvision data: MNIST, cifar10/, SVHN/   [data/torchvision]
#   DOWNLOAD  1 = download them if missing               [0]
#   IMNET     ImageNet-10 (or a full ImageNet) root      [data/imagenet10]
#   ATT_DIMS  attention widths                           ["64 128 256"]
#   OUT       results root                               [results]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
TV_DATA=${TV_DATA:-data/torchvision}
IMNET=${IMNET:-data/imagenet10}
ATT_DIMS=${ATT_DIMS:-"64 128 256"}
OUT=${OUT:-results}
DL=(); [[ "${DOWNLOAD:-0}" == 1 ]] && DL=(--download)
PY="bash sif.sh python3 -u"

log() { echo "[$(date '+%F %T')] $*"; }

# App. J: 3 repeats x width x depth x dataset x attention operator, then the table.
# The 6 runs Structured / att_dim 256 / depth 8 on SVHN and CIFAR-10 of the paper were
# completed with --batch_size 32 on 10 GB GPUs (see README.md).
attention() {
  for rep in 1 2 3; do
    for dim in $ATT_DIMS; do
      for depth in 4 8; do
        for db in mnist svhn cifar10; do
          for model in Softmax RFF RFF_tau Structured; do
            log "attention rep=$rep dim=$dim depth=$depth $db $model"
            $PY appJ_attention.py --dataset "$db" --model "$model" \
              --epochs 200 --batch_size 128 --lr 0.01 --weight_decay 1e-4 \
              --workers 8 --scheduler_factor 0.7 --scheduler_patience 3 \
              --init_tau 0.54 --att_dim "$dim" --att_depth "$depth" \
              --early_stop_patience 25 --data_path "$TV_DATA" "${DL[@]}" \
              --outputs "$OUT/attention"
          done
        done
      done
    done
  done
  $PY appJ_attention_table.py --root "$OUT/attention" | tee "$OUT/attention/table.txt"
}

# Tab. 15 (left): seven anchor initializations x 5 seeds, cached non-augmented features
anchor_init() {
  $PY table15_anchor_ablations.py --mode init_sweep --root_dataset "$IMNET" \
    --m 500 --batch_size 256 --lr 1e-3 --epochs 30 --patience 5 --init_tau 0.1 \
    --seeds 1 2 3 4 5 --features_on_gpu --outputs "$OUT/anchor_dynamics"
}

# Tab. 15 (right): I = 1..4 components x 3 seeds
anchor_components() {
  $PY table15_anchor_ablations.py --mode components --root_dataset "$IMNET" \
    --m 500 --batch_size 128 --lr 1e-3 --epochs 30 --patience 5 --init_tau 0.1 \
    --seeds 1 2 3 --components 1 2 3 4 --features_on_gpu --outputs "$OUT/anchor_dynamics"
}

SECTIONS=(attention anchor_init anchor_components)
if [ $# -eq 0 ]; then
  echo "usage: $0 <section> ..."; printf '  %s\n' "${SECTIONS[@]}"; exit 0
fi
for s in "$@"; do
  printf '%s\n' "${SECTIONS[@]}" | grep -qx "$s" || { echo "unknown section: $s"; exit 1; }
done
mkdir -p "$OUT"
for s in "$@"; do
  log "=== $s -> $OUT"
  "$s" 2>&1 | tee -a "$OUT/$s.log"
done
