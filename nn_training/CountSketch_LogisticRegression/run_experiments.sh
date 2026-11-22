#!/bin/bash
# Table 1 (ImageNet-10): frozen projections + logistic regression, then end-to-end LAPLEX
# and dense projections, five repetitions.  Run from this directory:
#
#   bash run_experiments.sh
#
# ImageNet-10 root: IMNET in ../../.env (see ../../.env.example), default ../data/imagenet10.
# Results: $OUT (default results/) / experiments_{I,II,III}.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
if [[ -f ../../.env ]]; then set -a; source ../../.env; set +a; fi
ROOT=${IMNET:-../data/imagenet10}
OUT=${OUT:-results}

for i in {1..5}; do

python logistic_regression.py \
          --root_dataset "$ROOT" \
          --dims 25 50 100 250 500 750 1000 \
          --batch_size 512 \
          --iters 1000 \
          --num_workers 8 \
          --outputs "$OUT"/experiments_I

python laplex.py \
          --root_dataset "$ROOT" \
          --dims 25 50 100 250 500 750 1000 \
          --batch_size 512 \
          --lr 1.e-3 \
          --epochs 100 \
          --num_workers 8 \
          --save_model 1 \
          --outputs "$OUT"/experiments_II \
          --patience 20

python dense.py \
          --root_dataset "$ROOT" \
          --dims 25 50 100 250 500 750 1000 \
          --batch_size 512 \
          --lr 1.e-3 \
          --epochs 100 \
          --num_workers 8 \
          --save_model 1 \
          --outputs "$OUT"/experiments_III \
          --patience 20

done

