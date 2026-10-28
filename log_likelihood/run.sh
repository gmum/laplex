#!/bin/bash
# App. I.2 (Tab. 9 and the raster-order circulant values in its text) and App. L (Tab. 14).
# Runs inside the Singularity image (sif.sh), on one CUDA GPU.
#
#   bash run.sh
#
# Environment (defaults in brackets):
#   IMNET   ImageNet-10 (or a full ImageNet) root, train/ and val/ by synset
#           [../nn_training/data/imagenet10]
#   DIV2K   DIV2K root (DIV2K_train_HR/, DIV2K_valid_HR/)  [data/DIV2K]
#   OUT     results root                              [results/density]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
IMNET=${IMNET:-../nn_training/data/imagenet10}
DIV2K=${DIV2K:-data/DIV2K}
OUT=${OUT:-results/density}
PY="bash sif.sh python3 -u"
IMNETTE=(--dataset imagenette --max_per_class 800 --epochs 20 --lr 3e-3 --batch_size 32
         --subsample 4000 --imagenet_root "$IMNET" --outputs "$OUT")
DIV=(--dataset div2k --epochs 20 --lr 2e-3 --batch_size 8 --subsample 200 --eval_bs 4
     --div2k_root "$DIV2K" --outputs "$OUT")

$PY density_models.py                                   # self-test against a dense Gaussian
# Tab. 14, plus the raster-order circulant models quoted in App. I.2
$PY table09_14_density.py "${IMNETTE[@]}" --orders raster \
  --models LAPLEX dense-LR PPCA Kronecker FFT-circulant circulant-K1 circulant-mix
# Tab. 9: circulant (K budget-matched, K = 1) and diagonal rows, permuted coordinates
$PY table09_14_density.py "${IMNETTE[@]}" --orders permuted --models circulant-mix circulant-K1 diagonal
$PY table09_14_density.py "${DIV[@]}" --orders permuted --models circulant-mix circulant-K1 diagonal
# App. I.2 text: raster-order circulant on DIV2K
$PY table09_14_density.py "${DIV[@]}" --orders raster --models circulant-mix circulant-K1
