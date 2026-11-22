#!/bin/bash
# App. L, Tab. 15. One CUDA GPU.
#
#   bash table15_downstream/run.sh       (from log_likelihood/; IMNET = ImageNet-10 root)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."
if [[ -f "$HERE/../../.env" ]]; then set -a; source "$HERE/../../.env"; set +a; fi   # see ../../.env.example
IMNET=${IMNET:-../nn_training/data/imagenet10}
PY="bash sif.sh python3 -u"
$PY table15_downstream/density_models.py                # self-test against a dense Gaussian
$PY table15_downstream/table15_density_downstream.py --max_per_class 800 --epochs 20 \
    --lr 3e-3 --batch_size 32 --subsample 4000 --rank 3 --imagenet_root "$IMNET" \
    --outputs table15_downstream/results
