#!/bin/bash
# App. I.2, Tabs. 9 and 10: every circulant and diagonal row. One CUDA GPU.
#
#   IMAGENETTE_ROOT=... DIV2K_ROOT=... bash table09_10_circulant/run.sh     (from log_likelihood/)
#
# Results: table09_10_circulant/results/; rows: python table09_10_circulant/print_tables.py
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."
PY="bash sif.sh python3 -u table09_10_circulant/_run_bccb.py"
R=table09_10_circulant/results
MIX="--lr 5e-4"

for order in raster permuted; do
  $PY --model full --order $order
  $PY --model mix --n-components 5 --order $order --warm-start $R/model_imagenette_full_$order.pt $MIX
  $PY --dataset div2k --model full --order $order
  $PY --dataset div2k --model mix --n-components 3 --order $order --warm-start $R/model_div2k_full_$order.pt $MIX
done
$PY --model diag --order raster
$PY --dataset div2k --model diag --order raster
