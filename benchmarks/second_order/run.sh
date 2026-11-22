#!/bin/bash
# App. K: second-order optimization with the exact LAPLEX curvature, with the settings of
# the paper.  Runs inside the Singularity image (../sif.sh), on one CUDA GPU.
#
#   bash second_order/run.sh            (from benchmarks/)
#
# Outputs: second_order/results/<mode>_<timestamp>/ and tables_<timestamp>/tables.md
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIF="bash $HERE/../sif.sh"
PY="python3 -u second_order"

$SIF $PY/fig22_table14_second_order.py --mode correctness                  # max. rel. error vs dense (text)
$SIF $PY/fig22_table14_second_order.py --mode convergence --n 512 --k 50000 # Fig. 22 left
$SIF $PY/fig22_table14_second_order.py --mode scaling --k_max_exp 25        # Fig. 22 right, Tab. 14
$SIF $PY/table11_13_second_order.py --n 512 --k 50000 --seeds 0 1 2         # Tabs. 11-13
