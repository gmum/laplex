#!/bin/bash
# Run the figure scripts one after another (never two benchmarks at once), inside
# the Singularity image.  Extra arguments are passed to every script.
#
#   bash run_all.sh                       # all GPU figures, then the CPU figures
#   GROUP=gpu bash run_all.sh --quick     # only GPU figures, development preset
#   FIGS="fig03_08_square fig12_gram" bash run_all.sh --tag H100
#
# GROUP: all | gpu | cpu.  FIGS overrides the list.  Logs: logs/<tag-or-job>/<script>.log
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

GPU_FIGS="fig01_15_runtime_memory fig03_08_square fig05_14_fft fig06_rff_value fig09_accuracy
fig10_batch fig11_asymmetric fig12_gram fig16_toeplitz_accuracy fig18_rff_gradient table03_anchor_cost"
CPU_FIGS="fig13_cpu_dense fig17_cpu_fft"
case "${GROUP:-all}" in
    gpu) DEFAULT="$GPU_FIGS" ;;
    cpu) DEFAULT="$CPU_FIGS" ;;
    *)   DEFAULT="$GPU_FIGS $CPU_FIGS" ;;
esac
FIGS="${FIGS:-$DEFAULT}"

LOGDIR="logs/${SLURM_JOB_ID:-local}-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$LOGDIR"
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} args=$* logs=$LOGDIR"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>/dev/null || true

status=0
for f in $FIGS; do
    echo "=== $(date +%H:%M:%S) $f"
    if bash "$HERE/sif.sh" python "$f.py" "$@" > "$LOGDIR/$f.log" 2>&1; then
        grep -E "stopped at|saved" "$LOGDIR/$f.log" | cut -c1-160
    else
        status=1
        echo "FAILED (exit $?), last lines of $LOGDIR/$f.log:"
        tail -5 "$LOGDIR/$f.log"
    fi
done
echo "=== $(date +%H:%M:%S) done, status=$status"
exit $status
