#!/bin/bash
# Run a command from the log_likelihood/ directory inside the Singularity image used for the
# paper (same image and GPU passthrough as ../benchmarks/sif.sh), plus a read-only bind of
# /shared/sets/datasets (ImageNet-10 and DIV2K can be linked from there) and a
# writable matplotlib cache.
# Without the image (or without singularity/apptainer) the command runs in the current
# Python environment instead.
#
#   bash sif.sh python density_models.py
#
# LAPLEX_SIF overrides the image; LAPLEX_SIF= (empty) always uses the current environment.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(dirname "$HERE")"
SIF="${LAPLEX_SIF-/shared/sets/singularity/python_image_2026-08-04.sif}"
DATASETS=/shared/sets/datasets
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$HERE/.cache/matplotlib"
mkdir -p "$MPLCONFIGDIR"
RUNTIME="$(command -v singularity || command -v apptainer || true)"
if [[ ! -f "$SIF" || -z "$RUNTIME" ]]; then
    cd "$HERE"
    exec "$@"
fi
BIND=(-B "$PROJECT")
[[ -d "$DATASETS" ]] && BIND+=(-B "$DATASETS:$DATASETS:ro")
exec "$RUNTIME" exec --nv "${BIND[@]}" --pwd "$HERE" "$SIF" "$@"
