#!/bin/bash
# Run a command from the benchmarks/ directory inside the Singularity image used for the
# paper, with GPU passthrough and the repository mounted.  Without the image (or without
# singularity/apptainer) the command runs in the current Python environment instead.
#
#   bash sif.sh python fig03_08_square.py --quick
#   srun --jobid=<JOB> --overlap bash sif.sh python fig03_08_square.py --quick
#
# LAPLEX_SIF overrides the image; LAPLEX_SIF= (empty) always uses the current environment.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(dirname "$HERE")"
SIF="${LAPLEX_SIF-/shared/sets/singularity/python_image_2026-08-04.sif}"
export PYTHONUNBUFFERED=1
# Stopping rule = "does it fit in device memory", not "does the caching allocator find a block";
# see common.py.  Must be set before torch initialises CUDA.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
RUNTIME="$(command -v singularity || command -v apptainer || true)"
if [[ ! -f "$SIF" || -z "$RUNTIME" ]]; then
    cd "$HERE"
    exec "$@"
fi
exec "$RUNTIME" exec --nv -B "$PROJECT" --pwd "$HERE" "$SIF" "$@"
