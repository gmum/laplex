#!/bin/bash
# Run a command from the benchmarks/ directory inside the container image (environment/laplex.def),
# with GPU passthrough and the repository mounted.  Without an image (LAPLEX_SIF unset or empty in
# ../.env) or without apptainer/singularity, the command runs in the current Python environment.
#
#   bash sif.sh python fig03_08_square.py --quick
#   srun --jobid=<JOB> --overlap bash sif.sh python fig03_08_square.py --quick
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(dirname "$HERE")"
# Machine-specific settings (container image, dataset paths): ../.env, see ../.env.example
if [[ -f "$PROJECT/.env" ]]; then set -a; source "$PROJECT/.env"; set +a; fi
SIF="${LAPLEX_SIF:-}"
export PYTHONUNBUFFERED=1
# Stopping rule = "does it fit in device memory", not "does the caching allocator find a block";
# see common.py.  Must be set before torch initialises CUDA.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
RUNTIME="$(command -v apptainer || command -v singularity || true)"
if [[ -z "$SIF" || ! -f "$SIF" || -z "$RUNTIME" ]]; then
    [[ -n "$SIF" && ! -f "$SIF" ]] && echo "sif.sh: LAPLEX_SIF=$SIF not found, using the current environment" >&2
    cd "$HERE"
    exec "$@"
fi
# The repository, LAPLEX_BIND and the dataset directories of .env (read-only) are visible inside.
BIND=(-B "$PROJECT")
IFS=':' read -r -a EXTRA <<<"${LAPLEX_BIND:-}"
for d in "${EXTRA[@]}"; do [[ -n "$d" && -e "$d" ]] && BIND+=(-B "$d"); done
for d in "${IMAGENET_ROOT:-}" "${IMAGENET_VAL_CSV:+$(dirname "${IMAGENET_VAL_CSV}")}" "${IMNET:-}" \
         "${DATA_ROOT:-}" "${VAL_ROOT:-}" "${TV_DATA:-}" "${DIV2K_ROOT:-}" "${IMAGENETTE_ROOT:-}"; do
    [[ -n "$d" && -d "$d" ]] && BIND+=(-B "$d:$d:ro")
done
exec "$RUNTIME" exec --nv "${BIND[@]}" --pwd "$HERE" "$SIF" "$@"
