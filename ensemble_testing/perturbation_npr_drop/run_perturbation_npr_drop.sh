#!/bin/bash
# Run the NPR-drop perturbation study across xsum, squad, and writing.
# This produces per-dataset CSVs, a combined CSV, and plots.
#
# Usage:
#   ./run_perturbation_npr_drop.sh [BASE_MODEL] [N_SAMPLES] [OUTDIR] [--cache_dir CACHE_DIR]
# Example:
#   ./run_perturbation_npr_drop.sh gpt2-medium 50 perturbation_npr_drop_results --cache_dir ./hf_cache

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

BASE_MODEL="${1:-gpt2-medium}"
N_SAMPLES="${2:-50}"
OUTDIR="${3:-perturbation_npr_drop_results}"
CACHE_DIR=""

if [ "$#" -gt 3 ]; then
  shift 3
else
  set --
fi
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cache_dir)
      if [ -n "$2" ]; then
        CACHE_DIR="$2"
        shift 2
      else
        echo "ERROR: --cache_dir requires a value"
        exit 1
      fi
      ;;
    *)
      echo "ERROR: Unknown argument: $1"
      exit 1
      ;;
  esac
done

echo "========================================"
echo "NPR Drop Perturbation Study"
echo "========================================"
echo "  Base model: $BASE_MODEL"
echo "  N samples:  $N_SAMPLES"
echo "  Output dir: $OUTDIR"
if [ -n "$CACHE_DIR" ]; then
  echo "  Cache dir:  $CACHE_DIR"
fi

echo "Running xsum, squad, and writing together..."

PYTHON_BIN=""
if command -v python3 &> /dev/null; then
  PYTHON_BIN="python3"
elif command -v python &> /dev/null; then
  PYTHON_BIN="python"
else
  echo "ERROR: Python not found in system PATH"
  exit 1
fi

$PYTHON_BIN ensemble_testing/perturbation_npr_drop/run_perturbation_npr_drop.py \
  --datasets xsum,squad \
  --base_model_name "$BASE_MODEL" \
  --n_samples "$N_SAMPLES" \
  --output_dir "$OUTDIR" \
  ${CACHE_DIR:+--cache_dir "$CACHE_DIR"}

echo "Done. Results are under $OUTDIR/combined"
