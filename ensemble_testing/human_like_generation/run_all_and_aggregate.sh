#!/bin/bash
# Run the human-like experiment across xsum, squad, and writing,
# then aggregate results into merged CSVs and plots.
#
# Usage:
#   ./run_all_and_aggregate.sh [BASE_MODEL] [N_SAMPLES] [BASELINES] [OUTDIR] [--model_path MODEL_DIR] [--cache_dir CACHE_DIR]
# Example:
#   ./run_all_and_aggregate.sh gpt2-medium 50 "likelihood,logrank,LRR,DetectGPT,NPR" human_like_results --cache_dir ./hf_cache

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

BASE_MODEL="${1:-gpt2-medium}"
N_SAMPLES="${2:-50}"
BASELINES="${3:-likelihood,logrank,LRR,DetectGPT,NPR}"
OUTDIR="${4:-human_like_results}"
MODEL_PATH=""
CACHE_DIR=""

if [ "$#" -gt 4 ]; then
  shift 4
else
  set --
fi
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_path)
      if [ -n "$2" ]; then
        MODEL_PATH="$2"
        shift 2
      else
        echo "ERROR: --model_path requires a value"
        exit 1
      fi
      ;;
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

echo "Running datasets xsum,squad,writing and aggregating outputs..."
./ensemble_testing/human_like_generation/run.sh \
  xsum,squad,writing \
  "$BASE_MODEL" \
  "$N_SAMPLES" \
  "$BASELINES" \
  "$OUTDIR" \
  ${MODEL_PATH:+--model_path "$MODEL_PATH"} \
  ${CACHE_DIR:+--cache_dir "$CACHE_DIR"}

PYTHON_BIN=""
if command -v python3 &> /dev/null; then
  PYTHON_BIN="python3"
elif command -v python &> /dev/null; then
  PYTHON_BIN="python"
else
  echo "ERROR: Python not found in system PATH"
  exit 1
fi

echo "Aggregating final CSVs and plots..."
$PYTHON_BIN ensemble_testing/human_like_generation/aggregate_human_like_results.py \
  --input_root "$OUTDIR" \
  --datasets xsum,squad,writing \
  --base_model_name "$BASE_MODEL" \
  --output_dir "$OUTDIR/combined"

echo "Done. Aggregated outputs are in $OUTDIR/combined"
