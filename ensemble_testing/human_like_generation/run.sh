#!/bin/bash
# Global runner for human-like text evaluation (works on Kaggle, local, etc.)
# Usage:
#   ./run.sh [DATASET|DATASET1,DATASET2,...] [BASE_MODEL] [N_SAMPLES] [BASELINES] [OUTDIR]
# Example:
#   ./run.sh xsum gpt2-medium 50 "likelihood,logrank,LRR" human_like_results

set -e

# Navigate to repo root (2 levels up from this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# Defaults (can be overridden by positional args)
DATASETS="${1:-xsum}"
BASE_MODEL="${2:-gpt2-medium}"
N_SAMPLES="${3:-50}"
BASELINES="${4:-likelihood,logrank,LRR}"
OUTDIR="${5:-human_like_results}"

echo "========================================"
echo "Human-Like Text Detector Robustness Test"
echo "========================================"
echo ""
echo "Configuration:"
echo "  Datasets:        $DATASETS"
echo "  Base Model:      $BASE_MODEL"
echo "  N Samples:       $N_SAMPLES"
echo "  Baselines:       $BASELINES"
echo "  Output Dir:      $OUTDIR"
echo ""

mkdir -p "$OUTDIR"

IFS=',' read -r -a DATASET_ARRAY <<< "$DATASETS"

# Determine Python executable path (system or venv)
if command -v python3 &> /dev/null; then
  PYTHON_BIN="python3"
elif command -v python &> /dev/null; then
  PYTHON_BIN="python"
else
  echo "ERROR: Python not found in system PATH"
  exit 1
fi

for DATASET in "${DATASET_ARRAY[@]}"; do
  DATASET="$(echo "$DATASET" | xargs)"
  if [ -z "$DATASET" ]; then
    continue
  fi

  DATASET_OUTDIR="$OUTDIR/$DATASET"
  mkdir -p "$DATASET_OUTDIR"

  echo ""
  echo "----------------------------------------"
  echo "Running dataset: $DATASET"
  echo "----------------------------------------"

  $PYTHON_BIN ensemble_testing/human_like_generation/generate_and_evaluate.py \
    --dataset "$DATASET" \
    --base_model_name "$BASE_MODEL" \
    --n_samples "$N_SAMPLES" \
    --baselines "$BASELINES" \
    --output_dir "$DATASET_OUTDIR"
done

echo ""
echo "========================================"
echo "Complete! Results in $OUTDIR"
echo "========================================"
