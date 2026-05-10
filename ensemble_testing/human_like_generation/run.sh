#!/bin/bash
# Global runner for human-like text evaluation (works on Kaggle, local, etc.)
# Usage:
#   ./run.sh [DATASET|DATASET1,DATASET2,...] [BASE_MODEL] [N_SAMPLES] [BASELINES] [OUTDIR] [--model_path MODEL_DIR]
# Example:
#   ./run.sh xsum gpt2-medium 50 "likelihood,logrank,LRR" human_like_results
#   ./run.sh xsum gpt2-medium 50 "likelihood,logrank,LRR,ensemble" human_like_results --model_path ./ensemble_models

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
MODEL_PATH=""

# Parse optional --model_path argument
if [ "$6" == "--model_path" ] && [ -n "$7" ]; then
  MODEL_PATH="$7"
fi

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
if [ -n "$MODEL_PATH" ]; then
  echo "  Model Path:      $MODEL_PATH"
fi
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

  CMD="$PYTHON_BIN ensemble_testing/human_like_generation/generate_and_evaluate.py \
    --dataset "$DATASET" \
    --base_model_name "$BASE_MODEL" \
    --n_samples "$N_SAMPLES" \
    --baselines "$BASELINES" \
    --output_dir "$DATASET_OUTDIR""

  if [ -n "$MODEL_PATH" ]; then
    CMD="$CMD --model_path "$MODEL_PATH""
  fi

  eval "$CMD"
done

echo ""
echo "========================================"
echo "Complete! Results in $OUTDIR"
echo "========================================"
