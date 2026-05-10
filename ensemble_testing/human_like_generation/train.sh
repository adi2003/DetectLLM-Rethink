#!/bin/bash
# Training script for ensemble classifier
# Creates training dataset, trains model, and saves it for later use
# Usage:
#   ./train.sh [DATASET] [BASE_MODEL] [N_SAMPLES] [EPOCHS] [LEARNING_RATE] [MODEL_DIR] [--cache_dir CACHE_DIR]
# Example:
#   ./train.sh xsum gpt2-medium 200 100 0.001 ./ensemble_models --cache_dir ./hf_cache

set -e

# Navigate to repo root (2 levels up from this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# Defaults
DATASET="${1:-xsum}"
BASE_MODEL="${2:-gpt2-medium}"
N_SAMPLES="${3:-200}"
EPOCHS="${4:-500}"
LEARNING_RATE="${5:-0.001}"
MODEL_DIR="${6:-./ensemble_models}"
CACHE_DIR=""

# Parse optional arguments
if [ "$#" -gt 6 ]; then
  shift 6
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
echo "ENSEMBLE CLASSIFIER TRAINING"
echo "========================================"
echo ""
echo "Configuration:"
echo "  Dataset:         $DATASET"
echo "  Base Model:      $BASE_MODEL"
echo "  Training Samples: $N_SAMPLES"
echo "  Epochs:          $EPOCHS"
echo "  Learning Rate:   $LEARNING_RATE"
echo "  Model Directory: $MODEL_DIR"
if [ -n "$CACHE_DIR" ]; then
  echo "  Cache Dir:       $CACHE_DIR"
fi
echo ""

mkdir -p "$MODEL_DIR"
if [ -n "$CACHE_DIR" ]; then
  mkdir -p "$CACHE_DIR"
fi

# Determine Python executable path
if command -v python3 &> /dev/null; then
  PYTHON_BIN="python3"
elif command -v python &> /dev/null; then
  PYTHON_BIN="python"
else
  echo "ERROR: Python not found in system PATH"
  exit 1
fi

echo "Starting training..."
echo "----------------------------------------"

$PYTHON_BIN ensemble_testing/human_like_generation/train_classifier.py \
  --dataset "$DATASET" \
  --base_model_name "$BASE_MODEL" \
  --n_samples "$N_SAMPLES" \
  --epochs "$EPOCHS" \
  --learning_rate "$LEARNING_RATE" \
  --model_dir "$MODEL_DIR" \
  --cache_dir "$CACHE_DIR"

echo ""
echo "========================================"
echo "Training Complete!"
echo "========================================"
echo ""
echo "Model saved in: $MODEL_DIR"
echo "Model files:"
echo "  - ensemble_${DATASET}_${BASE_MODEL}.pt"
echo "  - ensemble_${DATASET}_${BASE_MODEL}_stats.pt"
echo ""
echo "Use with run.sh:"
echo "  ./run.sh <dataset> <model> <samples> <baselines> <output_dir> --model_path $MODEL_DIR"
echo "========================================"
