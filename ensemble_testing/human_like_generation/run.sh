#!/bin/bash
# Global runner for human-like text evaluation (works on Kaggle, local, etc.)
# Usage:
#   ./run.sh [DATASET|DATASET1,DATASET2,...] [BASE_MODEL] [N_SAMPLES] [BASELINES] [OUTDIR] [--model_path MODEL_DIR] [--cache_dir CACHE_DIR]
# Example:
#   ./run.sh xsum gpt2-medium 50 "likelihood,logrank,LRR" human_like_results
#   ./run.sh xsum gpt2-medium 50 "likelihood,logrank,LRR,ensemble" human_like_results --model_path ./ensemble_models --cache_dir ./hf_cache

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
CACHE_DIR=""

# Parse optional arguments
if [ "$#" -gt 5 ]; then
  shift 5
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
if [ -n "$CACHE_DIR" ]; then
  echo "  Cache Dir:       $CACHE_DIR"
fi
echo ""

mkdir -p "$OUTDIR"
if [ -n "$CACHE_DIR" ]; then
  mkdir -p "$CACHE_DIR"
fi

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
  if [ -n "$CACHE_DIR" ]; then
    CMD="$CMD --cache_dir "$CACHE_DIR""
  fi

  eval "$CMD"
done

echo ""
echo "========================================"
echo "Complete! Results in $OUTDIR"
echo "========================================"
