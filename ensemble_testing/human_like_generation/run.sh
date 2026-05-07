#!/bin/bash
# Quick runner for human-like text evaluation across multiple configurations

set -e

echo "========================================"
echo "Human-Like Text Detector Robustness Test"
echo "========================================"

# Default settings
DATASET=${1:-xsum}
BASE_MODEL=${2:-gpt2-medium}
N_SAMPLES=${3:-100}
BASELINES=${4:-"likelihood,logrank,LRR"}
OUTPUT_DIR="human_like_results"

mkdir -p $OUTPUT_DIR

echo ""
echo "Configuration:"
echo "  Dataset:         $DATASET"
echo "  Base Model:      $BASE_MODEL"
echo "  N Samples:       $N_SAMPLES"
echo "  Baselines:       $BASELINES"
echo "  Output Dir:      $OUTPUT_DIR"
echo ""

python generate_and_evaluate.py \
  --dataset "$DATASET" \
  --base_model_name "$BASE_MODEL" \
  --n_samples $N_SAMPLES \
  --baselines "$BASELINES" \
  --output_dir "$OUTPUT_DIR"

echo ""
echo "========================================"
echo "Complete! Results in $OUTPUT_DIR"
echo "========================================"
