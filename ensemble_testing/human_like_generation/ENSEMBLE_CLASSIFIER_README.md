# Ensemble Classifier for Robust LLM Detection

This directory contains extensions to test whether combining multiple detection features can improve robustness against human-like text generation.

## Files

### New Files

1. **ensemble_classifier.py** - Binary classifier using PyTorch with sigmoid activation
   - `SigmoidEnsembleClassifier`: Neural network with 3 layers combining log likelihood, log rank, and entropy
   - `EnsembleTrainer`: Training and evaluation logic

2. **train_ensemble.py** - Main script for training and comparing detection methods
   - Generates human, normal LLM, and human-like LLM text
   - Extracts features: log likelihood, log rank, entropy
   - Trains the ensemble classifier
   - Evaluates and compares all methods

### Modified Files

- **generate_and_evaluate.py**: Added `entropy` to default baselines list

## Quick Start

### Step 1: Run baseline evaluation with entropy
```bash
python generate_and_evaluate.py \
  --dataset xsum \
  --base_model_name gpt2-medium \
  --n_samples 50 \
  --output_dir baseline_results
```

This will evaluate: likelihood, logrank, entropy, LRR, DetectGPT, NPR

### Step 2: Train and compare ensemble classifier
```bash
python train_ensemble.py \
  --dataset xsum \
  --base_model_name gpt2-medium \
  --n_samples 50 \
  --output_dir ensemble_results \
  --epochs 100
```

This will:
1. Generate all three types of text
2. Extract features for each
3. Train the ensemble classifier on combined features
4. Compare AUROC scores for:
   - Individual features (likelihood, logrank, entropy, LRR)
   - Ensemble classifier
   - Robustness on human-like text

## Output

Results are saved as JSON in the specified output directory:
- `baseline_results/comparison_*.json` - Individual feature scores from generate_and_evaluate.py
- `ensemble_results/ensemble_results_*.json` - Ensemble classifier comparison

## Key Results Reported

1. **Individual Feature Performance** (on normal LLM):
   - Log Likelihood AUROC
   - Log Rank AUROC
   - Entropy AUROC
   - LRR AUROC

2. **Ensemble Classifier Performance**:
   - vs Normal LLM
   - vs Human-like LLM
   - vs Combined (all LLM-generated)

3. **Robustness to Human-Like Text** (Human vs Human-like):
   - Log Likelihood AUROC
   - Log Rank AUROC
   - Entropy AUROC
   - LRR AUROC
   - Ensemble AUROC ← **Shows if ensemble is more robust**

## Architecture Details

The ensemble classifier:
- **Input**: [log_likelihood, log_rank, entropy] (normalized to [0,1])
- **Architecture**: Dense layer (3→64) → ReLU → Dense (64→32) → ReLU → Dense (32→1) → Sigmoid
- **Loss**: Binary Cross-Entropy
- **Optimizer**: Adam
- **Training Data**: Human text (label 0) + Normal LLM (label 1) + Human-like LLM (label 1)

## Expected Findings

If the ensemble approach works:
- Ensemble AUROC > individual feature AUROC
- Ensemble maintains high AUROC even on human-like text (showing robustness)
- Combined signal captures complementary information not available in individual features

## Customization

### Change perturbation model for DetectGPT/NPR
```bash
python train_ensemble.py --mask_filling_model_name t5-large
```

### Adjust training hyperparameters
```bash
python train_ensemble.py --epochs 200 --learning_rate 0.0001
```

### Test different dataset
```bash
python train_ensemble.py --dataset squad
```

### Use different LLM
```bash
python train_ensemble.py --base_model_name gpt2-large
```

## Dependencies

- torch
- numpy
- scikit-learn
- transformers
- datasets

All dependencies from the parent repo's requirements.txt
