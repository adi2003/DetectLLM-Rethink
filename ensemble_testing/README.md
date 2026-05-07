# Ensemble Detector & Hypothesis Testing Workflow

This directory implements the ensemble detector and statistical testing strategy outlined in the context.md comments.

## Workflow Overview

Aditya's proposed approach:

```
1. Generate human-like text and test detector robustness
   ↓ (if signals hold) → 2. Test across datasets/models/prompts
   ↓ (if signals break) → 4. Combine detectors into ensemble
   ↓
3. Run hypothesis tests to quantify statistical significance
   ↓
4. Validate ensemble generalization
```

## Directory Structure

```
ensemble_testing/
├── human_like_generation/          ← Step 1: Test if signals hold
│   ├── generate_and_evaluate.py     # Main script
│   ├── run.sh                       # Easy runner
│   └── README.md                    # Detailed docs
│
├── hypothesis_testing/              ← Step 3: Statistical tests (coming)
│   ├── bootstrap_auroc.py           # Bootstrap CI for AUROC
│   ├── permutation_tests.py         # Permutation tests for detector pairs
│   └── README.md
│
├── ensemble_methods/                ← Step 4: Ensemble models (coming)
│   ├── weighted_ensemble.py         # Simple weighted averaging
│   ├── meta_classifier.py           # Logistic regression ensemble
│   └── README.md
│
└── README.md                        ← This file
```

## Step 1: Human-Like Text Generation (Current)

**File**: `human_like_generation/generate_and_evaluate.py`

**What it does**:
1. Loads human text from a dataset
2. Generates text in two modes:
   - Normal: standard LLM generation (baseline)
   - Human-like: LLM instructed to "write naturally and sound human"
3. Runs all detectors on both
4. Compares AUROC scores
5. Checks if signals are robust (Δ AUROC < 5%)

**Example run**:
```bash
cd human_like_generation
python generate_and_evaluate.py --dataset xsum --base_model_name gpt2-medium --n_samples 100
```

**Expected output**:
- JSON file with per-detector AUROC comparison
- Summary: how many detectors are "robust" to human-like prompt

**Decision point**:
- ✓ All detectors robust → Signals are universal, proceed to Step 2 (cross-dataset/model testing)
- ✗ Some break → Signals are brittle, consider ensemble in Step 4

## Step 2: Cross-Validation (Not Yet Implemented)

Test robustness across:
- Different datasets (xsum, squad, writing, etc.)
- Different base models (gpt2-medium, gpt2-large, etc.)
- Different "human-like" prompts
- Different decoding settings (temperature, top-p, top-k)
- Length-stratified evaluation

## Step 3: Hypothesis Testing (Not Yet Implemented)

Once signals hold across conditions, quantify statistical significance:

**Bootstrap confidence intervals for AUROC**:
- Resample with replacement and recompute AUROC
- Report 95% CI
- Check if CI excludes 0.5 (chance performance)

**Permutation tests for detector pairs**:
- Null hypothesis: detector A and B are equally good
- Compute difference in AUROC
- Shuffle labels and recompute
- Calculate p-value

**Multiple comparison correction**:
- If testing many baselines, apply Bonferroni or FDR correction
- Prevents false positives from repeated testing

## Step 4: Ensemble Methods (Not Yet Implemented)

If Step 1 shows signals are brittle:

**Weighted ensemble**:
- Linear combination of detector scores
- Weights learned from validation set

**Meta-classifier**:
- Logistic regression on top of individual detector scores
- Train on one dataset, test on another

**Goal**: Combine weak signals into a stronger one

## Key Questions Answered

✓ **Do original signals hold for human-like text?** 
   → Test in Step 1

✓ **Are signals universal across datasets/models?**
   → Test in Step 2

✓ **Are observed gaps statistically significant?**
   → Test in Step 3

✓ **Can we ensemble brittle signals into a robust one?**
   → Test in Step 4

## Files to Create Next

1. `hypothesis_testing/bootstrap_auroc.py`
   - Bootstrap confidence intervals for AUROC
   - Permutation tests
   - Multiple comparison correction

2. `hypothesis_testing/analysis.py`
   - Read results from Step 1
   - Run bootstrap tests
   - Generate summary tables

3. `ensemble_methods/weighted_ensemble.py`
   - Learn weights from validation set
   - Evaluate on test set

4. `ensemble_methods/meta_classifier.py`
   - Train logistic regression
   - Cross-dataset evaluation

## Running the Full Workflow

```bash
# Step 1: Test human-like robustness
cd human_like_generation
python generate_and_evaluate.py --dataset xsum --n_samples 200

# Check results
cat human_like_results/comparison_xsum_gpt2-medium.json

# If all signals robust → proceed to Step 3
# If some break → prepare for Step 4 ensemble

# Step 3: Statistical testing (coming)
cd ../hypothesis_testing
python bootstrap_auroc.py --results ../human_like_generation/human_like_results

# Step 4: Ensemble (coming)
cd ../ensemble_methods
python weighted_ensemble.py --normal_results ... --human_like_results ...
```

## Configuration & Customization

**Instruction prompt** (in `human_like_generation/generate_and_evaluate.py`):
```python
instruction = "Write the following in a natural, human-like tone as if written by a person..."
```

**Robustness threshold** (default 0.05 = 5 percentage points):
```python
robust = abs(delta) < 0.05
```

Both can be modified for different definitions of "robust."

---

**Status**: Step 1 (human-like generation) ready. Steps 2-4 pending.

**Next**: Run Step 1 on xsum/squad/writing with gpt2-medium and gpt2-large, check if results hold.
