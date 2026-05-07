# Human-Like Text Generation & Detector Evaluation

This directory tests whether the original DetectLLM signals hold when the LLM is instructed to generate "human-like" text.

## Motivation

The paper's detectors are trained to separate human-written and normally-generated machine text. But what if we prompt the LLM to deliberately *sound human*? Do the signals still separate?

This is the first check in Aditya's proposed workflow:
1. Generate human text + LLM-generated human-like text
2. Run original detectors
3. If signals hold → test across datasets/models/prompts
4. If signals break → ensemble detectors together

## What the script does

`generate_and_evaluate.py`:
- Loads human text from standard datasets (xsum, squad, writing, etc.)
- Generates text in two modes:
  - **Normal**: Standard generation from the base model (original paper approach)
  - **Human-like**: Generation with instruction to "write naturally and sound human"
- Runs all detectors (likelihood, logrank, LRR, DetectGPT, NPR, etc.) on both
- Compares AUROC scores to see if separation degrades
- Saves detailed results as JSON

## Usage

```bash
# Basic usage with xsum dataset
python generate_and_evaluate.py \
  --dataset xsum \
  --base_model_name gpt2-medium \
  --n_samples 100

# With perturbation-based detectors
python generate_and_evaluate.py \
  --dataset xsum \
  --base_model_name gpt2-medium \
  --baselines likelihood,logrank,LRR,DetectGPT,NPR \
  --n_perturbation_list "5,10" \
  --n_samples 100

# Multiple datasets for robustness
python generate_and_evaluate.py --dataset xsum --n_samples 100
python generate_and_evaluate.py --dataset squad --n_samples 100
```

## Output

Results are saved as JSON in `human_like_results/`:
- `comparison_[dataset]_[model].json` contains:
  - Per-detector AUROC for both normal and human-like generation
  - Delta AUROC (change from normal to human-like)
  - Robustness flag (True if |delta| < 5 percentage points)
  - Summary: how many detectors are "robust" to the human-like prompt

## Expected Interpretation

- **Robust (Δ < 0.05)**: Detector still separates normal LLM from human-like LLM text
  - Signals likely universal, not just exploiting normal generation quirks
- **Breaks (|Δ| ≥ 0.05)**: Detector fails when LLM is instructed to sound human
  - Signal is brittle; ensemble may help

## Next Steps (If Signals Hold)

1. **Length-stratified analysis**: Compare by text length bins
2. **Cross-model**: Test if detector trained on model A works on model B  
3. **Cross-prompt**: Try different "make it human" prompts
4. **Decoding settings**: Test temperature, top-p, top-k variations
5. **Ensemble**: Combine signals that all hold

## Notes

- Requires the main DetectLLM code in parent directory
- GPU recommended for perturbation-based methods (DetectGPT, NPR)
- Instruction prompt can be customized in `HumanLikeTextGenerator._generate()`
