# Perturbation NPR Drop Study

This is a separate experiment tree for the perturbation follow-up.
It does not modify the current human-like generation experiment.

## What it does

The runner in this directory generates human text, normal LLM text, and human-like LLM text for:
- `xsum`
- `squad`
- `writing`

Then it computes NPR-style perturbation behavior and saves:
- per-example CSVs
- dataset-level summary CSVs
- a combined CSV across all datasets
- plots for NPR-drop comparisons

The key comparison is whether human-like LLM text is closer to human text than normal LLM text under perturbation.

## Main command

```bash
bash ensemble_testing/perturbation_npr_drop/run_perturbation_npr_drop.sh gpt2-medium 50 perturbation_npr_drop_results --cache_dir ./hf_cache
```

## Outputs

The script writes results to:
- `perturbation_npr_drop_results/<dataset>/feature_tables/`
- `perturbation_npr_drop_results/<dataset>/summaries/`
- `perturbation_npr_drop_results/<dataset>/plots/`
- `perturbation_npr_drop_results/combined/`

The combined folder includes:
- `all_datasets_<model>_feature_table.csv`
- `all_datasets_<model>_summary.csv`
- `all_datasets_<model>_comparison.csv`
- combined plots and a JSON report

## Requirements

You need:
- the base model name, such as `gpt2-medium`
- a cache directory, strongly recommended
- GPU support if you want perturbation to run reasonably fast

No ensemble model is needed for this perturbation study.
