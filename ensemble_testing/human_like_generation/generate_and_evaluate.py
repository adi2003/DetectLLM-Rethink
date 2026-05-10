
"""
Generate human-like text vs normal generation and evaluate detector signals.

This script:
1. Loads human text from standard datasets (xsum, squad, writing)
2. Generates text in two modes:
   - Normal: standard generation from the base model
   - Human-like: generation with instruction to make it sound human
3. Runs the original paper's detectors on both
4. Compares if the separation signals still hold
5. Saves per-example scores and aggregate metrics for analysis
"""

import argparse
import os
import json
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

# Add parent directory to path to use existing baselines
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from baselines.utils.preprocessing import preprocess_and_save
from baselines.utils.loadmodel import load_base_model_and_tokenizer, load_mask_filling_model
from baselines.sample_generate.generate import generate_data, generate_samples
from baselines.all_baselines import run_all_baselines
from baselines.utils.run_baseline import get_roc_metrics, get_precision_recall_metrics
from baselines.loss import get_ll
from baselines.rank import get_rank
from baselines.entropy import get_entropy
from ensemble_classifier import EnsembleTrainer
import torch
import datasets


class HumanLikeTextGenerator:
    """Generate text with instruction to sound human-like."""
    
    def __init__(self, model_config, args):
        self.model_config = model_config
        self.args = args
        self.base_model = model_config['base_model']
        self.base_tokenizer = model_config['base_tokenizer']
    
    def generate_normal(self, texts: List[str]) -> List[str]:
        """Generate normally from prompts."""
        return self._generate(texts, instruction=None)
    
    def generate_human_like(self, texts: List[str]) -> List[str]:
        """Generate with instruction to sound human-like."""
        instruction = "Write the following in a natural, human-like tone as if written by a person. Avoid robotic language and make it sound conversational:"
        new_instruction = "Rewrite the following to sound like it was written naturally by an intelligent person in casual real-world communication. Vary sentence length and structure, avoid overly polished or generic phrasing, remove robotic transitions, and keep the tone authentic, fluid, and slightly imperfect where appropriate:"
        return self._generate(texts, instruction=new_instruction)
    
    def _generate(self, texts: List[str], instruction: str = None) -> List[str]:
        """Generate text from prompts, optionally with instruction."""
        torch.manual_seed(42)
        np.random.seed(42)
        
        prompt_tokens = self.args.prompt_len
        DEVICE = self.args.DEVICE
        batch_size = max(1, int(self.args.batch_size))
        
        # If instruction provided, prepend it to each text
        if instruction:
            texts = [f"{instruction}\n{t}" for t in texts]

        decoded_all = []
        min_words = self.args.min_words

        # Process generation in mini-batches to avoid CUDA OOM on large n_samples.
        for start in range(0, len(texts), batch_size):
            end = min(start + batch_size, len(texts))
            batch_texts = texts[start:end]

            encoded = self.base_tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=prompt_tokens,
            ).to(DEVICE)

            decoded = ['' for _ in range(len(batch_texts))]

            # Generate until we have minimum word count
            tries = 0
            while (m := min(len(x.split()) for x in decoded)) < min_words:
                if tries != 0:
                    print(f"  batch {start}:{end} min words: {m}, needed {min_words}, regenerating (try {tries})")

                sampling_kwargs = {}
                if self.args.do_top_p:
                    sampling_kwargs['top_p'] = self.args.top_p
                elif self.args.do_top_k:
                    sampling_kwargs['top_k'] = self.args.top_k

                # Use new-token-based generation to avoid input vs max_length conflicts
                min_new = max(1, int(self.args.min_len))
                max_new = max(1, int(self.args.generation_len))
                outputs = self.base_model.generate(
                    **encoded,
                    min_new_tokens=min_new,
                    max_new_tokens=max_new,
                    temperature=self.args.temperature,
                    do_sample=True,
                    **sampling_kwargs,
                    pad_token_id=self.base_tokenizer.eos_token_id,
                    eos_token_id=self.base_tokenizer.eos_token_id
                )
                decoded = self.base_tokenizer.batch_decode(outputs, skip_special_tokens=True)

                # Remove instruction prefix if present
                if instruction:
                    decoded = [text.replace(instruction + "\n", "") for text in decoded]

                tries += 1
                if tries > 3:
                    break

            decoded_all.extend(decoded)

            if DEVICE.startswith("cuda"):
                del encoded
                if 'outputs' in locals():
                    del outputs
                torch.cuda.empty_cache()

        return decoded_all


def evaluate_detector_separation(real_scores: List[float], fake_scores: List[float], detector_name: str) -> Dict:
    """Compute AUROC and PR AUC for a detector."""
    fpr, tpr, roc_auc = get_roc_metrics(real_scores, fake_scores)
    precision, recall, pr_auc = get_precision_recall_metrics(real_scores, fake_scores)
    
    return {
        'detector': detector_name,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'n_samples': len(real_scores)
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate human-like text and evaluate detector robustness"
    )
    parser.add_argument('--dataset', type=str, default="xsum", help="Dataset to use")
    parser.add_argument('--dataset_key', type=str, default="document")
    parser.add_argument('--base_model_name', type=str, default="gpt2-medium")
    parser.add_argument('--mask_filling_model_name', type=str, default="t5-large")
    parser.add_argument('--n_samples', type=int, default=100, help="Number of examples to test")
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--prompt_len', type=int, default=30)
    parser.add_argument('--generation_len', type=int, default=200)
    parser.add_argument('--min_words', type=int, default=55)
    parser.add_argument('--min_len', type=int, default=150)
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--do_top_k', action='store_true')
    parser.add_argument('--top_k', type=int, default=40)
    parser.add_argument('--do_top_p', action='store_true')
    parser.add_argument('--top_p', type=float, default=0.96)
    parser.add_argument('--DEVICE', type=str, default='cuda')
    parser.add_argument('--cache_dir', type=str, default="")
    parser.add_argument('--output_dir', type=str, default="human_like_results")
    parser.add_argument('--baselines', type=str, default="likelihood,logrank,entropy,LRR,DetectGPT,NPR",
                       help="Comma-separated list of baselines to evaluate")
    parser.add_argument('--n_perturbation_list', type=str, default="5")
    parser.add_argument('--pct_words_masked', type=float, default=0.3)
    parser.add_argument('--span_length', type=int, default=2)
    parser.add_argument('--mask_top_p', type=float, default=1.0)
    parser.add_argument('--chunk_size', type=int, default=32, help='Chunk size for perturbation batching')
    parser.add_argument('--int8', action='store_true')
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--base_half', action='store_true')
    parser.add_argument('--buffer_size', type=int, default=1)
    parser.add_argument('--model_path', type=str, default=None, help="Path to saved ensemble model directory (for 'ensemble' baseline)")
    
    args = parser.parse_args()
    baselines = [x.strip() for x in args.baselines.split(',') if x.strip()]
    needs_mask_model = any(baseline in {'DetectGPT', 'NPR'} for baseline in baselines)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print("Human-Like Text Generation and Detector Signal Analysis")
    print("=" * 80)
    
    # Load models
    print(f"\nLoading base model: {args.base_model_name}")
    model_config = {'cache_dir': args.cache_dir}
    model_config = load_base_model_and_tokenizer(args, model_config)
    if needs_mask_model:
        model_config = load_mask_filling_model(args, args.mask_filling_model_name, model_config)
    
    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    if args.dataset == 'xsum':
        dataset = datasets.load_dataset('xsum', split='train', cache_dir=args.cache_dir)['document']
    elif args.dataset == 'squad':
        dataset = datasets.load_dataset('squad', split='train', cache_dir=args.cache_dir)['context']
    elif args.dataset == 'writing':
        # Simplified version, in real code would use custom_datasets
        dataset = datasets.load_dataset('xsum', split='train', cache_dir=args.cache_dir)['document']
    else:
        dataset = datasets.load_dataset(args.dataset, split='train', cache_dir=args.cache_dir)[args.dataset_key]
    
    # Filter and prepare dataset
    dataset = list(dict.fromkeys(dataset))  # Remove duplicates
    dataset = [x.strip() for x in dataset]
    dataset = [' '.join(x.split()) for x in dataset]  # Remove newlines
    
    # Truncate texts to fit within model's max sequence length
    # gpt2-medium has max_position_embeddings=1024, so we truncate at ~900 tokens to be safe
    max_tokens = 950
    tokenizer = model_config['base_tokenizer']
    
    def truncate_text(text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit."""
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_tokens,
            return_attention_mask=False,
        )
        return tokenizer.decode(encoded["input_ids"], skip_special_tokens=True)
    
    dataset = [truncate_text(x, max_tokens) for x in dataset]
    
    # Keep only long examples
    if args.dataset in ['writing', 'squad', 'xsum']:
        long_data = [x for x in dataset if len(x.split()) > 250]
        if len(long_data) > 0:
            dataset = long_data
    
    import random
    random.seed(0)
    random.shuffle(dataset)
    dataset = dataset[:args.n_samples]
    
    print(f"Using {len(dataset)} examples")
    
    # Initialize text generator
    generator = HumanLikeTextGenerator(model_config, args)
    
    # Generate texts in both modes
    print("\n" + "-" * 80)
    print("Generating normal text...")
    print("-" * 80)
    normal_generated = generator.generate_normal(dataset)
    
    print("\n" + "-" * 80)
    print("Generating human-like text...")
    print("-" * 80)
    human_like_generated = generator.generate_human_like(dataset)
    
    # Create data dicts for detector evaluation
    data_normal = {
        'original': dataset,
        'sampled': normal_generated
    }
    
    data_human_like = {
        'original': dataset,
        'sampled': human_like_generated
    }

    print(data_normal["original"][0])
    print(data_normal["sampled"][0])
    print(data_human_like["original"][0])
    print(data_human_like["sampled"][0])
    # print("hello")
    # print(data_normal)
    # print(type(data_normal["original"]))
    # print("hello")

    # if (data_normal["original"] and data_normal["sampled"] and
    #         data_human_like["original"] and data_human_like["sampled"]):
    #     print(data_normal["original"][0], flush=True)
    #     print(data_normal["sampled"][0], flush=True)
    #     print(data_human_like["original"][0], flush=True)
    #     print(data_human_like["sampled"][0], flush=True)
    # else:
    #     print(
    #         "Debug: one or more sample lists are empty.",
    #         flush=True
    #     )
    #     print(
    #         f"  normal original={len(data_normal['original'])}, "
    #         f"normal sampled={len(data_normal['sampled'])}, "
    #         f"human original={len(data_human_like['original'])}, "
    #         f"human sampled={len(data_human_like['sampled'])}",
    #         flush=True
    #     )
    
    # Parse args for run_all_baselines
    n_perturbation_list = [int(x) for x in args.n_perturbation_list.split(",")]
    baselines = [x.strip() for x in args.baselines.split(',')]
    
    # Prepare ensemble detector if requested and model exists
    ensemble_scores_normal = None
    ensemble_scores_human_like = None
    
    if 'ensemble' in baselines and args.model_path:
        print("\n" + "=" * 80)
        print("LOADING ENSEMBLE MODEL")
        print("=" * 80)
        try:
            trainer = EnsembleTrainer(device=args.DEVICE)
            base_model_clean = args.base_model_name.replace('/', '_')
            model_path = os.path.join(args.model_path, f"ensemble_{args.dataset}_{base_model_clean}.pt")
            stats_path = os.path.join(args.model_path, f"ensemble_{args.dataset}_{base_model_clean}_stats.pt")
            
            trainer.load(model_path, stats_path, device=args.DEVICE)
            
            # Extract features and evaluate ensemble
            print("\nExtracting features for ensemble evaluation...")
            
            # Normal generation
            print("  Normal generation...")
            normal_ll = []
            normal_lr = []
            normal_ent = []
            for text in data_normal['sampled']:
                try:
                    ll = get_ll(text, args, model_config)
                    lr = get_rank(text, args, model_config, log=True)
                    ent = get_entropy(text, args, model_config)
                    normal_ll.append(ll)
                    normal_lr.append(lr)
                    normal_ent.append(ent)
                except:
                    normal_ll.append(np.nan)
                    normal_lr.append(np.nan)
                    normal_ent.append(np.nan)
            
            # Human-like generation
            print("  Human-like generation...")
            human_like_ll = []
            human_like_lr = []
            human_like_ent = []
            for text in data_human_like['sampled']:
                try:
                    ll = get_ll(text, args, model_config)
                    lr = get_rank(text, args, model_config, log=True)
                    ent = get_entropy(text, args, model_config)
                    human_like_ll.append(ll)
                    human_like_lr.append(lr)
                    human_like_ent.append(ent)
                except:
                    human_like_ll.append(np.nan)
                    human_like_lr.append(np.nan)
                    human_like_ent.append(np.nan)
            
            # Get predictions
            X_normal, _ = trainer.prepare_features(normal_ll, normal_lr, normal_ent)
            X_human_like, _ = trainer.prepare_features(human_like_ll, human_like_lr, human_like_ent)
            
            ensemble_scores_normal = trainer.predict(X_normal)
            ensemble_scores_human_like = trainer.predict(X_human_like)
            
            print("Ensemble model loaded and evaluated successfully!")
        except Exception as e:
            print(f"Error loading ensemble model: {e}")
            baselines.remove('ensemble')
    elif 'ensemble' in baselines and not args.model_path:
        print("\nWarning: 'ensemble' baseline requested but --model_path not provided. Skipping ensemble.")
        baselines.remove('ensemble')
    
    # Run detectors on both
    print("\n" + "=" * 80)
    print(f"EVALUATING NORMAL GENERATION")
    print("=" * 80)
    results_normal = run_all_baselines(
        data_normal, args, n_perturbation_list, model_config, baselines=[b for b in baselines if b != 'ensemble']
    )
    
    # Add ensemble results if available
    if ensemble_scores_normal is not None:
        fpr, tpr, ensemble_auc = get_roc_metrics([0] * len(data_normal['original']), ensemble_scores_normal)
        results_normal.append({'name': 'ensemble_threshold', 'roc_auc': ensemble_auc})
    
    print("\n" + "=" * 80)
    print(f"EVALUATING HUMAN-LIKE GENERATION")
    print("=" * 80)
    results_human_like = run_all_baselines(
        data_human_like, args, n_perturbation_list, model_config, baselines=[b for b in baselines if b != 'ensemble']
    )
    
    # Add ensemble results if available
    if ensemble_scores_human_like is not None:
        fpr, tpr, ensemble_auc = get_roc_metrics([0] * len(data_human_like['original']), ensemble_scores_human_like)
        results_human_like.append({'name': 'ensemble_threshold', 'roc_auc': ensemble_auc})
    
    # Compare results
    print("\n" + "=" * 80)
    print("COMPARISON: NORMAL vs HUMAN-LIKE")
    print("=" * 80)
    
    comparison_results = {
        'dataset': args.dataset,
        'base_model': args.base_model_name,
        'n_samples': len(dataset),
        'prompt_len': args.prompt_len,
        'baselines_compared': baselines,
        'normal_generation': results_normal,
        'human_like_generation': results_human_like,
        'comparison_summary': []
    }
    
    # Match results by baseline name
    results_map_normal = {r['name']: r for r in results_normal}
    results_map_human = {r['name']: r for r in results_human_like}
    
    for baseline_name in baselines:
        matching_normal = [r for r in results_normal if r['name'].startswith(baseline_name)]
        matching_human = [r for r in results_human_like if r['name'].startswith(baseline_name)]
        
        if matching_normal and matching_human:
            for norm, human in zip(matching_normal, matching_human):
                auc_normal = norm.get('roc_auc', 0)
                auc_human = human.get('roc_auc', 0)
                delta = auc_human - auc_normal
                
                comparison_results['comparison_summary'].append({
                    'detector': norm['name'],
                    'normal_roc_auc': auc_normal,
                    'human_like_roc_auc': auc_human,
                    'delta_roc_auc': delta,
                    'robust': abs(delta) < 0.05  # Threshold: within 5% points
                })
                
                print(f"\n{norm['name']}:")
                print(f"  Normal:      AUROC = {auc_normal:.4f}")
                print(f"  Human-like:  AUROC = {auc_human:.4f}")
                print(f"  Delta:       {delta:+.4f} ({delta*100:+.2f}%)")
                print(f"  Robust:      {'✓ YES' if abs(delta) < 0.05 else '✗ NO'}")
    
    # Save results
    output_file = os.path.join(args.output_dir, f"comparison_{args.dataset}_{args.base_model_name}.json")
    with open(output_file, 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    print(f"\n\nResults saved to: {output_file}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    robust_count = sum(1 for c in comparison_results['comparison_summary'] if c['robust'])
    total_count = len(comparison_results['comparison_summary'])
    print(f"Robust detectors (Δ < 0.05): {robust_count}/{total_count}")
    
    if robust_count == total_count:
        print("\n✓ PASS: Original paper signals hold for human-like generation!")
    else:
        print("\n✗ FAIL: Some detectors break on human-like text.")


if __name__ == '__main__':
    main()
