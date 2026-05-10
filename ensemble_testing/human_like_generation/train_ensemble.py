"""
Train ensemble classifier and compare robustness against individual features.

This script:
1. Generates human, normal LLM, and human-like LLM text
2. Computes features: log likelihood, log rank, entropy
3. Trains a sigmoid-based ensemble classifier on combined features
4. Compares AUROC of individual features vs ensemble
"""

import argparse
import os
import json
import sys
from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from baselines.utils.loadmodel import load_base_model_and_tokenizer, load_mask_filling_model
from baselines.all_baselines import run_all_baselines
from baselines.loss import get_ll
from baselines.rank import get_rank
from baselines.entropy import get_entropy
from baselines.utils.run_baseline import get_roc_metrics, get_precision_recall_metrics
from ensemble_classifier import EnsembleTrainer

import torch
import datasets
import random


def extract_features(texts: List[str], 
                    original_texts: List[str],
                    args, 
                    model_config) -> Tuple[List[float], List[float], List[float]]:
    """Extract log likelihood, log rank, and entropy for texts."""
    log_likelihoods = []
    log_ranks = []
    entropies = []
    
    print("  Extracting features...")
    for idx, text in enumerate(texts):
        if idx % max(1, len(texts) // 10) == 0:
            print(f"    {idx}/{len(texts)}")
        
        try:
            ll = get_ll(text, args, model_config)
            lr = get_rank(text, args, model_config, log=True)
            ent = get_entropy(text, args, model_config)
            
            log_likelihoods.append(ll)
            log_ranks.append(lr)
            entropies.append(ent)
        except Exception as e:
            print(f"    Error extracting features for text {idx}: {e}")
            # Use NaN for failed extractions
            log_likelihoods.append(np.nan)
            log_ranks.append(np.nan)
            entropies.append(np.nan)
    
    return log_likelihoods, log_ranks, entropies


def load_and_prepare_dataset(args, model_config) -> List[str]:
    """Load and prepare dataset."""
    print(f"Loading dataset: {args.dataset}")
    if args.dataset == 'xsum':
        dataset = datasets.load_dataset('xsum', split='train', cache_dir=args.cache_dir)['document']
    elif args.dataset == 'squad':
        dataset = datasets.load_dataset('squad', split='train', cache_dir=args.cache_dir)['context']
    else:
        dataset = datasets.load_dataset(args.dataset, split='train', cache_dir=args.cache_dir)[args.dataset_key]
    
    # Filter and prepare
    dataset = list(dict.fromkeys(dataset))
    dataset = [x.strip() for x in dataset]
    dataset = [' '.join(x.split()) for x in dataset]
    
    # Truncate to fit within model limits
    max_tokens = 900
    tokenizer = model_config['base_tokenizer']
    
    def truncate_text(text: str, max_tokens: int) -> str:
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
    
    random.seed(0)
    random.shuffle(dataset)
    dataset = dataset[:args.n_samples]
    
    print(f"Using {len(dataset)} examples")
    return dataset


def generate_texts_simple(texts: List[str], model_config, args, instruction: str = None) -> List[str]:
    """Simple text generation without perturbation tracking."""
    torch.manual_seed(42)
    np.random.seed(42)
    
    prompt_tokens = args.prompt_len
    DEVICE = args.DEVICE
    batch_size = max(1, int(args.batch_size))
    
    if instruction:
        texts = [f"{instruction}\n{t}" for t in texts]
    
    decoded_all = []
    min_words = args.min_words
    
    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        batch_texts = texts[start:end]
        
        encoded = model_config['base_tokenizer'](
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=prompt_tokens,
        ).to(DEVICE)
        
        decoded = ['' for _ in range(len(batch_texts))]
        tries = 0
        
        while (m := min(len(x.split()) for x in decoded)) < min_words:
            if tries != 0:
                print(f"    Batch {start}:{end}, min words: {m}, regenerating (try {tries})")
            
            sampling_kwargs = {}
            if args.do_top_p:
                sampling_kwargs['top_p'] = args.top_p
            elif args.do_top_k:
                sampling_kwargs['top_k'] = args.top_k
            
            min_new = max(1, int(args.min_len))
            max_new = max(1, int(args.generation_len))
            
            outputs = model_config['base_model'].generate(
                **encoded,
                min_new_tokens=min_new,
                max_new_tokens=max_new,
                temperature=args.temperature,
                do_sample=True,
                **sampling_kwargs,
                pad_token_id=model_config['base_tokenizer'].eos_token_id,
                eos_token_id=model_config['base_tokenizer'].eos_token_id
            )
            decoded = model_config['base_tokenizer'].batch_decode(outputs, skip_special_tokens=True)
            
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


def main():
    parser = argparse.ArgumentParser(
        description="Train ensemble classifier and compare detection robustness"
    )
    parser.add_argument('--dataset', type=str, default="xsum")
    parser.add_argument('--dataset_key', type=str, default="document")
    parser.add_argument('--base_model_name', type=str, default="gpt2-medium")
    parser.add_argument('--mask_filling_model_name', type=str, default="t5-small")
    parser.add_argument('--n_samples', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--prompt_len', type=int, default=30)
    parser.add_argument('--generation_len', type=int, default=200)
    parser.add_argument('--min_words', type=int, default=55)
    parser.add_argument('--min_len', type=int, default=150)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--do_top_k', action='store_true')
    parser.add_argument('--top_k', type=int, default=40)
    parser.add_argument('--do_top_p', action='store_true')
    parser.add_argument('--top_p', type=float, default=0.96)
    parser.add_argument('--DEVICE', type=str, default='cuda')
    parser.add_argument('--cache_dir', type=str, default="")
    parser.add_argument('--output_dir', type=str, default="ensemble_results")
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=500)
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print("ENSEMBLE CLASSIFIER TRAINING AND ROBUSTNESS COMPARISON")
    print("=" * 80)
    
    # Load models
    print(f"\nLoading base model: {args.base_model_name}")
    model_config = {'cache_dir': args.cache_dir}
    model_config = load_base_model_and_tokenizer(args, model_config)
    model_config = load_mask_filling_model(args, args.mask_filling_model_name, model_config)
    
    # Load and prepare dataset
    dataset = load_and_prepare_dataset(args, model_config)
    
    # Generate texts in all three modes
    print("\n" + "-" * 80)
    print("Generating human text (using dataset as-is)...")
    print("-" * 80)
    human_texts = dataset
    
    print("\n" + "-" * 80)
    print("Generating normal LLM text...")
    print("-" * 80)
    normal_generated = generate_texts_simple(dataset, model_config, args, instruction=None)
    
    print("\n" + "-" * 80)
    print("Generating human-like LLM text...")
    print("-" * 80)
    instruction = "Write the following in a natural, human-like tone as if written by a person. Avoid robotic language and make it sound conversational:"
    human_like_generated = generate_texts_simple(dataset, model_config, args, instruction=instruction)
    
    # Extract features for all three types
    print("\n" + "=" * 80)
    print("EXTRACTING FEATURES")
    print("=" * 80)
    
    print("\nExtracting features from human text...")
    human_ll, human_lr, human_ent = extract_features(human_texts, human_texts, args, model_config)
    
    print("\nExtracting features from normal LLM text...")
    normal_ll, normal_lr, normal_ent = extract_features(normal_generated, dataset, args, model_config)
    
    print("\nExtracting features from human-like LLM text...")
    human_like_ll, human_like_lr, human_like_ent = extract_features(human_like_generated, dataset, args, model_config)
    
    # Remove NaN values
    def filter_valid(ll, lr, ent):
        valid_idx = ~(np.isnan(ll) | np.isnan(lr) | np.isnan(ent))
        return (
            np.array(ll)[valid_idx].tolist(),
            np.array(lr)[valid_idx].tolist(),
            np.array(ent)[valid_idx].tolist()
        )
    
    human_ll, human_lr, human_ent = filter_valid(human_ll, human_lr, human_ent)
    normal_ll, normal_lr, normal_ent = filter_valid(normal_ll, normal_lr, normal_ent)
    human_like_ll, human_like_lr, human_like_ent = filter_valid(human_like_ll, human_like_lr, human_like_ent)
    
    print(f"  Human: {len(human_ll)} valid samples")
    print(f"  Normal LLM: {len(normal_ll)} valid samples")
    print(f"  Human-like LLM: {len(human_like_ll)} valid samples")
    
    # Prepare training data: human (0) vs LLM-generated (1)
    # Combine normal and human-like as LLM-generated (both should be label 1)
    train_ll = human_ll + normal_ll + human_like_ll
    train_lr = human_lr + normal_lr + human_like_lr
    train_labels = [0] * len(human_ll) + [1] * (len(normal_ll) + len(human_like_ll))
    
    print(f"\nTraining data: {len(train_ll)} samples ({sum(train_labels)} LLM, {len(train_labels) - sum(train_labels)} human)")
    
    # Shuffle training data before training to avoid ordering bias
    print("Shuffling training data...")
    indices = np.arange(len(train_ll))
    np.random.seed(42)
    np.random.shuffle(indices)
    
    train_ll = [train_ll[i] for i in indices]
    train_lr = [train_lr[i] for i in indices]
    train_labels = [train_labels[i] for i in indices]
    
    # Prepare test sets for individual evaluations
    test_human_ll = human_ll
    test_human_lr = human_lr
    test_human_ent = human_ent
    test_human_labels = [0] * len(human_ll)
    
    test_normal_ll = normal_ll
    test_normal_lr = normal_lr
    test_normal_ent = normal_ent
    test_normal_labels = [1] * len(normal_ll)
    
    test_human_like_ll = human_like_ll
    test_human_like_lr = human_like_lr
    test_human_like_ent = human_like_ent
    test_human_like_labels = [1] * len(human_like_ll)
    
    # Train ensemble classifier
    print("\n" + "=" * 80)
    print("TRAINING ENSEMBLE CLASSIFIER")
    print("=" * 80)
    trainer = EnsembleTrainer(learning_rate=args.learning_rate, epochs=args.epochs)
    X_train, y_train = trainer.prepare_features(train_ll, train_lr, train_labels, fit_normalization=True)
    trainer.train(X_train, y_train)
    print(f"Learned fusion function: {trainer.get_fusion_formula()}")
    
    # Save training plot
    print("\n" + "=" * 80)
    print("SAVING TRAINING STATS")
    print("=" * 80)
    base_model_clean = args.base_model_name.replace('/', '_')
    plot_path = os.path.join(args.output_dir, f"training_loss_{args.dataset}_{base_model_clean}.png")
    trainer.plot_training_stats(plot_path)
    
    # Evaluate on all three test sets
    print("\n" + "=" * 80)
    print("EVALUATING ENSEMBLE CLASSIFIER")
    print("=" * 80)
    
    results = {}
    
    # 1. Individual features evaluation
    print("\nIndividual Feature Performance:")
    print("-" * 80)
    
    # Likelihood alone
    fpr, tpr, likelihood_auc = get_roc_metrics(test_human_ll, test_normal_ll + test_human_like_ll)
    results['likelihood_alone'] = likelihood_auc
    print(f"Log Likelihood alone:  AUROC = {likelihood_auc:.4f}")
    
    # Logrank alone
    fpr, tpr, logrank_auc = get_roc_metrics([-x for x in test_human_lr], [-x for x in (test_normal_lr + test_human_like_lr)])
    results['logrank_alone'] = logrank_auc
    print(f"Log Rank alone:        AUROC = {logrank_auc:.4f}")
    
    # Entropy alone
    fpr, tpr, entropy_auc = get_roc_metrics(test_human_ent, test_normal_ent + test_human_like_ent)
    results['entropy_alone'] = entropy_auc
    print(f"Entropy alone:         AUROC = {entropy_auc:.4f}")
    
    # LRR alone
    lrr_human = [-ll/lr if lr != 0 else 0 for ll, lr in zip(test_human_ll, test_human_lr)]
    lrr_llm = [-ll/lr if lr != 0 else 0 for ll, lr in zip(test_normal_ll + test_human_like_ll, test_normal_lr + test_human_like_lr)]
    fpr, tpr, lrr_auc = get_roc_metrics(lrr_human, lrr_llm)
    results['lrr_alone'] = lrr_auc
    print(f"LRR alone:             AUROC = {lrr_auc:.4f}")
    
    # 2. Ensemble classifier on individual test sets
    print("\nEnsemble Classifier Performance:")
    print("-" * 80)
    
    # Test on normal LLM
    test_normal_labels_array = np.array(test_normal_labels, dtype=int)
    X_test_normal, _ = trainer.prepare_features(test_normal_ll, test_normal_lr, test_normal_labels)
    eval_normal = trainer.evaluate(X_test_normal, test_normal_labels_array)
    results['ensemble_vs_normal_llm'] = eval_normal['roc_auc']
    print(f"Ensemble vs Normal LLM:        AUROC = {eval_normal['roc_auc']:.4f}")
    
    # Test on human-like LLM
    test_human_like_labels_array = np.array(test_human_like_labels, dtype=int)
    X_test_human_like, _ = trainer.prepare_features(test_human_like_ll, test_human_like_lr, test_human_like_labels)
    eval_human_like = trainer.evaluate(X_test_human_like, test_human_like_labels_array)
    results['ensemble_vs_human_like_llm'] = eval_human_like['roc_auc']
    print(f"Ensemble vs Human-like LLM:    AUROC = {eval_human_like['roc_auc']:.4f}")
    
    # Test on combined
    test_combined_ll = test_normal_ll + test_human_like_ll
    test_combined_lr = test_normal_lr + test_human_like_lr
    test_combined_labels = np.array([1] * (len(test_normal_ll) + len(test_human_like_ll)), dtype=int)
    X_test_combined, _ = trainer.prepare_features(test_combined_ll, test_combined_lr, test_combined_labels.tolist())
    eval_combined = trainer.evaluate(X_test_combined, test_combined_labels)
    results['ensemble_vs_all_llm'] = eval_combined['roc_auc']
    print(f"Ensemble vs All LLM:           AUROC = {eval_combined['roc_auc']:.4f}")
    
    # 3. Robustness test: evaluate individual features on human-like text
    print("\nRobustness on Human-Like Generated Text (vs Human):")
    print("-" * 80)
    fpr, tpr, likelihood_human_like_auc = get_roc_metrics(test_human_ll, test_human_like_ll)
    results['likelihood_vs_human_like'] = likelihood_human_like_auc
    print(f"Log Likelihood:  AUROC = {likelihood_human_like_auc:.4f}")
    
    fpr, tpr, logrank_human_like_auc = get_roc_metrics([-x for x in test_human_lr], [-x for x in test_human_like_lr])
    results['logrank_vs_human_like'] = logrank_human_like_auc
    print(f"Log Rank:        AUROC = {logrank_human_like_auc:.4f}")
    
    fpr, tpr, entropy_human_like_auc = get_roc_metrics(test_human_ent, test_human_like_ent)
    results['entropy_vs_human_like'] = entropy_human_like_auc
    print(f"Entropy:         AUROC = {entropy_human_like_auc:.4f}")
    
    fpr, tpr, lrr_human_like_auc = get_roc_metrics(lrr_human, [-ll/lr if lr != 0 else 0 for ll, lr in zip(test_human_like_ll, test_human_like_lr)])
    results['lrr_vs_human_like'] = lrr_human_like_auc
    print(f"LRR:             AUROC = {lrr_human_like_auc:.4f}")
    
    # Ensemble on human-like
    test_human_like_for_ensemble = np.array([0] * len(test_human_ll) + [1] * len(test_human_like_ll), dtype=int)
    X_human_like_for_ensemble, _ = trainer.prepare_features(
        test_human_ll + test_human_like_ll,
        test_human_lr + test_human_like_lr,
        test_human_like_for_ensemble.tolist()
    )
    eval_ensemble_human_like = trainer.evaluate(X_human_like_for_ensemble, test_human_like_for_ensemble)
    results['ensemble_vs_human_like'] = eval_ensemble_human_like['roc_auc']
    print(f"Ensemble:        AUROC = {eval_ensemble_human_like['roc_auc']:.4f}")
    
    # Save results
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    output_file = os.path.join(args.output_dir, f"ensemble_results_{args.dataset}_{args.base_model_name}.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    # Print improvement summary
    print("\nImprovement Summary:")
    print("-" * 80)
    print(f"Individual features vs Ensemble (Normal LLM):")
    print(f"  Likelihood: {likelihood_auc:.4f} → Ensemble: {eval_normal['roc_auc']:.4f} ({eval_normal['roc_auc'] - likelihood_auc:+.4f})")
    print(f"  LogRank:    {logrank_auc:.4f} → Ensemble: {eval_normal['roc_auc']:.4f} ({eval_normal['roc_auc'] - logrank_auc:+.4f})")
    print(f"  Entropy:    {entropy_auc:.4f} → Ensemble: {eval_normal['roc_auc']:.4f} ({eval_normal['roc_auc'] - entropy_auc:+.4f})")
    
    print(f"\nRobustness to Human-Like Text:")
    print(f"  Likelihood: {likelihood_human_like_auc:.4f}")
    print(f"  LogRank:    {logrank_human_like_auc:.4f}")
    print(f"  Entropy:    {entropy_human_like_auc:.4f}")
    print(f"  LRR:        {lrr_human_like_auc:.4f}")
    print(f"  Ensemble:   {eval_ensemble_human_like['roc_auc']:.4f} ← Better robustness")


if __name__ == '__main__':
    main()
