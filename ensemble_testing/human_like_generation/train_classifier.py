"""
Train and save ensemble classifier for LLM detection.

This script:
1. Generates human, normal LLM, and human-like LLM text
2. Computes features: log likelihood, log rank, entropy
3. Trains a sigmoid-based ensemble classifier
4. Saves the model for later use
"""

import argparse
import os
import json
import sys
from pathlib import Path
from typing import List
import numpy as np

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from baselines.utils.loadmodel import load_base_model_and_tokenizer
from baselines.loss import get_ll
from baselines.rank import get_rank
from ensemble_classifier import EnsembleTrainer

import torch
import datasets
import random


def extract_features(texts: List[str], 
                    args, 
                    model_config) -> tuple:
    """Extract log likelihood and log rank for texts."""
    log_likelihoods = []
    log_ranks = []
    
    print("  Extracting features...")
    for idx, text in enumerate(texts):
        if idx % max(1, len(texts) // 10) == 0:
            print(f"    {idx}/{len(texts)}")
        
        try:
            ll = get_ll(text, args, model_config)
            lr = get_rank(text, args, model_config, log=True)
            
            log_likelihoods.append(ll)
            log_ranks.append(lr)
        except Exception as e:
            print(f"    Error extracting features for text {idx}: {e}")
            log_likelihoods.append(np.nan)
            log_ranks.append(np.nan)
    
    return log_likelihoods, log_ranks


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
        description="Train ensemble classifier for LLM detection"
    )
    parser.add_argument('--dataset', type=str, default="xsum")
    parser.add_argument('--dataset_key', type=str, default="document")
    parser.add_argument('--base_model_name', type=str, default="gpt2-medium")
    parser.add_argument('--mask_filling_model_name', type=str, default="t5-small")
    parser.add_argument('--n_samples', type=int, default=200)
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
    parser.add_argument('--model_dir', type=str, default="./ensemble_models")
    parser.add_argument('--openai_model', type=str, default=None)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=100)
    
    args = parser.parse_args()
    
    os.makedirs(args.model_dir, exist_ok=True)
    
    print("=" * 80)
    print("ENSEMBLE CLASSIFIER TRAINING")
    print("=" * 80)
    
    # Load models
    print(f"\nLoading base model: {args.base_model_name}")
    model_config = {'cache_dir': args.cache_dir}
    model_config = load_base_model_and_tokenizer(args, model_config)
    # Note: mask filling model not needed for ensemble training (only uses likelihood and rank)
    
    # Load and prepare dataset
    dataset = load_and_prepare_dataset(args, model_config)
    
    # Generate texts in all three modes
    print("\n" + "-" * 80)
    print("Generating training data...")
    print("-" * 80)
    
    print("  Generating human text (using dataset as-is)...")
    human_texts = dataset
    
    print("  Generating normal LLM text...")
    normal_generated = generate_texts_simple(dataset, model_config, args, instruction=None)
    
    print("  Generating human-like LLM text...")
    instruction = "Write the following in a natural, human-like tone as if written by a person. Avoid robotic language and make it sound conversational:"
    human_like_generated = generate_texts_simple(dataset, model_config, args, instruction=instruction)
    
    # Extract features for all three types
    print("\n" + "=" * 80)
    print("EXTRACTING FEATURES")
    print("=" * 80)
    
    print("\nExtracting features from human text...")
    human_ll, human_lr = extract_features(human_texts, args, model_config)
    
    print("\nExtracting features from normal LLM text...")
    normal_ll, normal_lr = extract_features(normal_generated, args, model_config)
    
    print("\nExtracting features from human-like LLM text...")
    human_like_ll, human_like_lr = extract_features(human_like_generated, args, model_config)
    
    # Remove NaN values
    def filter_valid(ll, lr):
        valid_idx = ~(np.isnan(ll) | np.isnan(lr))
        return (
            np.array(ll)[valid_idx].tolist(),
            np.array(lr)[valid_idx].tolist(),
        )
    
    human_ll, human_lr = filter_valid(human_ll, human_lr)
    normal_ll, normal_lr = filter_valid(normal_ll, normal_lr)
    human_like_ll, human_like_lr = filter_valid(human_like_ll, human_like_lr)
    
    print(f"  Human: {len(human_ll)} valid samples")
    print(f"  Normal LLM: {len(normal_ll)} valid samples")
    print(f"  Human-like LLM: {len(human_like_ll)} valid samples")
    
    # Prepare training data: human (0) vs LLM-generated (1)
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
    
    # Train ensemble classifier
    print("\n" + "=" * 80)
    print("TRAINING ENSEMBLE CLASSIFIER")
    print("=" * 80)
    trainer = EnsembleTrainer(learning_rate=args.learning_rate, epochs=args.epochs)
    X_train, y_train = trainer.prepare_features(train_ll, train_lr, train_labels, fit_normalization=True)
    trainer.train(X_train, y_train)
    print(f"Learned fusion function: {trainer.get_fusion_formula()}")
    
    # Save model
    print("\n" + "=" * 80)
    print("SAVING MODEL")
    print("=" * 80)
    
    base_model_clean = args.base_model_name.replace('/', '_')
    model_path = os.path.join(args.model_dir, f"ensemble_{args.dataset}_{base_model_clean}.pt")
    stats_path = os.path.join(args.model_dir, f"ensemble_{args.dataset}_{base_model_clean}_stats.json")
    
    trainer.save(model_path, stats_path)
    
    # Save training metadata
    metadata = {
        'dataset': args.dataset,
        'base_model': args.base_model_name,
        'n_training_samples': len(train_ll),
        'n_human_samples': len(human_ll),
        'n_llm_samples': len(normal_ll) + len(human_like_ll),
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'architecture': 'quadratic sigmoid fusion over [log_likelihood, log_rank]',
        'feature_set': ['log_likelihood', 'log_rank'],
        'fusion_formula': trainer.get_fusion_formula(),
    }
    
    metadata_path = os.path.join(args.model_dir, f"ensemble_{args.dataset}_{base_model_clean}_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {metadata_path}")
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
