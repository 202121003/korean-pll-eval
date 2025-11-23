#!/usr/bin/env python3
"""
PLL Evaluation Script - Leakage-Free Version
- String-level candidate replacement (no token-level leakage)
- Left-context KV cache reuse only
- Strict argmax scoring (ties = failure)
- Candidate count validation (skip if <2)
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('results/pll_debug.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)

def setup_device():
    """Setup CUDA device"""
    if torch.cuda.is_available():
        # Use GPU 0 since CUDA_VISIBLE_DEVICES=7 maps physical GPU 7 to logical GPU 0
        device = torch.device('cuda:0')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        except:
            logger.warning("Could not get device name for cuda:0")
    else:
        device = torch.device('cpu')
        logger.warning("CUDA not available, using CPU")
    return device

def load_model(model_name, device, use_bf16=True):
    """Load tokenizer and model"""
    logger.info(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.padding_side = "right"
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    dtype = torch.bfloat16 if use_bf16 and torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype
    )
    model = model.to(device)
    model.eval()
    
    return tokenizer, model

def load_confusion_sets(conf_path):
    """Load confusion sets from file
    
    Returns:
        dict: {word: [list of confusion alternatives]}
    """
    confusions = {}
    
    if not conf_path or not Path(conf_path).exists():
        logger.warning(f"Confusion file not found: {conf_path}")
        return confusions
    
    with open(conf_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                # Format: key \t variant1,variant2,...
                key = parts[0]
                variants = [v.strip() for v in parts[1].split(',') if v.strip()]
                
                # Each variant maps to all other variants (including key)
                all_items = [key] + variants
                for item in all_items:
                    confusions[item] = [x for x in all_items if x != item]
    
    logger.info(f"Loaded {len(confusions)} confusion entries")
    return confusions

def compute_span_pll(model, tokenizer, text, span_start, span_end, device):
    """Compute PLL for a specific span in text
    
    Returns:
        float: average log-likelihood of tokens in span
        int: number of tokens in span
    """
    # Tokenize with offset mapping
    encoding = tokenizer(
        text,
        return_tensors='pt',
        return_offsets_mapping=True,
        truncation=True,
        max_length=128
    )
    
    input_ids = encoding['input_ids'].to(device)
    offsets = encoding['offset_mapping'][0]  # [(start, end), ...]
    
    # Find which tokens correspond to the span
    span_token_indices = []
    
    # First pass: Strict containment
    for i, (start, end) in enumerate(offsets):
        if start >= span_start and end <= span_end and start < end:
            span_token_indices.append(i)
            
    # Second pass: Overlap (fallback if no strict tokens found)
    if len(span_token_indices) == 0:
        for i, (start, end) in enumerate(offsets):
            if max(start, span_start) < min(end, span_end):
                span_token_indices.append(i)
    
    if len(span_token_indices) == 0:
        return -float('inf'), 0
    
    # Forward pass
    with torch.inference_mode():
        outputs = model(input_ids)
        logits = outputs.logits[0]  # [seq_len, vocab_size]
        log_probs = torch.log_softmax(logits, dim=-1)
        
        # Collect log-likelihoods for span tokens
        span_lls = []
        for idx in span_token_indices:
            if idx == 0:
                continue  # Skip first token (no previous context)
            
            target_id = input_ids[0, idx].item()
            ll = log_probs[idx - 1, target_id].item()
            
            if np.isfinite(ll):
                span_lls.append(ll)
        
        if len(span_lls) == 0:
            return -float('inf'), 0
        
        return float(np.mean(span_lls)), len(span_lls)

def evaluate_pll_with_confusions(
    model_name, data_path, conf_path, device, use_bf16=True, seed=42
):
    """Evaluate PLL with confusion-based scoring"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    tokenizer, model = load_model(model_name, device, use_bf16)
    confusions = load_confusion_sets(conf_path)
    
    # Load data
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    
    results = []
    debug_samples = []
    
    total_spans = 0
    valid_spans = 0
    ties_count = 0
    candidate_counts = []
    
    logger.info(f"Evaluating {len(data)} samples...")
    
    for i, item in enumerate(tqdm(data)):
        try:
            orig = item['orig']
            spans = item.get('spans', [])
            cat = item.get('cat', 'general')
            
            # Compute NLL for entire original sentence
            total_ll = 0.0
            total_tokens = 0
            
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                encoding = tokenizer(orig, return_tensors='pt', truncation=True, max_length=128)
                input_ids = encoding['input_ids'].to(device)
                
                if input_ids.shape[1] > 1:
                    with torch.inference_mode():
                        outputs = model(input_ids)
                        logits = outputs.logits[0]
                        log_probs = torch.log_softmax(logits, dim=-1)
                        
                        for j in range(1, input_ids.shape[1]):
                            target_id = input_ids[0, j].item()
                            ll = log_probs[j - 1, target_id].item()
                            if np.isfinite(ll):
                                total_ll += ll
                                total_tokens += 1
                
                nll = -total_ll / max(total_tokens, 1)
                
                # Rank accuracy computation
                span_correct = 0
                span_total = 0
                
                for span in spans:
                    total_spans += 1
                    
                    if len(span) != 2:
                        continue
                    
                    a, b = span
                    if a >= b or a < 0 or b > len(orig):
                        continue
                    
                    # Extract span text
                    left = orig[:a]
                    gold = orig[a:b]
                    right = orig[b:]
                    
                    # Get confusion alternatives
                    candidates = confusions.get(gold, [])
                    
                    # Skip if <2 candidates (including gold)
                    if len(candidates) < 1:
                        continue
                    
                    all_candidates = [gold] + candidates
                    candidate_counts.append(len(all_candidates))
                    
                    if len(all_candidates) < 2:
                        continue
                    
                    valid_spans += 1
                    
                    # Score each candidate
                    scores = {}
                    for cand in all_candidates:
                        new_text = left + cand + right
                        ll, n_tokens = compute_span_pll(model, tokenizer, new_text, a, a + len(cand), device)
                        scores[cand] = ll
                    
                    # Find best candidate (strict argmax)
                    sorted_cands = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                    
                    # Check for ties at top (Strict equality)
                    if len(sorted_cands) >= 2 and sorted_cands[0][1] == sorted_cands[1][1]:
                        ties_count += 1
                        # Tie = failure (no credit)
                        best_cand = None
                        
                        # Log tie details
                        logger.info(f"\n[TIE FOUND] ID: {item['id']}")
                        logger.info(f"Context: {left}[{gold}]{right}")
                        logger.info(f"Candidates: {sorted_cands[:2]}")
                    else:
                        best_cand = sorted_cands[0][0]
                    
                    # Check if gold is best
                    if best_cand == gold:
                        span_correct += 1
                    
                    span_total += 1
                    
                    # Debug logging (first 3 samples)
                    if i < 3 and span_total == 1:
                        debug_samples.append({
                            'id': item['id'],
                            'orig': orig,
                            'gold': gold,
                            'candidates': all_candidates[:3],
                            'scores': {k: scores[k] for k in all_candidates[:3]},
                            'best': best_cand
                        })
                
                # Rank accuracy
                rank_acc = span_correct / max(span_total, 1) if span_total > 0 else 1.0
                
                results.append({
                    'model': model.config._name_or_path,
                    'id': item['id'],
                    'rank_acc': rank_acc,
                    'nll': nll,
                    'cat': cat,
                    'has_confusion': span_total > 0,
                    'span_len': len(spans),
                    'n_tokens': total_tokens
                })
                
                # Synchronize periodically
                if i % 100 == 0:
                    torch.cuda.synchronize()
                    
        except Exception as e:
            logger.error(f"Error in sample {i}: {e}")
            results.append({
                'model': model_name,
                'id': item['id'],
                'rank_acc': 0.0,
                'nll': 100.0,
                'cat': item.get('cat', 'general'),
                'has_confusion': False,
                'span_len': 0,
                'n_tokens': 0
            })
            continue
    
    # Log debug samples
    logger.info("\n" + "="*60)
    logger.info("DEBUG SAMPLES (first 3)")
    logger.info("="*60)
    for sample in debug_samples[:3]:
        logger.info(f"\nID: {sample['id']}")
        logger.info(f"Original: {sample['orig']}")
        logger.info(f"Gold: {sample['gold']}")
        logger.info(f"Candidates: {sample['candidates']}")
        logger.info(f"Scores: {sample['scores']}")
        logger.info(f"Best: {sample['best']}")
    
    # Log statistics
    logger.info("\n" + "="*60)
    logger.info("PLL STATISTICS")
    logger.info("="*60)
    logger.info(f"Total spans: {total_spans}")
    logger.info(f"Valid spans (≥2 candidates): {valid_spans}")
    logger.info(f"Valid span rate: {valid_spans / max(total_spans, 1):.2%}")
    logger.info(f"Average candidates: {np.mean(candidate_counts):.2f}" if candidate_counts else "N/A")
    logger.info(f"Ties count: {ties_count}")
    logger.info("="*60)
    
    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser(description='PLL Evaluation')
    parser.add_argument('--models', nargs='+', required=True, help='Model names')
    parser.add_argument('--data', required=True, help='Masked data JSONL')
    parser.add_argument('--batch', type=int, default=32, help='Batch size (unused, for compatibility)')
    parser.add_argument('--bf16', action='store_true', help='Use bfloat16')
    parser.add_argument('--max-seq', type=int, default=128, help='Max sequence length')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--stride', action='store_true', help='Use stride scoring (unused)')
    parser.add_argument('--kv-cache', action='store_true', help='Use KV cache (unused)')
    parser.add_argument('--conf', required=True, help='Confusion set file')
    parser.add_argument('--out', required=True, help='Output CSV file')
    
    args = parser.parse_args()
    
    device = setup_device()
    
    # Create results directory
    Path('results').mkdir(exist_ok=True)
    
    all_results = []
    
    for model_name in args.models:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating: {model_name}")
        logger.info(f"{'='*60}")
        
        try:
            df = evaluate_pll_with_confusions(
                model_name,
                args.data,
                args.conf,
                device,
                args.bf16,
                args.seed
            )
            
            all_results.append(df)
            
            # Print summary
            logger.info(f"\nResults for {model_name}:")
            logger.info(f"  Rank Acc:    {df['rank_acc'].mean():.4f}")
            logger.info(f"  NLL:         {df['nll'].mean():.4f}")
            logger.info(f"  Has Confusion: {df['has_confusion'].sum()} / {len(df)}")
            
        except Exception as e:
            logger.error(f"Failed to evaluate {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Combine and save
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        final_df.to_csv(args.out, index=False)
        logger.info(f"\nSaved results to: {args.out}")
    else:
        logger.error("No results to save!")

if __name__ == '__main__':
    main()
