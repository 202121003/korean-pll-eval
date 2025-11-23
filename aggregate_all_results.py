#!/usr/bin/env python3
"""
모든 PLL 평가 결과를 집계하여 table_context_means.csv 및 table_context_full.csv 생성
"""
import pandas as pd
import numpy as np
import os
from pathlib import Path

# 모델 파라미터 매핑 (실제 파라미터 수)
model_params = {
    'Qwen/Qwen2-0.5B': 0.494,
    'Qwen/Qwen2-1.5B': 1.544,
    'Qwen/Qwen2-7B': 7.616,
    'Qwen/Qwen2-7B-Instruct': 7.616,
    'Qwen/Qwen2.5-0.5B': 0.494,
    'Qwen/Qwen2.5-1.5B': 1.544,
    'Qwen/Qwen2.5-3B': 3.086,
    'Qwen/Qwen2.5-7B': 7.616,
    'Qwen/Qwen3-1.7B': 2.032,
    'Qwen/Qwen3-4B': 4.022,
    'Qwen/Qwen3-4B-Instruct-2507': 4.022,
    'Qwen/Qwen3-7B': 8.0,  # 존재하지 않음
    'Qwen/Qwen3-8B': 8.191,
    'trillionlabs/Tri-0.5B-Base': 0.472,
    'trillionlabs/Tri-1.9B-Base': 1.809,
    'trillionlabs/Tri-7B-Base': 7.527,
    'trillionlabs/Tri-7B': 7.527,
}

def aggregate_results():
    """모든 PLL 평가 결과를 집계"""
    
    # CSV 파일 목록
    csv_files = [
        # Qwen2.5 family (0.5B / 1.5B / 3B / 7B)
        ('results/pll_qwen25_small_short.csv', 'short'),
        ('results/pll_qwen25_small_long.csv', 'long'),
        ('results/pll_qwen25_medium_short.csv', 'short'),
        ('results/pll_qwen25_medium_long.csv', 'long'),
        ('results/pll_qwen25_large_short.csv', 'short'),
        ('results/pll_qwen25_large_long.csv', 'long'),
        ('results/pll_qwen25_7b_short.csv', 'short'),
        ('results/pll_qwen25_7b_long.csv', 'long'),
        
        # Qwen2 family (0.5B / 1.5B / 7B in single CSV)
        ('results/pll_qwen2_all_short.csv', 'short'),
        ('results/pll_qwen2_all_long.csv', 'long'),
        
        # Qwen3 family (1.7B / 4B legacy short + new long, 8B short/long)
        ('results/pll_qwen3_medium_short.csv', 'short'),
        ('results/pll_qwen3_1.7b_long.csv', 'long'),
        ('results/pll_qwen3_4b_long.csv', 'long'),
        ('results/pll_qwen3_8b_short.csv', 'short'),
        ('results/pll_qwen3_8b_long.csv', 'long'),
        
        # Tri family (0.5B / 1.9B / 7B in single CSV)
        ('results/pll_tri_all_short.csv', 'short'),
        ('results/pll_tri_all_long.csv', 'long'),
    ]
    
    all_results = []
    processed_files = 0
    
    for csv_file, context in csv_files:
        # 파일이 없으면 건너뛰기 (7B 평가가 아직 완료되지 않은 경우)
        if not os.path.exists(csv_file):
            print(f"  Skipping: {csv_file} (not found)")
            continue
            
        try:
            df = pd.read_csv(csv_file)
            processed_files += 1
            
            # 각 모델별로 집계
            for model in df['model'].unique():
                # Instruct 모델 스킵 (Base 버전이 있는 경우만)
                if 'Instruct' in model:
                    base_name = model.replace('-Instruct', '').split('-')[0]
                    # Qwen2-7B, Qwen3-7B는 새로 평가하므로 Instruct 스킵
                    if 'Qwen2-7B' in model or 'Qwen3' in model:
                        continue
                
                model_df = df[df['model'] == model]
                
                # rank_acc와 nll 평균 계산
                rank_acc_mean = model_df['rank_acc'].mean()
                nll_mean = model_df['nll'].mean()
                
                # 타이 카운트 계산 (rank_acc == 0.5인 경우)
                tie_count = (model_df['rank_acc'] == 0.5).sum()
                total_samples = len(model_df)
                tie_rate = tie_count / total_samples if total_samples > 0 else 0
                
                # 파라미터 정보 추가
                params = model_params.get(model, 0)
                
                all_results.append({
                    'model': model,
                    'context': context,
                    'rank_acc_mean': rank_acc_mean,
                    'nll_mean': nll_mean,
                    'tie_count': tie_count,
                    'total_samples': total_samples,
                    'tie_rate': tie_rate,
                    'params': params
                })
        except Exception as e:
            print(f"  Error processing {csv_file}: {e}")
            continue
    
    print(f"\n✓ Processed: {processed_files} CSV files")
    
    return pd.DataFrame(all_results)

def main():
    print("Aggregating all PLL evaluation results...")
    
    # 결과 집계
    df = aggregate_results()
    
    if len(df) == 0:
        print("No results to aggregate!")
        return
    
    # table_context_means.csv 저장 (기존 형식)
    output_df = df[['model', 'context', 'rank_acc_mean', 'nll_mean']].copy()
    output_df.to_csv('results/table_context_means.csv', index=False)
    print(f"✓ Saved: results/table_context_means.csv ({len(output_df)} rows)")
    
    # 타이 정보 포함 전체 결과 저장
    df.to_csv('results/table_context_full.csv', index=False)
    print(f"✓ Saved: results/table_context_full.csv ({len(df)} rows)")
    
    # 결과 요약 출력
    print("\n" + "="*70)
    print("Results Summary:")
    print("="*70)
    print(f"Total models evaluated: {df['model'].nunique()}")
    print(f"Total evaluations: {len(df)}")
    print("="*70)
    
    # 최고/최저 성능 출력
    best_short = df[df['context'] == 'short'].nlargest(1, 'rank_acc_mean')
    best_long = df[df['context'] == 'long'].nlargest(1, 'rank_acc_mean')
    
    if len(best_short) > 0:
        print(f"\nBest (Short): {best_short.iloc[0]['model']}")
        print(f"  Rank Acc: {best_short.iloc[0]['rank_acc_mean']:.3f}")
        print(f"  NLL: {best_short.iloc[0]['nll_mean']:.2f}")
    
    if len(best_long) > 0:
        print(f"\nBest (Long): {best_long.iloc[0]['model']}")
        print(f"  Rank Acc: {best_long.iloc[0]['rank_acc_mean']:.3f}")
        print(f"  NLL: {best_long.iloc[0]['nll_mean']:.2f}")
    
    print("="*70)

if __name__ == '__main__':
    main()
