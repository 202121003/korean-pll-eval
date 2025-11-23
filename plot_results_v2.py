#!/usr/bin/env python3
"""
파라미터 규모별 모델 성능 및 타이 비율 시각화 (모델별 분리)
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def plot_by_model_family():
    """모델 패밀리별 성능 비교 그래프"""
    df = pd.read_csv('results/final_summary.csv')
    
    # 모델 패밀리 추출
    df['family'] = df['model'].apply(lambda x:
        'Qwen2' if 'Qwen2-' in x and 'Qwen2.5' not in x
        else 'Qwen2.5' if 'Qwen2.5' in x
        else 'Qwen3' if 'Qwen3' in x
        else 'Tri'
    )
    
    families = ['Qwen2', 'Qwen2.5', 'Qwen3', 'Tri']
    colors = {'short': '#2E86AB', 'long': '#A23B72'}
    
    # 각 모델 패밀리별로 별도 그래프 생성
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, family in enumerate(families):
        ax = axes[idx]
        subset = df[df['family'] == family].sort_values('params')
        
        # Short와 Long context 데이터 분리
        short_data = subset[subset['context'] == 'short']
        long_data = subset[subset['context'] == 'long']
        
        # 라인 플롯
        if len(short_data) > 0:
            ax.plot(short_data['params'], short_data['rank_acc_mean'], 
                   marker='o', label='Short Context', color=colors['short'],
                   linewidth=2.5, markersize=10, alpha=0.9)
        
        if len(long_data) > 0:
            ax.plot(long_data['params'], long_data['rank_acc_mean'], 
                   marker='s', label='Long Context', color=colors['long'],
                   linewidth=2.5, markersize=10, alpha=0.9)
        
        # 각 포인트에 실제 파라미터 수 라벨 추가
        for _, row in short_data.iterrows():
            ax.annotate(f'{row["params"]:.2f}B', 
                       xy=(row['params'], row['rank_acc_mean']),
                       xytext=(0, 10), textcoords='offset points',
                       ha='center', fontsize=8, color=colors['short'])
        
        for _, row in long_data.iterrows():
            ax.annotate(f'{row["params"]:.2f}B', 
                       xy=(row['params'], row['rank_acc_mean']),
                       xytext=(0, -15), textcoords='offset points',
                       ha='center', fontsize=8, color=colors['long'])
        
        ax.set_xlabel('Parameters (Billions)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Rank Accuracy', fontsize=12, fontweight='bold')
        ax.set_title(f'{family} Performance vs Model Size', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='lower right')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim([0.90, 0.97])
        
    plt.tight_layout()
    plt.savefig('results/performance_by_family.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/performance_by_family.png")
    plt.close()

def plot_all_models_comparison():
    """전체 모델 비교 (Context별 분리)"""
    df = pd.read_csv('results/table_context_full.csv')
    
    df['family'] = df['model'].apply(lambda x:
        'Qwen2' if 'Qwen2-' in x and 'Qwen2.5' not in x
        else 'Qwen2.5' if 'Qwen2.5' in x
        else 'Qwen3' if 'Qwen3' in x
        else 'Tri'
    )
    
    colors = {'Qwen2': '#1f77b4', 'Qwen2.5': '#ff7f0e', 
              'Qwen3': '#2ca02c', 'Tri': '#d62728'}
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    # Short Context
    ax = axes[0]
    short_df = df[df['context'] == 'short']
    for family in ['Qwen2', 'Qwen2.5', 'Qwen3', 'Tri']:
        subset = short_df[short_df['family'] == family].sort_values('params')
        ax.plot(subset['params'], subset['rank_acc_mean'],
               marker='o', label=family, color=colors[family],
               linewidth=2.5, markersize=10, alpha=0.8)
    
    ax.set_xlabel('Parameters (Billions)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Rank Accuracy', fontsize=13, fontweight='bold')
    ax.set_title('Short Context Performance (KLUE)', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([0.92, 0.97])
    
    # Long Context
    ax = axes[1]
    long_df = df[df['context'] == 'long']
    for family in ['Qwen2', 'Qwen2.5', 'Qwen3', 'Tri']:
        subset = long_df[long_df['family'] == family].sort_values('params')
        ax.plot(subset['params'], subset['rank_acc_mean'],
               marker='s', label=family, color=colors[family],
               linewidth=2.5, markersize=10, alpha=0.8)
    
    ax.set_xlabel('Parameters (Billions)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Rank Accuracy', fontsize=13, fontweight='bold')
    ax.set_title('Long Context Performance (Kowiki)', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([0.91, 0.97])
    
    plt.tight_layout()
    plt.savefig('results/performance_all_models.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/performance_all_models.png")
    plt.close()

def plot_nll_comparison():
    """NLL 비교 그래프"""
    df = pd.read_csv('results/table_context_full.csv')
    
    df['family'] = df['model'].apply(lambda x:
        'Qwen2' if 'Qwen2-' in x and 'Qwen2.5' not in x
        else 'Qwen2.5' if 'Qwen2.5' in x
        else 'Qwen3' if 'Qwen3' in x
        else 'Tri'
    )
    
    colors = {'Qwen2': '#1f77b4', 'Qwen2.5': '#ff7f0e', 
              'Qwen3': '#2ca02c', 'Tri': '#d62728'}
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    # Short Context
    ax = axes[0]
    short_df = df[df['context'] == 'short']
    for family in ['Qwen2', 'Qwen2.5', 'Qwen3', 'Tri']:
        subset = short_df[short_df['family'] == family].sort_values('params')
        ax.plot(subset['params'], subset['nll_mean'],
               marker='o', label=family, color=colors[family],
               linewidth=2.5, markersize=10, alpha=0.8)
    
    ax.set_xlabel('Parameters (Billions)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Negative Log-Likelihood', fontsize=13, fontweight='bold')
    ax.set_title('NLL vs Model Size (Short Context)', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.invert_yaxis()  # Lower is better
    
    # Long Context  
    ax = axes[1]
    long_df = df[df['context'] == 'long']
    for family in ['Qwen2', 'Qwen2.5', 'Qwen3', 'Tri']:
        subset = long_df[long_df['family'] == family].sort_values('params')
        ax.plot(subset['params'], subset['nll_mean'],
               marker='s', label=family, color=colors[family],
               linewidth=2.5, markersize=10, alpha=0.8)
    
    ax.set_xlabel('Parameters (Billions)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Negative Log-Likelihood', fontsize=13, fontweight='bold')
    ax.set_title('NLL vs Model Size (Long Context)', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.invert_yaxis()  # Lower is better
    
    plt.tight_layout()
    plt.savefig('results/nll_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/nll_comparison.png")
    plt.close()

def plot_tie_rates():
    """타이 비율 분석"""
    df = pd.read_csv('results/table_context_full.csv')
    
    df['family'] = df['model'].apply(lambda x:
        'Qwen2' if 'Qwen2-' in x and 'Qwen2.5' not in x
        else 'Qwen2.5' if 'Qwen2.5' in x
        else 'Qwen3' if 'Qwen3' in x
        else 'Tri'
    )
    
    colors = {'Qwen2': '#1f77b4', 'Qwen2.5': '#ff7f0e', 
              'Qwen3': '#2ca02c', 'Tri': '#d62728'}
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    
    for family in ['Qwen2', 'Qwen2.5', 'Qwen3', 'Tri']:
        for context, marker in [('short', 'o'), ('long', 's')]:
            subset = df[(df['family'] == family) & (df['context'] == context)].sort_values('params')
            linestyle = '-' if context == 'short' else '--'
            label = f'{family} ({context})'
            ax.plot(subset['params'], subset['tie_rate'] * 100,
                   marker=marker, label=label, color=colors[family],
                   linewidth=2, markersize=8, linestyle=linestyle, alpha=0.7)
    
    ax.set_xlabel('Parameters (Billions)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Tie Rate (%)', fontsize=13, fontweight='bold')
    ax.set_title('Tie Rate vs Model Size', fontsize=15, fontweight='bold')
    ax.legend(fontsize=9, ncol=2, loc='upper left')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('results/tie_rates.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/tie_rates.png")
    plt.close()

def main():
    print("Generating improved visualizations with actual parameter counts...")
    print()
    
    # 결과 재집계 (실제 파라미터 수 반영)
    print("Re-aggregating results with actual parameter counts...")
    import subprocess
    subprocess.run(['python3', 'scripts/aggregate_all_results.py'], check=True)
    print()
    
    plot_by_model_family()
    plot_all_models_comparison()
    plot_nll_comparison()
    plot_tie_rates()
    
    print("\n✓ All visualizations complete!")
    print("\nGenerated files:")
    print("  - results/performance_by_family.png (모델 패밀리별 분리)")
    print("  - results/performance_all_models.png (전체 비교: Short vs Long)")
    print("  - results/nll_comparison.png (NLL 비교)")
    print("  - results/tie_rates.png (타이 비율)")

if __name__ == '__main__':
    main()
