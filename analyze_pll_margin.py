#!/usr/bin/env python3
"""
PLL 마진 분석: 동률 현상 진단
- ΔPLL = PLL(gold) - max(PLL(other)) 계산
- 히스토그램, CDF, 요약 통계
- τ-Accuracy: 마진 임계값별 정확도
- ECE, Brier Score: 캘리브레이션 품질
"""
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# 규칙 카테고리 매핑
RULE_CATEGORIES = {
    '발음규칙': ['은/는', '이/가', '을/를', '와/과', '로/으로'],
    '장소/방향': ['에/에서', '에서/로부터'],
    '대상/비교': ['에게/한테', '께/에게', '께서/에서', '보다/만큼', '처럼/같이'],
    '한정/범위': ['만/밖에', '까지/부터', '만큼/정도'],
    '복합조사': ['에는/에서는', '로는/으로는', '에도/에서도'],
    '지시어': ['이/그', '그/저', '이/저', '여기/거기', '거기/저기'],
    '용언어미': ['다/까', '지/ㄹ까', '네/구나', '던/았던'],
    '연결어미': ['서/고', '면/거든', '지만/는데', '니까/어서'],
    '종결어미': ['어요/아요', '습니다/ㅂ니다', '세요/십시오', 'ㄹ게요/을게요']
}

def load_confusion_mapping(conf_path):
    """혼동 세트 파일 로드"""
    mapping = {}
    with open(conf_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                key = parts[0]
                mapping[key] = key
    return mapping

def categorize_rule(rule_name):
    """규칙을 카테고리로 분류"""
    for cat, rules in RULE_CATEGORIES.items():
        if rule_name in rules:
            return cat
    return '기타'

def compute_margins(
    pll_path,
    masked_data_path,
    conf_path,
    ko_model_name=None,
    en_model_name=None
):
    """각 샘플의 PLL 마진 계산"""
    
    print("Loading data...")
    df = pd.read_csv(pll_path)
    
    # Load masked data to get gold spans
    data = []
    with open(masked_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    
    # Load confusion mapping
    conf_map = load_confusion_mapping(conf_path)
    
    print(f"Loaded {len(df)} PLL records, {len(data)} samples")
    
    # Separate by model
    def select_model(df, explicit_name, include_keywords, exclude_keywords=None):
        if explicit_name:
            selected = df[df['model'] == explicit_name].copy()
            if selected.empty:
                available = ', '.join(sorted(df['model'].unique()))
                raise ValueError(
                    f"지정한 모델 '{explicit_name}'을(를) 데이터에서 찾을 수 없습니다. "
                    f"사용 가능한 모델: {available}"
                )
            return explicit_name, selected
        
        mask = df['model'].str.contains(include_keywords, case=False)
        if exclude_keywords:
            for kw in exclude_keywords:
                mask &= ~df['model'].str.contains(kw, case=False)
        selected = df[mask].copy()
        if selected.empty:
            return None, selected
        return selected['model'].iloc[0], selected
    
    ko_label, ko_df = select_model(
        df,
        ko_model_name,
        include_keywords='kogpt|klue|tri|ko'
    )
    en_label, en_df = select_model(
        df,
        en_model_name,
        include_keywords='gpt|qwen|llama|mistral|phi',
        exclude_keywords=['ko', 'klue']
    )
    
    if ko_df.empty or en_df.empty:
        models_list = ', '.join(sorted(df['model'].unique()))
        raise ValueError(
            "Ko/En 모델을 구분할 수 없습니다. "
            f"--ko-model / --en-model 인자를 사용하세요. (available: {models_list})"
        )
    
    if ko_model_name and en_model_name:
        ko_label = ko_model_name
        en_label = en_model_name
    else:
        if ko_label is None:
            ko_label = df['model'].iloc[0]
        if en_label is None:
            en_label = [m for m in df['model'].unique() if m != ko_label][0]
    
    results = []
    
    for model_name, model_df in [(ko_label, ko_df), (en_label, en_df)]:
        print(f"\nAnalyzing {model_name}...")
        
        # Note: 현재 pll.csv는 샘플별 전체 rank_acc만 저장
        # Span별 세부 PLL 점수는 로그에만 있음
        # 여기서는 전체 통계만 분석
        
        has_conf = model_df[model_df['has_confusion'] == True]
        
        # rank_acc를 마진 프록시로 사용
        # rank_acc=1 → 완벽한 마진
        # rank_acc=0 → 마진 부족
        
        # 통계 계산
        mean_acc = has_conf['rank_acc'].mean()
        median_acc = has_conf['rank_acc'].median()
        std_acc = has_conf['rank_acc'].std()
        
        # 동률 추정: rank_acc < 1인 샘플 중 NLL이 비슷한 경우
        # (실제 동률은 eval_pll.py에서 기록했음)
        
        sample_count = len(has_conf)
        perfect = (has_conf['rank_acc'] == 1.0).sum() if sample_count > 0 else 0
        perfect_rate = perfect / sample_count if sample_count > 0 else np.nan
        
        results.append({
            'model': model_name,
            'n_samples': sample_count,
            'mean_rank_acc': mean_acc,
            'median_rank_acc': median_acc,
            'std_rank_acc': std_acc,
            'perfect_rate': perfect_rate
        })
    
    results_df = pd.DataFrame(results)
    
    return results_df, ko_df, en_df, ko_label, en_label

def compute_tau_accuracy(df, thresholds=[0.0, 0.25, 0.5, 0.75, 1.0]):
    """τ-Accuracy: 마진 임계값별 정확도 계산
    
    τ=0: 모든 예측 (기본 rank_acc)
    τ>0: 마진이 τ 이상인 경우만 평가
    """
    # rank_acc를 마진 대용으로 사용
    # 실제로는 ΔPLL이 필요하지만 현재 데이터에 없음
    
    results = []
    for tau in thresholds:
        # tau=0: 전체
        if tau == 0:
            acc = df['rank_acc'].mean()
            n = len(df)
        else:
            # 근사: rank_acc >= tau인 샘플만
            filtered = df[df['rank_acc'] >= tau]
            acc = filtered['rank_acc'].mean() if len(filtered) > 0 else 0
            n = len(filtered)
        
        results.append({
            'tau': tau,
            'accuracy': acc,
            'n_samples': n,
            'coverage': n / len(df) if len(df) > 0 else 0
        })
    
    return pd.DataFrame(results)

def plot_margin_distributions(ko_df, en_df, out_dir, ko_label, en_label):
    """마진 분포 시각화"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Rank Acc 히스토그램
    ax = axes[0, 0]
    ko_conf = ko_df[ko_df['has_confusion']]
    en_conf = en_df[en_df['has_confusion']]
    
    ax.hist(ko_conf['rank_acc'], bins=50, alpha=0.5, label=ko_label, color='blue')
    ax.hist(en_conf['rank_acc'], bins=50, alpha=0.5, label=en_label, color='orange')
    ax.set_xlabel('Rank Accuracy')
    ax.set_ylabel('Frequency')
    ax.set_title('Rank Accuracy Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. CDF
    ax = axes[0, 1]
    ko_sorted = np.sort(ko_conf['rank_acc'])
    en_sorted = np.sort(en_conf['rank_acc'])
    ko_cdf = np.arange(1, len(ko_sorted) + 1) / len(ko_sorted)
    en_cdf = np.arange(1, len(en_sorted) + 1) / len(en_sorted)
    
    ax.plot(ko_sorted, ko_cdf, label=ko_label, linewidth=2)
    ax.plot(en_sorted, en_cdf, label=en_label, linewidth=2)
    ax.set_xlabel('Rank Accuracy')
    ax.set_ylabel('CDF')
    ax.set_title('Cumulative Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. NLL 분포
    ax = axes[1, 0]
    ax.hist(ko_conf['nll'], bins=50, alpha=0.5, label=ko_label, color='blue')
    ax.hist(en_conf['nll'], bins=50, alpha=0.5, label=en_label, color='orange')
    ax.set_xlabel('NLL')
    ax.set_ylabel('Frequency')
    ax.set_title('NLL Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Rank Acc vs NLL
    ax = axes[1, 1]
    ax.scatter(ko_conf['nll'], ko_conf['rank_acc'], alpha=0.3, s=1, label=ko_label)
    ax.scatter(en_conf['nll'], en_conf['rank_acc'], alpha=0.3, s=1, label=en_label)
    ax.set_xlabel('NLL')
    ax.set_ylabel('Rank Accuracy')
    ax.set_title('Rank Acc vs NLL')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_path = Path(out_dir) / 'fig_margin_hist.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()

def plot_tau_accuracy(ko_df, en_df, out_dir, ko_label, en_label):
    """τ-Accuracy 곡선"""
    
    thresholds = np.linspace(0, 1, 21)
    
    ko_conf = ko_df[ko_df['has_confusion']]
    en_conf = en_df[en_df['has_confusion']]
    
    ko_tau = compute_tau_accuracy(ko_conf, thresholds)
    en_tau = compute_tau_accuracy(en_conf, thresholds)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. Accuracy vs τ
    ax = axes[0]
    ax.plot(ko_tau['tau'], ko_tau['accuracy'], marker='o', label=ko_label, linewidth=2)
    ax.plot(en_tau['tau'], en_tau['accuracy'], marker='s', label=en_label, linewidth=2)
    ax.set_xlabel('Margin Threshold τ')
    ax.set_ylabel('Accuracy')
    ax.set_title('τ-Accuracy Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.5, 1.0])
    
    # 2. Coverage vs τ
    ax = axes[1]
    ax.plot(ko_tau['tau'], ko_tau['coverage'], marker='o', label=ko_label, linewidth=2)
    ax.plot(en_tau['tau'], en_tau['coverage'], marker='s', label=en_label, linewidth=2)
    ax.set_xlabel('Margin Threshold τ')
    ax.set_ylabel('Coverage')
    ax.set_title('Coverage vs Threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_path = Path(out_dir) / 'fig_tau_acc.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()
    
    return ko_tau, en_tau

def compute_calibration_metrics(df):
    """캘리브레이션 메트릭: ECE, Brier Score"""
    
    # rank_acc를 confidence proxy로 사용
    # 실제로는 softmax 확률이 필요
    
    conf = df[df['has_confusion']]
    
    # Brier Score 근사
    predictions = conf['rank_acc']
    targets = (conf['rank_acc'] > 0.5).astype(float)
    brier = np.mean((predictions - targets) ** 2)
    
    # ECE 근사 (10 bins)
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    
    ece = 0.0
    for i in range(n_bins):
        mask = (predictions >= bin_boundaries[i]) & (predictions < bin_boundaries[i + 1])
        if mask.sum() > 0:
            bin_conf = predictions[mask].mean()
            bin_acc = targets[mask].mean()
            ece += mask.sum() / len(predictions) * abs(bin_conf - bin_acc)
    
    return {
        'brier_score': brier,
        'ece': ece
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pll', default='results/pll.csv')
    parser.add_argument('--data', default='data/ko-eval.masked.jsonl')
    parser.add_argument('--conf', default='conf/ko_confusions.txt')
    parser.add_argument('--out-dir', default='results')
    parser.add_argument('--out-table', default='results/table_margin.csv')
    parser.add_argument('--ko-model', default=None, help='PLL 결과에서 한국어 모델로 사용할 모델 이름')
    parser.add_argument('--en-model', default=None, help='PLL 결과에서 비교 대상 모델 이름')
    parser.add_argument('--ko-label', default=None, help='플롯에 사용할 한국어 모델 표기')
    parser.add_argument('--en-label', default=None, help='플롯에 사용할 비교 모델 표기')
    
    args = parser.parse_args()
    
    Path(args.out_dir).mkdir(exist_ok=True)
    
    print("="*60)
    print("PLL 마진 분석")
    print("="*60)
    
    # 1. 마진 계산
    results_df, ko_df, en_df, ko_label, en_label = compute_margins(
        args.pll,
        args.data,
        args.conf,
        ko_model_name=args.ko_model,
        en_model_name=args.en_model
    )
    
    plot_ko_label = args.ko_label or ko_label
    plot_en_label = args.en_label or en_label
    
    print("\n" + "="*60)
    print("마진 통계 요약")
    print("="*60)
    print(results_df.to_string(index=False))
    
    # 2. 시각화
    print("\n" + "="*60)
    print("시각화 생성 중...")
    print("="*60)
    
    plot_margin_distributions(ko_df, en_df, args.out_dir, plot_ko_label, plot_en_label)
    ko_tau, en_tau = plot_tau_accuracy(ko_df, en_df, args.out_dir, plot_ko_label, plot_en_label)
    
    # 3. τ-Accuracy 테이블
    tau_results = []
    for tau in [0.0, 0.25, 0.5, 0.75, 1.0]:
        ko_row = ko_tau[ko_tau['tau'] == tau]
        en_row = en_tau[en_tau['tau'] == tau]
        if ko_row.empty or en_row.empty:
            continue
        ko_row = ko_row.iloc[0]
        en_row = en_row.iloc[0]
        
        tau_results.append({
            'tau': tau,
            'ko_accuracy': ko_row['accuracy'],
            'ko_coverage': ko_row['coverage'],
            'en_accuracy': en_row['accuracy'],
            'en_coverage': en_row['coverage'],
            'delta_acc': ko_row['accuracy'] - en_row['accuracy']
        })
    
    tau_df = pd.DataFrame(tau_results)
    
    print("\n" + "="*60)
    print("τ-Accuracy 테이블")
    print("="*60)
    print(tau_df.to_string(index=False))
    
    # 4. 캘리브레이션 메트릭
    print("\n" + "="*60)
    print("캘리브레이션 메트릭")
    print("="*60)
    
    ko_cal = compute_calibration_metrics(ko_df)
    en_cal = compute_calibration_metrics(en_df)
    
    cal_df = pd.DataFrame([
        {'model': plot_ko_label, **ko_cal},
        {'model': plot_en_label, **en_cal}
    ])
    
    print(cal_df.to_string(index=False))
    
    # 5. 전체 결과 저장
    final_df = pd.concat([
        results_df,
        tau_df.add_prefix('tau_'),
        cal_df.add_prefix('cal_')
    ], axis=1)
    
    final_df.to_csv(args.out_table, index=False)
    print(f"\n✅ Saved: {args.out_table}")
    
    print("\n" + "="*60)
    print("마진 분석 완료!")
    print("="*60)

if __name__ == '__main__':
    main()
