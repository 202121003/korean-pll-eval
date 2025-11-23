# Korean PLL Evaluation (tri_vs_qwen)

한국어 PLL(단문 KLUE / 장문 KoWiki) 평가 실험을 위한 최소 레포 정리본입니다. 모델 체크포인트와 대규모 데이터는 포함하지 않고, 샘플 데이터와 요약 결과, 실행 스크립트만 제공합니다.

## 구조
```
tri_vs_qwen/
├── data/                  # 샘플 단문/장문 + 마스킹본
├── conf/                  # 혼동 집합 (은/는, 이/가, …)
├── scripts/               # 마스킹, PLL 평가, 집계, 플롯
├── results/               # 요약 CSV (table_main/restored_summary 등)
├── outputs/               # 예시 그림 일부
├── Makefile               # 파이프라인 실행
└── env.yml                # Conda 환경 정의
```

## 환경
```bash
conda env create -f env.yml
conda activate ko-ctx
export CUDA_VISIBLE_DEVICES=7  # GPU 지정 시
```

## 실행 예시
- 샘플 데이터 마스킹: `make mask`
- PLL 평가(샘플 기준): `make pll`
- 집계: `make agg`
- 플롯: `make plot`

## 포함 데이터/결과
- `data/ko-eval.txt`, `ko-eval.masked.jsonl` (단문 샘플)
- `data/kowiki_long.txt`, `kowiki_long.masked.jsonl` (장문 샘플)
- `results/table_main.csv`, `restored_summary.csv`, `final_summary.csv` (요약본)

## 포함하지 않는 것
- 모델 가중치(예: Qwen/Tri)와 68k/255k 전체 데이터
- 대규모 원시 결과 CSV, 디버그 로그, 체크포인트

## 재현 시 유의
- 실제 논문 수치는 별도 대규모 데이터와 동일한 모델 가중치가 필요합니다. HuggingFace에서 모델을 받아 `scripts/eval_pll*.py`를 사용하세요.
- long/short 입력을 교체하려면 `data/`의 샘플을 동일 이름으로 대체한 뒤 `make mask`부터 재실행하세요.
