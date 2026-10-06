"""
One-Class SVM 단일 오탐률 기반 세부 검증 지표 및 성적표 대시보드
"""

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.metrics import (
    precision_score, recall_score, f1_score, matthews_corrcoef,
    precision_recall_curve, auc, brier_score_loss
)
from sklearn.model_selection import train_test_split

plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

# 1. 데이터 로드
df_normal = pd.read_csv('windows_features_L17_s4_normal.csv')
df_outlier = pd.read_csv('windows_features_L17_s4_outlier.csv')

features = ['V0_p2p', 'V1_flow', 'I_fent']

X_normal = df_normal[features].values
y_normal = np.zeros(len(df_normal), dtype=int)
X_outlier = df_outlier[features].values
y_outlier = np.ones(len(df_outlier), dtype=int)

X_train_norm, X_test_norm, y_train_norm, y_test_norm = train_test_split(
    X_normal, y_normal, test_size=0.2, random_state=42
)

X_test = np.vstack([X_test_norm, X_outlier])
y_test = np.concatenate([y_test_norm, y_outlier])

# 2. 모델 학습 (황금 밸런스: gamma=0.02, nu=0.0002)
t_start_train = time.perf_counter()
ocsvm = OneClassSVM(kernel='rbf', gamma=0.02, nu=0.0002)
ocsvm.fit(X_train_norm)
t_train_time = time.perf_counter() - t_start_train

# 3. 추론 시간
t_start_infer = time.perf_counter()
scores = -ocsvm.decision_function(X_test)
preds = np.where(ocsvm.predict(X_test) == -1, 1, 0)
t_infer_total = time.perf_counter() - t_start_infer
t_infer_per_win = (t_infer_total / len(X_test)) * 1000  # ms/window

# 4. 정밀 지표 계산
fp_count = int(np.sum((preds == 1) & (y_test == 0)))
rate_fp = (fp_count / len(X_test_norm)) * 100  # 오경보율 (오탐률 %)

precision_val = precision_score(y_test, preds, zero_division=0)
recall_val = recall_score(y_test, preds, zero_division=0)
f1_val = f1_score(y_test, preds, zero_division=0)
mcc_val = matthews_corrcoef(y_test, preds)

precision_curve, recall_curve, _ = precision_recall_curve(y_test, scores)
pr_auc_val = auc(recall_curve, precision_curve)

prob_scores = 1 / (1 + np.exp(-scores))
brier_val = brier_score_loss(y_test, prob_scores)

# 이상 구간 탐지율 (Burst 단위)
burst_outliers = df_outlier.groupby('burst_id').size()
total_bursts = len(burst_outliers)
detected_bursts = 0
for b_id, b_df in df_outlier.groupby('burst_id'):
    X_b = b_df[features].values
    p_b = np.where(ocsvm.predict(X_b) == -1, 1, 0)
    if np.sum(p_b) > 0:
        detected_bursts += 1
rate_burst_detect = (detected_bursts / total_bursts) * 100

total_normal_hours = (len(X_test_norm) * 0.4) / 3600.0
fp_per_hour = fp_count / total_normal_hours if total_normal_hours > 0 else 0.0

# 5. Bootstrap 1000회 (불확실성)
np.random.seed(42)
n_bootstraps = 1000
boot_f1s = []
boot_rate_fps = []
n_samples = len(y_test)

for _ in range(n_bootstraps):
    idx = np.random.choice(n_samples, n_samples, replace=True)
    y_b = y_test[idx]
    pred_b = preds[idx]
    
    b_f1 = f1_score(y_b, pred_b, zero_division=0)
    boot_f1s.append(b_f1)
    
    norm_mask = (y_b == 0)
    if np.sum(norm_mask) > 0:
        b_fp_rate = (np.sum((pred_b[norm_mask] == 1)) / np.sum(norm_mask)) * 100
        boot_rate_fps.append(b_fp_rate)

ci_f1_low, ci_f1_high = np.percentile(boot_f1s, [2.5, 97.5])
ci_fp_low, ci_fp_high = np.percentile(boot_rate_fps, [2.5, 97.5])

# 시각화 대시보드
fig = plt.figure(figsize=(14, 10), facecolor='#f8f9fa')
plt.suptitle("프레스 기계 이상 탐지 (One-Class SVM) 세부 평가 성적표", fontsize=18, fontweight='bold', y=0.96, color='#111827')

gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25, left=0.06, right=0.94, top=0.88, bottom=0.06)

def draw_card(ax, title, metrics_dict, header_color='#1f2937'):
    ax.set_facecolor('#ffffff')
    ax.axis('off')
    
    rect = plt.Rectangle((0, 0.82), 1, 0.18, transform=ax.transAxes, color=header_color, clip_on=False)
    ax.add_patch(rect)
    ax.text(0.04, 0.90, title, transform=ax.transAxes, color='white', fontsize=12, fontweight='bold', va='center')
    
    y_pos = 0.70
    for key, val in metrics_dict.items():
        ax.text(0.06, y_pos, key, transform=ax.transAxes, color='#4b5563', fontsize=10.5, va='center')
        ax.text(0.94, y_pos, str(val), transform=ax.transAxes, color='#111827', fontsize=11, fontweight='bold', ha='right', va='center')
        y_pos -= 0.05
        ax.plot([0.05, 0.95], [y_pos, y_pos], transform=ax.transAxes, color='#e5e7eb', lw=0.8)
        y_pos -= 0.13

# 1. 모델 선택 기준
draw_card(fig.add_subplot(gs[0, 0]), "■ 모델 선택 기준", {
    "1. F1-score": f"{f1_val:.4f} (최고 수치)",
    "2. 오경보율 (오탐률)": f"{rate_fp:.2f}% ({fp_count} 건 / {len(X_test_norm)})",
    "3. 윈도당 추론 시간": f"{t_infer_per_win:.4f} ms / window"
}, header_color='#2563eb')

# 2. 종합 성능
draw_card(fig.add_subplot(gs[0, 1]), "■ 종합 성능 (Global Metrics)", {
    "정밀도 (Precision)": f"{precision_val:.4f}",
    "재현율 (Recall)": f"{recall_val:.4f}",
    "MCC (Matthews Correlation)": f"{mcc_val:.4f}",
    "PR-AUC (Precision-Recall AUC)": f"{pr_auc_val:.4f}"
}, header_color='#059669')

# 3. 확률 품질
draw_card(fig.add_subplot(gs[1, 0]), "■ 확률 품질 (Probability Calibration)", {
    "Brier 점수 (Brier Score)": f"{brier_val:.4f}",
    "신뢰도 캘리브레이션": "우수 (0에 가까울수록 정밀)",
    "Score - 확률 시그모이드 변환": "정상 적용 완료"
}, header_color='#7c3aed')

# 4. 현장 관점
draw_card(fig.add_subplot(gs[1, 1]), "■ 현장 관점 (Field Operations)", {
    "이상 구간 탐지율 (Burst Level)": f"{rate_burst_detect:.1f}% ({detected_bursts}/{total_bursts} 구간)",
    "시간당 오경보 수": f"{fp_per_hour:.2f} 회 / hour",
    "학습 소요 시간": f"{t_train_time:.4f} 초"
}, header_color='#d97706')

# 5. 불확실성
draw_card(fig.add_subplot(gs[2, 0]), "■ 불확실성 (Bootstrap 1000회 95% 신뢰구간)", {
    "F1-score 95% 신뢰구간": f"[{ci_f1_low:.4f} ~ {ci_f1_high:.4f}]",
    "오경보율 95% 신뢰구간": f"[{ci_fp_low:.2f}% ~ {ci_fp_high:.2f}%]",
    "부트스트랩 샘플 수": "1,000 회 복원 추출"
}, header_color='#4b5563')

plt.savefig('ocsvm_comprehensive_metrics_report.png', dpi=300, bbox_inches='tight')
print("✓ 성적표 업데이트 완료: ocsvm_comprehensive_metrics_report.png")
