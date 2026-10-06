"""
마할라노비스 거리(MnD) + One-Class SVM 앙상블 듀얼 엔진 파이프라인
================================================================================
핵심 변경사항:
1. 시계열 인덱스 기준 분할 (Raw 상위 15,000행 학습 / 하위 5,000행 테스트)
2. 마할라노비스 거리 기반 3σ/6σ 사전경보 시스템
3. MnD + OCSVM 앙상블 가중 결합 위험도 지수 (Ensemble Risk Index)
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.metrics import (
    confusion_matrix, classification_report, f1_score,
    precision_score, recall_score, matthews_corrcoef,
    precision_recall_curve, auc, brier_score_loss
)
from scipy.spatial.distance import mahalanobis

plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

# ============================================================================
# 1. Raw 데이터 로드 및 시계열 인덱스 기준 분할
# ============================================================================

NORMALIZATION_PARAMS = {
    'V0_p2p_mean': 0.236445, 'V0_p2p_std': 0.085320,
    'V1_flow_mean': 0.084717, 'V1_flow_std': 0.119349,
    'I_fent_mean': 0.051404, 'I_fent_std': 0.018462
}

L = 17
STRIDE = 4
FS = 10.0
GAP_THRESHOLD = 500
LOWFREQ_THRESHOLD = 1.2

def extract_features_from_window(v0, v1, cur):
    p2p = float(np.max(v0) - np.min(v0))
    
    fft_v1 = np.abs(np.fft.rfft(v1)) ** 2
    sum_v1 = np.sum(fft_v1)
    if sum_v1 < 1e-10:
        flow = 0.0
    else:
        norm_v1 = fft_v1 / sum_v1
        freqs = np.fft.rfftfreq(len(v1), 1/FS)
        flow = float(np.sum(norm_v1[freqs < LOWFREQ_THRESHOLD]))
    
    fft_i = np.abs(np.fft.rfft(cur)) ** 2
    sum_i = np.sum(fft_i)
    if sum_i < 1e-10:
        fent = 0.0
    else:
        norm_i = fft_i / sum_i
        pxx_safe = norm_i + 1e-10
        fent = float(-np.sum(pxx_safe * np.log2(pxx_safe)))
    
    return p2p, flow, fent

def raw_df_to_features(df):
    """DataFrame을 받아 Burst 분할 후 윈도우 특징 추출 및 Z-score 정규화"""
    df = df.copy()
    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
    df['time_diff_ms'] = df['TimeStamp'].diff().dt.total_seconds() * 1000
    df['is_gap'] = df['time_diff_ms'] > GAP_THRESHOLD
    df['is_gap'] = df['is_gap'].fillna(False)
    df['burst_id'] = df['is_gap'].cumsum()
    
    rows = []
    window_id = 0
    for burst_id, burst_df in df.groupby('burst_id'):
        if len(burst_df) < L:
            continue
        v0_arr = burst_df['AI0_Vibration'].values
        v1_arr = burst_df['AI1_Vibration'].values
        cur_arr = burst_df['AI2_Current'].values
        ts_arr = burst_df['TimeStamp'].values
        
        for start_idx in range(0, len(burst_df) - L + 1, STRIDE):
            v0_w = v0_arr[start_idx:start_idx + L]
            v1_w = v1_arr[start_idx:start_idx + L]
            cur_w = cur_arr[start_idx:start_idx + L]
            
            p2p, flow, fent = extract_features_from_window(v0_w, v1_w, cur_w)
            
            p2p_n = (p2p - NORMALIZATION_PARAMS['V0_p2p_mean']) / NORMALIZATION_PARAMS['V0_p2p_std']
            flow_n = (flow - NORMALIZATION_PARAMS['V1_flow_mean']) / NORMALIZATION_PARAMS['V1_flow_std']
            fent_n = (fent - NORMALIZATION_PARAMS['I_fent_mean']) / NORMALIZATION_PARAMS['I_fent_std']
            
            rows.append({
                'window_id': window_id, 'burst_id': burst_id,
                'start_time': ts_arr[start_idx],
                'V0_p2p': p2p_n, 'V1_flow': flow_n, 'I_fent': fent_n,
            })
            window_id += 1
    return pd.DataFrame(rows)

# ============================================================================
# 시계열 인덱스 기준 Raw 데이터 분할 (상위 15,000행 / 하위 5,000행)
# ============================================================================

print("=" * 80)
print("1단계: 시계열 인덱스 기준 Raw 데이터 분할")
print("=" * 80)

df_raw_normal = pd.read_csv('press_data_normal.csv')
df_raw_outlier = pd.read_csv('outlier_data.csv')

TRAIN_CUTOFF = 15000

df_train_raw = df_raw_normal.iloc[:TRAIN_CUTOFF]      # 상위 15,000행 (학습)
df_test_normal_raw = df_raw_normal.iloc[TRAIN_CUTOFF:] # 하위 5,000행 (테스트-정상)
df_test_outlier_raw = df_raw_outlier                   # 이상 전체 600행 (테스트-이상)

print(f"학습용 Raw 데이터: 0 ~ {TRAIN_CUTOFF-1} 행 ({len(df_train_raw)} 행)")
print(f"평가용 정상 Raw 데이터: {TRAIN_CUTOFF} ~ {len(df_raw_normal)-1} 행 ({len(df_test_normal_raw)} 행)")
print(f"평가용 이상 Raw 데이터: outlier_data.csv 전체 ({len(df_test_outlier_raw)} 행)")

# ============================================================================
# 특징 추출 (각각 독립적으로)
# ============================================================================

print("\n" + "=" * 80)
print("2단계: 특징 추출 (학습/테스트 각각 독립)")
print("=" * 80)

df_train_feat = raw_df_to_features(df_train_raw)
df_test_normal_feat = raw_df_to_features(df_test_normal_raw)
df_test_outlier_feat = raw_df_to_features(df_test_outlier_raw)

features = ['V0_p2p', 'V1_flow', 'I_fent']

X_train = df_train_feat[features].values
X_test_norm = df_test_normal_feat[features].values
X_test_outlier = df_test_outlier_feat[features].values

X_test = np.vstack([X_test_norm, X_test_outlier])
y_test = np.concatenate([np.zeros(len(X_test_norm)), np.ones(len(X_test_outlier))]).astype(int)

print(f"학습 윈도우 수: {len(X_train)} 개 (정상만)")
print(f"평가 정상 윈도우 수: {len(X_test_norm)} 개")
print(f"평가 이상 윈도우 수: {len(X_test_outlier)} 개")
print(f"평가 총 윈도우 수: {len(X_test)} 개")

# ============================================================================
# 2. 듀얼 엔진 학습 (마할라노비스 + One-Class SVM)
# ============================================================================

print("\n" + "=" * 80)
print("3단계: 듀얼 엔진 학습 (Mahalanobis Distance + One-Class SVM)")
print("=" * 80)

t_start = time.perf_counter()

# (A) 마할라노비스 거리 학습: 정상 중심점(mu)과 공분산(Sigma) 계산
mu_train = np.mean(X_train, axis=0)
cov_train = np.cov(X_train, rowvar=False)
cov_inv = np.linalg.inv(cov_train)

print(f"[마할라노비스] 정상 중심점 (mu): V0_p2p={mu_train[0]:.4f}, V1_flow={mu_train[1]:.4f}, I_fent={mu_train[2]:.4f}")
print(f"[마할라노비스] 공분산 행렬 크기: {cov_train.shape}")

# (B) One-Class SVM 학습
ocsvm = OneClassSVM(kernel='rbf', gamma=0.02, nu=0.0002)
ocsvm.fit(X_train)

t_train_time = time.perf_counter() - t_start
print(f"[One-Class SVM] gamma=0.02, nu=0.0002 학습 완료")
print(f"듀얼 엔진 총 학습 시간: {t_train_time:.4f} 초")

# ============================================================================
# 3. 듀얼 엔진 추론 (Ensemble Risk Index)
# ============================================================================

print("\n" + "=" * 80)
print("4단계: 듀얼 엔진 앙상블 추론")
print("=" * 80)

t_start_infer = time.perf_counter()

# (A) 마할라노비스 거리 계산
mnd_scores = np.array([mahalanobis(x, mu_train, cov_inv) for x in X_test])

# (B) One-Class SVM Decision Score
ocsvm_raw_scores = -ocsvm.decision_function(X_test)

# (C) 앙상블 가중합 위험도 지수 (Ensemble Risk Index)
# 두 스코어를 Min-Max 0~1 정규화 후 가중 결합
mnd_norm = (mnd_scores - mnd_scores.min()) / (mnd_scores.max() - mnd_scores.min() + 1e-10)
ocsvm_norm = (ocsvm_raw_scores - ocsvm_raw_scores.min()) / (ocsvm_raw_scores.max() - ocsvm_raw_scores.min() + 1e-10)

W_MND = 0.5   # 마할라노비스 가중치 (설명력)
W_SVM = 0.5   # One-Class SVM 가중치 (정밀도)

ensemble_risk = W_MND * mnd_norm + W_SVM * ocsvm_norm  # 0~1 사이의 앙상블 위험 지수

t_infer = time.perf_counter() - t_start_infer

# ============================================================================
# 4. 마할라노비스 3σ / 6σ 사전경보 경계
# ============================================================================

# 정상 학습 데이터 자체의 마할라노비스 거리 분포 확인
mnd_train = np.array([mahalanobis(x, mu_train, cov_inv) for x in X_train])
mnd_train_mean = np.mean(mnd_train)
mnd_train_std = np.std(mnd_train)

# 경보 경계 설정
TH_CAUTION = 3.0    # 3σ: 정상의 99.7% 바깥 → 주의 시작
TH_CRITICAL = 6.0   # 6σ: 정상의 사실상 100% 바깥 → 확정 위험

# 3단계 Zone 판정
zone_labels = []
for d in mnd_scores:
    if d <= TH_CAUTION:
        zone_labels.append('Normal')
    elif d <= TH_CRITICAL:
        zone_labels.append('Caution_Warning')
    else:
        zone_labels.append('Critical_Anomaly')

# OCSVM 단독 판정 (비교용)
ocsvm_preds = np.where(ocsvm.predict(X_test) == -1, 1, 0)

# 앙상블 판정 (MnD 3σ 기준 + OCSVM 동의 시 이상)
ensemble_preds = np.where((mnd_scores > TH_CAUTION) & (ocsvm_preds == 1), 1, 0)

print(f"마할라노비스 정상 학습 거리 분포: mean={mnd_train_mean:.4f}, std={mnd_train_std:.4f}")
print(f"3σ 주의 경계 (Caution): d_M > {TH_CAUTION:.1f}")
print(f"6σ 위험 경계 (Critical): d_M > {TH_CRITICAL:.1f}")

# ============================================================================
# 5. 성능 평가 (3가지 모델 비교: MnD 단독 / OCSVM 단독 / 앙상블)
# ============================================================================

print("\n" + "=" * 80)
print("5단계: 3가지 모델 성능 비교 (시계열 인덱스 분할 기준)")
print("=" * 80)

# MnD 단독 판정 (3σ 초과 시 이상)
mnd_preds = np.where(mnd_scores > TH_CAUTION, 1, 0)

models = {
    'Mahalanobis 3-sigma': mnd_preds,
    'One-Class SVM': ocsvm_preds,
    'Ensemble (MnD+SVM)': ensemble_preds,
}

for name, preds in models.items():
    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    mcc = matthews_corrcoef(y_test, preds)
    fp_rate = (fp / len(X_test_norm)) * 100
    
    print(f"\n--- {name} ---")
    print(f"  혼동행렬: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print(f"  정밀도={prec:.4f}, 재현율={rec:.4f}, F1={f1:.4f}, MCC={mcc:.4f}")
    print(f"  오경보율={fp_rate:.2f}% ({fp}건/{len(X_test_norm)}), 미탐지율={fn}건/{len(X_test_outlier)}")

# ============================================================================
# 6. 마할라노비스 3σ/6σ 3단계 Zone 분포
# ============================================================================

print("\n" + "=" * 80)
print("6단계: 마할라노비스 3σ/6σ 사전경보 Zone 분포표")
print("=" * 80)

df_result = pd.DataFrame({
    'true_label': y_test,
    'mnd_score': mnd_scores,
    'ocsvm_score': ocsvm_raw_scores,
    'ensemble_risk': ensemble_risk,
    'risk_zone': zone_labels
})

ct = pd.crosstab(df_result['risk_zone'], df_result['true_label'], margins=True)
ct.columns = ['Actual Normal(0)', 'Actual Outlier(1)', 'Total']
print(ct)

# ============================================================================
# 7. 시각화 대시보드 (3D 원본 공간 + 마할라노비스 Zone + 앙상블)
# ============================================================================

fig = plt.figure(figsize=(18, 12))

# Subplot 1: 3D 피처 공간 + 마할라노비스 Zone 컬러링
ax1 = fig.add_subplot(2, 2, 1, projection='3d')
colors_zone = {'Normal': 'green', 'Caution_Warning': 'darkorange', 'Critical_Anomaly': 'red'}
for zone, col in colors_zone.items():
    mask = np.array(zone_labels) == zone
    ax1.scatter(X_test[mask, 0], X_test[mask, 1], X_test[mask, 2],
                c=col, label=zone, alpha=0.7, s=25)
ax1.set_xlabel('V0_p2p')
ax1.set_ylabel('V1_flow')
ax1.set_zlabel('I_fent')
ax1.set_title('3D Feature Space + Mahalanobis Zone', fontsize=12, fontweight='bold')
ax1.legend(loc='upper right', fontsize=9)
ax1.view_init(elev=25, azim=135)

# Subplot 2: 마할라노비스 거리 분포 히스토그램 + 3σ/6σ 경계선
ax2 = fig.add_subplot(2, 2, 2)
ax2.hist(mnd_scores[y_test == 0], bins=50, alpha=0.7, color='blue', label='Normal (Test)', density=True)
ax2.hist(mnd_scores[y_test == 1], bins=20, alpha=0.7, color='red', label='Outlier (Test)', density=True)
ax2.axvline(x=TH_CAUTION, color='darkorange', linewidth=2.5, linestyle='--', label=f'3-sigma Caution ({TH_CAUTION})')
ax2.axvline(x=TH_CRITICAL, color='darkred', linewidth=2.5, label=f'6-sigma Critical ({TH_CRITICAL})')
ax2.set_xlabel('Mahalanobis Distance')
ax2.set_ylabel('Density')
ax2.set_title('Mahalanobis Distance Distribution + Warning Boundaries', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)

# Subplot 3: 앙상블 위험 지수 분포
ax3 = fig.add_subplot(2, 2, 3)
ax3.scatter(mnd_scores[y_test == 0], ocsvm_raw_scores[y_test == 0], c='blue', alpha=0.4, s=15, label='Normal')
ax3.scatter(mnd_scores[y_test == 1], ocsvm_raw_scores[y_test == 1], c='red', alpha=0.9, s=40, marker='^', label='Outlier')
ax3.axvline(x=TH_CAUTION, color='darkorange', linewidth=2, linestyle='--', label='3-sigma')
ax3.set_xlabel('Mahalanobis Distance (d_M)')
ax3.set_ylabel('OCSVM Decision Score (-f(x))')
ax3.set_title('Dual Engine Score Map (MnD vs OCSVM)', fontsize=12, fontweight='bold')
ax3.legend(fontsize=9)

# Subplot 4: 성능 비교 막대 그래프
ax4 = fig.add_subplot(2, 2, 4)
model_names = list(models.keys())
f1_scores = [f1_score(y_test, models[n], zero_division=0) for n in model_names]
prec_scores = [precision_score(y_test, models[n], zero_division=0) for n in model_names]
rec_scores = [recall_score(y_test, models[n], zero_division=0) for n in model_names]

x_pos = np.arange(len(model_names))
width = 0.25
ax4.bar(x_pos - width, prec_scores, width, label='Precision', color='#2563eb', alpha=0.8)
ax4.bar(x_pos, rec_scores, width, label='Recall', color='#dc2626', alpha=0.8)
ax4.bar(x_pos + width, f1_scores, width, label='F1-Score', color='#059669', alpha=0.8)
ax4.set_xticks(x_pos)
ax4.set_xticklabels(model_names, fontsize=9)
ax4.set_ylim(0, 1.15)
ax4.set_title('Model Performance Comparison', fontsize=12, fontweight='bold')
ax4.legend()

for i, (p, r, f) in enumerate(zip(prec_scores, rec_scores, f1_scores)):
    ax4.text(i - width, p + 0.02, f'{p:.3f}', ha='center', fontsize=8)
    ax4.text(i, r + 0.02, f'{r:.3f}', ha='center', fontsize=8)
    ax4.text(i + width, f + 0.02, f'{f:.3f}', ha='center', fontsize=8)

plt.tight_layout()
plt.savefig('ensemble_mnd_ocsvm_dashboard.png', dpi=300)
print("\n✓ 앙상블 듀얼 엔진 대시보드 저장 완료: ensemble_mnd_ocsvm_dashboard.png")

# ============================================================================
# 8. 종합 세부 평가 성적표 (앙상블 기준)
# ============================================================================

print("\n" + "=" * 80)
print("7단계: 앙상블 듀얼 엔진 종합 세부 평가 성적표")
print("=" * 80)

best_preds = ensemble_preds
best_name = "Ensemble (MnD + OCSVM)"

cm_final = confusion_matrix(y_test, best_preds)
tn, fp, fn, tp = cm_final.ravel()
prec_final = precision_score(y_test, best_preds, zero_division=0)
rec_final = recall_score(y_test, best_preds, zero_division=0)
f1_final = f1_score(y_test, best_preds, zero_division=0)
mcc_final = matthews_corrcoef(y_test, best_preds)

# PR-AUC (앙상블 위험 지수 기반)
pr_prec, pr_rec, _ = precision_recall_curve(y_test, ensemble_risk)
pr_auc_val = auc(pr_rec, pr_prec)

# Brier Score
prob_ensemble = np.clip(ensemble_risk, 0, 1)
brier_val = brier_score_loss(y_test, prob_ensemble)

# 이상 구간 탐지율 (Burst 단위)
burst_ids_outlier = df_test_outlier_feat['burst_id'].unique()
total_bursts = len(burst_ids_outlier)
detected_bursts = 0
for b_id in burst_ids_outlier:
    mask = df_test_outlier_feat['burst_id'] == b_id
    X_b = df_test_outlier_feat.loc[mask, features].values
    mnd_b = np.array([mahalanobis(x, mu_train, cov_inv) for x in X_b])
    ocsvm_b = np.where(ocsvm.predict(X_b) == -1, 1, 0)
    if np.any((mnd_b > TH_CAUTION) & (ocsvm_b == 1)):
        detected_bursts += 1
rate_burst = (detected_bursts / total_bursts) * 100

# 시간당 오경보 수
total_normal_hours = (len(X_test_norm) * 0.4) / 3600.0
fp_per_hour = fp / total_normal_hours if total_normal_hours > 0 else 0.0

# Bootstrap 95% CI
np.random.seed(42)
boot_f1s, boot_fps = [], []
for _ in range(1000):
    idx = np.random.choice(len(y_test), len(y_test), replace=True)
    b_f1 = f1_score(y_test[idx], best_preds[idx], zero_division=0)
    boot_f1s.append(b_f1)
    nm = (y_test[idx] == 0)
    if np.sum(nm) > 0:
        boot_fps.append((np.sum(best_preds[idx][nm] == 1) / np.sum(nm)) * 100)
ci_f1_lo, ci_f1_hi = np.percentile(boot_f1s, [2.5, 97.5])
ci_fp_lo, ci_fp_hi = np.percentile(boot_fps, [2.5, 97.5])

t_infer_per_win = (t_infer / len(X_test)) * 1000

print(f"■ 모델: {best_name}")
print(f"■ 데이터 분할: 시계열 인덱스 기준 (상위 15,000행 학습 / 하위 5,000행 + 이상 600행 테스트)")
print(f"\n[모델 선택 기준]")
print(f"  1. F1-score: {f1_final:.4f}")
print(f"  2. 오경보율: {(fp / len(X_test_norm)) * 100:.2f}% ({fp}건/{len(X_test_norm)})")
print(f"  3. 윈도당 추론 시간: {t_infer_per_win:.4f} ms/window")
print(f"\n[종합 성능]")
print(f"  정밀도: {prec_final:.4f}")
print(f"  재현율: {rec_final:.4f}")
print(f"  MCC: {mcc_final:.4f}")
print(f"  PR-AUC: {pr_auc_val:.4f}")
print(f"\n[확률 품질]")
print(f"  Brier 점수: {brier_val:.4f}")
print(f"\n[현장 관점]")
print(f"  이상 구간 탐지율 (Burst): {rate_burst:.1f}% ({detected_bursts}/{total_bursts} 구간)")
print(f"  시간당 오경보 수: {fp_per_hour:.2f} 회/hr")
print(f"  듀얼 엔진 학습 시간: {t_train_time:.4f} 초")
print(f"\n[불확실성 Bootstrap 1000회 95% CI]")
print(f"  F1-score 95% CI: [{ci_f1_lo:.4f} ~ {ci_f1_hi:.4f}]")
print(f"  오경보율 95% CI: [{ci_fp_lo:.2f}% ~ {ci_fp_hi:.2f}%]")

print("\n[혼동행렬]")
print(f"  {cm_final}")
print("\n[분류 상세 리포트]")
print(classification_report(y_test, best_preds, target_names=['Normal (0)', 'Outlier (1)'], digits=4))
print("=" * 80)
