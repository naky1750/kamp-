"""
Burst-level Group K-Fold 및 약점 검증 스크립트
================================================================================
목적: 시계열 윈도우 오버랩에 의한 데이터 누수(Data Leakage) 검증 및 과적합 테스트
"""

import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import GroupKFold

# 1. 데이터 로드
df_normal = pd.read_csv('windows_features_L17_s4_normal.csv')
df_outlier = pd.read_csv('windows_features_L17_s4_outlier.csv')

features = ['V0_p2p', 'V1_flow', 'I_fent']

# 2. Group K-Fold (burst_id 기준으로 정상 데이터를 분할하여 데이터 누수 차단)
gkf = GroupKFold(n_splits=5)
X_norm = df_normal[features].values
groups_norm = df_normal['burst_id'].values

X_outlier = df_outlier[features].values
y_outlier = np.ones(len(df_outlier), dtype=int)

print("================================================================================")
print("1. Burst-level Group 5-Fold Cross Validation (데이터 누수 방지 평가)")
print("================================================================================")

cv_precisions, cv_recalls, cv_f1s, cv_fps = [], [], [], []

# 황금 밸런스 파라미터 (gamma=0.02, nu=0.0002)
gamma_val = 0.02
nu_val = 0.0002

for fold, (train_idx, test_idx) in enumerate(gkf.split(X_norm, groups=groups_norm), 1):
    X_train_cv = X_norm[train_idx]
    X_test_norm_cv = X_norm[test_idx]
    y_test_norm_cv = np.zeros(len(X_test_norm_cv), dtype=int)
    
    # Test Set: 해당 Fold의 정상 + 전체 이상 데이터
    X_test_cv = np.vstack([X_test_norm_cv, X_outlier])
    y_test_cv = np.concatenate([y_test_norm_cv, y_outlier])
    
    # One-Class SVM 학습 (Train에는 해당 Fold의 정상 burst만 포함)
    ocsvm = OneClassSVM(kernel='rbf', gamma=gamma_val, nu=nu_val)
    ocsvm.fit(X_train_cv)
    
    preds_raw = ocsvm.predict(X_test_cv)
    y_pred_cv = np.where(preds_raw == -1, 1, 0)
    
    cm = confusion_matrix(y_test_cv, y_pred_cv)
    tn, fp, fn, tp = cm.ravel()
    prec, rec, f1, _ = precision_recall_fscore_support(y_test_cv, y_pred_cv, average='binary', zero_division=0)
    
    cv_precisions.append(prec)
    cv_recalls.append(rec)
    cv_f1s.append(f1)
    cv_fps.append(fp)
    
    print(f"Fold {fold}: Train Burst={len(np.unique(groups_norm[train_idx]))}개, Test Norm={len(X_test_norm_cv)}개 | "
          f"Precision={prec:.4f}, Recall={rec:.4f}, FP(오탐)={fp}건, FN(미탐)={fn}건")

print("-" * 80)
print(f"Group K-Fold 평균 Precision: {np.mean(cv_precisions):.4f}")
print(f"Group K-Fold 평균 Recall:    {np.mean(cv_recalls):.4f}")
print(f"Group K-Fold 평균 F1-Score:  {np.mean(cv_f1s):.4f}")
print(f"Group K-Fold 평균 FP (오탐):  {np.mean(cv_fps):.2f} 건")
print("================================================================================")
