"""
One-Class SVM 프레스 기계 이상 탐지 모델
================================================================================
목적: 정상 데이터만으로 One-Class SVM 학습 후, 남은 정상 + 이상 데이터로 평가
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# 1. 특징 데이터셋 로드
df_normal = pd.read_csv('windows_features_L17_s4_normal.csv')
df_outlier = pd.read_csv('windows_features_L17_s4_outlier.csv')

features = ['V0_p2p', 'V1_flow', 'I_fent']

X_normal = df_normal[features].values
y_normal = np.zeros(len(df_normal), dtype=int)  # 0: 정상

X_outlier = df_outlier[features].values
y_outlier = np.ones(len(df_outlier), dtype=int)  # 1: 이상

# 2. Train / Test Split (정상 데이터의 80%만 학습에 사용)
X_train_norm, X_test_norm, y_train_norm, y_test_norm = train_test_split(
    X_normal, y_normal, test_size=0.2, random_state=42
)

# Test Set 구축: 남은 정상 20% + 이상 데이터 전체 100%
X_test = np.vstack([X_test_norm, X_outlier])
y_test = np.concatenate([y_test_norm, y_outlier])

print("================================================================================")
print("One-Class SVM 데이터셋 분할")
print("================================================================================")
print(f"학습용 정상 데이터 (Train Normal): {len(X_train_norm)} 개")
print(f"평가용 데이터 (Test Total):      {len(X_test)} 개 (정상: {len(X_test_norm)}개, 이상: {len(X_outlier)}개)")

# 3. One-Class SVM 모델 생성 및 학습
# nu=0.03 (정상 데이터 중 약 3% 미만을 이상으로 볼 오차 수용범위)
ocsvm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.03)
ocsvm.fit(X_train_norm)

# 4. 예측 수행
# OneClassSVM predict: 정상 = +1, 이상 = -1
preds_raw = ocsvm.predict(X_test)
y_pred = np.where(preds_raw == -1, 1, 0)  # 이상(-1)을 1로, 정상(+1)을 0으로 변환

# Decision function 점수 (값이 작거나 음수일수록 이상일 확률 높음)
scores = -ocsvm.decision_function(X_test)

# 5. 성능 평가
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
auc_roc = roc_auc_score(y_test, scores)

print("\n================================================================================")
print("One-Class SVM 성능 평가 결과")
print("================================================================================")
print(f"Confusion Matrix:\n{cm}")
print(f"\n- True Positive (이상 탐지 성공):  {tp} / {len(X_outlier)}")
print(f"- False Positive (정상을 이상으로 오탐): {fp} / {len(X_test_norm)}")
print(f"- False Negative (이상을 정상으로 놓침): {fn} / {len(X_outlier)}")
print(f"- True Negative (정상 판정 성공):  {tn} / {len(X_test_norm)}")
print("--------------------------------------------------------------------------------")
print(f"Precision (정밀도): {precision:.4f}")
print(f"Recall (재현율/탐지율): {recall:.4f}")
print(f"F1-Score:             {f1:.4f}")
print(f"ROC-AUC Score:        {auc_roc:.4f}")
print("================================================================================")
print("\n[분류 상세 리포트]")
print(classification_report(y_test, y_pred, target_names=['Normal (0)', 'Outlier (1)']))

# 6. 시각화 (PCA 2D projection & decision region)
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_norm)
X_test_pca = pca.transform(X_test)

ocsvm_2d = OneClassSVM(kernel='rbf', gamma='scale', nu=0.03)
ocsvm_2d.fit(X_train_pca)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Subplot 1: 3D Scatter (Original features)
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.scatter(X_test_norm[:, 0], X_test_norm[:, 1], X_test_norm[:, 2], c='blue', label='Normal (Test)', alpha=0.5, s=20)
ax1.scatter(X_outlier[:, 0], X_outlier[:, 1], X_outlier[:, 2], c='red', label='Outlier (Test)', alpha=0.9, s=40, marker='^')
ax1.set_xlabel('V0_p2p')
ax1.set_ylabel('V1_flow')
ax1.set_zlabel('I_fent')
ax1.set_title('3D Feature Space & Test Data')
ax1.legend()

# Subplot 2: PCA 2D Decision Boundary
xx, yy = np.meshgrid(
    np.linspace(X_test_pca[:, 0].min() - 1, X_test_pca[:, 0].max() + 1, 300),
    np.linspace(X_test_pca[:, 1].min() - 1, X_test_pca[:, 1].max() + 1, 300)
)
Z = ocsvm_2d.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

axes[1].contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu, alpha=0.4)
axes[1].contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
axes[1].scatter(X_test_pca[y_test == 0, 0], X_test_pca[y_test == 0, 1], c='blue', label='Normal', alpha=0.5, s=20)
axes[1].scatter(X_test_pca[y_test == 1, 0], X_test_pca[y_test == 1, 1], c='red', label='Outlier', alpha=0.9, s=40, marker='^')
axes[1].set_title('One-Class SVM Decision Boundary (PCA 2D)')
axes[1].set_xlabel('PCA Component 1')
axes[1].set_ylabel('PCA Component 2')
axes[1].legend()

plt.tight_layout()
plt.savefig('ocsvm_evaluation_result.png', dpi=300)
print("\n✓ 시각화 결과 저장 완료: ocsvm_evaluation_result.png")
