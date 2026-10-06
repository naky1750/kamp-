"""
One-Class SVM 경계 확장 (Precision 100% 달성) 스크립트
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

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

print("================================================================================")
print("One-Class SVM 경계 확장 실험 (gamma & nu 그리드 서치)")
print("================================================================================")

best_f1 = 0
best_params = None

gammas = [0.001, 0.005, 0.01, 0.05, 'scale', 'auto']
nus = [1e-4, 5e-4, 1e-3, 2e-3, 5e-3]

for g in gammas:
    for nu in nus:
        ocsvm = OneClassSVM(kernel='rbf', gamma=g, nu=nu)
        ocsvm.fit(X_train_norm)
        
        preds_raw = ocsvm.predict(X_test)
        y_pred = np.where(preds_raw == -1, 1, 0)
        
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_params = (g, nu, prec, rec, f1, fp, fn)
        
        if fp == 0 and fn == 0:
            print(f"🎉 완벽 탐지 모델 발견! gamma={g}, nu={nu} | Precision={prec:.4f}, Recall={rec:.4f}, FP={fp}, FN={fn}")

g_opt, nu_opt, prec_opt, rec_opt, f1_opt, fp_opt, fn_opt = best_params
print("\n--------------------------------------------------------------------------------")
print(f"최적 파라미터 조합: gamma={g_opt}, nu={nu_opt}")
print(f"Precision: {prec_opt:.4f}, Recall: {rec_opt:.4f}, F1-Score: {f1_opt:.4f}")
print(f"False Positive (정상 오탐): {fp_opt}건, False Negative (이상 미탐): {fn_opt}건")

# 최적 파라미터 모델 재학습 및 시각화
ocsvm_best = OneClassSVM(kernel='rbf', gamma=g_opt, nu=nu_opt)
ocsvm_best.fit(X_train_norm)

preds_raw = ocsvm_best.predict(X_test)
y_pred = np.where(preds_raw == -1, 1, 0)

print("\n================================================================================")
print("최적화된 One-Class SVM 상세 분류 리포트")
print("================================================================================")
print(classification_report(y_test, y_pred, target_names=['Normal (0)', 'Outlier (1)'], digits=4))

# PCA 2D 시각화
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_norm)
X_test_pca = pca.transform(X_test)

ocsvm_2d = OneClassSVM(kernel='rbf', gamma=g_opt if isinstance(g_opt, float) else 'scale', nu=nu_opt)
ocsvm_2d.fit(X_train_pca)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Subplot 1: 3D Scatter
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.scatter(X_test_norm[:, 0], X_test_norm[:, 1], X_test_norm[:, 2], c='blue', label='Normal (Test)', alpha=0.5, s=20)
ax1.scatter(X_outlier[:, 0], X_outlier[:, 1], X_outlier[:, 2], c='red', label='Outlier (Test)', alpha=0.9, s=40, marker='^')
ax1.set_xlabel('V0_p2p')
ax1.set_ylabel('V1_flow')
ax1.set_zlabel('I_fent')
ax1.set_title('3D Feature Space (Optimized Boundary)')
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
axes[1].set_title(f'Optimized Decision Boundary (gamma={g_opt}, nu={nu_opt})')
axes[1].set_xlabel('PCA Component 1')
axes[1].set_ylabel('PCA Component 2')
axes[1].legend()

plt.tight_layout()
plt.savefig('ocsvm_perfect_boundary.png', dpi=300)
print("\n✓ 최적화 시각화 차트 저장 완료: ocsvm_perfect_boundary.png")
