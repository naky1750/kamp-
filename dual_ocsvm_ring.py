"""
이중 One-Class SVM (Dual-Parameter Ring Boundary) 사전경보 시스템 (영문 라벨)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
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

# (1) 1차 타이트 모델: 점 산도 핏팅 (gamma=0.05, nu=0.005)
ocsvm_strict = OneClassSVM(kernel='rbf', gamma=0.05, nu=0.005)
ocsvm_strict.fit(X_train_norm)

# (2) 2차 널널한 모델: 넉넉한 외곽 경계 (gamma=0.005, nu=0.0001)
ocsvm_relaxed = OneClassSVM(kernel='rbf', gamma=0.005, nu=0.0001)
ocsvm_relaxed.fit(X_train_norm)

pred_strict = ocsvm_strict.predict(X_test)     # +1 (안쪽), -1 (바깥)
pred_relaxed = ocsvm_relaxed.predict(X_test)   # +1 (안쪽), -1 (바깥)

dual_ring_status = []
for p_s, p_r in zip(pred_strict, pred_relaxed):
    if p_s == 1 and p_r == 1:
        dual_ring_status.append('0_Normal')
    elif p_s == -1 and p_r == 1:
        dual_ring_status.append('1_Caution_Warning_Zone')
    else:
        dual_ring_status.append('2_Critical_Anomaly')

df_res = pd.DataFrame({
    'true_label': y_test,
    'pred_strict': pred_strict,
    'pred_relaxed': pred_relaxed,
    'status': dual_ring_status
})

print("================================================================================")
print("이중 OCSVM (nu & gamma 보정) 3단계 링 판정 결과")
print("================================================================================")
ct = pd.crosstab(df_res['status'], df_res['true_label'], margins=True)
ct.columns = ['Actual Normal(0)', 'Actual Outlier(1)', 'Total']
print(ct)

# 2D PCA 시각화
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_norm)
X_test_pca = pca.transform(X_test)

ocsvm_strict_2d = OneClassSVM(kernel='rbf', gamma=0.05, nu=0.005)
ocsvm_strict_2d.fit(X_train_pca)

ocsvm_relaxed_2d = OneClassSVM(kernel='rbf', gamma=0.005, nu=0.0001)
ocsvm_relaxed_2d.fit(X_train_pca)

fig, ax = plt.subplots(figsize=(10, 7))

xx, yy = np.meshgrid(
    np.linspace(X_test_pca[:, 0].min() - 0.5, X_test_pca[:, 0].max() + 0.5, 400),
    np.linspace(X_test_pca[:, 1].min() - 0.5, X_test_pca[:, 1].max() + 0.5, 400)
)

Z_s = ocsvm_strict_2d.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
Z_r = ocsvm_relaxed_2d.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# 1차 타이트 경계선 (Inner Boundary)
c1 = ax.contour(xx, yy, Z_s, levels=[0], colors=['orange'], linewidths=2.5, linestyles='--')
# 2차 널널한 경계선 (Outer Boundary)
c2 = ax.contour(xx, yy, Z_r, levels=[0], colors=['darkred'], linewidths=3.0)

colors_map = {
    '0_Normal': 'green',
    '1_Caution_Warning_Zone': 'darkorange',
    '2_Critical_Anomaly': 'red'
}

for status_val, col in colors_map.items():
    idx_mask = (df_res['status'] == status_val).values
    ax.scatter(X_test_pca[idx_mask, 0], X_test_pca[idx_mask, 1], c=col, label=status_val, alpha=0.7, s=30)

ax.set_title('Dual OCSVM Ring Boundary System\n(Inner: Strict gamma=0.05 | Outer: Relaxed gamma=0.005)', fontsize=13, fontweight='bold')
ax.set_xlabel('PCA Component 1')
ax.set_ylabel('PCA Component 2')
ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig('ocsvm_dual_ring_dashboard.png', dpi=300)
print("\n✓ 이중 OCSVM 링 경계 시각화 차트 저장 완료: ocsvm_dual_ring_dashboard.png")
