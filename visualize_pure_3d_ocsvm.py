"""
PCA 차원축소 없는 순수 3차원 피처 공간 상의 One-Class SVM RBF 커널 경계 시각화
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split

# 1. 데이터 로드 (원본 3개 피처)
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

# 2. PCA 없이 원본 3D 공간 상에서 One-Class SVM RBF 학습
ocsvm_3d = OneClassSVM(kernel='rbf', gamma=0.02, nu=0.0002)
ocsvm_3d.fit(X_train_norm)

# 3. 3D 시각화 구성
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(1, 1, 1, projection='3d')

# (1) 정상 테스트 데이터 산점도
ax.scatter(
    X_test_norm[:, 0], X_test_norm[:, 1], X_test_norm[:, 2],
    c='#2563eb', label='Normal Data (Test)', alpha=0.5, s=20
)

# (2) 이상 테스트 데이터 산점도
ax.scatter(
    X_outlier[:, 0], X_outlier[:, 1], X_outlier[:, 2],
    c='#dc2626', label='Outlier Data (Test)', alpha=0.9, s=50, marker='^'
)

# 4. 3D Decision Surface Mesh 생성 (PCA 없이 원본 공간)
x_min, x_max = X_normal[:, 0].min() - 0.5, X_normal[:, 0].max() + 0.5
y_min, y_max = X_normal[:, 1].min() - 0.5, X_normal[:, 1].max() + 0.5
z_min, z_max = X_normal[:, 2].min() - 0.5, X_normal[:, 2].max() + 0.5

# 3D Grid
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 40),
    np.linspace(y_min, y_max, 40)
)

# mean Z 축 평면에서의 RBF 커널 결정 경계 등고선 슬라이스 시각화
z_mid = X_normal[:, 2].mean()
grid_mid = np.c_[xx.ravel(), yy.ravel(), np.full(xx.size, z_mid)]
Z_mid = ocsvm_3d.decision_function(grid_mid).reshape(xx.shape)

ax.contourf(xx, yy, Z_mid, zdir='z', offset=z_min, cmap=plt.cm.PuBu, alpha=0.4)
ax.contour(xx, yy, Z_mid, levels=[0], zdir='z', offset=z_min, colors='darkred', linewidths=2)

ax.set_title("Pure 3D Feature Space Boundary (No PCA Projection)\nOriginal Features: V0_p2p, V1_flow, I_fent", fontsize=14, fontweight='bold')
ax.set_xlabel('V0_p2p (Vibration 0 Peak-to-Peak)', labelpad=10)
ax.set_ylabel('V1_flow (Vibration 1 Low-Freq Ratio)', labelpad=10)
ax.set_zlabel('I_fent (Current Spectral Entropy)', labelpad=10)
ax.legend(loc='upper right', fontsize=11)

# 보기 좋은 시각 각도 조절
ax.view_init(elev=25, azim=135)

plt.tight_layout()
plt.savefig('pure_3d_ocsvm_boundary.png', dpi=300)
print("✓ PCA 없는 순수 3차원 피처 공간 시각화 완성: pure_3d_ocsvm_boundary.png")
