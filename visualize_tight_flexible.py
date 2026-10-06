"""
점산도 굴곡을 밀착 추종하는 Tight & Flexible One-Class SVM 경계 비교 (영문 라벨)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
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

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_norm)
X_test_pca = pca.transform(X_test)

fig, axes = plt.subplots(2, 2, figsize=(16, 13))
axes = axes.ravel()

configs = [
    ("1. Overly Smooth Elliptical (gamma=0.001, nu=0.0001)", 0.001, 0.0001, 0.0),
    ("2. Tight Non-Linear Curve (gamma=0.05, nu=0.0005)", 0.05, 0.0005, 0.0),
    ("3. Tight Curve + Threshold Offset -0.1 (gamma=0.05)", 0.05, 0.001, -0.1),
    ("4. GOLDILOCKS BOUNDARY (gamma=0.02, nu=0.0002)", 0.02, 0.0002, 0.0),
]

xx, yy = np.meshgrid(
    np.linspace(X_test_pca[:, 0].min() - 0.5, X_test_pca[:, 0].max() + 0.5, 300),
    np.linspace(X_test_pca[:, 1].min() - 0.5, X_test_pca[:, 1].max() + 0.5, 300)
)

for idx, (title, g, nu, threshold_offset) in enumerate(configs):
    ocsvm = OneClassSVM(kernel='rbf', gamma=g, nu=nu)
    ocsvm.fit(X_train_pca)
    
    scores = ocsvm.decision_function(X_test_pca)
    y_pred = np.where(scores < threshold_offset, 1, 0)
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)
    
    Z = ocsvm.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax = axes[idx]
    ax.contourf(xx, yy, Z, levels=np.linspace(Z.min(), threshold_offset, 7), cmap=plt.cm.PuBu, alpha=0.3)
    ax.contour(xx, yy, Z, levels=[threshold_offset], linewidths=2.5, colors='darkred')
    
    ax.scatter(X_test_pca[y_test == 0, 0], X_test_pca[y_test == 0, 1], c='blue', label='Normal (0)', alpha=0.4, s=15)
    ax.scatter(X_test_pca[y_test == 1, 0], X_test_pca[y_test == 1, 1], c='red', label='Outlier (1)', alpha=0.9, s=35, marker='^')
    
    ax.set_title(f"{title}\nPrecision: {prec:.4f} | Recall: {rec:.4f} | FP: {fp} samples", fontsize=11, fontweight='bold')
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig('ocsvm_tight_boundary_comparison.png', dpi=300)
print("✓ 깔끔한 라벨링 비교 그래프 저장 완료: ocsvm_tight_boundary_comparison.png")
