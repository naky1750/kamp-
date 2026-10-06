"""
공정 조건 위험도 분석 및 2단계 사전경보(Dual Threshold) 이상 탐지 시스템
================================================================================
목적:
1. 2단계 경계 (정상 🟢 - 주의 🟡 - 위험 🔴) 설정으로 현장 작업자 사전 경보 구현
2. False Positive / False Negative 집중 공정 조건 및 변수 간 상호작용 분석
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import OneClassSVM
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.decomposition import PCA

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

# 2. One-Class SVM 모델 (점 산도 추종 파라미터: gamma=0.02, nu=0.0002)
ocsvm = OneClassSVM(kernel='rbf', gamma=0.02, nu=0.0002)
ocsvm.fit(X_train_norm)

# raw decision function score (클수록 이상에 가까움: -decision_function)
scores = -ocsvm.decision_function(X_test)

# 3. 2단계 경계 (Dual Threshold) 정의
# Score < 0.0: 정상 (Normal)
# 0.0 <= Score < 0.3: 주의 / 사전경보 (Caution / Warning Zone - 2번째 경계!)
# Score >= 0.3: 위험 / 불량 (Critical Anomaly)
th_warning = 0.0
th_critical = 0.3

zone_labels = []
for s in scores:
    if s < th_warning:
        zone_labels.append('0_Normal')
    elif s < th_critical:
        zone_labels.append('1_Caution_Warning')
    else:
        zone_labels.append('2_Critical_Anomaly')

df_analysis = pd.DataFrame(X_test, columns=features)
df_analysis['true_label'] = y_test
df_analysis['decision_score'] = scores
df_analysis['risk_zone'] = zone_labels

# 4. 널널한 경계에서의 FP / FN 및 공정 변수 상호작용 분석
# 변수간 상호작용 항(Interaction features) 생성
df_analysis['inter_V0_V1'] = df_analysis['V0_p2p'] * df_analysis['V1_flow']
df_analysis['inter_V0_I'] = df_analysis['V0_p2p'] * df_analysis['I_fent']
df_analysis['inter_V1_I'] = df_analysis['V1_flow'] * df_analysis['I_fent']

print("================================================================================")
print("1. 3단계 공정 위험도(Risk Zone) 분포 분석")
print("================================================================================")
zone_summary = pd.crosstab(df_analysis['risk_zone'], df_analysis['true_label'], margins=True)
zone_summary.columns = ['실제 정상(0)', '실제 이상(1)', '전체(Total)']
print(zone_summary)

print("\n================================================================================")
print("2. 변수간 상호작용 및 공정 변수 영역별 평균")
print("================================================================================")
group_means = df_analysis.groupby('risk_zone')[features + ['inter_V0_V1', 'inter_V0_I', 'inter_V1_I']].mean()
print(group_means.round(4))

# 5. 의사결정나무(Decision Tree)를 이용한 주의/위험 공정 조건 룰 추출
tree = DecisionTreeClassifier(max_depth=3)
tree.fit(df_analysis[features], df_analysis['risk_zone'])
tree_rules = export_text(tree, feature_names=features)

print("\n================================================================================")
print("3. 작업자용 주요 공정 주의/위험 도출 룰 (Rule Extraction)")
print("================================================================================")
print(tree_rules)

# 6. 시각화 (Dual Threshold 경계선 및 3단계 Zone 대시보드)
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_norm)
X_test_pca = pca.transform(X_test)

ocsvm_2d = OneClassSVM(kernel='rbf', gamma=0.02, nu=0.0002)
ocsvm_2d.fit(X_train_pca)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Subplot 1: 3단계 Zone 분포 Scatter Plot
colors = {'0_Normal': 'green', '1_Caution_Warning': 'orange', '2_Critical_Anomaly': 'red'}
for zone, color in colors.items():
    sub = df_analysis[df_analysis['risk_zone'] == zone]
    sub_pca = X_test_pca[df_analysis['risk_zone'] == zone]
    axes[0].scatter(sub_pca[:, 0], sub_pca[:, 1], c=color, label=zone, alpha=0.6, s=30)
    
axes[0].set_title('Process Risk Zone Distribution (2D PCA Projection)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('PCA Component 1')
axes[0].set_ylabel('PCA Component 2')
axes[0].legend()

# Subplot 2: 2단계 경계 (Dual Threshold Contour)
xx, yy = np.meshgrid(
    np.linspace(X_test_pca[:, 0].min() - 0.5, X_test_pca[:, 0].max() + 0.5, 300),
    np.linspace(X_test_pca[:, 1].min() - 0.5, X_test_pca[:, 1].max() + 0.5, 300)
)
Z = -ocsvm_2d.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Contour levels: Green (Normal), Yellow/Orange (Warning), Red (Critical)
contour_fill = axes[1].contourf(xx, yy, Z, levels=[-10, 0.0, 0.3, 100], colors=['#e6ffe6', '#fff5cc', '#ffe6e6'])
c1 = axes[1].contour(xx, yy, Z, levels=[0.0], colors=['darkorange'], linestyles='--', linewidths=2, label='1st Boundary (Warning)')
c2 = axes[1].contour(xx, yy, Z, levels=[0.3], colors=['darkred'], linewidths=2.5, label='2nd Boundary (Critical)')

axes[1].scatter(X_test_pca[y_test == 0, 0], X_test_pca[y_test == 0, 1], c='blue', alpha=0.3, s=15, label='Actual Normal')
axes[1].scatter(X_test_pca[y_test == 1, 0], X_test_pca[y_test == 1, 1], c='red', alpha=0.9, s=40, marker='^', label='Actual Outlier')

axes[1].set_title('Dual-Threshold Boundary Map for Operators', fontsize=13, fontweight='bold')
axes[1].set_xlabel('PCA Component 1')
axes[1].set_ylabel('PCA Component 2')
axes[1].legend(loc='upper right')

plt.tight_layout()
plt.savefig('process_warning_dashboard.png', dpi=300)
print("✓ 2단계 사전경보 대시보드 저장 완료: process_warning_dashboard.png")
