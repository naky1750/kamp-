"""
One-Class SVM nu 파라미터 튜닝 실험
"""
import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split

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

print("nu 파라미터 변화에 따른 One-Class SVM 성능 비교:")
print(f"{'nu':<8} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'FP (오탐)':<10} | {'FN (미탐)':<10}")
print("-" * 65)

for nu in [0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1]:
    ocsvm = OneClassSVM(kernel='rbf', gamma='scale', nu=nu)
    ocsvm.fit(X_train_norm)
    
    preds_raw = ocsvm.predict(X_test)
    y_pred = np.where(preds_raw == -1, 1, 0)
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)
    
    print(f"{nu:<8.3f} | {prec:<10.4f} | {rec:<10.4f} | {f1:<10.4f} | {fp:<10} | {fn:<10}")
