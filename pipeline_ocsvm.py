"""
프레스 기계 이상 탐지 통합 파이프라인 (Raw CSV → 특징 추출 → One-Class SVM 추론)
================================================================================
사용 방법:
1. 학습 및 추론:
   python pipeline_ocsvm.py --train_csv press_data_normal.csv --test_csv outlier_data.csv

2. 추론 전용:
   python pipeline_ocsvm.py --test_csv outlier_data.csv --model_path ocsvm_model.joblib
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import joblib
from typing import Tuple, List, Dict

# ============================================================================
# 1. 특징 추출 설정 및 파라미터
# ============================================================================

NORMALIZATION_PARAMS = {
    'V0_p2p_mean': 0.236445,
    'V0_p2p_std': 0.085320,
    'V1_flow_mean': 0.084717,
    'V1_flow_std': 0.119349,
    'I_fent_mean': 0.051404,
    'I_fent_std': 0.018462
}

L = 17                  # 윈도우 길이 (1.7초)
STRIDE = 4              # 윈도우 이동 간격 (0.4초)
FS = 10.0               # 샘플링 주파수 10Hz
GAP_THRESHOLD = 500     # Burst 분할 간격 (500ms)
LOWFREQ_THRESHOLD = 1.2 # 저주파 기준 (Hz)


def extract_features_from_window(v0: np.ndarray, v1: np.ndarray, cur: np.ndarray) -> Tuple[float, float, float]:
    """17개 샘플 윈도우에서 3개 핵심 특징(V0_p2p, V1_flow, I_fent) 추출"""
    # 1) V0_p2p (진동0 Peak-to-Peak)
    p2p = float(np.max(v0) - np.min(v0))
    
    # 2) V1_flow (진동1 저주파 비율 0~1.2Hz)
    fft_v1 = np.abs(np.fft.rfft(v1)) ** 2
    sum_v1 = np.sum(fft_v1)
    if sum_v1 < 1e-10:
        flow = 0.0
    else:
        norm_v1 = fft_v1 / sum_v1
        freqs = np.fft.rfftfreq(len(v1), 1/FS)
        flow = float(np.sum(norm_v1[freqs < LOWFREQ_THRESHOLD]))
        
    # 3) I_fent (전류 스펙트럼 엔트로피)
    fft_i = np.abs(np.fft.rfft(cur)) ** 2
    sum_i = np.sum(fft_i)
    if sum_i < 1e-10:
        fent = 0.0
    else:
        norm_i = fft_i / sum_i
        pxx_safe = norm_i + 1e-10
        fent = float(-np.sum(pxx_safe * np.log2(pxx_safe)))
        
    return p2p, flow, fent


def raw_csv_to_features(csv_path: str) -> pd.DataFrame:
    """Raw CSV 경로를 받아 Burst 분할 후 윈도우 슬라이딩 특징 데이터프레임으로 변환"""
    df = pd.read_csv(csv_path)
    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
    
    # Burst 분할
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
            
            # Z-Score 정규화
            p2p_norm = (p2p - NORMALIZATION_PARAMS['V0_p2p_mean']) / NORMALIZATION_PARAMS['V0_p2p_std']
            flow_norm = (flow - NORMALIZATION_PARAMS['V1_flow_mean']) / NORMALIZATION_PARAMS['V1_flow_std']
            fent_norm = (fent - NORMALIZATION_PARAMS['I_fent_mean']) / NORMALIZATION_PARAMS['I_fent_std']
            
            rows.append({
                'window_id': window_id,
                'burst_id': burst_id,
                'start_time': ts_arr[start_idx],
                'V0_p2p': p2p_norm,
                'V1_flow': flow_norm,
                'I_fent': fent_norm,
            })
            window_id += 1
            
    return pd.DataFrame(rows)


# ============================================================================
# 2. 통합 파이프라인 클래스
# ============================================================================

from sklearn.svm import OneClassSVM

class PressAnomalyPipeline:
    def __init__(self, gamma: float = 0.02, nu: float = 0.0002):
        self.gamma = gamma
        self.nu = nu
        self.model = OneClassSVM(kernel='rbf', gamma=self.gamma, nu=self.nu)
        self.is_fitted = False
        
    def fit(self, train_raw_csv: str):
        """정상 Raw CSV 데이터를 전처리하여 One-Class SVM 학습"""
        print(f"🔄 [1/2] 학습 데이터 특징 변환 중: {train_raw_csv}")
        df_feat = raw_csv_to_features(train_raw_csv)
        X_train = df_feat[['V0_p2p', 'V1_flow', 'I_fent']].values
        
        print(f"🧠 [2/2] One-Class SVM 학습 중 ({len(X_train)}개 윈도우, gamma={self.gamma}, nu={self.nu})...")
        self.model.fit(X_train)
        self.is_fitted = True
        print("✓ 학습 완료!")
        
    def predict_raw_csv(self, test_raw_csv: str, output_csv: str = None) -> pd.DataFrame:
        """Raw CSV 데이터를 입력받아 전처리 및 이상 여부 판정"""
        if not self.is_fitted:
            raise ValueError("모델이 아직 학습되지 않았습니다. fit()을 실행하거나 저장된 모델을 로드하세요.")
            
        print(f"\n🔍 [추론] Raw CSV 데이터 전처리 중: {test_raw_csv}")
        df_feat = raw_csv_to_features(test_raw_csv)
        X_test = df_feat[['V0_p2p', 'V1_flow', 'I_fent']].values
        
        # OCSVM 예측 (1: 정상, -1: 이상)
        preds_raw = self.model.predict(X_test)
        scores = -self.model.decision_function(X_test)  # 클수록 이상일 확률 높음
        
        # 0: 정상, 1: 이상 변환
        df_feat['anomaly_pred'] = np.where(preds_raw == -1, 1, 0)
        df_feat['anomaly_score'] = scores
        df_feat['status_label'] = np.where(preds_raw == -1, 'OUTLIER (이상)', 'NORMAL (정상)')
        
        total_win = len(df_feat)
        outlier_win = (df_feat['anomaly_pred'] == 1).sum()
        normal_win = total_win - outlier_win
        outlier_ratio = (outlier_win / total_win) * 100 if total_win > 0 else 0
        
        print("\n================================================================================")
        print(f"📊 진단 분석 리포트: {os.path.basename(test_raw_csv)}")
        print("================================================================================")
        print(f"- 전체 윈도우 수: {total_win} 개")
        print(f"- 정상 판정 윈도우: {normal_win} 개 ({100-outlier_ratio:.1f}%)")
        print(f"- 이상 판정 윈도우: {outlier_win} 개 ({outlier_ratio:.1f}%)")
        print("--------------------------------------------------------------------------------")
        if outlier_win > 0:
            print(f"⚠️ 경고: 입력된 설비 데이터에서 총 {outlier_win}개의 이상 윈도우가 감지되었습니다!")
        else:
            print("✅ 정상: 모든 공정 윈도우가 정상 범위 내에 존재합니다.")
        print("================================================================================\n")
        
        if output_csv:
            df_feat.to_csv(output_csv, index=False, encoding='utf-8-sig')
            print(f"💾 진단 결과 CSV 저장 완료: {output_csv}")
            
        return df_feat

    def save_model(self, filepath: str = "ocsvm_model.joblib"):
        """학습된 모델 저장"""
        joblib.dump(self, filepath)
        print(f"💾 모델 저장 완료: {filepath}")

    @staticmethod
    def load_model(filepath: str = "ocsvm_model.joblib"):
        """저장된 모델 로드"""
        pipeline = joblib.load(filepath)
        print(f"📂 모델 로드 완료: {filepath}")
        return pipeline


# ============================================================================
# 3. CLI 실행 메인 함수
# ============================================================================

if __name__ == "__main__":
    import sys
    
    print("================================================================================")
    print("프레스 기계 이상 탐지 통합 파이프라인 (One-Class SVM)")
    print("================================================================================")
    
    # 기본 실행 예시 (학습: press_data_normal.csv, 추론: outlier_data.csv)
    train_file = "press_data_normal.csv"
    test_file = "outlier_data.csv"
    
    pipeline = PressAnomalyPipeline(gamma=0.02, nu=0.0002)
    
    # 1. 학습
    pipeline.fit(train_file)
    pipeline.save_model("ocsvm_pipeline.joblib")
    
    # 2. 추론 (이상 데이터 테스트)
    result_outlier = pipeline.predict_raw_csv(test_file, output_csv="outlier_diagnosis_result.csv")
    
    # 3. 추론 (정상 데이터 테스트)
    result_normal = pipeline.predict_raw_csv(train_file, output_csv="normal_diagnosis_result.csv")
