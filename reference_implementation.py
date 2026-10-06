"""
프레스 기계 이상 탐지: 특징 추출 참고 구현코드
================================================================
목적: Raw CSV → 3개 특징 시계열 (V0_p2p, V1_flow, I_fent)

작성자: Data Science Team
최종 수정: 2026-10-06
================================================================
"""

import numpy as np
import pandas as pd
import json
from typing import List, Dict, Tuple

# ============================================================================
# 1. 정규화 파라미터 정의 (또는 JSON에서 로드)
# ============================================================================

NORMALIZATION_PARAMS = {
    'V0_p2p_mean': 0.236445,
    'V0_p2p_std': 0.085320,
    'V1_flow_mean': 0.084717,
    'V1_flow_std': 0.119349,
    'I_fent_mean': 0.051404,
    'I_fent_std': 0.018462
}

# ============================================================================
# 2. 기본 파라미터
# ============================================================================

L = 17              # 윈도우 길이 (샘플)
STRIDE = 4          # 윈도우 간격 (샘플)
FS = 10.0           # 샘플링 주파수 (Hz)
GAP_THRESHOLD = 500 # burst 분할 기준 (ms)
LOWFREQ_THRESHOLD = 1.2  # 저주파 범위 (Hz)


# ============================================================================
# 3. 특징 계산 함수들
# ============================================================================

def calculate_p2p(signal_window: np.ndarray) -> float:
    """
    V0_p2p: 진동0의 피크-피크 값
    
    Args:
        signal_window: 17개 샘플의 신호 배열
    
    Returns:
        float: 최대값 - 최소값
    """
    signal = np.asarray(signal_window, dtype=float)
    if len(signal) == 0:
        return 0.0
    
    p2p = np.max(signal) - np.min(signal)
    return float(p2p)


def calculate_flow(signal_window: np.ndarray, fs: float = FS) -> float:
    """
    V1_flow: 진동1의 저주파 비율 (0~1.2Hz 파워 합)
    
    Args:
        signal_window: 17개 샘플의 신호 배열
        fs: 샘플링 주파수 (Hz)
    
    Returns:
        float: 0~1 사이의 저주파 비율
    """
    signal = np.asarray(signal_window, dtype=float)
    if len(signal) < 2:
        return 0.0
    
    # FFT 수행
    fft_complex = np.fft.rfft(signal)
    fft_mag = np.abs(fft_complex)
    
    # 주파수 축 생성
    freqs = np.fft.rfftfreq(len(signal), 1/fs)
    
    # 파워 스펙트럼 계산
    pxx = fft_mag ** 2
    
    # 정규화 (전체 합 = 1)
    pxx_sum = np.sum(pxx)
    if pxx_sum < 1e-10:
        return 0.0
    
    pxx_norm = pxx / pxx_sum
    
    # 저주파 범위 선택
    lowfreq_mask = freqs < LOWFREQ_THRESHOLD
    
    # 저주파 파워 합
    flow = np.sum(pxx_norm[lowfreq_mask])
    
    return float(flow)


def calculate_fent(signal_window: np.ndarray, fs: float = FS) -> float:
    """
    I_fent: 전류의 스펙트럼 엔트로피 (Shannon entropy)
    
    엔트로피가 높으면: 많은 주파수가 골고루 분포 (이상 신호)
    엔트로피가 낮으면: 특정 주파수에 집중 (정상 신호)
    
    Args:
        signal_window: 17개 샘플의 신호 배열
        fs: 샘플링 주파수 (Hz)
    
    Returns:
        float: 0~log2(N) 범위의 엔트로피 값
    """
    signal = np.asarray(signal_window, dtype=float)
    if len(signal) < 2:
        return 0.0
    
    # FFT 수행
    fft_mag = np.abs(np.fft.rfft(signal))
    
    # 파워 스펙트럼 계산 및 정규화
    pxx = fft_mag ** 2
    pxx_sum = np.sum(pxx)
    if pxx_sum < 1e-10:
        return 0.0
    
    pxx_norm = pxx / pxx_sum
    
    # Shannon Entropy: H = -Σ p_i * log2(p_i)
    # 0으로 나누기 방지: p_i에 작은 값 추가
    pxx_safe = pxx_norm + 1e-10
    
    fent = -np.sum(pxx_safe * np.log2(pxx_safe))
    
    return float(fent)


def normalize_value(raw_value: float, mean: float, std: float) -> float:
    """
    Z-score 정규화: (값 - 평균) / 표준편차
    
    Args:
        raw_value: 원본 값
        mean: 평균 (정상 데이터 기준)
        std: 표준편차 (정상 데이터 기준)
    
    Returns:
        float: 정규화된 값
    """
    if std < 1e-10:
        return 0.0
    
    return float((raw_value - mean) / std)


# ============================================================================
# 4. Burst 분할 함수
# ============================================================================

def split_bursts(df: pd.DataFrame, gap_threshold_ms: float = GAP_THRESHOLD) -> List[pd.DataFrame]:
    """
    시간 기반으로 데이터를 연속 구간(burst)으로 분할
    
    500ms 이상의 시간 간격이 있으면 burst 끝으로 판정
    길이 17 이상인 burst만 반환
    
    Args:
        df: TimeStamp 열이 datetime 형식인 DataFrame
        gap_threshold_ms: burst 분할 기준 시간 간격 (ms)
    
    Returns:
        List[pd.DataFrame]: 각 burst의 DataFrame 리스트
    """
    # 시간 간격 계산
    df['time_diff_ms'] = df['TimeStamp'].diff().dt.total_seconds() * 1000
    
    # 공백 판정
    df['is_gap'] = df['time_diff_ms'] > gap_threshold_ms
    df['is_gap'] = df['is_gap'].fillna(False)
    
    # burst 그룹화
    df['burst_id'] = df['is_gap'].cumsum()
    
    # 각 burst 추출
    bursts = []
    for burst_id, burst_df in df.groupby('burst_id'):
        if len(burst_df) >= L:
            bursts.append(burst_df.reset_index(drop=True))
    
    return bursts


# ============================================================================
# 5. 메인 처리 함수
# ============================================================================

def process_raw_data(input_csv_path: str, 
                     output_csv_path: str,
                     norm_params: Dict = None) -> pd.DataFrame:
    """
    Raw CSV 데이터에서 3개 특징을 추출하여 시계열 데이터로 변환
    
    처리 과정:
    1. 입력 CSV 로드
    2. 시간 기반 burst 분할
    3. 각 burst에서 L=17, stride=4 윈도우 슬라이딩
    4. 각 윈도우에서 3개 특징 추출
    5. Z-score 정규화
    6. 출력 CSV 저장
    
    Args:
        input_csv_path: 입력 CSV 경로 (TimeStamp, AI0_Vibration, AI1_Vibration, AI2_Current, Equipment_state)
        output_csv_path: 출력 CSV 경로
        norm_params: 정규화 파라미터 dict (기본값: NORMALIZATION_PARAMS)
    
    Returns:
        pd.DataFrame: 처리된 데이터 (CSV로도 저장됨)
    
    Example:
        >>> result = process_raw_data(
        ...     'press_data_normal.csv',
        ...     'windows_features_L17_s4.csv'
        ... )
        >>> print(f"생성됨: {len(result)}개 윈도우")
    """
    
    if norm_params is None:
        norm_params = NORMALIZATION_PARAMS
    
    print(f"[1/5] 입력 파일 로드 중: {input_csv_path}")
    df = pd.read_csv(input_csv_path)
    
    # TimeStamp 파싱
    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
    
    print(f"      → {len(df)} 행 로드됨")
    
    # Burst 분할
    print(f"[2/5] Burst 분할 중 (gap > {GAP_THRESHOLD}ms)")
    bursts = split_bursts(df)
    print(f"      → {len(bursts)}개 burst 생성 (L ≥ {L})")
    
    # 윈도우 처리
    print(f"[3/5] 윈도우 슬라이딩 중 (L={L}, stride={STRIDE})")
    results = []
    window_id = 0
    
    for burst_id, burst_df in enumerate(bursts):
        signal_v0 = burst_df['AI0_Vibration'].values
        signal_v1 = burst_df['AI1_Vibration'].values
        signal_i = burst_df['AI2_Current'].values
        
        # stride만큼 이동하며 윈도우 생성
        for start_idx in range(0, len(burst_df) - L + 1, STRIDE):
            window_v0 = signal_v0[start_idx:start_idx + L]
            window_v1 = signal_v1[start_idx:start_idx + L]
            window_i = signal_i[start_idx:start_idx + L]
            
            # 특징 계산
            v0_p2p_raw = calculate_p2p(window_v0)
            v1_flow_raw = calculate_flow(window_v1)
            i_fent_raw = calculate_fent(window_i)
            
            # 정규화
            v0_norm = normalize_value(
                v0_p2p_raw,
                norm_params['V0_p2p_mean'],
                norm_params['V0_p2p_std']
            )
            v1_norm = normalize_value(
                v1_flow_raw,
                norm_params['V1_flow_mean'],
                norm_params['V1_flow_std']
            )
            i_norm = normalize_value(
                i_fent_raw,
                norm_params['I_fent_mean'],
                norm_params['I_fent_std']
            )
            
            timestamp = burst_df.iloc[start_idx]['TimeStamp']
            label = burst_df.iloc[start_idx]['Equipment_state']
            
            results.append({
                'timestamp': timestamp,
                'window_id': window_id,
                'burst_id': burst_id,
                'V0_p2p': v0_norm,
                'V1_flow': v1_norm,
                'I_fent': i_norm,
                'label': int(label)
            })
            
            window_id += 1
    
    print(f"      → {window_id}개 윈도우 생성")
    
    # 결과 DataFrame 생성
    print(f"[4/5] 결과 정리 중")
    result_df = pd.DataFrame(results)
    
    # 저장
    print(f"[5/5] CSV 저장 중: {output_csv_path}")
    result_df.to_csv(output_csv_path, index=False)
    
    print(f"\n✓ 완료!")
    print(f"  입력:  {len(df)} 행")
    print(f"  burst: {len(bursts)}개")
    print(f"  출력:  {len(result_df)} 윈도우")
    print(f"  파일:  {output_csv_path}")
    
    return result_df


# ============================================================================
# 6. 사용 예시
# ============================================================================

if __name__ == '__main__':
    # 정상 데이터 처리
    print("=" * 80)
    print("정상 데이터 처리")
    print("=" * 80)
    normal_result = process_raw_data(
        input_csv_path='press_data_normal.csv',
        output_csv_path='windows_features_L17_s4_normal.csv'
    )
    
    print(f"\n정상 데이터 통계 (정규화 후):")
    print(f"  V0_p2p:  mean={normal_result['V0_p2p'].mean():.3f}, "
          f"std={normal_result['V0_p2p'].std():.3f}")
    print(f"  V1_flow: mean={normal_result['V1_flow'].mean():.3f}, "
          f"std={normal_result['V1_flow'].std():.3f}")
    print(f"  I_fent:  mean={normal_result['I_fent'].mean():.3f}, "
          f"std={normal_result['I_fent'].std():.3f}")
    
    print("\n" + "=" * 80)
    print("이상 데이터 처리")
    print("=" * 80)
    outlier_result = process_raw_data(
        input_csv_path='outlier_data.csv',
        output_csv_path='windows_features_L17_s4_outlier.csv'
    )
    
    print(f"\n이상 데이터 통계 (정규화 후):")
    print(f"  V0_p2p:  mean={outlier_result['V0_p2p'].mean():.3f}, "
          f"std={outlier_result['V0_p2p'].std():.3f}")
    print(f"  V1_flow: mean={outlier_result['V1_flow'].mean():.3f}, "
          f"std={outlier_result['V1_flow'].std():.3f}")
    print(f"  I_fent:  mean={outlier_result['I_fent'].mean():.3f}, "
          f"std={outlier_result['I_fent'].std():.3f}")
    
    print("\n" + "=" * 80)
    print("분리 지표 확인")
    print("=" * 80)
    
    normal_mean = normal_result[['V0_p2p', 'V1_flow', 'I_fent']].mean()
    outlier_mean = outlier_result[['V0_p2p', 'V1_flow', 'I_fent']].mean()
    
    print(f"\n특징별 차이 (이상 - 정상):")
    for feat in ['V0_p2p', 'V1_flow', 'I_fent']:
        diff = outlier_mean[feat] - normal_mean[feat]
        direction = "↑ 증가" if diff > 0 else "↓ 감소"
        print(f"  {feat:<8} {diff:+.3f}  {direction}")
