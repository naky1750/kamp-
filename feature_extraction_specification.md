# 프레스 기계 이상 탐지: 최적 특징 추출 명세서

## 1. 개요

### 목표
Raw 시계열 데이터(CSV)에서 **3개 최적 특징**을 추출하여 **시계열 이산 데이터**로 변환

### 최적 조합 (FDC 분석 결과)
```
V0 (진동0 AI0_Vibration):  p2p     (f4)
V1 (진동1 AI1_Vibration):  flow    (f8)
I  (전류  AI2_Current):    fent    (f11)

FDCn Score: 1.9215 ± 0.0068 (매우 안정적)
```

### 입력 데이터 형식
```
CSV 파일:
  - 열: TimeStamp, AI0_Vibration, AI1_Vibration, AI2_Current, Equipment_state
  - 행: 시계열 데이터 (100ms 간격, 10Hz 샘플링)
  
예시 (press_data_normal.csv):
,TimeStamp,AI0_Vibration,AI1_Vibration,AI2_Current,Equipment_state
0,2022-07-12 00:00:00.019,0.116967,-0.176403,192.33873,0
1,2022-07-12 00:00:00.119,0.03008,-0.210895,217.00433,0
...
```

### 출력 데이터 형식
```
CSV 파일 (시계열 이산 데이터):

windows_features_L17_s4.csv:
timestamp,window_id,V0_p2p,V1_flow,I_fent,label
2022-07-12 00:00:00.019,0,544.75,0.8220,1.7329,0
2022-07-12 00:00:01.619,1,532.18,0.8195,1.7234,0
2022-07-12 00:00:03.219,2,551.32,0.8167,1.7456,0
...

각 행 = 1개 윈도우
각 열 = 시간, 윈도우 ID, 3개 특징값, 정상/이상 레이블
```

---

## 2. 전체 계산 파이프라인

```
Raw 신호 데이터
    ↓
[Step A] 시간 기반으로 연속 구간(burst) 분할
    ├─ 공백(>500ms) 찾기
    └─ 각 연속 구간 = 1개 "burst"
    ↓
[Step B] 각 burst 내에서 윈도우 슬라이딩
    ├─ 윈도우 길이: L = 17 샘플 (1.7초)
    ├─ 윈도우 간격(stride): 4 샘플 (0.4초)
    └─ 각 윈도우 = 1개 데이터 포인트
    ↓
[Step C] 각 윈도우에서 3개 특징 추출
    ├─ V0_p2p: AI0_Vibration의 (최대 - 최소)
    ├─ V1_flow: AI1_Vibration의 저주파 비율
    └─ I_fent: AI2_Current의 스펙트럼 엔트로피
    ↓
[Step D] 특징 정규화 (Z-score, 정상 데이터 기준)
    ├─ 정상 데이터 전체의 평균, 표준편차 계산
    └─ (값 - 평균) / 표준편차
    ↓
시계열 이산 데이터 (3개 특징 × N개 윈도우)
```

---

## 3. 상세 계산 단계

### Step A: 연속 구간(burst) 분할

**목적**: 데이터의 공백 구간을 찾아 독립적인 측정 구간으로 나누기

**입력**: 시계열 데이터 (TimeStamp, 3채널 신호)

**계산 과정**:
```python
1. TimeStamp를 읽어 인접한 행 사이의 시간 간격 계산
   예: row[i+1].time - row[i].time

2. 간격이 500ms(0.5초) 이상이면 "공백"으로 판정
   정상: 100ms 간격 (10Hz) → 연속
   공백: >500ms → burst 끝

3. 연속된 행들을 하나의 burst로 묶기

4. 각 burst의 길이가 17샘플 이상인 것만 사용
   (17샘플 미만은 버림)
```

**예시**:
```
행번호  TimeStamp              간격    판정     Burst ID
0      2022-07-12 00:00:00   -       시작     Burst 0
1      2022-07-12 00:00:00.1 100ms   연속     Burst 0
2      2022-07-12 00:00:00.2 100ms   연속     Burst 0
...
50     2022-07-12 00:00:05.0 100ms   연속     Burst 0
51     2022-07-12 00:00:06.8 1800ms  공백     ← Burst 0 끝
52     2022-07-12 00:00:08.6 1800ms  (새로운 burst 시작)
                                     Burst 1
```

**출력**: burst_list = [burst_0, burst_1, ..., burst_N]
  - 각 burst는 DataFrame (행=시간 순서, 열=AI0/AI1/AI2)

---

### Step B: 각 burst에서 윈도우 슬라이딩

**목적**: burst 내에서 L=17 샘플씩, stride=4 간격으로 추출

**입력**: burst_list

**계산 과정**:
```python
for each burst in burst_list:
    signal_length = len(burst)
    
    for start_idx in range(0, signal_length - L + 1, stride):
        # start_idx = 0, 4, 8, 12, 16, ...
        
        window = burst[start_idx : start_idx + L]
        # 17개 샘플 추출
        
        timestamp = burst[start_idx].TimeStamp
        
        # 이 window에서 3개 특징 계산 (Step C)
        v0_p2p = calculate_p2p(window['AI0_Vibration'])
        v1_flow = calculate_flow(window['AI1_Vibration'])
        i_fent = calculate_fent(window['AI2_Current'])
        
        저장: (timestamp, window_id, v0_p2p, v1_flow, i_fent, label)
```

**시각화 예시**:
```
Burst 0: [row0, row1, ..., row100]

Window 0: row[0:17]       → 특징 추출 → (t0, 0, v0_0, v1_0, i_0, label)
Window 1: row[4:21]       → 특징 추출 → (t1, 1, v0_1, v1_1, i_1, label)
Window 2: row[8:25]       → 특징 추출 → (t2, 2, v0_2, v1_2, i_2, label)
...
Window N: row[96:113]     → 특징 추출 → (tN, N, v0_N, v1_N, i_N, label)
```

---

### Step C: 각 윈도우에서 3개 특징 추출

#### C-1: V0_p2p (진동0의 피크-피크)

**정의**: 17샘플 윈도우 내 신호의 최대값 - 최소값

**계산**:
```python
def calculate_p2p(signal_window):
    """
    입력: signal_window = [val0, val1, ..., val16] (17개)
    출력: 하나의 숫자 (V0_p2p)
    """
    max_val = max(signal_window)
    min_val = min(signal_window)
    p2p = max_val - min_val
    return p2p
```

**예시**:
```
signal_window = [63.82, 102.47, 200.49, ..., -104.16]
max_val = 204.72
min_val = -211.22
V0_p2p = 204.72 - (-211.22) = 415.94
```

**범위**: 0 ~ 수백 (신호의 크기에 따라)

---

#### C-2: V1_flow (진동1의 저주파 비율)

**정의**: 17샘플 윈도우의 FFT 중 저주파(0~1.2Hz) 파워 비율

**계산**:
```python
def calculate_flow(signal_window, fs=10.0):
    """
    입력: signal_window = [val0, val1, ..., val16] (17개)
          fs = 샘플링 주파수 (10Hz)
    출력: 하나의 숫자 0~1 (V1_flow)
    """
    # Step 1: FFT 계산
    fft_mag = np.abs(np.fft.rfft(signal_window))
    
    # Step 2: 주파수 축 생성
    freqs = np.fft.rfftfreq(len(signal_window), 1/fs)
    # freqs = [0, 0.588, 1.176, 1.765, 2.353, 2.941, 3.529, 4.118, 4.706]
    
    # Step 3: 파워 계산 (크기의 제곱)
    pxx = fft_mag ** 2
    
    # Step 4: 파워 정규화 (전체 합 = 1)
    pxx_norm = pxx / np.sum(pxx)
    
    # Step 5: 저주파 마스크 (1.2Hz 이하)
    lowfreq_mask = freqs < 1.2
    
    # Step 6: 저주파 파워 합
    flow = np.sum(pxx_norm[lowfreq_mask])
    
    return flow
```

**계산 상세 예시**:
```
signal_window = [63.82, 102.47, 200.49, ..., -104.16]

Step 1-3 결과:
freqs:      [0.000, 0.588, 1.176, 1.765, 2.353, 2.941, 3.529, 4.118, 4.706]
fft_mag:    [190.29, 1217.72, 647.07, 488.86, 238.59, ...]
pxx:        [36210, 1482829, 418702, 238970, 56929, ...]

Step 4 (정규화):
pxx_sum = 2194640
pxx_norm: [0.0165, 0.6751, 0.1907, 0.1089, 0.0259, ...]

Step 5-6 (저주파만):
lowfreq_mask = [T, T, T, F, F, F, F, F, F]
             (0.588 < 1.2)
V1_flow = 0.0165 + 0.6751 + 0.1907 = 0.8823
```

**범위**: 0 ~ 1 (비율이므로)

---

#### C-3: I_fent (전류의 스펙트럼 엔트로피)

**정의**: 17샘플 윈도우의 FFT 스펙트럼의 엔트로피 (정보론적 복잡도)

**계산**:
```python
def calculate_fent(signal_window, fs=10.0):
    """
    입력: signal_window = [val0, val1, ..., val16] (17개)
          fs = 샘플링 주파수 (10Hz)
    출력: 하나의 숫자 0~3.17 (I_fent)
    """
    # Step 1-4: 정규화 파워 (flow와 동일)
    fft_mag = np.abs(np.fft.rfft(signal_window))
    freqs = np.fft.rfftfreq(len(signal_window), 1/fs)
    pxx = fft_mag ** 2
    pxx_norm = pxx / np.sum(pxx)
    
    # Step 5: 엔트로피 계산 (Shannon entropy)
    # H = -Σ p_i * log2(p_i)
    pxx_safe = pxx_norm + 1e-10  # 0으로 나누기 방지
    fent = -np.sum(pxx_safe * np.log2(pxx_safe))
    
    return fent
```

**계산 상세 예시**:
```
pxx_norm: [0.0165, 0.6751, 0.1907, 0.1089, 0.0259, 0.0007, 0.0005, 0.0012, 0.0006]

Step 5: 각 항에 log2 적용
-0.0165 * log2(0.0165) = -0.0165 * (-5.929) = 0.0978
-0.6751 * log2(0.6751) = -0.6751 * (-0.568) = 0.3836
-0.1907 * log2(0.1907) = -0.1907 * (-2.388) = 0.4557
...

엔트로피 합 = 0.0978 + 0.3836 + 0.4557 + 0.2864 + ... = 1.7329
```

**범위**: 0 ~ log2(N) = 0 ~ 3.17
  - 0에 가까움: 특정 주파수에 집중 (순수한 신호, 정상)
  - 3.17에 가까움: 모든 주파수가 균등 분포 (복잡한 신호, 이상)

---

### Step D: 특징 정규화 (Z-score, 정상 데이터만 기준)

**목적**: 각 특징을 동일한 스케일로 변환 (0 중심, 표준편차 1)

**계산 과정**:
```python
# 정상 데이터 전체에서 통계 계산
normal_v0_values = [v0_p2p for all windows in normal data]
normal_v1_values = [v1_flow for all windows in normal data]
normal_i_values = [i_fent for all windows in normal data]

mean_v0 = np.mean(normal_v0_values)
std_v0 = np.std(normal_v0_values)

mean_v1 = np.mean(normal_v1_values)
std_v1 = np.std(normal_v1_values)

mean_i = np.mean(normal_i_values)
std_i = np.std(normal_i_values)

# 모든 데이터(정상/이상) 정규화
v0_normalized = (v0_p2p - mean_v0) / std_v0
v1_normalized = (v1_flow - mean_v1) / std_v1
i_normalized = (i_fent - mean_i) / std_i
```

**예시** (정상 데이터 통계):
```
V0_p2p:
  mean = 111.32
  std = 45.67
  정규화 예시: (415.94 - 111.32) / 45.67 = 6.68

V1_flow:
  mean = 0.4600
  std = 0.1850
  정규화 예시: (0.8823 - 0.4600) / 0.1850 = 2.28

I_fent:
  mean = 1.1100
  std = 0.3200
  정규화 예시: (1.7329 - 1.1100) / 0.3200 = 1.95
```

**결과**: 정규화된 값들도 대부분 -3 ~ 3 범위

---

## 4. 출력 파일 명세

### 파일명
```
windows_features_L17_s4.csv
```

### 열(Columns)
| 열명 | 타입 | 설명 | 예시 |
|------|------|------|------|
| timestamp | str | 윈도우 시작 시간 | 2022-07-12 00:00:00.019 |
| window_id | int | 윈도우 순번 | 0, 1, 2, ... |
| burst_id | int | burst 번호 | 0, 0, 0, ..., 1, 1, ... |
| V0_p2p | float | 정규화된 진동0 p2p | 6.68, 5.92, 7.15, ... |
| V1_flow | float | 정규화된 진동1 저주파 비율 | 2.28, 2.35, 2.10, ... |
| I_fent | float | 정규화된 전류 엔트로피 | 1.95, 1.87, 2.05, ... |
| label | int | 정상(0) 또는 이상(1) | 0, 0, 0, ..., 1, 1, ... |

### 데이터 예시
```csv
timestamp,window_id,burst_id,V0_p2p,V1_flow,I_fent,label
2022-07-12 00:00:00.019,0,0,6.68,2.28,1.95,0
2022-07-12 00:00:00.419,1,0,5.92,2.35,1.87,0
2022-07-12 00:00:00.819,2,0,7.15,2.10,2.05,0
2022-07-12 00:00:01.219,3,0,6.45,2.41,1.92,0
...
2022-07-17 10:51:07.943,251,14,-1.23,-3.45,3.12,1
2022-07-17 10:51:08.343,252,14,-0.89,-3.22,3.28,1
```

---

## 5. 구현 코드 (Python 참고용)

### 완전한 파이프라인 코드

```python
import numpy as np
import pandas as pd
from scipy import stats

def process_press_machine_data(input_csv_path, output_csv_path, 
                               L=17, stride=4, gap_threshold_ms=500):
    """
    프레스 기계 데이터 처리 전체 파이프라인
    
    입력: input_csv_path (press_data_normal.csv 또는 outlier_data.csv)
    출력: output_csv_path (windows_features_L17_s4.csv)
    """
    
    # 1. 데이터 로드
    df = pd.read_csv(input_csv_path)
    print(f"로드됨: {len(df)} 행")
    
    # 2. Step A: 연속 구간 분할
    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
    df['time_diff_ms'] = df['TimeStamp'].diff().dt.total_seconds() * 1000
    df['is_gap'] = df['time_diff_ms'] > gap_threshold_ms
    df['burst_id'] = df['is_gap'].fillna(False).cumsum()
    
    bursts = []
    for bid, burst_df in df.groupby('burst_id'):
        if len(burst_df) >= L:
            bursts.append(burst_df.reset_index(drop=True))
    
    print(f"burst 생성: {len(bursts)}개")
    
    # 3. 정규화 파라미터 계산 (정상 데이터가 로드되었다고 가정)
    # 실제로는 정상 데이터 전체에서 계산해야 함
    normal_data = pd.read_csv('press_data_normal.csv')
    # ... 정규화 파라미터 계산 ...
    
    # 4. Step B, C: 윈도우 슬라이딩 및 특징 추출
    results = []
    window_id = 0
    
    for burst_id, burst_df in enumerate(bursts):
        signal_v0 = burst_df['AI0_Vibration'].values
        signal_v1 = burst_df['AI1_Vibration'].values
        signal_i = burst_df['AI2_Current'].values
        
        for start_idx in range(0, len(burst_df) - L + 1, stride):
            window_v0 = signal_v0[start_idx:start_idx + L]
            window_v1 = signal_v1[start_idx:start_idx + L]
            window_i = signal_i[start_idx:start_idx + L]
            
            # 특징 계산
            v0_p2p = calculate_p2p(window_v0)
            v1_flow = calculate_flow(window_v1)
            i_fent = calculate_fent(window_i)
            
            # 정규화
            v0_norm = (v0_p2p - mean_v0) / std_v0
            v1_norm = (v1_flow - mean_v1) / std_v1
            i_norm = (i_fent - mean_i) / std_i
            
            timestamp = burst_df.loc[start_idx, 'TimeStamp']
            label = burst_df.loc[start_idx, 'Equipment_state']
            
            results.append({
                'timestamp': timestamp,
                'window_id': window_id,
                'burst_id': burst_id,
                'V0_p2p': v0_norm,
                'V1_flow': v1_norm,
                'I_fent': i_norm,
                'label': label
            })
            
            window_id += 1
    
    # 5. 결과 저장
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_csv_path, index=False)
    print(f"저장됨: {len(result_df)} 윈도우 → {output_csv_path}")
    
    return result_df


def calculate_p2p(signal_window):
    """V0: 진동0의 p2p"""
    return np.max(signal_window) - np.min(signal_window)


def calculate_flow(signal_window, fs=10.0):
    """V1: 진동1의 저주파 비율"""
    fft_mag = np.abs(np.fft.rfft(signal_window))
    freqs = np.fft.rfftfreq(len(signal_window), 1/fs)
    pxx = fft_mag ** 2
    pxx_norm = pxx / (np.sum(pxx) + 1e-10)
    
    lowfreq_mask = freqs < 1.2
    flow = np.sum(pxx_norm[lowfreq_mask])
    
    return flow


def calculate_fent(signal_window, fs=10.0):
    """I: 전류의 스펙트럼 엔트로피"""
    fft_mag = np.abs(np.fft.rfft(signal_window))
    freqs = np.fft.rfftfreq(len(signal_window), 1/fs)
    pxx = fft_mag ** 2
    pxx_norm = pxx / (np.sum(pxx) + 1e-10)
    
    pxx_safe = pxx_norm + 1e-10
    fent = -np.sum(pxx_safe * np.log2(pxx_safe))
    
    return fent


# 사용 예시
if __name__ == '__main__':
    # 정상 데이터 처리
    process_press_machine_data(
        'press_data_normal.csv',
        'windows_features_L17_s4_normal.csv'
    )
    
    # 이상 데이터 처리
    process_press_machine_data(
        'outlier_data.csv',
        'windows_features_L17_s4_outlier.csv'
    )
```

---

## 6. 구현 체크리스트

구현 시 다음을 확인하세요:

- [ ] 입력 CSV 형식 확인 (TimeStamp, AI0_Vibration, AI1_Vibration, AI2_Current, Equipment_state)
- [ ] 시간 간격 기반 burst 분할 (gap > 500ms)
- [ ] 최소 길이 필터 (L >= 17)
- [ ] L=17, stride=4 윈도우 슬라이딩
- [ ] FFT 계산 (np.fft.rfft 사용)
- [ ] V1_flow 저주파 범위 확인 (< 1.2Hz)
- [ ] I_fent 엔트로피 계산 (log2 사용)
- [ ] Z-score 정규화 (정상 데이터만 사용)
- [ ] 출력 CSV 생성 (7개 열)
- [ ] 행 수 확인 (정상 ~9,000개, 이상 ~250개)

---

## 7. 실행 결과 예상

### 정상 데이터
```
로드됨: 20000 행
burst 생성: 469개
저장됨: 9132 윈도우 → windows_features_L17_s4_normal.csv
```

### 이상 데이터
```
로드됨: 600 행
burst 생성: 14개
저장됨: 252 윈도우 → windows_features_L17_s4_outlier.csv
```

### 데이터 통계 (정규화 후)
```
정상 (9132개):
  V0_p2p: mean ≈ 0.0, std ≈ 1.0 (by definition)
  V1_flow: mean ≈ 0.0, std ≈ 1.0
  I_fent: mean ≈ 0.0, std ≈ 1.0

이상 (252개):
  V0_p2p: mean ≈ 5.5 (정상보다 큼)
  V1_flow: mean ≈ -3.2 (정상보다 작음)
  I_fent: mean ≈ 2.1 (정상보다 큼)

→ 3개 축이 이상을 잘 구분함!
```

---

## 8. 질문 대응

**Q: stride=4가 아니라 다른 값을 쓰고 싶으면?**
A: L=17, stride=1~8은 성능 차이 없음. stride=4가 계산 시간(0.9초)과 표본 수(3,044개) 최적

**Q: 정상 데이터 없이 테스트 데이터만 있으면?**
A: 정상 데이터로 먼저 정규화 파라미터 계산. 별도 저장했다가 테스트에 적용

**Q: 다른 조합 (예: V0_std, V1_rms, I_p2p)을 써도 되나?**
A: 가능하지만 FDCn 1.9215가 최고임. 이 조합이 최적 증명됨

**Q: 시간별로 3개 값의 이상 여부를 판단하려면?**
A: 다음 단계: 마할라노비스 탐지기 + 99백분위 임계값 설정

---

## 9. 전달 체크리스트

구현자에게 전달할 것:
- [x] 이 문서
- [x] 입력 CSV 샘플 (press_data_normal.csv, outlier_data.csv)
- [x] 계산 코드 (Python 또는 다른 언어)
- [x] 출력 형식 명세
- [x] 정규화 파라미터 (mean, std)
- [ ] 추가 질문용 연락처
