# 프레스 기계 이상 탐지: 구현 가이드 (간편판)

## 📋 한눈에 보기

```
Raw CSV 입력
    ↓
① Burst 분할 (시간 간격 > 500ms)
    ↓
② 윈도우 슬라이딩 (L=17, stride=4)
    ↓
③ 3개 특징 추출 및 정규화
    ├─ V0_p2p   = (p2p - 0.2364) / 0.0853
    ├─ V1_flow  = (flow - 0.0847) / 0.1193
    └─ I_fent   = (fent - 0.0514) / 0.0185
    ↓
시계열 이산 데이터 (CSV)
```

---

## 🔧 정확한 계산식

### 1️⃣ V0_p2p (진동0의 피크-피크)

```
raw_p2p = max(signal[0:17]) - min(signal[0:17])
normalized = (raw_p2p - 0.2364) / 0.0853
```

**예시**:
```python
# 17샘플 신호
signal = [63.82, 102.47, 200.49, ..., -104.16]

# 계산
max_val = 204.72
min_val = -211.22
raw_p2p = 204.72 - (-211.22) = 415.94

# 정규화
V0_p2p = (415.94 - 0.2364) / 0.0853 = 4868.68
```

### 2️⃣ V1_flow (진동1의 저주파 비율)

```
Step 1. FFT: fft_mag = |FFT(signal[0:17])|
Step 2. 주파수: freqs = [0, 0.588, 1.176, 1.765, ...]Hz
Step 3. 파워: pxx = fft_mag²
Step 4. 정규화: pxx_norm = pxx / sum(pxx)
Step 5. 마스크: lowfreq = (freqs < 1.2)
Step 6. 합: raw_flow = sum(pxx_norm[lowfreq])
Step 7. 정규화: normalized = (raw_flow - 0.0847) / 0.1193
```

**예시**:
```python
import numpy as np

signal = [63.82, 102.47, 200.49, ..., -104.16]  # 17개
fs = 10.0  # 샘플링 주파수 (Hz)

# Step 1-4
fft_mag = np.abs(np.fft.rfft(signal))
freqs = np.fft.rfftfreq(17, 1/10.0)
pxx = fft_mag ** 2
pxx_norm = pxx / np.sum(pxx)

# Step 5-6
lowfreq_mask = freqs < 1.2
raw_flow = np.sum(pxx_norm[lowfreq_mask])
# raw_flow ≈ 0.8220

# Step 7
V1_flow = (0.8220 - 0.0847) / 0.1193 = 6.16
```

### 3️⃣ I_fent (전류의 스펙트럼 엔트로피)

```
Step 1-4. (위와 동일) pxx_norm 계산
Step 5. 엔트로피: fent = -sum(pxx_norm * log2(pxx_norm))
Step 6. 정규화: normalized = (fent - 0.0514) / 0.0185
```

**예시**:
```python
# Step 1-4 (위와 동일)
pxx_norm = [0.0165, 0.6751, 0.1907, 0.1089, ...]

# Step 5 (Shannon Entropy)
fent = -sum(pxx_norm * log2(pxx_norm))
# fent ≈ 1.7329

# Step 6
I_fent = (1.7329 - 0.0514) / 0.0185 = 91.08
```

---

## 📊 정규화 파라미터

| 특징 | mean | std |
|------|------|-----|
| V0_p2p | 0.236445 | 0.085320 |
| V1_flow | 0.084717 | 0.119349 |
| I_fent | 0.051404 | 0.018462 |

**파라미터 저장 위치**: `normalization_parameters.json`

```json
{
  "V0_p2p_mean": 0.236445,
  "V0_p2p_std": 0.085320,
  "V1_flow_mean": 0.084717,
  "V1_flow_std": 0.119349,
  "I_fent_mean": 0.051404,
  "I_fent_std": 0.018462
}
```

---

## 📥 입력 CSV 형식

**파일명**: `press_data_normal.csv` 또는 `outlier_data.csv`

```csv
,TimeStamp,AI0_Vibration,AI1_Vibration,AI2_Current,Equipment_state
0,2022-07-12 00:00:00.019,0.116967,-0.176403,192.33873,0
1,2022-07-12 00:00:00.119,0.03008,-0.210895,217.00433,0
2,2022-07-12 00:00:00.219,-0.045794,-0.029424,197.51062,0
...
```

**필수 열**:
- `TimeStamp` (datetime)
- `AI0_Vibration` (float) ← 진동 센서 0
- `AI1_Vibration` (float) ← 진동 센서 1
- `AI2_Current` (float) ← 전류 센서
- `Equipment_state` (0=정상, 1=이상)

---

## 📤 출력 CSV 형식

**파일명**: `windows_features_L17_s4.csv`

```csv
timestamp,window_id,burst_id,V0_p2p,V1_flow,I_fent,label
2022-07-12 00:00:00.019,0,0,6.68,2.28,1.95,0
2022-07-12 00:00:00.419,1,0,5.92,2.35,1.87,0
2022-07-12 00:00:00.819,2,0,7.15,2.10,2.05,0
...
2022-07-17 10:51:07.943,251,14,-1.23,-3.45,3.12,1
```

**출력 열**:
- `timestamp` (str): 윈도우 시작 시간
- `window_id` (int): 윈도우 순번 (0부터)
- `burst_id` (int): burst 그룹 번호
- `V0_p2p` (float): 정규화된 진동0 p2p
- `V1_flow` (float): 정규화된 진동1 저주파 비율
- `I_fent` (float): 정규화된 전류 엔트로피
- `label` (0 or 1): 정상(0) 또는 이상(1)

---

## 🔑 핵심 파라미터

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| L | 17 | 윈도우 길이 (샘플 수) |
| stride | 4 | 윈도우 이동 간격 (샘플 수) |
| gap_threshold | 500 ms | burst 분할 기준 |
| fs | 10.0 Hz | 샘플링 주파수 |
| lowfreq_threshold | 1.2 Hz | flow 저주파 범위 |

---

## ✅ 구현 체크리스트

### 입력 검증
- [ ] CSV 파일 로드 확인
- [ ] 필수 5개 열 존재 확인
- [ ] TimeStamp 파싱 확인 (datetime으로 변환)
- [ ] Equipment_state 값 확인 (0 또는 1)

### Burst 분할
- [ ] 인접 행의 시간 차이 계산
- [ ] 500ms 이상 간격 찾기
- [ ] 각 burst를 별도 그룹으로 분리
- [ ] 길이 17 이상인 burst만 선택

### 윈도우 슬라이딩
- [ ] L=17, stride=4로 정확히 슬라이딩
- [ ] 윈도우 수 확인:
  - 정상: ~3,000개 (정상 데이터 9,132개에서)
  - 이상: ~250개 (이상 데이터 252개에서)

### 특징 추출
- [ ] V0_p2p: max - min
- [ ] V1_flow: FFT → 저주파 파워 합
- [ ] I_fent: 스펙트럼 엔트로피 (-sum(p*log2(p)))

### 정규화
- [ ] 정규화 파라미터 로드 (또는 JSON에서)
- [ ] Z-score 변환 적용: (값 - mean) / std
- [ ] 정규화 후 값의 범위 확인 (-3 ~ +5 정도)

### 출력
- [ ] CSV 파일 생성
- [ ] 7개 열 확인
- [ ] 행 수 확인
- [ ] 첫 몇 행 샘플 검증

---

## 🧮 Python 구현 스켈레톤

```python
import numpy as np
import pandas as pd
import json

# 1. 정규화 파라미터 로드
with open('normalization_parameters.json') as f:
    params = json.load(f)

# 2. 입력 CSV 로드
df = pd.read_csv('press_data_normal.csv')
df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])

# 3. Burst 분할
df['time_diff_ms'] = df['TimeStamp'].diff().dt.total_seconds() * 1000
df['is_gap'] = df['time_diff_ms'] > 500
df['burst_id'] = df['is_gap'].fillna(False).cumsum()

results = []
window_id = 0

for burst_id, burst_df in df.groupby('burst_id'):
    if len(burst_df) < 17:
        continue
    
    # 4. 윈도우 슬라이딩
    for start in range(0, len(burst_df) - 17 + 1, 4):
        w_v0 = burst_df.iloc[start:start+17]['AI0_Vibration'].values
        w_v1 = burst_df.iloc[start:start+17]['AI1_Vibration'].values
        w_i = burst_df.iloc[start:start+17]['AI2_Current'].values
        
        # 5. 특징 계산
        v0_p2p_raw = np.max(w_v0) - np.min(w_v0)
        
        # (V1_flow 계산: 위의 FFT 코드 참고)
        # (I_fent 계산: 위의 엔트로피 코드 참고)
        
        # 6. 정규화
        v0_norm = (v0_p2p_raw - params['V0_p2p_mean']) / params['V0_p2p_std']
        v1_norm = (v1_flow_raw - params['V1_flow_mean']) / params['V1_flow_std']
        i_norm = (i_fent_raw - params['I_fent_mean']) / params['I_fent_std']
        
        # 7. 결과 저장
        results.append({
            'timestamp': burst_df.iloc[start]['TimeStamp'],
            'window_id': window_id,
            'burst_id': burst_id,
            'V0_p2p': v0_norm,
            'V1_flow': v1_norm,
            'I_fent': i_norm,
            'label': burst_df.iloc[start]['Equipment_state']
        })
        
        window_id += 1

# 8. 출력 CSV 생성
result_df = pd.DataFrame(results)
result_df.to_csv('windows_features_L17_s4.csv', index=False)
```

---

## 📞 주요 함수 (의사코드)

### calculate_p2p()
```
입력: signal_window (17개 float 배열)
출력: 하나의 float

return max(signal_window) - min(signal_window)
```

### calculate_flow()
```
입력: signal_window (17개 float 배열), fs=10.0
출력: 하나의 float (0~1)

fft_mag = FFT(signal_window)의 크기
freqs = FFT 주파수축
pxx = fft_mag²
pxx_norm = pxx / sum(pxx)
return sum(pxx_norm[freqs < 1.2])
```

### calculate_fent()
```
입력: signal_window (17개 float 배열), fs=10.0
출력: 하나의 float (0~3.17)

fft_mag = FFT(signal_window)의 크기
pxx = fft_mag²
pxx_norm = pxx / sum(pxx)
return -sum(pxx_norm * log2(pxx_norm))
```

### normalize()
```
입력: raw_value, mean, std
출력: normalized_value

return (raw_value - mean) / std
```

---

## 🎯 최종 확인

✅ **정상 데이터로 출력 시**:
```
총 행 수: ~3,044개 윈도우
V0_p2p 범위: -2 ~ +2 (정규화 후)
V1_flow 범위: -1 ~ +6
I_fent 범위: -3 ~ +4
label: 모두 0
```

✅ **이상 데이터로 출력 시**:
```
총 행 수: ~252개 윈도우
V0_p2p 범위: +4 ~ +8 (정상보다 큼)
V1_flow 범위: -4 ~ -1 (정상보다 작음)
I_fent 범위: +2 ~ +4 (정상보다 큼)
label: 모두 1
```

→ **3개 축이 정상/이상을 잘 구분!**

---

## 📎 참고 파일

- `feature_extraction_specification.md` (상세 설명)
- `normalization_parameters.json` (파라미터)
- `press_data_normal.csv` (입력 샘플)
- `outlier_data.csv` (입력 샘플)
