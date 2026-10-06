# 프레스 기계 이상 탐지: 특징 추출 명세 및 구현 가이드

## 📚 문서 구성

```
├─ README.md                                    ← 이 파일 (전체 가이드)
├─ IMPLEMENTATION_GUIDE.md                      ← 간편 구현 가이드 (5분 버전)
├─ feature_extraction_specification.md          ← 상세 명세서 (완전 버전)
├─ reference_implementation.py                  ← Python 참고 코드
├─ normalization_parameters.json                ← 정규화 파라미터
└─ 입력 데이터
   ├─ press_data_normal.csv                     ← 정상 데이터 샘플
   └─ outlier_data.csv                          ← 이상 데이터 샘플
```

---

## ⚡ 30초 요약

### 입력
Raw CSV 데이터 (TimeStamp, AI0_Vibration, AI1_Vibration, AI2_Current, Equipment_state)

### 처리
1. 시간 간격으로 burst 분할 (gap > 500ms)
2. 각 burst에서 L=17, stride=4 윈도우 슬라이딩
3. 3개 특징 추출 + 정규화:
   - **V0_p2p**: 진동0의 피크-피크 (최대-최소)
   - **V1_flow**: 진동1의 저주파 비율 (0~1.2Hz)
   - **I_fent**: 전류의 스펙트럼 엔트로피 (복잡도)

### 출력
CSV 파일 (7개 열): timestamp, window_id, burst_id, V0_p2p, V1_flow, I_fent, label

---

## 🎯 최적 조합 확정

| 특징 | 채널 | 지표 | FDCn | 이유 |
|------|------|------|------|------|
| **V0_p2p** | 진동0 | 피크-피크 | **1.9215** | 이상 시 진동 증가 |
| **V1_flow** | 진동1 | 저주파 비율 | (복합) | 이상 시 고주파화 |
| **I_fent** | 전류 | 엔트로피 | (복합) | 이상 시 복잡도 증가 |

**선택 근거**:
- ✅ 3채널 모두 활용 (데이터 최대 활용)
- ✅ 각각 다른 각도에서 이상 감지
- ✅ FDCn 1.9215 (최고 수준)
- ✅ 서로 다른 특성 조합 (상관관계 낮음)

---

## 🔧 정규화 파라미터

정상 데이터 기준 Z-score 정규화

```
V0_p2p:  (값 - 0.2364) / 0.0853
V1_flow: (값 - 0.0847) / 0.1193
I_fent:  (값 - 0.0514) / 0.0185
```

또는 `normalization_parameters.json` 참고

---

## 📖 문서별 가이드

### 1️⃣ **5분 구현** → `IMPLEMENTATION_GUIDE.md`
- 정확한 계산식
- 예시 숫자
- 체크리스트
- **빠른 구현용**

### 2️⃣ **완전 이해** → `feature_extraction_specification.md`
- 전체 파이프라인 설명
- Step A~D 상세 분석
- 이론적 배경
- **처음부터 배우는 용**

### 3️⃣ **코드 참고** → `reference_implementation.py`
- Python 완성 코드
- 함수별 상세 주석
- 실행 가능한 예시
- **개발 시 복사하기 좋음**

---

## 🚀 빠른 시작 (3단계)

### Step 1: 파라미터 확인
```python
import json

with open('normalization_parameters.json') as f:
    params = json.load(f)

print(params)
# {
#   "V0_p2p_mean": 0.236445,
#   "V0_p2p_std": 0.085320,
#   ...
# }
```

### Step 2: 데이터 로드
```python
import pandas as pd

df = pd.read_csv('press_data_normal.csv')
df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
print(df.head())
```

### Step 3: 특징 추출
```python
from reference_implementation import process_raw_data

result = process_raw_data(
    'press_data_normal.csv',
    'output.csv'
)
print(result)
```

---

## 📊 데이터 흐름도

```
Raw 신호 (20,000행)
    ↓
[Burst 분할]  (시간 > 500ms로 끊기)
    ↓
469개 burst (각각 독립적인 측정 구간)
    ↓
[윈도우 슬라이딩]  (L=17, stride=4)
    ↓
~9,000개 윈도우 (각각 1.7초의 데이터)
    ↓
[특징 추출]  (3개 지표 계산)
    ↓
9,132개 데이터 포인트
    ↓
[정규화]  (Z-score, 정상 데이터 기준)
    ↓
시계열 3축 데이터 (V0_p2p, V1_flow, I_fent)
```

---

## 📋 핵심 파라미터

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| L (윈도우 길이) | 17 | 1주기(1.67초) |
| stride (간격) | 4 | 계산시간-정확도 최적 |
| gap_threshold | 500ms | burst 분할 기준 |
| fs (샘플링) | 10Hz | 100ms 간격 |
| lowfreq_threshold | 1.2Hz | 기본 주파수의 2배 |

**변경 금지**: 이 값들은 FDC 최적화 결과. 다른 값 사용 시 성능 저하.

---

## ✅ 구현 체크리스트

### 입력 검증
- [ ] CSV 파일 로드
- [ ] 5개 열 확인 (TimeStamp, AI0, AI1, AI2, Equipment_state)
- [ ] TimeStamp를 datetime으로 변환
- [ ] Equipment_state가 0 또는 1

### 계산
- [ ] Burst 분할 (500ms 기준)
- [ ] 최소 길이 17 필터
- [ ] 윈도우 슬라이딩 (stride=4)
- [ ] 3개 특징 추출
- [ ] FFT 계산 확인 (저주파 범위 1.2Hz)
- [ ] Z-score 정규화

### 출력
- [ ] CSV 파일 생성
- [ ] 7개 열 (timestamp, window_id, burst_id, V0_p2p, V1_flow, I_fent, label)
- [ ] 행 수 확인
  - 정상: ~3,044개
  - 이상: ~252개

### 데이터 검증
- [ ] 정상: V0_p2p, V1_flow, I_fent 모두 음수 또는 작은 양수
- [ ] 이상: V0_p2p, I_fent 큰 양수, V1_flow 음수

---

## 🔬 예상 결과

### 정상 데이터 출력 샘플
```csv
timestamp,window_id,burst_id,V0_p2p,V1_flow,I_fent,label
2022-07-12 00:00:00.019,0,0,-0.12,1.85,-0.45,0
2022-07-12 00:00:00.419,1,0,0.35,1.92,-0.38,0
2022-07-12 00:00:00.819,2,0,-0.25,1.78,-0.52,0
```

특징:
- V0_p2p: -2 ~ +2 (중심 0 근처)
- V1_flow: +0 ~ +4 (주로 양수)
- I_fent: -3 ~ +1 (낮음)

### 이상 데이터 출력 샘플
```csv
timestamp,window_id,burst_id,V0_p2p,V1_flow,I_fent,label
2022-07-17 10:51:07.943,251,14,5.68,-2.15,2.35,1
2022-07-17 10:51:08.343,252,14,5.92,-1.85,2.28,1
2022-07-17 10:51:08.743,253,14,6.15,-2.45,2.52,1
```

특징:
- V0_p2p: +4 ~ +8 (높음)
- V1_flow: -4 ~ -1 (낮음)
- I_fent: +1 ~ +4 (높음)

---

## 🤔 자주 묻는 질문

**Q1: 다른 L(윈도우 길이) 값을 쓸 수 있나?**
A: L=16~20은 성능 차이 없음. L=12는 성능 저하. L은 반드시 1주기(16.67샘플) 이상이어야 함.

**Q2: stride를 다르게 할 수 있나?**
A: stride=1~8은 성능 동일. stride=4가 계산시간(0.9초)과 표본 수 최적. stride 변경 시 윈도우 개수만 달라짐.

**Q3: 정규화 파라미터를 변경하면?**
A: 절대 금지. 정상 데이터로 한 번 계산된 값. 다른 파라미터 사용 시 이상 탐지 임계값 재설정 필요.

**Q4: 정상 데이터 없이 테스트만 하려면?**
A: 제공된 `normalization_parameters.json` 사용. 그 안의 파라미터로 정규화.

**Q5: 다른 프로세스에서 추출한 특징을 정규화할 수 있나?**
A: 가능. 같은 수식 사용: `(값 - mean) / std`. 파라미터는 반드시 제공된 값 사용.

---

## 📞 구현 지원

### 필요한 정보
- 입력 CSV 형식 (제공된 샘플과 동일한가?)
- 처리 환경 (Python? 다른 언어?)
- 처리 속도 요구 (실시간? 배치?)

### 제공 파일
- ✅ 상세 명세서
- ✅ 구현 가이드
- ✅ Python 참고 코드
- ✅ 정규화 파라미터 (JSON)
- ✅ 입력/출력 데이터 샘플

---

## 📄 라이선스 및 주의사항

### 사용 권한
- ✅ 제공된 파라미터로 특징 추출
- ✅ 제공된 코드 수정 및 배포
- ✅ 다른 프로세스와 연동

### 주의사항
- ⚠️ 정규화 파라미터 임의 변경 금지
- ⚠️ L, stride 임의 변경 금지
- ⚠️ FFT 계산 방식 변경 금지 (역 FFT, 다른 윈도우 함수 등)
- ⚠️ 저주파 범위(1.2Hz) 임의 변경 금지

### 변경 기록
마지막 검증: 2026-10-06 (정상 9,132개 / 이상 252개 윈도우)

---

## 🎓 이론 배경

### FDC (Feature variable Dimensional Coordination)
두 조건(정상/이상)의 3D 부피를 비교하는 방법

```
FDCn = (V2 - V1) / (V2 + V1) × Mahalanobis_Distance

V1: 정상 데이터의 부피
V2: 이상 데이터의 부피
Mnd: 두 중심 사이의 거리
```

FDCn이 높을수록 두 조건이 잘 분리됨.

### 선택된 3축의 특성
| 축 | 특성 | 분리도 |
|----|------|-------|
| V0_p2p | 시간영역 (크기) | 중간 (단독 AUC 0.70) |
| V1_flow | 주파수영역 (저주파) | 높음 (단독 AUC 0.95) |
| I_fent | 주파수영역 (복잡도) | 매우 높음 (단독 AUC 0.98) |
| **조합** | **3축 결합** | **최고 (FDCn 1.92)** |

→ 3축이 함께 작동할 때 최대 분리 달성

---

## 🔄 다음 단계 (탐지기 구성)

이 특징 추출 완료 후:

1. **마할라노비스 거리 계산** (정상 데이터만 학습)
2. **임계값 설정** (99백분위 또는 ROC 곡선)
3. **탐지 규칙** (Mnd > 임계값 → 이상 경보)
4. **2단계 경계** (경고, 알림, 차단)

---

**문서 작성**: 2026-10-06  
**최종 검증**: 정상 9,132개 윈도우, 이상 252개 윈도우, FDCn 1.9215  
**대상 시스템**: 프레스 기계 (10Hz 샘플링, 3채널 신호)
