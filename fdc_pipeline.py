"""
프레스 데이터 FDC(Feature variable Dimensional Coordination) 파이프라인 v2
- 연속 구간(버스트) 안에서만 슬라이딩 윈도우 -> 채널(3)별 특징 추출
- 시간영역 8종 (+ --freq 이면 주파수 4종) x 3채널 = 24개(또는 36개) 특징
- 3개 특징 조합(2024개 또는 7140개)마다 V1, V2, MnD -> FDCn (정상/이상 개수 맞춤 보정 포함)
- 채널 구성 보고서, 단일채널 한정 비교, 정상만으로 학습한 마할라노비스 탐지율 평가

사용:
  python fdc_pipeline.py --L 17 --stride 4 --freq            # 특징 테이블 + FDC 순위표 저장
  python fdc_pipeline.py --L 17 --stride 4 --freq --report   # 채널 구성 보고서
  python fdc_pipeline.py --stride 4 --freq --sweep           # 윈도우 길이 스윕
  python fdc_pipeline.py --stride 4 --freq --stability       # 이상 구간 1개씩 빼며 순위 안정성(복원추출 없음)
  python fdc_pipeline.py --L 17 --freq --scope               # 채널 수별(1/2/3채널) 비교
"""
import argparse, itertools, time, warnings
import numpy as np, pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import skew, kurtosis, spearmanr
from scipy.spatial import ConvexHull

warnings.filterwarnings("ignore")
UP, OUT = "/mnt/user-data/uploads", "/mnt/user-data/outputs"
CH = ["AI0_Vibration", "AI1_Vibration", "AI2_Current"]
CH_SHORT = ["v0", "v1", "cur"]
FT_TIME = ["mean", "median", "std", "rms", "p2p", "skew", "kurt", "p2r"]
FT_FREQ = ["flow", "fhigh", "fcent", "fent"]
GAP_SEC = 0.15  # 샘플 간격이 이보다 크면 새 연속 구간(버스트)

USE_FREQ, NAMES, COMBOS = False, [], None


def configure(freq):
    """특징 집합 설정. 이후 NAMES, COMBOS가 바뀐다."""
    global USE_FREQ, NAMES, COMBOS
    USE_FREQ = freq
    fts = FT_TIME + (FT_FREQ if freq else [])
    NAMES = [f"{c}_{f}" for c in CH_SHORT for f in fts]  # 채널 우선 순서
    COMBOS = np.array(list(itertools.combinations(range(len(NAMES)), 3)))


configure(False)


def channel_of(name): return name.split("_")[0]
def is_freq(name): return name.split("_")[1] in FT_FREQ


# ---------------------------------------------------------------- 1. 연속 구간/윈도우
def load(path, burst_offset=0):
    df = pd.read_csv(path, parse_dates=["TimeStamp"])
    dt = df["TimeStamp"].diff().dt.total_seconds().fillna(99)
    df["burst_id"] = (dt > GAP_SEC).cumsum() + burst_offset
    return df


def time_features(w):
    rms = np.sqrt((w ** 2).mean(-1))
    f = np.stack([w.mean(-1), np.median(w, -1), w.std(-1), rms, np.ptp(w, -1),
                  skew(w, axis=-1), kurtosis(w, axis=-1), np.abs(w).max(-1) / rms], -1)
    return f  # (n, 3, 8)


def freq_features(w):
    """DC 제거 후 FFT. 대역을 Hz로 고정해서 L이 달라도 같은 물리 대역을 본다.
    flow: 0~1.2Hz 에너지 비율(기본주기 0.6Hz 부근), fhigh: 2.3Hz 이상 비율(나이퀴스트 5Hz까지),
    fcent: 스펙트럼 중심(Hz), fent: 정규화 스펙트럼 엔트로피(0~1)."""
    L = w.shape[-1]
    x = w - w.mean(-1, keepdims=True)
    P = (np.abs(np.fft.rfft(x, axis=-1)) ** 2)[..., 1:]
    f = np.fft.rfftfreq(L, d=0.1)[1:]
    p = P / (P.sum(-1, keepdims=True) + 1e-12)
    ent = -(p * np.log(p + 1e-12)).sum(-1) / np.log(len(f))
    return np.stack([p[..., f <= 1.2].sum(-1), p[..., f >= 2.3].sum(-1), (p * f).sum(-1), ent], -1)  # (n, 3, 4)


def window_features(w):
    """w: (n_win, 3채널, L) -> (n_win, 24 또는 36). 루프 없이 한 번에."""
    f = time_features(w)
    if USE_FREQ:
        f = np.concatenate([f, freq_features(w)], -1)
    return f.reshape(len(w), -1)


def extract_windows(df, L, stride, label):
    """연속 구간 경계를 넘지 않고, 구간 안에서만 stride로 민다. 길이 < L 인 구간은 버린다."""
    rows = []
    for bid, g in df.groupby("burst_id"):
        v = g[CH].to_numpy()
        if len(v) < L:
            continue
        w = sliding_window_view(v, L, axis=0)[::stride]
        t0 = g["TimeStamp"].to_numpy()[: len(v) - L + 1][::stride]
        feat = pd.DataFrame(window_features(w), columns=NAMES)
        feat.insert(0, "start_time", t0); feat.insert(0, "burst_id", bid)
        rows.append(feat)
    out = pd.concat(rows, ignore_index=True)
    out.insert(0, "label", label)
    return out.replace([np.inf, -np.inf], np.nan).dropna()


def build(L, stride):
    n = load(f"{UP}/press_data_normal.csv", 0)
    o = load(f"{UP}/outlier_data.csv", 100000)
    return extract_windows(n, L, stride, 0), extract_windows(o, L, stride, 1)


# ---------------------------------------------------------------- 2. FDC
def _hull_vol(P):
    try: return ConvexHull(P).volume
    except Exception: return np.nan


def fdc_table(Xn, Xo, n_repeat=3, seed=0):
    """정상을 이상 개수에 맞춰 n_repeat번 뽑아 V1, MnD를 평균. FDCn=(V2-V1)/(V2+V1)*MnD."""
    rng = np.random.default_rng(seed); m = len(Xo)
    V2 = np.array([_hull_vol(Xo[:, c]) for c in COMBOS])
    Co, mo = np.cov(Xo.T), Xo.mean(0)
    v1s, mnds = [], []
    for _ in range(n_repeat):
        sub = Xn[rng.choice(len(Xn), m, replace=False)]
        v1s.append([_hull_vol(sub[:, c]) for c in COMBOS])
        C = (np.cov(sub.T) + Co) / 2  # 개수가 같으므로 합동 공분산 = 평균
        d = (sub.mean(0) - mo)[COMBOS]
        Cc = C[COMBOS[:, :, None], COMBOS[:, None, :]]
        mnds.append(np.sqrt(np.einsum("ij,ijk,ik->i", d, np.linalg.pinv(Cc), d)))
    V1, MnD = np.nanmean(v1s, 0), np.mean(mnds, 0)
    t = pd.DataFrame({"combo": ["-".join(NAMES[i] for i in c) for c in COMBOS],
                      "V1": V1, "V2": V2, "MnD": MnD, "FDCn": (V2 - V1) / (V2 + V1) * MnD})
    t["FDCn_norm"] = (t.FDCn - t.FDCn.min()) / (t.FDCn.max() - t.FDCn.min())
    t["combo_idx"] = range(len(t))
    t["n_channels"] = [len({channel_of(NAMES[i]) for i in c}) for c in COMBOS]
    t["n_freq"] = [sum(is_freq(NAMES[i]) for i in c) for c in COMBOS]
    return t.sort_values("FDCn", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------- 3. 검증
def cv_balanced_acc(Xn, gn, Xo, go, idxs, n_normal=600, seed=1):
    """지도 검증: SVM, 연속 구간 단위 3-fold."""
    from sklearn.svm import SVC
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import balanced_accuracy_score
    rs = np.random.default_rng(seed)
    idx = rs.choice(len(Xn), min(len(Xn), n_normal), replace=False)
    X = np.vstack([Xn[idx], Xo]); y = np.r_[np.zeros(len(idx)), np.ones(len(Xo))]; g = np.r_[gn[idx], go]
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=0); res = []
    for ci in idxs:
        cols, s = list(COMBOS[ci]), []
        for tr, te in cv.split(X, y, g):
            m = make_pipeline(StandardScaler(), SVC(class_weight="balanced")).fit(X[tr][:, cols], y[tr])
            s.append(balanced_accuracy_score(y[te], m.predict(X[te][:, cols])))
        res.append(np.mean(s))
    return np.array(res)


def detect_rate(Xn, gn, Xo, idxs, q=99):
    """정상만으로 학습한 마할라노비스 탐지기. 임계값=학습 정상의 q백분위.
    반환: (이상 탐지율, 보류 정상의 오경보율). 정상은 연속 구간 단위 3-fold."""
    from sklearn.model_selection import GroupKFold
    out = []
    for ci in idxs:
        cols = list(COMBOS[ci]); A, B = Xn[:, cols], Xo[:, cols]; dr, far = [], []
        for tr, te in GroupKFold(n_splits=3).split(A, groups=gn):
            mu = A[tr].mean(0); P = np.linalg.pinv(np.cov(A[tr].T))
            sc = lambda Z: np.sqrt(np.einsum("ij,jk,ik->i", Z - mu, P, Z - mu))
            thr = np.percentile(sc(A[tr]), q)
            dr.append((sc(B) > thr).mean()); far.append((sc(A[te]) > thr).mean())
        out.append((np.mean(dr), np.mean(far)))
    return np.array(out)


# ---------------------------------------------------------------- 4. 보고서
def topk_overlap(a, b, k=50): return len(set(a.combo[:k]) & set(b.combo[:k]))
def rank_corr(a, b):
    m = a.set_index("combo").FDCn.to_frame("a").join(b.set_index("combo").FDCn.rename("b")).dropna()
    return spearmanr(m.a, m.b)[0]


def channel_report(L, stride, n_repeat):
    fn, fo = build(L, stride)
    Xn, Xo = fn[NAMES].to_numpy(), fo[NAMES].to_numpy(); gn, go = fn.burst_id.to_numpy(), fo.burst_id.to_numpy()
    t = fdc_table(Xn, Xo, n_repeat)
    t.to_csv(f"{OUT}/fdc_ranking_L{L}_s{stride}{'_freq' if USE_FREQ else ''}.csv", index=False)
    print(f"L={L}, stride={stride}, 특징 {len(NAMES)}개, 조합 {len(t)}개, 정상 {len(fn)} / 이상 {len(fo)}")
    for k in (30, 100):
        s = t.head(k)
        print(f"\n[FDCn 상위 {k}] 조합이 걸친 채널 수:", s.n_channels.value_counts().sort_index().to_dict(),
              "| 조합 안의 주파수 특징 수:", s.n_freq.value_counts().sort_index().to_dict())
        cnt = pd.Series([channel_of(NAMES[i]) for c in s.combo_idx for i in COMBOS[c]]).value_counts()
        print("  채널별 등장 비율:", {k2: f"{v / (3 * k):.0%}" for k2, v in cnt.items()})
    print("\n[채널 한정 최고 조합 vs 교차 채널 최고 조합]")
    rows = []
    for name, mask in [(c, t.combo.map(lambda s, c=c: all(channel_of(x) == c for x in s.split("-")))) for c in CH_SHORT] \
                      + [("교차채널(2개 이상)", t.n_channels >= 2)]:
        r = t[mask].iloc[0]
        a = cv_balanced_acc(Xn, gn, Xo, go, [r.combo_idx])[0]; d, f = detect_rate(Xn, gn, Xo, [r.combo_idx])[0]
        rows.append((name, r.combo, round(r.FDCn_norm, 2), round(a, 3), f"{d:.0%}", f"{f:.1%}"))
    print(pd.DataFrame(rows, columns=["범위", "최고 조합", "FDCn", "CV정확도", "탐지율", "오경보율"]).to_string(index=False))
    top = t.combo_idx[:30].tolist(); rnd = np.random.default_rng(0).choice(len(t), 60, replace=False).tolist()
    print("\n[정상만으로 학습한 탐지기, 상위30 vs 무작위60 평균 (탐지율 / 오경보율)]")
    for nm, ids in [("FDCn 상위30", top), ("무작위60", rnd)]:
        r = detect_rate(Xn, gn, Xo, ids).mean(0); print(f"  {nm}: {r[0]:.1%} / {r[1]:.1%}")


def sweep(stride, n_repeat):
    Ls, tabs, info = [12, 16, 17, 20, 25, 33], {}, {}
    for L in Ls:
        t0 = time.time(); fn, fo = build(L, stride); tf = time.time() - t0
        Xn, Xo = fn[NAMES].to_numpy(), fo[NAMES].to_numpy()
        t0 = time.time(); t = fdc_table(Xn, Xo, n_repeat); tc = time.time() - t0
        top = t.combo_idx[:30].tolist(); gn, go = fn.burst_id.to_numpy(), fo.burst_id.to_numpy()
        d, f = detect_rate(Xn, gn, Xo, top).mean(0)
        tabs[L] = t
        info[L] = dict(정상=len(fn), 이상=len(Xo), 이상구간수=fo.burst_id.nunique(), 특징추출_s=round(tf, 2),
                       FDC계산_s=round(tc, 1), 상위30_CV정확도=round(cv_balanced_acc(Xn, gn, Xo, go, top).mean(), 3),
                       탐지율=f"{d:.0%}", 오경보율=f"{f:.1%}")
    print(pd.DataFrame(info).T.to_string())
    base = tabs[17]
    print("\n(L 안에서의 순위 안정성은 --stability 로 확인: 이상 구간을 하나씩 빼는 방식, 복원추출 없음)")
    print("\n[L=17 대비 순위 유사도]")
    for L in Ls:
        print(f"  L={L:>2}: 상위50 겹침 {topk_overlap(base, tabs[L]):>2}/50, 순위상관 {rank_corr(base, tabs[L]):.3f}")


def jackknife(fn, fo, n_repeat, base=None, seed=0):
    """이상 연속 구간을 하나씩 빼고 순위가 얼마나 흔들리는지. 있는 구간만 쓰고 복원추출은 하지 않는다."""
    Xn = fn[NAMES].to_numpy()
    base = base if base is not None else fdc_table(Xn, fo[NAMES].to_numpy(), n_repeat, seed)
    ov, rc = [], []
    for b in fo.burst_id.unique():
        t = fdc_table(Xn, fo[fo.burst_id != b][NAMES].to_numpy(), n_repeat, seed)
        ov.append(topk_overlap(base, t)); rc.append(rank_corr(base, t))
    return np.mean(ov), np.min(ov), np.mean(rc)


def stability_study(stride, n_repeat=1):
    """윈도우 길이별로 (a) 이상 구간 1개를 뺄 때의 흔들림, (b) L=17과의 차이를 나란히 본다."""
    Ls, tabs, res = [12, 16, 17, 20, 25, 33], {}, {}
    for L in Ls:
        fn, fo = build(L, stride)
        base = fdc_table(fn[NAMES].to_numpy(), fo[NAMES].to_numpy(), n_repeat); tabs[L] = base
        jm, jmin, jr = jackknife(fn, fo, n_repeat, base)
        res[L] = {"이상구간수": fo.burst_id.nunique(), "1개빼면 상위50겹침(평균)": round(jm, 1),
                  "1개빼면 최소": int(jmin), "1개빼면 순위상관": round(jr, 3)}
    print(pd.DataFrame(res).T.to_string())
    print("\n[L=17 대비]")
    for L in Ls:
        print(f"  L={L:>2}: 상위50 겹침 {topk_overlap(tabs[17], tabs[L]):>2}/50, 순위상관 {rank_corr(tabs[17], tabs[L]):.3f}")


def scope_report(L, stride, n_repeat):
    """조합이 걸친 채널 수별로 최고 성능 비교. 3채널 모두 = 채널마다 특징 1개씩(12x12x12)."""
    fn, fo = build(L, stride)
    Xn, Xo = fn[NAMES].to_numpy(), fo[NAMES].to_numpy(); gn, go = fn.burst_id.to_numpy(), fo.burst_id.to_numpy()
    t = fdc_table(Xn, Xo, n_repeat); rows = []
    for nm, m in [("제약 없음", t.n_channels >= 1), ("3채널 모두 포함", t.n_channels == 3),
                  ("정확히 2채널", t.n_channels == 2), ("1채널만", t.n_channels == 1)]:
        ids = t[m].combo_idx[:30].tolist(); d, f = detect_rate(Xn, gn, Xo, ids).mean(0)
        rows.append((nm, int(m.sum()), round(t[m].FDCn_norm[:30].mean(), 2),
                     round(cv_balanced_acc(Xn, gn, Xo, go, ids).mean(), 3), f"{d:.0%}", f"{f:.1%}", t[m].combo.iloc[0]))
    print(pd.DataFrame(rows, columns=["범위", "조합수", "상위30 FDCn평균", "CV정확도", "탐지율", "오경보율", "1위 조합"]).to_string(index=False))
    print("\n[3채널 모두 포함 상위 8]")
    print(t[t.n_channels == 3].head(8)[["combo", "V1", "V2", "MnD", "FDCn_norm"]].to_string(index=False))



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=17); ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--repeat", type=int, default=3); ap.add_argument("--freq", action="store_true")
    ap.add_argument("--sweep", action="store_true"); ap.add_argument("--report", action="store_true")
    ap.add_argument("--stability", action="store_true"); ap.add_argument("--scope", action="store_true")
    a = ap.parse_args(); configure(a.freq)
    if a.sweep: sweep(a.stride, a.repeat)
    elif a.stability: stability_study(a.stride)
    elif a.scope: scope_report(a.L, a.stride, a.repeat)
    else: channel_report(a.L, a.stride, a.repeat)
