"""BSQ-rate (bitrate for the same quality), Zvezdakova, Kulikov, Zvezdakov, Vatolin, Proc. ISP RAS 32(1):89-108, 2020.
Steps: points ordered by actual bitrate; points that break monotonicity are removed; axes inverted (bitrate as a
function of quality); linear interpolation; integration over the overlap of the quality ranges; BSQ = S_test / S_ref.
Reported here as (BSQ - 1) * 100 %, i.e. the average bitrate difference at equal quality."""
import numpy as np

def _monotone(rate, q, higher_better=True):
    o = np.argsort(rate); r = np.asarray(rate, float)[o]; y = np.asarray(q, float)[o]
    if not higher_better: y = -y
    keep_r, keep_y = [], []
    for ri, yi in zip(r, y):
        if not np.isfinite(yi) or not np.isfinite(ri): continue
        if not keep_y or yi > keep_y[-1]: keep_r.append(ri); keep_y.append(yi)
    yk = np.array(keep_y); return np.array(keep_r), (yk if higher_better else -yk)

def bsq_rate(rate_ref, q_ref, rate_test, q_test, higher_better=True, lo=None, hi=None, min_ratio=None, n=400):
    ra, ya = _monotone(rate_ref, q_ref, higher_better); rt, yt = _monotone(rate_test, q_test, higher_better)
    if len(ra) < 2 or len(rt) < 2: return np.nan
    a = max(ya.min(), yt.min()); b = min(ya.max(), yt.max())
    if lo is not None: a = max(a, lo)
    if hi is not None: b = min(b, hi)
    if b <= a or (min_ratio is not None and b / a < min_ratio): return np.nan
    g = np.linspace(a, b, n)
    sa = np.trapezoid(np.interp(g, *(ya, ra) if higher_better else (ya[::-1], ra[::-1])), g)
    st = np.trapezoid(np.interp(g, *(yt, rt) if higher_better else (yt[::-1], rt[::-1])), g)
    return (st / sa - 1) * 100

def bsq_loc(k_ref, r_ref, k_test, r_test):
    """BSQ-rate at equal relative ATE (lower is better), integration over r in [1.05, 3] and the overlap."""
    return bsq_rate(k_ref, r_ref, k_test, r_test, higher_better=False, lo=1.05, hi=3.0, min_ratio=10 ** 0.05)
