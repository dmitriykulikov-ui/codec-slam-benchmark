"""Analysis for the codec–SLAM rate–accuracy benchmark.

Conventions
- relative ATE  r = ATE(config) / median ATE(uncompressed) for the same (sequence, system)
- a run FAILS if no trajectory, ATE is NaN, or the estimate covers < 80 % of the GT duration
- per (sequence, system, codec, q): median r over repetitions, failure fraction
- a rate point is "usable" if failure fraction < 0.5; unusable points get r = R_FAIL
"""
import numpy as np, pandas as pd
from scipy.interpolate import PchipInterpolator

def _pava_decreasing(y):
    """Pool-adjacent-violators: best non-increasing fit (equal weights)."""
    y = list(-np.asarray(y, float)); blocks = []
    for v in y:
        blocks.append([v, 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            v2, n2 = blocks.pop(); v1, n1 = blocks.pop()
            blocks.append([(v1 * n1 + v2 * n2) / (n1 + n2), n1 + n2])
    return -np.concatenate([[v] * n for v, n in blocks])

R_FAIL = 30.0
COV_MIN = 0.8

def load(runs_csv, codec_csv=None):
    runs = pd.read_csv(runs_csv) if isinstance(runs_csv, str) else runs_csv.copy()
    # coverage is judged relative to the same estimator on the uncompressed reference (EuRoC initialization
    # periods leave ~0.85 absolute coverage even without compression)
    cref = runs[(runs.codec == 'raw') & runs.ate_m.notna()].groupby(['seq', 'system']).coverage.median().rename('cov_ref')
    runs = runs.join(cref, on=['seq', 'system'])
    runs['cov_rel'] = runs.coverage / runs.cov_ref
    runs['fail'] = runs.ate_m.isna() | (runs.cov_rel < COV_MIN)
    raw = (runs[(runs.codec == 'raw') & ~runs.fail].groupby(['seq', 'system']).ate_m.median().rename('ate_raw'))
    runs = runs.join(raw, on=['seq', 'system'])
    runs['rel'] = np.where(runs.fail, np.nan, runs.ate_m / runs.ate_raw)
    return runs

def points(runs):
    g = runs.groupby(['seq', 'system', 'codec', 'q']).agg(
        kbps=('kbps_total', 'median'), psnr=('psnr_y', 'median'), ssim=('ssim_y', 'median'),
        rel=('rel', 'median'), fail=('fail', 'mean'), n=('fail', 'size')).reset_index()
    g['rel_eff'] = np.where(g.fail >= 0.5, R_FAIL, g.rel.fillna(R_FAIL))
    return g

def _monotone_curve(kbps, rel):
    """Isotonic fit: error must not decrease as bitrate decreases. Returns sorted (log10 rate, log10 rel)."""
    o = np.argsort(kbps); lr = np.log10(np.asarray(kbps)[o]); y = np.log10(np.asarray(rel)[o])
    y = _pava_decreasing(y)
    return lr, y

def rate_at_tolerance(kbps, rel, tau):
    """Lowest bitrate (kbps) at which the monotone error curve is <= tau (log-linear interpolation)."""
    lr, y = _monotone_curve(kbps, rel); t = np.log10(tau)
    if y[-1] > t: return np.nan               # never reaches tolerance within tested range
    if y[0] <= t: return 10 ** lr[0]          # tolerance met even at the lowest tested rate (censored)
    i = np.where(y <= t)[0][0]
    x0, x1, y0, y1 = lr[i - 1], lr[i], y[i - 1], y[i]
    return 10 ** (x0 + (t - y0) * (x1 - x0) / (y1 - y0)) if y1 != y0 else 10 ** x1

def bd_rate_loc(k_a, r_a, k_t, r_t, lo=np.log10(1.05), hi=np.log10(3.0), n=200):
    """Bjontegaard-style rate difference (%) of test vs anchor at equal localization error.
    Integrates log-rate as a function of log relative ATE over the overlapping error interval within [lo, hi]."""
    def inv(k, r):
        lr, y = _monotone_curve(k, r)
        # collapse plateaus: keep the lowest rate per error level
        df = pd.DataFrame({'y': y, 'lr': lr}).groupby('y').lr.min().reset_index().sort_values('y')
        return df.y.values, df.lr.values
    ya, la = inv(k_a, r_a); yt, lt = inv(k_t, r_t)
    if len(ya) < 2 or len(yt) < 2: return np.nan
    a, b = max(ya.min(), yt.min(), lo), min(ya.max(), yt.max(), hi)
    if b - a < 0.05: return np.nan
    fa, ft = PchipInterpolator(ya, la), PchipInterpolator(yt, lt)
    g = np.linspace(a, b, n)
    return (10 ** (np.trapezoid(ft(g) - fa(g), g) / (b - a)) - 1) * 100

def summarize(pts, anchor='x264', taus=(1.25, 2.0)):
    rows = []
    for (seq, sysn), d in pts[pts.codec != 'raw'].groupby(['seq', 'system']):
        A = d[d.codec == anchor]
        for cdc, dd in d.groupby('codec'):
            row = dict(seq=seq, system=sysn, codec=cdc)
            for t in taus: row[f'R@{t}'] = rate_at_tolerance(dd.kbps.values, dd.rel_eff.values, t)
            fails = dd[dd.fail >= 0.5]
            row['fail_kbps'] = fails.kbps.max() if len(fails) else np.nan   # highest bitrate with failure
            row['bd_vs_' + anchor] = (0.0 if cdc == anchor else
                                     bd_rate_loc(A.kbps.values, A.rel_eff.values, dd.kbps.values, dd.rel_eff.values))
            rows.append(row)
    return pd.DataFrame(rows)
