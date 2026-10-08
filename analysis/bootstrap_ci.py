"""Hierarchical bootstrap for BD-rate_loc and R_tau medians.
Level 1: resample sequences (within estimator x dataset) with replacement.
Level 2: within each rate point, resample the repeated runs (ORB-SLAM3: 3 runs) with replacement.
The uncompressed reference ATE used for normalization is held fixed (conditional on the reference)."""
import numpy as np, pandas as pd
import analysis

def _prep(R):
    """dict[(system,dataset,seq,codec)] -> list of (kbps, rel_array, fail_array) per q"""
    d = {}
    for (s, ds, seq, c, q), g in R[R.codec != 'raw'].groupby(['system', 'dataset', 'seq', 'codec', 'q']):
        d.setdefault((s, ds, seq, c), []).append((g.kbps_total.median(), g.rel.values, g.fail.values.astype(bool)))
    return d

def _curve(pts, rng):
    k, r = [], []
    for kbps, rel, fail in pts:
        i = rng.integers(0, len(rel), len(rel)) if rng is not None else np.arange(len(rel))
        rr, ff = rel[i], fail[i]
        k.append(kbps)
        r.append(analysis.R_FAIL if ff.mean() >= 0.5 else np.nanmedian(np.where(ff, np.nan, rr)) if (~ff).any() else analysis.R_FAIL)
    return np.array(k), np.array(r)

def bootstrap(R, n_boot=1000, seed=0, anchor='x264', taus=(1.25,)):
    rng = np.random.default_rng(seed)
    D = _prep(R)
    groups = {}
    for (s, ds, seq, c) in D: groups.setdefault((s, ds), set()).add(seq)
    codecs = sorted({k[3] for k in D})
    out = []
    for (s, ds), seqs in groups.items():
        seqs = sorted(seqs)
        for b in range(n_boot):
            samp = rng.choice(seqs, len(seqs), replace=True)
            vals = {c: {'bd': [], **{f'R@{t}': [] for t in taus}} for c in codecs}
            for seq in samp:
                if (s, ds, seq, anchor) not in D: continue
                ka, ra = _curve(D[(s, ds, seq, anchor)], rng)
                for c in codecs:
                    if (s, ds, seq, c) not in D: continue
                    k, r = (ka, ra) if c == anchor else _curve(D[(s, ds, seq, c)], rng)
                    vals[c]['bd'].append(0.0 if c == anchor else analysis.bd_rate_loc(ka, ra, k, r))
                    for t in taus: vals[c][f'R@{t}'].append(analysis.rate_at_tolerance(k, r, t))
            for c, v in vals.items():
                if not v['bd']: continue
                row = dict(system=s, dataset=ds, codec=c, b=b, bd=np.nanmedian(v['bd']) if np.isfinite(v['bd']).any() else np.nan)
                for t in taus:
                    a = np.array(v[f'R@{t}'], float); row[f'R@{t}'] = np.nanmedian(a) if np.isfinite(a).any() else np.nan
                out.append(row)
    B = pd.DataFrame(out)
    ci = B.groupby(['system', 'dataset', 'codec']).agg(
        bd_lo=('bd', lambda x: np.nanpercentile(x, 2.5)), bd_hi=('bd', lambda x: np.nanpercentile(x, 97.5)),
        bd_pneg=('bd', lambda x: np.nanmean(x < 0)),
        r125_lo=('R@1.25', lambda x: np.nanpercentile(x, 2.5) if np.isfinite(x).any() else np.nan),
        r125_hi=('R@1.25', lambda x: np.nanpercentile(x, 97.5) if np.isfinite(x).any() else np.nan)).reset_index()
    return B, ci
