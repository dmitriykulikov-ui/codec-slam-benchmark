"""Diagnostics quoted in the paper (failure classification, sensitivity analyses, run-to-run spread, correlations).
Executed into the same namespace as build_assets.py / final_pipeline.py; compute_diagnostics() returns LaTeX macros."""
import os
import numpy as np, pandas as pd
from scipy.stats import spearmanr

ORB_TIMEOUT_S = 880.0      # ORB-SLAM3 container limit is 900 s (1500 s on KITTI); runs without trajectory beyond 880 s = timeout

def _fmt_pct(x): return f'{100 * x:.0f}'

def compute_diagnostics(vi, dp, ct, R, P, S, results_dir=None):
    D = {}
    fr = ct[ct.codec == 'raw'].set_index('seq').frames
    dur = fr / np.where(fr.index.str.startswith('K'), 10.0, 20.0)
    o = R[R.system == 'orbslam3'].copy(); o['dur'] = o.seq.map(dur)
    # failure classification of ORB-SLAM3 runs without trajectory (from wall-clock time)
    nt = o[o.ate_m.isna()]
    to = nt[nt.wall_s >= ORB_TIMEOUT_S]; cr = nt[(nt.wall_s < ORB_TIMEOUT_S) & (nt.wall_s < 0.5 * nt.dur)]
    D.update(NumORBNoTraj=str(len(nt)), NumORBTimeout=str(len(to)), NumORBCrash=str(len(cr)))
    def tab(Sx):
        out = {}
        for (s, d), g in Sx[Sx.codec != 'raw'].groupby(['system', 'dataset']):
            for c in g.codec.unique(): out[(s, d, c)] = (agg_bd(cell(Sx, s, d, c))[0], agg_rtau(cell(Sx, s, d, c), 'R@1.25')[0])
        return out
    T0 = tab(S)
    excl = set(map(tuple, pd.concat([to, cr])[['seq', 'codec', 'q', 'system', 'rep']].values))
    mask = [tuple(x) not in excl for x in vi[['seq', 'codec', 'q', 'system', 'rep']].values]
    Rx, Px, Sx = prepare(vi[mask], dp); Sx = add_censoring(Sx, Px); T1 = tab(Sx)
    dbd = [abs(T0[k][0] - T1[k][0]) for k in T0 if k in T1 and np.isfinite(T0[k][0]) and np.isfinite(T1[k][0])]
    drt = [abs(np.log(T0[k][1] / T1[k][1])) for k in T0 if k in T1 and np.isfinite(T0[k][1]) and np.isfinite(T1[k][1])]
    D['CrashSensBD'] = f'{max(dbd):.0f}'; D['CrashSensRt'] = f'{100 * (np.exp(max(drt)) - 1):.0f}'
    # r_fail sensitivity
    dd = []
    for rf in (10.0, 100.0):
        old = analysis.R_FAIL; analysis.R_FAIL = rf
        Rr, Pr, Sr = prepare(vi, dp); Sr = add_censoring(Sr, Pr); Tr = tab(Sr); analysis.R_FAIL = old
        dd += [abs(T0[k][0] - Tr[k][0]) for k in T0 if k in Tr and np.isfinite(T0[k][0]) and np.isfinite(Tr[k][0])]
    D['RfailSensBD'] = f'{max(dd):.0f}'
    # ORB-SLAM3 run-to-run spread on the grid
    g = R[(R.system == 'orbslam3') & ~R.fail].groupby(['seq', 'codec', 'q']).ate_m.agg(['count', 'min', 'max', 'mean'])
    g = g[g['count'] == 3]; sp = (g['max'] - g['min']) / g['mean']
    D.update(ORBSpreadMed=_fmt_pct(sp.median()), ORBSpreadNcfg=str(len(g)), ORBSpreadOver=_fmt_pct((sp > 0.25).mean()))
    # pilot spread (room1-room3, configurations with three Basalt repetitions), including off-grid pilot points
    pil = vi[vi.seq.isin(['room1', 'room2', 'room3'])]
    b3 = pil[pil.system == 'basalt'].groupby(['seq', 'codec', 'q']).ate_m.agg(['count', 'min', 'max', 'mean']); b3 = b3[b3['count'] == 3]
    o3 = pil[(pil.system == 'orbslam3')].groupby(['seq', 'codec', 'q']).ate_m.agg(['count', 'min', 'max', 'mean'])
    o3 = o3[(o3['count'] == 3) & o3.index.isin(b3.index)]
    so = (o3['max'] - o3['min']) / o3['mean']; sb = (b3['max'] - b3['min']) / b3['mean']
    D.update(PilotORBN=str(len(so)), PilotORBMed=_fmt_pct(so.median()), PilotORBMax=_fmt_pct(so.max()),
             PilotBasN=str(len(sb)), PilotBasMed=f'{100 * sb.median():.1f}', PilotBasMax=_fmt_pct(sb.max()))
    # wall-clock time: pacing overrun and correlation with relative error (time normalized per sequence)
    wd = (o.wall_s / o.dur).groupby(o.dataset).median()
    D['WallOverMin'] = _fmt_pct(wd.min() - 1); D['WallOverMax'] = _fmt_pct(wd.max() - 1)
    ok = o[~o.fail & o.rel.notna()].copy(); ok['wn'] = ok.wall_s / ok.groupby('seq').wall_s.transform('median')
    for ds_, k in (('TUM VI', 'TUM'), ('EuRoC', 'EUR'), ('KITTI', 'KIT')):
        gg = ok[ok.dataset == ds_]; D['WallRho' + k] = f'{spearmanr(gg.wn, gg.rel).statistic:.2f}'.replace('-', '$-$')
    # deviations of individual uncompressed ORB-SLAM3 runs from the median of their repetitions
    u = o[o.codec == 'raw'].copy(); u['dev'] = (u.ate_m / u.groupby('seq').ate_m.transform('median') - 1).abs()
    D.update(RefORBRuns=str(len(u)), RefORBDevOver=_fmt_pct((u.dev > 0.25).mean()))
    # MH_02 partial trajectories by PSNR
    m2 = o[(o.seq == 'MH_02_easy') & (o.codec != 'raw')].copy(); m2['pb'] = pd.cut(m2.psnr_y, [0, 30, 33, 36, 40, 99])
    fb = m2.groupby('pb', observed=True).fail.mean()
    D.update(MHtwoFailLowMin=_fmt_pct(min(fb.iloc[0], fb.iloc[1])), MHtwoFailLowMax=_fmt_pct(max(fb.iloc[0], fb.iloc[1])),
             MHtwoFailMid=_fmt_pct(fb.iloc[2]), MHtwoFailHigh=_fmt_pct(fb.iloc[3]))
    # SSIM / RPE consistency
    for s, k in (('orbslam3', 'ORB'), ('basalt', 'Bas'), ('dpvo', 'DPVO')):
        pp = P[(P.system == s) & (P.codec != 'raw')]; D['RhoSSIM' + k] = f'{spearmanr(pp.ssim, pp.rel_eff).statistic:.2f}'.replace('-', '$-$')
    for s, k in (('orbslam3', 'ORB'), ('basalt', 'Bas')):
        x = R[(R.system == s) & ~R.fail & R.rel.notna() & R.rpe1m_m.notna()].copy()
        ref = R[(R.system == s) & (R.codec == 'raw')].groupby('seq').rpe1m_m.median()
        x['rrpe'] = x.rpe1m_m / x.seq.map(ref); x = x[x.codec != 'raw']
        D['RhoRPE' + k] = f'{spearmanr(x.rel, x.rrpe).statistic:.2f}'
    # x264 scene-cut audit (optional table)
    if results_dir and os.path.exists(os.path.join(results_dir, 'x264_scenecut_audit.csv')):
        sc = pd.read_csv(os.path.join(results_dir, 'x264_scenecut_audit.csv')); vals = []
        for seq, g2 in sc.groupby('seq'):
            a = g2[g2['mode'] == 'nosc']; b = g2[g2['mode'] == 'default']
            vals.append(bsq.bsq_rate(a.bytes.values, a.psnr.values, b.bytes.values, b.psnr.values))
        D['SceneCutBSQMin'] = f'{min(vals):+.1f}'.replace('-', '$-$').replace('+', '$+$'); D['SceneCutBSQMax'] = f'{max(vals):+.1f}'.replace('-', '$-$').replace('+', '$+$')
    return D
