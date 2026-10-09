"""End-to-end generation of all manuscript assets (figures, tables, numeric macros) from run tables.
Usage (python, repo root): exec(open('analysis/build_assets.py').read()); exec(open('analysis/final_pipeline.py').read());
out = build_all(runs_vi, runs_dpvo, codec_table, n_boot=1000)"""
import numpy as np, pandas as pd
import bootstrap_ci

INTER = ['x264', 'x265', 'svtav1', 'vvc', 'vvc_ld', 'nvh264', 'nvhevc']
RAW_MBPS = {'TUM VI': 83.886, 'EuRoC': 115.507, 'KITTI': 72.579}

def wr(d, path='paper/generated_numbers.tex'):
    with open(path, 'a') as f:
        for k, v in d.items(): f.write('\\newcommand{\\%s}{%s}\n' % (k, v))

def summ_at(vi, dp, th):
    old = analysis.COV_MIN; analysis.COV_MIN = th
    try:
        R_, P_, S_ = prepare(vi, dp)
    finally:
        analysis.COV_MIN = old
    return R_, S_

def sensitivity(vi, dp, R):
    R7, S7 = summ_at(vi, dp, 0.7); _, S8 = summ_at(vi, dp, 0.8)
    m = lambda S: S[S.codec != 'raw'].groupby(['system', 'dataset', 'codec']).bd_vs_x264.median()
    cmp = pd.concat({'0.8': m(S8), '0.7': m(S7)}, axis=1)
    tv = R[(R.system == 'orbslam3') & (R.dataset == 'TUM VI')]
    d = (cmp['0.8'] - cmp['0.7']).abs()
    out = {'FailRateORBseven': f"{100 * R7[R7.system == 'orbslam3'].fail.mean():.1f}",
           'SensTUM': f"{d.loc['orbslam3', 'TUM VI'].max():.0f}",
           'NoTrajTUM': f"{int(tv[tv.fail].ate_m.isna().sum())}", 'FailTUM': f"{int(tv.fail.sum())}",
           'SensEURxTwoSixFive': f"{cmp.loc[('orbslam3', 'EuRoC', 'x265'), '0.7']:+.0f}".replace('-', '$-$').replace('+', '$+$')}
    for ds_, key in (('EuRoC', 'SensEURMax'), ('KITTI', 'SensKITMax')):
        if ('orbslam3', ds_) in d.index.droplevel(2): out[key] = f"{d.loc['orbslam3', ds_].max():.0f}"
    return out, cmp

def practical_table(S, ct, path):
    """Worst case over ORB-SLAM3 and Basalt of the censoring-aware median R_tau, and log hours per TB."""
    dss = [d for d in ['TUM VI', 'EuRoC', 'KITTI'] if d in S.dataset.unique()]
    def worst(d, c, t):
        ms = [agg_rtau(cell(S, s, d, c), t)[0] for s in ('orbslam3', 'basalt')]
        ms = [m for m in ms if not np.isnan(m)]
        return max(ms) / 1000 if ms else np.nan
    cc = ct[ct.codec != 'raw'].copy(); px = {'TUM VI': 512 * 512, 'EuRoC': 752 * 480, 'KITTI': 1226 * 370}
    cc['mpps'] = cc.frames / cc[['enc_s_cam0', 'enc_s_cam1']].mean(axis=1) * cc.seq.map(dataset).map(px) / 1e6
    thr = cc.groupby('codec').mpps.median()
    hrs = lambda r: f'{1e12 * 8 / (r * 1e6) / 3600:,.0f}'.replace(',', '{,}')
    L = [r'\begin{table}[t]',
         r'\caption{Practical budget per configuration: worst case over ORB-SLAM3 and Basalt of the median rate at tolerance (stereo, Mbit/s; censoring as in Table~\ref{tab:rtau}; n.r.: not reached in the tested range for at least one estimator) and the stereo log duration that fits into 1~TB at that rate. Enc.: median encoding throughput per camera stream in megapixels per second over all encodings (shared workstation, indicative only). Uncompressed: %s.}' % ', '.join(f'{RAW_MBPS[d]:.1f}~Mbit/s ({d.replace("TUM VI", "TUM~VI")}, {1e12 * 8 / (RAW_MBPS[d] * 1e6) / 3600:.0f}~h/TB)' for d in dss),
         r'\label{tab:practical}', r'\centering\footnotesize\setlength{\tabcolsep}{3pt}', r'\resizebox{\columnwidth}{!}{%',
         r'\begin{tabular}{@{}l' + 'rrr' * len(dss) + 'r@{}}', r'\toprule',
         ' & ' + ' & '.join(r'\multicolumn{3}{c}{%s}' % d.replace('TUM VI', 'TUM~VI') for d in dss) + r' & \\',
         ''.join(r'\cmidrule(lr){%d-%d}' % (2 + 3 * i, 4 + 3 * i) for i in range(len(dss))),
         'Configuration & ' + ' & '.join([r'$R_{1.25}$ & h/TB & $R_{2}$'] * len(dss)) + r' & Enc. \\', r'\midrule']
    out = {}
    for c in [c for c in mr.ORDER if c in S.codec.unique()]:
        cells = [mr.LAB[c]]
        for d in dss:
            r1, r2 = worst(d, c, 'R@1.25'), worst(d, c, 'R@2.0'); out[(d, c)] = (r1, r2)
            cells += (['n.r.', '--'] if not np.isfinite(r1) else [mr.fmt_rate(r1 * 1000), hrs(r1)]) + ['n.r.' if not np.isfinite(r2) else mr.fmt_rate(r2 * 1000)]
        cells.append(f'{thr[c]:.0f}' if thr[c] >= 10 else f'{thr[c]:.1f}')
        L.append(' & '.join(cells) + r' \\')
    L += [r'\bottomrule', r'\end{tabular}}', r'\end{table}']
    open(path, 'w').write('\n'.join(L) + '\n')
    return out, None

def extra_macros(R, P, S):
    ref = R[R.codec == 'raw'].groupby(['system', 'dataset', 'seq']).ate_m.median().groupby(['system', 'dataset']).agg(['min', 'max'])
    rng = lambda s, d: f"{ref.loc[(s, d), 'min']:.3f}--{ref.loc[(s, d), 'max']:.3f}"
    low = P[P.codec != 'raw'].groupby(['dataset', 'codec', 'q']).kbps.median().reset_index().sort_values('kbps').groupby(['dataset', 'codec']).first()
    ex = {'RefORBTUM': rng('orbslam3', 'TUM VI'), 'RefBasTUM': rng('basalt', 'TUM VI'), 'RefDPVO': rng('dpvo', 'TUM VI'),
          'FailXTwoSixFour': f"{low.loc[('TUM VI', 'x264'), 'kbps']:.0f}", 'FailXTwoSixFive': f"{low.loc[('TUM VI', 'x265'), 'kbps']:.0f}",
          'LowSVT': f"{low.loc[('TUM VI', 'svtav1'), 'kbps']:.0f}", 'LowVVC': f"{low.loc[('TUM VI', 'vvc'), 'kbps']:.0f}", 'FailRegimeKbps': '200'}
    if ('orbslam3', 'EuRoC') in ref.index: ex.update(RefORBEUR=rng('orbslam3', 'EuRoC'), RefBasEUR=rng('basalt', 'EuRoC'))
    return ex

# measured once on room1 cam0 (setup/enc_checks.sh): gray->yuv420p->gray round trip, keyframe counts, MKV overhead
CHECKS = {'RangeRoundTripPSNR': '56.7', 'VVCRAKeyframes': '71', 'XTwoSixFourKeyframes': '101', 'CheckFrames': '2821',
          'MKVOverheadVVC': '1.5', 'MKVOverheadXTwoSixFour': '1.0'}

def build_all(vi, dp, ct, n_boot=1000, seed=1):
    R, P, S = prepare(vi, dp); S = add_censoring(S, P); PT = psnr_at_tol(P, S)
    figs = {'curves_tumvi': fig_curves(P, 'TUM VI', ['orbslam3', 'basalt', 'dpvo'], 'paper/figures/curves_tumvi.pdf'),
            'curves_euroc': fig_curves(P, 'EuRoC', ['orbslam3', 'basalt'], 'paper/figures/curves_euroc.pdf')}
    if (P.dataset == 'KITTI').any():
        figs['curves_kitti'] = fig_curves(P, 'KITTI', ['orbslam3', 'basalt'], 'paper/figures/curves_kitti.pdf')
    figs['rtau'] = fig_rtau(S, 'paper/figures/rtau.pdf')
    figs['psnr'], rho = fig_psnr(P, 'paper/figures/psnr.pdf')
    for k, f in figs.items(): f.savefig(f'prev_{k}.png', dpi=200, bbox_inches='tight')
    B, ci = bootstrap_ci.bootstrap(R, n_boot=n_boot, seed=seed)
    table_bd_ci(S, ci, 'paper/table_bd.tex'); table_rtau(S, 'paper/table_rtau.tex'); table_psnr(PT, 'paper/table_psnr.tex')
    rt_w, sp = practical_table(S, ct, 'paper/table_practical.tex')
    nums = gen_numbers(R, P, S, PT, rho, ct, 'paper/generated_numbers.tex', 'PLACEHOLDER')
    ex = extra_macros(R, P, S); ex.update(CHECKS); ex.update(globals().get('DIAG', {}))
    def bdc(s, d, c): return agg_bd(cell(S, s, d, c))[0]
    def rtc(s, d, c, t='R@1.25'): return agg_rtau(cell(S, s, d, c), t)[0]
    VI, IND = ['orbslam3', 'basalt'], ['TUM VI', 'EuRoC']
    mi = [bdc(s, d, 'mjpeg') for s in VI for d in IND]; mi = [m for m in mi if np.isfinite(m)]
    ex['MJPEGFactorMin'] = f"{1 + min(mi) / 100:.1f}"; ex['MJPEGFactorMax'] = f"{1 + max(mi) / 100:.0f}"
    ex['MJPEGBDMin'] = f"{min(mi):.0f}"; ex['MJPEGBDMax'] = f"{max(mi):.0f}"
    rr = [rtc(s, d, 'mjpeg') / rtc(s, d, 'x264') for s in VI for d in IND if np.isfinite(rtc(s, d, 'x264'))]
    ex['MJPEGRtRatioMin'] = f"{min(rr):.0f}"; ex['MJPEGRtRatioMax'] = f"{max(rr):.0f}"
    if 'KITTI' in S.dataset.values:
        ex['MJPEGKITBoundORB'] = f"{rtc('orbslam3', 'KITTI', 'mjpeg') / rtc('orbslam3', 'KITTI', 'x264'):.1f}"
        ex['MJPEGKITRatioBas'] = f"{rtc('basalt', 'KITTI', 'mjpeg') / rtc('basalt', 'KITTI', 'x264'):.1f}"
        kb = cell(S, 'basalt', 'KITTI', 'mjpeg').bd_vs_x264.dropna()
        ex['MJPEGKITBasSeq'] = ' and '.join(f'{v:+.0f}\\%' for v in sorted(kb)).replace('+', '$+$')
    fk = R[R.dataset == 'KITTI']
    if len(fk):
        ex['FailRateBasaltKIT'] = f"{100 * fk[fk.system == 'basalt'].fail.mean():.1f}"; ex['FailRateORBKIT'] = f"{100 * fk[fk.system == 'orbslam3'].fail.mean():.1f}"
        ref = R[(R.codec == 'raw') & (R.dataset == 'KITTI')].groupby(['system', 'seq']).ate_m.median()
        ex['RefORBKIT'] = f"{ref['orbslam3'].min():.2f}--{ref['orbslam3'].max():.2f}"; ex['RefBasKIT'] = f"{ref['basalt'].min():.2f}--{ref['basalt'].max():.2f}"
    rt1 = {(s, d, c): rtc(s, d, c) for s in VI for d in IND for c in INTER}
    fin = {k: v for k, v in rt1.items() if np.isfinite(v)}
    ex['RtauVIMin'] = f"{min(fin.values()) / 1000:.1f}"; ex['RtauVIMax'] = f"{max(fin.values()) / 1000:.1f}"
    ex['RtauVINR'] = str(sum(1 for v in rt1.values() if not np.isfinite(v)))
    rt2 = {(s, d, c): rtc(s, d, c, 'R@2.0') for s in VI for d in IND for c in INTER}
    ex['RtwoVIMin'] = f"{min(rt2.values()) / 1000:.2f}"; ex['RtwoVIMax'] = f"{max(rt2.values()) / 1000:.1f}"
    ex['CRMin'] = f"{10 * np.floor(min(RAW_MBPS[k[1]] * 1000 / v for k, v in fin.items()) / 10):.0f}"
    ex['CRtwoMin'] = f"{50 * np.floor(min(RAW_MBPS[k[1]] * 1000 / v for k, v in rt2.items()) / 50):.0f}"
    rk = [rtc(s, 'KITTI', c) for s in VI for c in INTER] if 'KITTI' in S.dataset.values else []
    rk = [v for v in rk if np.isfinite(v)]
    if rk: ex['RtauKITMin'] = f"{min(rk) / 1000:.1f}"; ex['RtauKITMax'] = f"{max(rk) / 1000:.1f}"
    low = P[P.codec != 'raw'].groupby(['dataset', 'codec', 'q']).kbps.median().reset_index().sort_values('kbps').groupby(['dataset', 'codec']).first()
    if ('TUM VI', 'vvc_ld') in low.index: ex['LowVVCLD'] = f"{low.loc[('TUM VI', 'vvc_ld'), 'kbps']:.0f}"
    vl = [bdc(s, d, 'vvc_ld') for s in VI for d in IND]
    ex['VVCLDSaveMin'] = f"{-max(vl):.0f}"; ex['VVCLDSaveMax'] = f"{-min(vl):.0f}"
    vv = [bdc(s, 'TUM VI', 'vvc') for s in VI + ['dpvo']]
    ex['VVCSaveMin'] = f"{-max(vv):.0f}"; ex['VVCSaveMax'] = f"{-min(vv):.0f}"
    spr = PT.groupby(['system', 'dataset', 'codec']).psnr125.median().dropna().groupby(['system', 'dataset']).agg(lambda v: v.max() - v.min())
    ex['PSNRSpreadMin'] = f"{spr.min():.1f}"; ex['PSNRSpreadMax'] = f"{spr.max():.1f}"
    ex['PSNRSpreadORBEUR'] = f"{spr.get(('orbslam3', 'EuRoC'), np.nan):.1f}"
    ex['BootN'] = f'{n_boot:,}'.replace(',', '{,}')
    sens, cmp = sensitivity(vi, dp, R); ex.update(sens)
    seqs = sorted(R[R.dataset == 'EuRoC'].seq.unique()); ex['EuRoCSeqList'] = ', '.join(s.replace('_', '\\_') for s in seqs)
    ks = sorted(R[R.dataset == 'KITTI'].seq.unique()); ex['KITTISeqList'] = ', '.join(s[1:] for s in ks)
    lines = open('paper/generated_numbers.tex').read().replace(r'\newcommand{\FindingOne}{PLACEHOLDER}', '')
    open('paper/generated_numbers.tex', 'w').write(lines)
    wr(ex)
    DSK = {'TUM VI': 'TUM', 'EuRoC': 'EUR', 'KITTI': 'KIT'}; sg = lambda x: '0' if round(x) == 0 else f'{x:+.0f}'.replace('-', '$-$').replace('+', '$+$')
    with open('paper/generated_numbers.tex', 'a') as f:
        for _, r in ci.iterrows():
            if np.isfinite(r.bd_lo): f.write('\\expandafter\\def\\csname bdci@%s@%s@%s\\endcsname{[%s, %s]}\n' % (r.system, DSK[r.dataset], r.codec, sg(r.bd_lo), sg(r.bd_hi)))
        f.write('\\newcommand{\\bdci}[3]{\\csname bdci@#1@#2@#3\\endcsname}\n')
    BQ, BP, TB, exb, dis = bsq_assets(P, S, ct); write_bsq_macros(exb)
    plt.close('all')
    return dict(BQ=BQ, BP=BP, TB=TB, bsq_macros=exb, bsq_dis=dis, R=R, P=P, S=S, PT=PT, rho=rho, nums={**nums, **ex}, ci=ci, B=B, cmp=cmp, rt_worst=rt_w, speed=sp)


# ---------- BSQ-rate (Zvezdakova et al., 2020) on PSNR and on localization error ----------
import bsq

def _agg_col(v):
    v = np.asarray(v, float); n = len(v); k = int(np.isfinite(v).sum())
    if n == 0 or k < 3 or k < n / 2: return np.nan, k, n
    return float(np.median(v[np.isfinite(v)])), k, n

def bsq_assets(P, S, ct, path='paper/table_bsq.tex'):
    rows = []
    for (seq, sy), d in P[P.codec != 'raw'].groupby(['seq', 'system']):
        A = d[d.codec == 'x264']
        for c, dd in d.groupby('codec'):
            rows.append(dict(seq=seq, system=sy, codec=c, dataset=dd.dataset.iloc[0],
                             bsq_loc=0.0 if c == 'x264' else bsq.bsq_loc(A.kbps.values, A.rel_eff.values, dd.kbps.values, dd.rel_eff.values)))
    BQ = pd.DataFrame(rows)
    E = ct[ct.codec != 'raw'].copy(); E['dataset'] = E.seq.map(dataset); rows = []
    for seq, d in E.groupby('seq'):
        A = d[d.codec == 'x264']
        for c, dd in d.groupby('codec'):
            rows.append(dict(seq=seq, codec=c, dataset=dd.dataset.iloc[0],
                             bsq_psnr=0.0 if c == 'x264' else bsq.bsq_rate(A.kbps_total.values, A.psnr_y.values, dd.kbps_total.values, dd.psnr_y.values)))
    BP = pd.DataFrame(rows)
    T = {}
    for (sy, ds, c), g in BQ.groupby(['system', 'dataset', 'codec']): T[(sy, ds, c)] = _agg_col(g.bsq_loc)
    for (ds, c), g in BP.groupby(['dataset', 'codec']): T[('psnr', ds, c)] = _agg_col(g.bsq_psnr)
    def fmt(key, c):
        if c == 'x264': return '0'
        if key not in T: return 'n.d.'
        m, k, n = T[key]
        if not np.isfinite(m): return r'n/a\,\textsuperscript{%d/%d}' % (k, n)
        return _sg(m) + ('' if k == n else r'\,\textsuperscript{%d/%d}' % (k, n))
    cols = [('TUM VI', ['psnr', 'orbslam3', 'basalt', 'dpvo']), ('EuRoC', ['psnr', 'orbslam3', 'basalt']), ('KITTI', ['psnr', 'orbslam3', 'basalt'])]
    hdr = {'psnr': 'PSNR', 'orbslam3': 'ORB', 'basalt': 'Bas.', 'dpvo': 'DPVO'}
    nc = sum(len(x[1]) for x in cols)
    L = [r'\begin{table}[t]',
         r'\caption{BSQ-rate~\cite{zvezdakova2020bsq} relative to x264 (\%): average bitrate difference at equal quality, with linear interpolation of the rate as a function of quality, removal of points that break monotonicity, and integration over the overlap of the quality ranges. PSNR: quality is the luma PSNR of the decoded images (image fidelity). ORB, Bas., DPVO: quality is the relative ATE of ORB-SLAM3, Basalt and DPVO, integrated over relative errors 1.05--3 as for $\mathrm{BD\text{-}rate}_{\mathrm{loc}}$. Median over sequences; $k/n$ and n/a as in Table~\ref{tab:bd}; n.d.: not evaluated.}',
         r'\label{tab:bsq}', r'\centering\footnotesize\setlength{\tabcolsep}{2.5pt}', r'\resizebox{\columnwidth}{!}{%',
         r'\begin{tabular}{@{}l' + 'r' * nc + '@{}}', r'\toprule',
         ' & ' + ' & '.join(r'\multicolumn{%d}{c}{%s}' % (len(s), d.replace('TUM VI', 'TUM~VI')) for d, s in cols) + r' \\']
    i = 2; cm = ''
    for d, s in cols: cm += r'\cmidrule(lr){%d-%d}' % (i, i + len(s) - 1); i += len(s)
    L += [cm, 'Configuration & ' + ' & '.join(hdr[x] for _, s in cols for x in s) + r' \\', r'\midrule']
    for c in [c for c in mr.ORDER if c not in ('raw', 'x264')]:
        cells = [mr.LAB[c]]
        for d, s in cols:
            for x in s: cells.append(fmt((x, d, c) if x == 'psnr' else (x, d, c), c))
        L.append(' & '.join(cells) + r' \\')
    L += [r'\bottomrule', r'\end{tabular}}', r'\end{table}']
    open(path, 'w').write('\n'.join(L) + '\n')
    from scipy.stats import spearmanr
    M = S.merge(BQ, on=['seq', 'system', 'codec', 'dataset']); M = M[(M.codec != 'x264') & M.bd_vs_x264.notna() & M.bsq_loc.notna()]
    ex = {'BSQBDRho': f"{spearmanr(M.bd_vs_x264, M.bsq_loc).statistic:.2f}", 'BSQBDN': str(len(M)),
          'BSQBDSign': str(int((np.sign(M.bd_vs_x264) != np.sign(M.bsq_loc)).sum())),
          'BSQBDAbsDiff': f"{(M.bsq_loc - M.bd_vs_x264).abs().median():.0f}"}
    med = lambda key: T.get(key, (np.nan,))[0]
    mjp = [med(('psnr', d, 'mjpeg')) for d in ['TUM VI', 'EuRoC', 'KITTI']]
    mjl = [med((s, d, 'mjpeg')) for s in ['orbslam3', 'basalt'] for d in ['TUM VI', 'EuRoC', 'KITTI']]; mjl = [v for v in mjl if np.isfinite(v)]
    ex.update(BSQMJPSNRMin=f'{min(mjp):.0f}', BSQMJPSNRMax=f'{max(mjp):.0f}', BSQMJLocMin=f'{min(mjl):.0f}', BSQMJLocMax=f'{max(mjl):.0f}')
    # combinations where the PSNR-based and the localization-based medians disagree in sign
    dis = []
    for (sy, ds, c), (m, k, n) in T.items():
        if sy == 'psnr' or c in ('x264', 'mjpeg') or not np.isfinite(m): continue
        mp = med(('psnr', ds, c))
        if np.isfinite(mp) and np.sign(mp) != np.sign(m) and abs(m) >= 5: dis.append((sy, ds, c))
    ex['BSQSignDis'] = str(len(dis)); ex['BSQSignDisKIT'] = str(sum(1 for x in dis if x[1] == 'KITTI'))
    ntot = sum(1 for (sy, ds, c), v in T.items() if sy != 'psnr' and c not in ('x264', 'mjpeg') and np.isfinite(v[0]))
    ex['BSQSignTot'] = str(ntot)
    gap = [abs(med((s, d, c)) - med(('psnr', d, c))) for s in ['orbslam3', 'basalt', 'dpvo'] for d in ['TUM VI', 'EuRoC', 'KITTI'] for c in INTER
           if np.isfinite(med((s, d, c))) and np.isfinite(med(('psnr', d, c)))]
    ex['BSQGapMed'] = f'{np.median(gap):.0f}'; ex['BSQGapMax'] = f'{max(gap):.0f}'
    for (sy, ds, c), (m, k, n) in T.items():
        if np.isfinite(m): ex_key = 'bsq@%s@%s@%s' % (sy, {'TUM VI': 'TUM', 'EuRoC': 'EUR', 'KITTI': 'KIT'}[ds], c); ex[ex_key] = _sg(m)
    return BQ, BP, T, ex, dis

def write_bsq_macros(ex, path='paper/generated_numbers.tex'):
    with open(path, 'a') as f:
        for k, v in ex.items():
            if '@' in k: f.write('\\expandafter\\def\\csname %s\\endcsname{%s}\n' % (k, v))
            else: f.write('\\newcommand{\\%s}{%s}\n' % (k, v))
        f.write('\\newcommand{\\bsq}[3]{\\csname bsq@#1@#2@#3\\endcsname}\n')
