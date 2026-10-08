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
    """Encoder speed and log storage at the worst-case (over VI estimators) median R_1.25."""
    e = ct[ct.codec != 'raw'].copy(); e['dataset'] = e.seq.map(dataset)
    e['fps'] = e.frames / ((e.enc_s_cam0 + e.enc_s_cam1) / 2)
    sp = e.groupby(['dataset', 'codec']).fps.median()
    rt = S[S.system.isin(['orbslam3', 'basalt'])].groupby(['dataset', 'codec', 'system'])['R@1.25'].median().groupby(['dataset', 'codec']).max() / 1000
    dss = [d for d in ['TUM VI', 'EuRoC', 'KITTI'] if d in S.dataset.unique()]
    L = [r'\begin{table}[t]',
         r'\caption{Practical budget per configuration. Rate: worst case over ORB-SLAM3 and Basalt of the median $R_{1.25}$ (stereo, Mbit/s); hours: stereo log duration that fits into 1~TB at that rate; speed: median encoding speed per camera stream (frames/s) over all rate points, measured on a shared 28-core workstation and including PNG decoding (indicative only; NVENC on the GPU). Uncompressed: %s.}' % ', '.join(f'{RAW_MBPS[d]:.1f}~Mbit/s ({d.replace("TUM VI", "TUM~VI")}, {1e12 * 8 / (RAW_MBPS[d] * 1e6) / 3600:.0f}~h/TB)' for d in dss),
         r'\label{tab:practical}', r'\centering\footnotesize\setlength{\tabcolsep}{3pt}', r'\resizebox{\columnwidth}{!}{%',
         r'\begin{tabular}{@{}l' + 'rrr' * len(dss) + '@{}}', r'\toprule',
         ' & ' + ' & '.join(r'\multicolumn{3}{c}{%s}' % d.replace('TUM VI', 'TUM~VI') for d in dss) + r' \\',
         ''.join(r'\cmidrule(lr){%d-%d}' % (2 + 3 * i, 4 + 3 * i) for i in range(len(dss))),
         'Configuration & ' + ' & '.join(['Mbit/s & h/TB & fps'] * len(dss)) + r' \\', r'\midrule']
    for c in [c for c in mr.ORDER if c in S.codec.unique()]:
        cells = [mr.LAB[c]]
        for d in dss:
            r = rt.get((d, c), np.nan); f = sp.get((d, c), np.nan)
            cells += (['--', '--'] if not np.isfinite(r) else [mr.fmt_rate(r * 1000), f'{1e12 * 8 / (r * 1e6) / 3600:,.0f}'.replace(',', '{,}')]) + ['--' if not np.isfinite(f) else f'{f:.0f}']
        L.append(' & '.join(cells) + r' \\')
    L += [r'\bottomrule', r'\end{tabular}}', r'\end{table}']
    open(path, 'w').write('\n'.join(L) + '\n')
    return rt, sp

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
    ex = extra_macros(R, P, S)
    mj = S[(S.codec == 'mjpeg')].groupby(['system', 'dataset']).bd_vs_x264.median().dropna()
    mi = mj[mj.index.get_level_values(1) != 'KITTI']
    ex['MJPEGFactorMin'] = f"{1 + mi.min() / 100:.0f}"; ex['MJPEGFactorMax'] = f"{1 + mi.max() / 100:.0f}"
    mk = mj[mj.index.get_level_values(1) == 'KITTI']
    if len(mk): ex['MJPEGFactorKITMin'] = f"{1 + mk.min() / 100:.1f}"; ex['MJPEGFactorKITMax'] = f"{1 + mk.max() / 100:.1f}"
    fk = R[R.dataset == 'KITTI']
    if len(fk):
        ex['FailRateBasaltKIT'] = f"{100 * fk[fk.system == 'basalt'].fail.mean():.1f}"; ex['FailRateORBKIT'] = f"{100 * fk[fk.system == 'orbslam3'].fail.mean():.1f}"
        ref = R[(R.codec == 'raw') & (R.dataset == 'KITTI')].groupby(['system', 'seq']).ate_m.median()
        ex['RefORBKIT'] = f"{ref['orbslam3'].min():.2f}--{ref['orbslam3'].max():.2f}"; ex['RefBasKIT'] = f"{ref['basalt'].min():.2f}--{ref['basalt'].max():.2f}"
    vv = S[(S.codec == 'vvc') & (S.dataset == 'TUM VI')].groupby('system').bd_vs_x264.median()
    ex['VVCSaveMin'] = f"{-vv.max():.0f}"; ex['VVCSaveMax'] = f"{-vv.min():.0f}"
    rt = S[S.codec.isin(INTER) & S.system.isin(['orbslam3', 'basalt']) & (S.dataset != 'KITTI')].groupby(['system', 'dataset', 'codec'])['R@1.25'].median()
    rk = S[S.codec.isin(INTER) & S.system.isin(['orbslam3', 'basalt']) & (S.dataset == 'KITTI')].groupby(['system', 'codec'])['R@1.25'].median().dropna()
    if len(rk): ex['RtauKITMin'] = f"{rk.min() / 1000:.1f}"; ex['RtauKITMax'] = f"{rk.max() / 1000:.1f}"
    ex['RtauVIMin'] = f"{rt.min() / 1000:.1f}"; ex['RtauVIMax'] = f"{rt.max() / 1000:.1f}"
    cr = pd.Series({k: RAW_MBPS[k[1]] * 1000 / v for k, v in rt.items()})
    ex['CRMin'] = f"{10 * np.floor(cr.min() / 10):.0f}"
    low = P[P.codec != 'raw'].groupby(['dataset', 'codec', 'q']).kbps.median().reset_index().sort_values('kbps').groupby(['dataset', 'codec']).first()
    if ('TUM VI', 'vvc_ld') in low.index: ex['LowVVCLD'] = f"{low.loc[('TUM VI', 'vvc_ld'), 'kbps']:.0f}"
    vl = S[(S.codec == 'vvc_ld') & (S.dataset != 'KITTI') & S.system.isin(['orbslam3', 'basalt'])].groupby(['system', 'dataset']).bd_vs_x264.median()
    if len(vl): ex['VVCLDSaveMin'] = f"{-vl.max():.0f}"; ex['VVCLDSaveMax'] = f"{-vl.min():.0f}"
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
    plt.close('all')
    return dict(R=R, P=P, S=S, PT=PT, rho=rho, nums={**nums, **ex}, ci=ci, B=B, cmp=cmp, rt_worst=rt_w, speed=sp)
