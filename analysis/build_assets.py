"""Build all figures, tables and numeric macros for the manuscript from run tables.
Usage (python kernel): exec(open('build_assets.py').read()); build(run_tables)"""
import numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt
from scipy.stats import spearmanr
import analysis, make_results as mr

GRID = {'x264': [23, 30, 36, 40, 44, 47], 'x265': [23, 30, 36, 42, 46, 50], 'svtav1': [30, 40, 48, 55, 60, 63],
        'vvc': [27, 32, 37, 42, 47, 51], 'vvc_ld': [27, 32, 37, 42, 47, 51], 'nvh264': [25, 30, 35, 40, 45, 50], 'nvhevc': [25, 30, 35, 40, 45, 50],
        'mjpeg': [2, 5, 10, 15, 22, 31], 'raw': [0]}
DPVO_GRID = {'x264': [30, 36, 40, 44], 'x265': [30, 36, 42, 46], 'svtav1': [40, 48, 55, 60], 'vvc': [32, 37, 42, 47], 'vvc_ld': [32, 37, 42, 47], 'raw': [0]}

def on_grid(df, grid):
    return df[[q in grid.get(c, []) for c, q in zip(df.codec, df.q)]]

TITLE = {'orbslam3': 'ORB-SLAM3 (stereo-inertial)', 'basalt': 'Basalt (stereo-inertial)', 'dpvo': 'DPVO (monocular)'}

def dataset(seq): return 'KITTI' if seq.startswith('K') else ('EuRoC' if seq.startswith('MH_') else 'TUM VI')

def prepare(runs_vi, runs_dpvo):
    vi = on_grid(runs_vi, GRID); dp = on_grid(runs_dpvo, DPVO_GRID)
    R = analysis.load(pd.concat([vi, dp], ignore_index=True))
    R['dataset'] = R.seq.map(dataset)
    P = analysis.points(R); P['dataset'] = P.seq.map(dataset)
    S = analysis.summarize(P); S['dataset'] = S.seq.map(dataset)
    return R, P, S

def ptab_cell(v, fmt):
    v = v.dropna()
    return '--' if v.empty else fmt(v)

def table_bd(S, path):
    cols = [('orbslam3', 'TUM VI'), ('orbslam3', 'EuRoC'), ('orbslam3', 'KITTI'), ('basalt', 'TUM VI'), ('basalt', 'EuRoC'), ('basalt', 'KITTI'), ('dpvo', 'TUM VI')]
    cols = [c for c in cols if ((S.system == c[0]) & (S.dataset == c[1])).any()]
    nseq = {c: S[(S.system == c[0]) & (S.dataset == c[1])].seq.nunique() for c in cols}
    L = [r'\begin{table*}[t]',
         r'\caption{$\mathrm{BD\text{-}rate}_{\mathrm{loc}}$ relative to x264 (\%): average bitrate difference at equal localization error over relative errors 1.05--3. Negative values mean fewer bits than x264 for the same trajectory error. Median over sequences, range in brackets; $n$ = number of sequences. ``n.d.'': configuration not evaluated for this estimator; ``--'': curves do not overlap sufficiently. The ORB-SLAM3 EuRoC values depend on the failure criterion (Section~\ref{sec:sensitivity}).}',
         r'\label{tab:bd}', r'\centering\footnotesize\setlength{\tabcolsep}{4pt}',
         r'\begin{tabular}{@{}l' + 'c' * len(cols) + '@{}}', r'\toprule',
         ' & ' + ' & '.join(r'%s, %s ($n{=}%d$)' % (mr.SYSSHORT[s], d.replace('TUM VI', 'TUM~VI'), nseq[(s, d)]) for s, d in cols) + r' \\', r'\midrule']
    for c in [c for c in mr.ORDER if c in S.codec.unique()]:
        cells = [mr.LAB[c]]
        for s, d in cols:
            v = S[(S.system == s) & (S.dataset == d) & (S.codec == c)].bd_vs_x264
            if c == 'x264': cells.append('0 (anchor)'); continue
            if v.empty: cells.append('n.d.'); continue
            cells.append(ptab_cell(v, lambda v: f'{v.median():+.0f} [{v.min():+.0f}, {v.max():+.0f}]'.replace('-', '$-$').replace('+', '$+$')))
        L.append(' & '.join(cells) + r' \\')
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table*}']
    open(path, 'w').write('\n'.join(L) + '\n')

def table_rtau(S, path):
    cols = [('orbslam3', 'TUM VI'), ('orbslam3', 'EuRoC'), ('orbslam3', 'KITTI'), ('basalt', 'TUM VI'), ('basalt', 'EuRoC'), ('basalt', 'KITTI'), ('dpvo', 'TUM VI')]
    cols = [c for c in cols if ((S.system == c[0]) & (S.dataset == c[1])).any()]
    L = [r'\begin{table*}[t]',
         r'\caption{Rate at tolerance (stereo bitrate in Mbit/s; DPVO uses only the left stream but the same stereo rate is reported for comparability): lowest bitrate at which the median relative ATE is at most 1.25 ($R_{1.25}$) and 2 ($R_{2}$). Median over sequences. Values marked $^\dagger$ are censored at the lowest tested rate for at least one sequence; ``--'': tolerance not reached at the highest tested rate for most sequences. Uncompressed: 83.9 (TUM~VI), 115.5 (EuRoC) and 72.6~Mbit/s (KITTI).}',
         r'\label{tab:rtau}', r'\centering\footnotesize\setlength{\tabcolsep}{4pt}', r'\resizebox{\textwidth}{!}{%',
         r'\begin{tabular}{@{}l' + 'cc' * len(cols) + '@{}}', r'\toprule',
         ' & ' + ' & '.join(r'\multicolumn{2}{c}{%s, %s}' % (mr.SYSSHORT[s], d.replace('TUM VI', 'TUM~VI')) for s, d in cols) + r' \\',
         ''.join(r'\cmidrule(lr){%d-%d}' % (2 + 2 * i, 3 + 2 * i) for i in range(len(cols))),
         'Configuration & ' + ' & '.join([r'$R_{1.25}$ & $R_{2}$'] * len(cols)) + r' \\', r'\midrule']
    for c in [c for c in mr.ORDER if c in S.codec.unique()]:
        cells = [mr.LAB[c]]
        for s, d in cols:
            sub = S[(S.system == s) & (S.dataset == d) & (S.codec == c)]
            if sub.empty: cells += ['n.d.', 'n.d.']; continue
            for t in ('R@1.25', 'R@2.0'):
                v = sub[t]
                if v.isna().mean() > 0.5: cells.append('--'); continue
                cens = bool(sub[f'cens_{t}'].any()) if f'cens_{t}' in sub else False
                cells.append(mr.fmt_rate(v.median()) + (r'$^\dagger$' if cens else ''))
        L.append(' & '.join(cells) + r' \\')
    L += [r'\bottomrule', r'\end{tabular}}', r'\end{table*}']
    open(path, 'w').write('\n'.join(L) + '\n')

def add_censoring(S, P):
    S = S.copy()
    mins = P[P.codec != 'raw'].groupby(['seq', 'system', 'codec']).kbps.min()
    for t in ('R@1.25', 'R@2.0'):
        S[f'cens_{t}'] = [bool(np.isclose(r, mins.get((a, b, c), np.nan))) for r, a, b, c in zip(S[t], S.seq, S.system, S.codec)]
    return S

def psnr_at_tol(P, S):
    rows = []
    for (seq, s, cdc), d in P[P.codec != 'raw'].groupby(['seq', 'system', 'codec']):
        r = S[(S.seq == seq) & (S.system == s) & (S.codec == cdc)].iloc[0]
        o = d.sort_values('kbps'); lr = np.log10(o.kbps.values)
        def at(R):
            if not np.isfinite(R) or not (lr[0] <= np.log10(R) <= lr[-1]): return np.nan
            return float(np.interp(np.log10(R), lr, o.psnr.values))
        rows.append(dict(seq=seq, system=s, codec=cdc, dataset=dataset(seq), psnr125=at(r['R@1.25']), psnr2=at(r['R@2.0'])))
    return pd.DataFrame(rows)

def fig_curves(P, ds, systems, path):
    d = P[P.dataset == ds]
    fig, axes = plt.subplots(1, len(systems), figsize=(7.1 if len(systems) == 3 else 4.9, 2.5), sharey=True,
                             gridspec_kw=dict(wspace=0.06))
    axes = np.atleast_1d(axes)
    for ax, s in zip(axes, systems):
        ds_ = d[(d.system == s) & (d.codec != 'raw')]
        for c in mr.ORDER:
            dd = ds_[ds_.codec == c]
            if dd.empty: continue
            agg = dd.groupby('q').agg(kbps=('kbps', 'median'), rel=('rel_eff', 'median')).sort_values('kbps')
            ok = agg[agg.rel < analysis.R_FAIL]; bad = agg[agg.rel >= analysis.R_FAIL]
            ax.plot(ok.kbps / 1000, ok.rel, mr.LS.get(c, '-'), marker='o', ms=2.6, lw=1.05, color=mr.COL[c], label=mr.LAB[c], zorder=3)
            if len(bad): ax.scatter(bad.kbps / 1000, np.full(len(bad), 20.0), marker='x', s=16, lw=0.9, color=mr.COL[c], zorder=4)
        ax.axhline(1, color='0.45', lw=0.7, ls=':'); ax.axhspan(1, 1.25, color='0.92', zorder=0, lw=0)
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_title({'orbslam3': 'ORB-SLAM3 (stereo)', 'basalt': 'Basalt (stereo VO)'}.get(s, TITLE[s]) if ds == 'KITTI' else TITLE[s], loc='left')
        ax.set_xlabel('Stereo bitrate (Mbit/s)'); ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(mr.mbps_fmt))
        ax.set_xlim(0.018, 30)
    a0 = axes[0]; a0.set_ylabel('Relative ATE (median over sequences)'); a0.set_ylim(0.7, 26)
    a0.set_yticks([1, 1.25, 2, 5, 10, 20]); a0.set_yticklabels(['1', '1.25', '2', '5', '10', 'fail'])
    a0.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    a0.text(0.03, 0.04, 'lower = better', transform=a0.transAxes, fontsize=6, color='0.35')
    h, l = [], []
    for ax in axes:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l: h.append(hh); l.append(ll)
    fig.legend(h, l, loc='lower center', ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.2 if len(systems) < 3 else -0.15), fontsize=6, handlelength=1.6, columnspacing=1.0)
    fig.savefig(path, bbox_inches='tight'); return fig

def fig_rtau(S, path):
    systems = [s for s in mr.SYS if s in S.system.unique()]
    codecs = [c for c in mr.ORDER if c in S.codec.unique()]
    fig, axes = plt.subplots(1, len(systems), figsize=(7.1, 2.2), sharey=True, gridspec_kw=dict(wspace=0.06))
    MK = {'TUM VI': 'o', 'EuRoC': '^', 'KITTI': 's'}
    for ax, s in zip(np.atleast_1d(axes), systems):
        for i, c in enumerate(codecs):
            sub = S[(S.system == s) & (S.codec == c)]
            if sub.empty:
                ax.text(0.06, i, 'n.d.', va='center', fontsize=5.5, color='0.5', transform=ax.get_yaxis_transform()); continue
            for j, (dsn, dd) in enumerate(sub.groupby('dataset')):
                v = dd['R@1.25'].dropna() / 1000
                ax.scatter(v, np.full(len(v), i) + {'TUM VI': -0.2, 'EuRoC': 0.0, 'KITTI': 0.2}[dsn], s=9, marker=MK[dsn],
                           facecolor=mr.COL[c], edgecolor='none', alpha=0.8, zorder=3)
            v = sub['R@1.25'].dropna() / 1000
            if len(v): ax.plot([v.median()] * 2, [i - 0.32, i + 0.32], color=mr.COL[c], lw=1.8, zorder=4)
            else: ax.text(0.06, i, 'not reached', va='center', fontsize=5.5, color='0.5', transform=ax.get_yaxis_transform())
        ax.set_xscale('log'); ax.set_title(mr.SYSSHORT[s], loc='left'); ax.set_xlabel(r'$R_{1.25}$, stereo (Mbit/s)')
        ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(mr.mbps_fmt)); ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter()); ax.set_xlim(0.05, 15); ax.margins(y=0.05)
    a0 = np.atleast_1d(axes)[0]
    a0.set_yticks(range(len(codecs))); a0.set_yticklabels([mr.LAB[c] for c in codecs]); a0.invert_yaxis()
    from matplotlib.lines import Line2D
    hl = [Line2D([], [], ls='', marker='o', color='0.4', ms=3.5, label='TUM VI sequence'),
          Line2D([], [], ls='', marker='^', color='0.4', ms=3.5, label='EuRoC sequence'),
          Line2D([], [], ls='', marker='s', color='0.4', ms=3.2, label='KITTI sequence'),
          Line2D([], [], color='0.4', lw=1.8, label='median')]
    np.atleast_1d(axes)[-1].legend(handles=hl, loc='lower right', frameon=False, fontsize=5.5)
    a0.text(0.97, 0.98, 'left = fewer bits', transform=a0.transAxes, ha='right', va='top', fontsize=5.5, color='0.35')
    fig.savefig(path, bbox_inches='tight'); return fig

def gen_numbers(R, P, S, PT, rho, codec_tables, path, finding):
    DS = {'TUM VI': 'TUM', 'EuRoC': 'EUR', 'KITTI': 'KIT'}
    L = []
    def cs(name, val): L.append(r'\expandafter\def\csname %s\endcsname{%s}' % (name, val))
    for (s, d, c), g in S.groupby(['system', 'dataset', 'codec']):
        b = g.bd_vs_x264.dropna()
        cs(f'bd@{s}@{DS[d]}@{c}', '--' if b.empty else f'{b.median():+.0f}'.replace('-', '$-$').replace('+', '$+$'))
        for t, k in (('R@1.25', '125'), ('R@2.0', '2')):
            v = g[t].dropna(); cs(f'rt@{s}@{DS[d]}@{c}@{k}', '--' if v.empty else f'{v.median() / 1000:.2f}')
    for (s, d, c), g in PT.groupby(['system', 'dataset', 'codec']):
        v = g.psnr125.dropna(); cs(f'pt@{s}@{DS[d]}@{c}', '--' if v.empty else f'{v.median():.1f}')
    for s, r_ in rho.items(): cs(f'rho@{s}', f'{r_:.2f}'.replace('-', '{-}'))
    L.append(r'\newcommand{\bd}[3]{\csname bd@#1@#2@#3\endcsname}')
    L.append(r'\newcommand{\rt}[4]{\csname rt@#1@#2@#3@#4\endcsname}')
    L.append(r'\newcommand{\pt}[3]{\csname pt@#1@#2@#3\endcsname}')
    L.append(r'\newcommand{\rhoof}[1]{\csname rho@#1\endcsname}')
    nseq_t = R[R.dataset == 'TUM VI'].seq.nunique(); nseq_e = R[R.dataset == 'EuRoC'].seq.nunique(); nseq_k = R[R.dataset == 'KITTI'].seq.nunique()
    words = {0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten', 11: 'eleven', 12: 'twelve', 13: 'thirteen', 14: 'fourteen', 15: 'fifteen'}
    ct = codec_tables
    n_enc = len(ct[ct.codec != 'raw']) + len(ct[ct.codec == 'raw'])
    r_ = ct[ct.codec == 'raw'].set_index('seq'); durs = r_.frames / np.where(r_.index.str.startswith('K'), 10.0, 20.0)
    nums = {'NumCodecs': words[R[R.codec != 'raw'].codec.nunique()], 'NumRuns': f'{len(R):,}'.replace(',', '{,}'), 'NumSeqs': words[nseq_t + nseq_e + nseq_k], 'NumSeqsKITTI': words[nseq_k],
            'NumSeqsTUM': words[nseq_t], 'NumSeqsEuRoC': words[nseq_e], 'NumEncodings': f'{n_enc}',
            'SeqDurMin': f'{durs.min():.0f}', 'SeqDurMax': f'{durs.max():.0f}',
            'NumFramesTotal': f'{int(ct[ct.codec == "raw"].frames.sum()):,}'.replace(',', '{,}'),
            'FFmpegVersion': 'master N-127233, 7 Oct.~2026', 'FindingOne': finding,
            'NumRunsORB': f'{(R.system == "orbslam3").sum():,}'.replace(',', '{,}'),
            'NumRunsBasalt': f'{(R.system == "basalt").sum():,}'.replace(',', '{,}'),
            'NumRunsDPVO': f'{(R.system == "dpvo").sum()}',
            'FailRateORB': f'{100 * R[R.system == "orbslam3"].fail.mean():.1f}',
            'FailRateBasalt': f'{100 * R[R.system == "basalt"].fail.mean():.1f}',
            'FailRateDPVO': f'{100 * R[R.system == "dpvo"].fail.mean():.1f}'}
    for k, v in nums.items(): L.append(r'\newcommand{\%s}{%s}' % (k, v))
    open(path, 'w').write('\n'.join(L) + '\n')
    return nums

def table_psnr(PT, path):
    cols = [('orbslam3', 'TUM VI'), ('orbslam3', 'EuRoC'), ('orbslam3', 'KITTI'), ('basalt', 'TUM VI'), ('basalt', 'EuRoC'), ('basalt', 'KITTI'), ('dpvo', 'TUM VI')]
    cols = [c for c in cols if ((PT.system == c[0]) & (PT.dataset == c[1])).any()]
    L = [r'\begin{table}[t]', r'\caption{Luma PSNR (dB) at which the relative ATE reaches 1.25, interpolated on the rate--PSNR curve at $R_{1.25}$; median over sequences. If PSNR were a sufficient proxy for localization quality, each column would be nearly constant.}',
         r'\label{tab:psnr}', r'\centering\footnotesize\setlength{\tabcolsep}{3pt}', r'\resizebox{\columnwidth}{!}{%',
         r'\begin{tabular}{@{}l' + 'c' * len(cols) + '@{}}', r'\toprule',
         ' & ' + ' & '.join(r'\shortstack{%s\\%s}' % (mr.SYSSHORT[s], d.replace('TUM VI', 'TUM~VI')) for s, d in cols) + r' \\', r'\midrule']
    for c in [c for c in mr.ORDER if c in PT.codec.unique()]:
        cells = [mr.LAB[c]]
        for s, d in cols:
            v = PT[(PT.system == s) & (PT.dataset == d) & (PT.codec == c)].psnr125.dropna()
            cells.append('n.d.' if not ((PT.system == s) & (PT.dataset == d) & (PT.codec == c)).any() else ('--' if v.empty else f'{v.median():.1f}'))
        L.append(' & '.join(cells) + r' \\')
    spread = []
    for s, d in cols:
        v = PT[(PT.system == s) & (PT.dataset == d)].groupby('codec').psnr125.median().dropna()
        spread.append(f'{v.max() - v.min():.1f}' if len(v) > 1 else '--')
    L += [r'\midrule', 'Spread (max$-$min) & ' + ' & '.join(spread) + r' \\', r'\bottomrule', r'\end{tabular}}', r'\end{table}']
    open(path, 'w').write('\n'.join(L) + '\n')

def fig_psnr(P, path):
    systems = [s for s in mr.SYS if s in P.system.unique()]
    fig, axes = plt.subplots(1, len(systems), figsize=(7.1, 2.3), sharey=True, gridspec_kw=dict(wspace=0.06))
    rho = {}
    for ax, s in zip(np.atleast_1d(axes), systems):
        d = P[(P.system == s) & (P.codec != 'raw')]
        for c in mr.ORDER:
            dd = d[d.codec == c]
            if dd.empty: continue
            ok = dd[dd.rel_eff < analysis.R_FAIL]; bad = dd[dd.rel_eff >= analysis.R_FAIL]
            ax.scatter(ok.psnr, ok.rel_eff.clip(upper=14), s=5, color=mr.COL[c], alpha=0.7, lw=0, label=mr.LAB[c])
            ax.scatter(bad.psnr, np.full(len(bad), 20.0), s=10, marker='x', lw=0.6, color=mr.COL[c])
        rho[s] = spearmanr(d.psnr, d.rel_eff)[0]
        ax.set_yscale('log'); ax.set_title(mr.SYSSHORT[s] + rf', $\rho={rho[s]:.2f}$', loc='left'); ax.set_xlabel('Luma PSNR (dB)')
        ax.axhline(1, color='0.45', lw=0.7, ls=':'); ax.margins(0.04)
    a0 = np.atleast_1d(axes)[0]
    a0.set_ylabel('Relative ATE'); a0.set_ylim(0.4, 26)
    a0.set_yticks([0.5, 1, 2, 5, 10, 20]); a0.set_yticklabels(['0.5', '1', '2', '5', '10', 'fail']); a0.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    h, l = a0.get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.2), fontsize=6, markerscale=1.8, columnspacing=1.0)
    fig.savefig(path, bbox_inches='tight'); return fig, rho


def table_bd_ci(S, ci, path):
    """BD-rate table: median over sequences with hierarchical-bootstrap 95% CI (see bootstrap_ci.py)."""
    cols = [('orbslam3', 'TUM VI'), ('orbslam3', 'EuRoC'), ('orbslam3', 'KITTI'), ('basalt', 'TUM VI'), ('basalt', 'EuRoC'), ('basalt', 'KITTI'), ('dpvo', 'TUM VI')]
    cols = [c for c in cols if ((S.system == c[0]) & (S.dataset == c[1])).any()]
    nseq = {c: S[(S.system == c[0]) & (S.dataset == c[1])].seq.nunique() for c in cols}
    f = lambda x: '0' if round(x) == 0 else f'{x:+.0f}'.replace('-', '$-$').replace('+', '$+$')
    L = [r'\begin{table*}[t]',
         r'\caption{$\mathrm{BD\text{-}rate}_{\mathrm{loc}}$ relative to x264 (\%): average bitrate difference at equal localization error over relative errors 1.05--3. Negative values mean fewer bits than x264 for the same trajectory error. Median over sequences; brackets: 95\% hierarchical-bootstrap confidence interval of the median (sequences and repeated runs resampled, \BootN{} resamples, Section~\ref{sec:bd}); $n$ = number of sequences. VVC-RA: random-access GOP; VVC-LD: low-delay GOP without reordering. ``n.d.'': configuration not evaluated for this estimator. The ORB-SLAM3 EuRoC values depend on the failure criterion (Section~\\ref{sec:sensitivity}).}',
         r'\label{tab:bd}', r'\centering\footnotesize\setlength{\tabcolsep}{3.5pt}', r'\resizebox{\textwidth}{!}{%',
         r'\begin{tabular}{@{}l' + 'c' * len(cols) + '@{}}', r'\toprule',
         ' & ' + ' & '.join(r'%s, %s ($n{=}%d$)' % (mr.SYSSHORT[s], d.replace('TUM VI', 'TUM~VI'), nseq[(s, d)]) for s, d in cols) + r' \\', r'\midrule']
    for c in [c for c in mr.ORDER if c in S.codec.unique()]:
        cells = [mr.LAB[c]]
        for s, d in cols:
            v = S[(S.system == s) & (S.dataset == d) & (S.codec == c)].bd_vs_x264.dropna()
            if c == 'x264': cells.append('0 (anchor)'); continue
            if v.empty: cells.append('n.d.'); continue
            q = ci[(ci.system == s) & (ci.dataset == d) & (ci.codec == c)]
            if q.empty or not np.isfinite(q.bd_lo.iloc[0]): cells.append(f(v.median())); continue
            cells.append(f'{f(v.median())} [{f(q.bd_lo.iloc[0])}, {f(q.bd_hi.iloc[0])}]')
        L.append(' & '.join(cells) + r' \\')
    L += [r'\bottomrule', r'\end{tabular}}', r'\end{table*}']
    open(path, 'w').write('\n'.join(L) + '\n')
