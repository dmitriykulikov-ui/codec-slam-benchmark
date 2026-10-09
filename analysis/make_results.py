"""Build figures, tables and generated_numbers.tex for the manuscript.
Inputs: RUNS (list of runs.csv paths), CODEC_CSV (codec.csv); outputs into paper/.
Run inside the python kernel after apply_figure_style() is available."""
import numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt
from scipy.stats import spearmanr
import analysis

ORDER = ['x264', 'x265', 'svtav1', 'vvc', 'vvc_ld', 'nvh264', 'nvhevc', 'mjpeg']
LAB = {'x264': 'H.264 (x264)', 'x265': 'HEVC (x265)', 'svtav1': 'AV1 (SVT-AV1)', 'vvc': 'VVC-RA (VVenC)', 'vvc_ld': 'VVC-LD (VVenC)',
       'nvh264': 'H.264-HW (NVENC)', 'nvhevc': 'HEVC-HW (NVENC)', 'mjpeg': 'Motion JPEG'}
COL = {'x264': '#4C72B0', 'x265': '#DD8452', 'svtav1': '#55A868', 'vvc': '#C44E52', 'vvc_ld': '#7A1F22',
       'nvh264': '#7FA7D9', 'nvhevc': '#F2B279', 'mjpeg': '#8172B3'}
LS = {'nvh264': '--', 'nvhevc': '--', 'vvc_ld': '-.'}
SYS = ['orbslam3', 'basalt', 'dpvo']
SYSLAB = {'orbslam3': 'ORB-SLAM3 (stereo-inertial, descriptors)', 'basalt': 'Basalt (stereo-inertial, KLT flow)',
          'dpvo': 'DPVO (monocular, learned)'}
SYSSHORT = {'orbslam3': 'ORB-SLAM3', 'basalt': 'Basalt', 'dpvo': 'DPVO'}

def mbps_fmt(v, _): return f'{v:g}'

def fig_rate_curves(P, path):
    systems = [s for s in SYS if s in P.system.unique()]
    fig, axes = plt.subplots(1, len(systems), figsize=(7.1, 2.55), sharey=True, gridspec_kw=dict(wspace=0.06))
    for ax, s in zip(np.atleast_1d(axes), systems):
        d = P[(P.system == s) & (P.codec != 'raw')]
        for c in ORDER:
            dd = d[d.codec == c]
            if dd.empty: continue
            agg = dd.groupby('q').agg(kbps=('kbps', 'median'), rel=('rel_eff', 'median'), fail=('fail', 'mean')).sort_values('kbps')
            ok = agg[agg.rel < analysis.R_FAIL]
            ax.plot(ok.kbps / 1000, ok.rel, LS.get(c, '-'), marker='o', ms=2.8, lw=1.1, color=COL[c], label=LAB[c], zorder=3)
            bad = agg[agg.rel >= analysis.R_FAIL]
            if len(bad): ax.scatter(bad.kbps / 1000, np.full(len(bad), 20.0), marker='x', s=18, lw=0.9, color=COL[c], zorder=4)
        ax.axhline(1, color='0.45', lw=0.7, ls=':'); ax.axhspan(1, 1.25, color='0.92', zorder=0, lw=0)
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_title(SYSLAB[s], loc='left')
        ax.set_xlabel('Stereo bitrate (Mbit/s)'); ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(mbps_fmt))
        ax.margins(x=0.04)
    a0 = np.atleast_1d(axes)[0]
    a0.set_ylabel('Relative ATE (median)'); a0.set_ylim(0.7, 26)
    a0.set_yticks([1, 1.25, 2, 5, 10, 20]); a0.set_yticklabels(['1', '1.25', '2', '5', '10', 'fail'])
    a0.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    a0.text(0.03, 0.04, 'lower = better', transform=a0.transAxes, fontsize=6, color='0.35')
    h, l = a0.get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=7, frameon=False, bbox_to_anchor=(0.5, -0.13), fontsize=6, handlelength=1.6, columnspacing=1.0)
    fig.savefig(path, bbox_inches='tight'); return fig

def fig_rtau(S, path, tau='R@1.25'):
    systems = [s for s in SYS if s in S.system.unique()]
    fig, axes = plt.subplots(1, len(systems), figsize=(7.1, 2.1), sharey=True, gridspec_kw=dict(wspace=0.06))
    codecs = [c for c in ORDER if c in S.codec.unique()]
    for ax, s in zip(np.atleast_1d(axes), systems):
        for i, c in enumerate(codecs):
            v = S[(S.system == s) & (S.codec == c)][tau].dropna() / 1000
            if v.empty:
                ax.text(0.05, i, 'n.d.', va='center', fontsize=5.5, color='0.5'); continue
            ax.scatter(v, np.full(len(v), i) + np.linspace(-0.12, 0.12, len(v)), s=8, color=COL[c], alpha=0.7, lw=0, zorder=3)
            ax.plot([v.median()] * 2, [i - 0.3, i + 0.3], color=COL[c], lw=1.8, zorder=4)
        ax.set_xscale('log'); ax.set_title(SYSSHORT[s], loc='left'); ax.set_xlabel(r'$R_{1.25}$ (Mbit/s)')
        ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(mbps_fmt)); ax.margins(x=0.08, y=0.06)
    a0 = np.atleast_1d(axes)[0]
    a0.set_yticks(range(len(codecs))); a0.set_yticklabels([LAB[c] for c in codecs]); a0.invert_yaxis()
    a0.text(0.97, 0.03, 'left = fewer bits', transform=a0.transAxes, ha='right', fontsize=6, color='0.35')
    fig.savefig(path, bbox_inches='tight'); return fig

def fig_psnr(P, path):
    systems = [s for s in SYS if s in P.system.unique()]
    fig, axes = plt.subplots(1, len(systems), figsize=(7.1, 2.35), sharey=True, gridspec_kw=dict(wspace=0.06))
    rho = {}
    for ax, s in zip(np.atleast_1d(axes), systems):
        d = P[(P.system == s) & (P.codec != 'raw')]
        for c in ORDER:
            dd = d[d.codec == c]
            if dd.empty: continue
            ok = dd[dd.rel_eff < analysis.R_FAIL]; bad = dd[dd.rel_eff >= analysis.R_FAIL]
            ax.scatter(ok.psnr, ok.rel_eff, s=6, color=COL[c], alpha=0.75, lw=0, label=LAB[c])
            ax.scatter(bad.psnr, np.full(len(bad), 20.0), s=12, marker='x', lw=0.7, color=COL[c])
        r_, p_ = spearmanr(d.psnr, d.rel_eff); rho[s] = r_
        ax.text(0.97, 0.95, rf'Spearman $\rho={r_:.2f}$', transform=ax.transAxes, ha='right', va='top', fontsize=6)
        ax.set_yscale('log'); ax.set_title(SYSSHORT[s], loc='left'); ax.set_xlabel('Luma PSNR (dB)')
        ax.axhline(1, color='0.45', lw=0.7, ls=':'); ax.margins(0.04)
    a0 = np.atleast_1d(axes)[0]
    a0.set_ylabel('Relative ATE'); a0.set_ylim(0.5, 26)
    a0.set_yticks([1, 2, 5, 10, 20]); a0.set_yticklabels(['1', '2', '5', '10', 'fail']); a0.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    h, l = a0.get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=7, frameon=False, bbox_to_anchor=(0.5, -0.14), fontsize=6, markerscale=1.6, columnspacing=1.0)
    fig.savefig(path, bbox_inches='tight'); return fig, rho

def matched_psnr_contrast(P, lo=36.0, hi=39.0):
    """Median relative error per codec among rate points with PSNR in [lo, hi)."""
    d = P[(P.codec != 'raw') & (P.psnr >= lo) & (P.psnr < hi)]
    return d.groupby(['system', 'codec']).rel_eff.median().unstack(0)

def fmt_rate(v):
    if not np.isfinite(v): return '--'
    return f'{v/1000:.2f}' if v < 1000 else f'{v/1000:.1f}'

def table_main(S, path):
    systems = [s for s in SYS if s in S.system.unique()]
    codecs = [c for c in ORDER if c in S.codec.unique()]
    lines = [r'\begin{table*}[t]', r'\caption{Localization-based codec comparison. $R_{1.25}$ and $R_{2}$: lowest stereo bitrate (Mbit/s) at which the median relative ATE stays within 1.25$\times$ and 2$\times$ of the uncompressed result; $\mathrm{BD\text{-}rate}_{\mathrm{loc}}$: average bitrate difference to x264 at equal localization error (negative = fewer bits). Values are medians over sequences; brackets give the range over sequences. ``--'': tolerance not reached in the tested range or curve overlap too small.}',
             r'\label{tab:main}', r'\centering\footnotesize\setlength{\tabcolsep}{3.2pt}',
             r'\begin{tabular}{@{}l' + 'rrr' * len(systems) + '@{}}', r'\toprule',
             ' & ' + ' & '.join(r'\multicolumn{3}{c}{%s}' % SYSSHORT[s] for s in systems) + r' \\',
             ''.join(r'\cmidrule(lr){%d-%d}' % (2 + 3 * i, 4 + 3 * i) for i in range(len(systems))),
             'Configuration & ' + ' & '.join([r'$R_{1.25}$ & $R_{2}$ & BD (\%)'] * len(systems)) + r' \\', r'\midrule']
    for c in codecs:
        cells = [LAB[c]]
        for s in systems:
            d = S[(S.system == s) & (S.codec == c)]
            if d.empty: cells += ['n.d.'] * 3; continue
            cells += [fmt_rate(d['R@1.25'].median()), fmt_rate(d['R@2.0'].median())]
            b = d.bd_vs_x264.dropna()
            cells.append('0 (anchor)' if c == 'x264' else (f'{b.median():+.0f} [{b.min():+.0f}, {b.max():+.0f}]' if len(b) else '--'))
        lines.append(' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table*}']
    open(path, 'w').write('\n'.join(lines) + '\n')
