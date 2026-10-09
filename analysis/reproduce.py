"""Regenerate every figure, table and numeric macro of the paper from the released CSV files.
Usage (from the repository root):  python analysis/reproduce.py [--n-boot 1000] [--seed 1] [--out paper]
Writes <out>/generated_numbers.tex, <out>/table_*.tex and <out>/figures/*.pdf, then checks the per-sequence summary
against results/summary_per_sequence.csv."""
import argparse, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import matplotlib; matplotlib.use('Agg')

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--n-boot', type=int, default=1000); ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--out', default='paper'); a = ap.parse_args()
    runs = pd.read_csv(os.path.join(ROOT, 'results/runs_orbslam3_basalt.csv'))
    runs_dpvo = pd.read_csv(os.path.join(ROOT, 'results/runs_dpvo.csv'))
    enc = pd.read_csv(os.path.join(ROOT, 'results/encodings.csv'))
    os.makedirs(os.path.join(a.out, 'figures'), exist_ok=True)
    g = {'__name__': 'codecslam_reproduce'}
    for f in ('build_assets.py', 'final_pipeline.py'):
        src = open(os.path.join(HERE, f)).read().replace("'paper/", "'" + a.out.rstrip('/') + "/")
        exec(compile(src, f, 'exec'), g)
    out = g['build_all'](runs, runs_dpvo, enc, n_boot=a.n_boot, seed=a.seed)
    ref = pd.read_csv(os.path.join(ROOT, 'results/summary_per_sequence.csv'))
    S = out['S']; key = ['seq', 'system', 'codec']
    m = ref.merge(S, on=key, suffixes=('_ref', ''))
    for col in ('R@1.25', 'R@2.0', 'bd_vs_x264'):
        if col + '_ref' in m:
            d = np.nanmax(np.abs(m[col + '_ref'] - m[col])) if len(m) else np.nan
            print(f'{col}: max abs difference to released summary = {d:.3g}')
    out['BQ'].to_csv(os.path.join(a.out, 'bsq_per_sequence.csv'), index=False)
    print('done:', len(S), 'per-sequence rows')

if __name__ == '__main__':
    main()
