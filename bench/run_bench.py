#!/usr/bin/env python3
"""Pilot: rate–localization curves for VIO on TUM-VI under lossy video coding.

For each (sequence, codec, rate point): encode both cameras of the 8-bit anchor,
decode back to PNGs with the original timestamp names, measure bitrate / PSNR,
run each SLAM system N times in Docker, evaluate ATE vs mocap GT, delete frames.
Results are appended to results/runs.csv (one row per SLAM run) and
results/codec.csv (one row per encoding).
"""
import argparse, csv, json, os, shutil, subprocess, sys, time, glob, re, uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np

ROOT = Path('/home/dmitriy/work/codec-slam')
FF = ROOT / 'tools/ffmpeg-btbn/bin/ffmpeg'
FF_NV = ROOT / 'tools/ffmpeg-n81/bin/ffmpeg'   # NVENC needs a release build for driver 580
CALIB = ROOT / 'calib'
DATA = ROOT / 'data/tumvi'
ANCHOR = ROOT / 'data/tumvi_anchor8'
TMP = ROOT / ('tmp_' + os.environ.get('RESDIR', 'results'))
RES = ROOT / os.environ.get('RESDIR', 'results')
FPS = 20
GOP = 40  # 2 s at 20 Hz, low-delay (no B-frames)
_cur = __import__('threading').local()
def _G(): return getattr(_cur, 'gop', GOP)
def _F(): return getattr(_cur, 'fps', FPS)
def fps_of(seq): return 10 if seq.startswith('K') else 20
def gop_of(seq): return 2 * fps_of(seq)          # intra period 2 s

CODECS = {
    'x264':   dict(rates=[23, 30, 36, 40, 44, 47], ext='mkv',
                   args=lambda q: ['-c:v', 'libx264', '-preset', 'medium', '-bf', '0', '-g', str(_G()),
                                   '-crf', str(q), '-pix_fmt', 'yuv420p']),
    'x265':   dict(rates=[23, 30, 36, 42, 46, 50], ext='mkv',
                   args=lambda q: ['-c:v', 'libx265', '-preset', 'medium', '-pix_fmt', 'yuv420p',
                                   '-x265-params', f'bframes=0:keyint={_G()}:min-keyint={_G()}:crf={q}:log-level=error']),
    'svtav1': dict(rates=[30, 40, 48, 55, 60, 63], ext='mkv',
                   args=lambda q: ['-c:v', 'libsvtav1', '-preset', '8', '-g', str(_G()), '-crf', str(q),
                                   '-svtav1-params', 'pred-struct=1', '-pix_fmt', 'yuv420p']),
    'vvc':    dict(rates=[27, 32, 37, 42, 47, 51], ext='mkv',
                   args=lambda q: ['-c:v', 'libvvenc', '-preset', 'faster', '-qp', str(q), '-period', '2',
                                   '-pix_fmt', 'yuv420p10le']),
    # standalone VVenC, low-delay (GOP 8, no reordering), intra period 2 s; encoded via vvencFFapp in Docker
    'vvc_ld': dict(rates=[27, 32, 37, 42, 47, 51], ext='266', ext_enc='vvencld'),
    'nvh264': dict(rates=[25, 30, 35, 40, 45, 50], ext='mkv', ff='nv',
                   args=lambda q: ['-c:v', 'h264_nvenc', '-preset', 'p5', '-tune', 'll', '-rc', 'constqp', '-qp', str(q),
                                   '-bf', '0', '-g', str(_G()), '-pix_fmt', 'yuv420p']),
    'nvhevc': dict(rates=[25, 30, 35, 40, 45, 50], ext='mkv', ff='nv',
                   args=lambda q: ['-c:v', 'hevc_nvenc', '-preset', 'p5', '-tune', 'll', '-rc', 'constqp', '-qp', str(q),
                                   '-bf', '0', '-g', str(_G()), '-pix_fmt', 'yuv420p']),
    'mjpeg':  dict(rates=[2, 5, 10, 15, 22, 31], ext='mkv',
                   args=lambda q: ['-c:v', 'mjpeg', '-q:v', str(q), '-pix_fmt', 'yuvj420p']),
}

def sh(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)

def docker_run(cmd, timeout):
    """docker run with a unique name; on timeout kill the container (killing the client alone leaves it running)."""
    name = 'cs_' + uuid.uuid4().hex[:10]
    cmd = cmd[:2] + ['--name', name, '--init'] + cmd[2:]
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=timeout); return False
    except subprocess.TimeoutExpired:
        subprocess.run(['docker', 'kill', name], capture_output=True); return True

def frames(d):
    return sorted(Path(d).glob('*.png'))

# ---------------------------------------------------------------- anchor
def ds(seq):
    if seq.startswith('K'): return 'kitti'
    return 'euroc' if seq.startswith(('MH_', 'V1_', 'V2_')) else 'tumvi'
HW = {'tumvi': (512, 512), 'euroc': (752, 480), 'kitti': (1226, 370)}

def make_anchor(seq):
    """Source PNG -> 8-bit gray PNG (same names), EuRoC layout with imu/GT/csv copied."""
    src = (DATA / f'dataset-{seq}_512_16/mav0') if ds(seq) == 'tumvi' else (ROOT / f'data/euroc/{seq}/mav0')
    dst = ANCHOR / seq / 'mav0'
    if (dst / 'DONE').exists():
        return dst
    assert ds(seq) != 'kitti', 'run kitti_prep.py first' 
    for cam in ('cam0', 'cam1'):
        (dst / cam / 'data').mkdir(parents=True, exist_ok=True)
        shutil.copy(src / cam / 'data.csv', dst / cam / 'data.csv')
        for s in ('sensor.yaml',):
            if (src / cam / s).exists(): shutil.copy(src / cam / s, dst / cam / s)
        names = [p.name for p in frames(src / cam / 'data')]
        # ffmpeg glob -> numbered 8-bit pngs, then rename to timestamps
        tmpd = dst / cam / 'tmp'; tmpd.mkdir(exist_ok=True)
        sh([str(FF), '-v', 'error', '-y', '-framerate', str(_F()), '-pattern_type', 'glob',
            '-i', str(src / cam / 'data' / '*.png'), '-pix_fmt', 'gray', str(tmpd / '%06d.png')])
        out = frames(tmpd); assert len(out) == len(names), (seq, cam, len(out), len(names))
        for o, n in zip(out, names): o.rename(dst / cam / 'data' / n)
        tmpd.rmdir()
    gtdir = 'mocap0' if ds(seq) == 'tumvi' else 'state_groundtruth_estimate0'
    for d in ('imu0', gtdir):
        shutil.copytree(src / d, dst / d, dirs_exist_ok=True)
    gt_to_tum(dst / gtdir / 'data.csv', ANCHOR / seq / 'gt_tum.txt')
    (dst / 'DONE').write_text('ok')
    return dst

def gt_to_tum(csvp, out):
    a = np.loadtxt(csvp, delimiter=',', comments='#')
    # mocap0 / state_groundtruth_estimate0: t[ns], px,py,pz, qw,qx,qy,qz, ...
    with open(out, 'w') as f:
        for r in a:
            f.write(f'{r[0]*1e-9:.9f} {r[1]:.6f} {r[2]:.6f} {r[3]:.6f} {r[5]:.7f} {r[6]:.7f} {r[7]:.7f} {r[4]:.7f}\n')

# ---------------------------------------------------------------- coding
def vvenc_ld(pngdir, bs, q, work, seq):
    """8-bit gray PNGs -> yuv420p raw -> vvencFFapp low-delay (lowdelay_faster.cfg, QP q, IntraPeriod=GOP)."""
    yuv = work / (bs.stem + '.yuv')
    sh([str(FF), '-v', 'error', '-y', '-framerate', str(_F()), '-pattern_type', 'glob', '-i', str(pngdir / '*.png'),
        '-pix_fmt', 'yuv420p', '-f', 'rawvideo', str(yuv)])
    w, h = HW[ds(seq)]
    try:
        sh(['docker', 'run', '--rm', '-u', f'{os.getuid()}:{os.getgid()}', '-w', '/w', '-v', f'{work}:/w', 'codecslam/vvenc',
            '/opt/vvenc/bin/release-static/vvencFFapp', '-c', '/opt/vvenc/cfg/experimental/lowdelay_faster.cfg',
            '-i', f'/w/{yuv.name}', '--SourceWidth', str(w), '--SourceHeight', str(h), '--FrameRate', str(_F()),
            '--InputBitDepth', '8', '--InputChromaFormat', '420', '--QP', str(q), '--IntraPeriod', str(_G() if _G() % 8 == 0 else 8 * round(_G() / 8)),
            '--Threads', '4', '-o', '', '-b', f'/w/{bs.name}'])
    finally:
        yuv.unlink(missing_ok=True)

def encode_decode(seq, codec, q, work):
    """Returns dict with bitrate & quality; writes decoded EuRoC tree into work/mav0."""
    anc = ANCHOR / seq / 'mav0'
    info = dict(seq=seq, codec=codec, q=q)
    tot_bytes, n = 0, 0
    psnr, ssim = [], []
    for cam in ('cam0', 'cam1'):
        names = [p.name for p in frames(anc / cam / 'data')]
        bs = work / f'{cam}.{CODECS[codec]["ext"]}'
        t0 = time.time()
        ffbin = FF_NV if CODECS[codec].get('ff') == 'nv' else FF
        if CODECS[codec].get('ext_enc') == 'vvencld':
            vvenc_ld(anc / cam / 'data', bs, q, work, seq)
        else:
            sh([str(ffbin), '-v', 'error', '-y', '-framerate', str(_F()), '-pattern_type', 'glob',
                '-i', str(anc / cam / 'data' / '*.png'), *CODECS[codec]['args'](q), str(bs)])
        info[f'enc_s_{cam}'] = round(time.time() - t0, 2)
        tot_bytes += bs.stat().st_size; n = len(names)
        dd = work / 'mav0' / cam / 'data'; dd.mkdir(parents=True, exist_ok=True)
        sh([str(FF), '-v', 'error', '-y', '-i', str(bs), '-fps_mode', 'passthrough', '-pix_fmt', 'gray', '-start_number', '0', str(dd / '%06d.png')])
        out = frames(dd); assert len(out) == len(names), (seq, codec, q, cam, len(out), len(names))
        for o, nm in zip(out, names): o.rename(dd / nm)
        shutil.copy(anc / cam / 'data.csv', work / 'mav0' / cam / 'data.csv')
        # quality vs anchor (luma)
        r = subprocess.run([str(FF), '-v', 'info', '-framerate', str(_F()), '-pattern_type', 'glob', '-i', str(dd / '*.png'), '-framerate', str(_F()), '-pattern_type', 'glob',
                            '-i', str(anc / cam / 'data' / '*.png'),
                            '-lavfi', '[0:v]format=gray[a];[1:v]format=gray[b];[a]split[a1][a2];[b]split[b1][b2];'
                                      '[a1][b1]psnr;[a2][b2]ssim', '-f', 'null', '-'],
                           capture_output=True, text=True)
        m = re.search(r'PSNR.*?average:([\d.inf]+)', r.stderr); psnr.append(float(m.group(1)) if m else np.nan)
        m = re.search(r'SSIM.*?All:([\d.]+)', r.stderr); ssim.append(float(m.group(1)) if m else np.nan)
    for d in ('imu0', 'mocap0', 'state_groundtruth_estimate0'):
        if (anc / d).exists(): os.symlink(anc / d, work / 'mav0' / d)
    dur = n / _F()
    hw = HW[ds(seq)][0] * HW[ds(seq)][1]
    info.update(kbps_total=round(tot_bytes * 8 / dur / 1000, 2), bpp=round(tot_bytes * 8 / (2 * n * hw), 5),
                psnr_y=round(float(np.mean(psnr)), 3), ssim_y=round(float(np.mean(ssim)), 5), frames=n)
    return info

def anchor_info(seq):
    anc = ANCHOR / seq / 'mav0'
    n = len(frames(anc / 'cam0/data')); hw = HW[ds(seq)][0] * HW[ds(seq)][1]
    return dict(seq=seq, codec='raw', q=0, kbps_total=round(2 * n * hw * 8 / (n / fps_of(seq)) / 1000, 1),
                bpp=8.0, psnr_y=np.inf, ssim_y=1.0, frames=n)

# ---------------------------------------------------------------- SLAM
def run_orbslam3(seq, mav0, out):
    out.mkdir(parents=True, exist_ok=True)
    E = '/opt/ORB_SLAM3/Examples/Stereo-Inertial'
    if ds(seq) == 'tumvi':
        run = (f'{E}/stereo_inertial_tum_vi /opt/ORB_SLAM3/Vocabulary/ORBvoc.txt {E}/TUM-VI.yaml /data/mav0/cam0/data '
               f'/data/mav0/cam1/data {E}/TUM_TimeStamps/dataset-{seq}_512.txt {E}/TUM_IMU/dataset-{seq}_512.txt traj')
    elif ds(seq) == 'kitti':
        a = ANCHOR / seq
        run = ('( ' + f'mkdir -p /seq && ln -s /data/mav0/cam0/data /seq/image_0 && ln -s /data/mav0/cam1/data /seq/image_1 && cp {a}/times.txt /seq/ && '
               f'/opt/ORB_SLAM3/Examples/Stereo/stereo_kitti /opt/ORB_SLAM3/Vocabulary/ORBvoc.txt /calib/KITTI04-12_1226.yaml /seq; '
               "[ -f f_traj_ns.txt ] && awk '{printf \"%.9f\", $1/1e9; for(i=2;i<=NF;i++) printf \" %s\", $i; print \"\"}' f_traj_ns.txt > f_traj.txt" + ' )')
    else:
        short = seq[:2] + seq[3:5]  # MH_01_easy -> MH01
        run = f'{E}/stereo_inertial_euroc /opt/ORB_SLAM3/Vocabulary/ORBvoc.txt {E}/EuRoC.yaml /data {E}/EuRoC_TimeStamps/{short}.txt traj'
    cmd = ['docker', 'run', '--rm', '--cpus=4', '-v', f'{mav0}:/data/mav0:ro', '-v', f'{ANCHOR}:{ANCHOR}:ro',
           '-v', f'{CALIB}:/calib:ro', '-v', f'{out}:/out', '-w', '/out', ('codecslam/orbslam3k' if ds(seq) == 'kitti' else 'codecslam/orbslam3'), 'bash', '-c', run + ' > log.txt 2>&1; chmod -R a+rw /out']
    t0 = time.time(); docker_run(cmd, 900 if ds(seq) != 'kitti' else 1500)
    est = out / 'f_traj.txt'
    return est if est.exists() else None, time.time() - t0

def run_basalt(seq, mav0, out):
    out.mkdir(parents=True, exist_ok=True)
    if ds(seq) == 'kitti':
        a = ANCHOR / seq
        cmd = ['docker', 'run', '--rm', '--cpus=4', '-v', f'{mav0}:/data/mav0:ro', '-v', f'{ANCHOR}:{ANCHOR}:ro',
               '-v', f'{out}:/out', '-w', '/out', 'codecslam/basalt', 'bash', '-c',
               f'mkdir -p /seq && ln -s /data/mav0/cam0/data /seq/image_0 && ln -s /data/mav0/cam1/data /seq/image_1 && cp {a}/times.txt /seq/ && '
               f'/root/.local/bin/basalt_vio --dataset-path /seq --dataset-type kitti --cam-calib {a}/basalt_calib.json '
               '--config-path /root/.local/etc/basalt/kitti_config.json --show-gui 0 --use-imu 0 --save-trajectory tum > log.txt 2>&1; chmod -R a+rw /out']
        t0 = time.time(); docker_run(cmd, 900)
        est = out / 'trajectory.txt'
        return est if est.exists() else None, time.time() - t0
    cmd = ['docker', 'run', '--rm', '--cpus=4', '-v', f'{mav0}:/data/mav0:ro', '-v', f'{ANCHOR}:{ANCHOR}:ro',
           '-v', f'{out}:/out', '-w', '/out', 'codecslam/basalt', 'bash', '-c',
           '/root/.local/bin/basalt_vio --dataset-path /data --dataset-type euroc '
           + ('--cam-calib /root/.local/etc/basalt/tumvi_512_eucm_calib.json --config-path /root/.local/etc/basalt/tumvi_512_config.json '
              if ds(seq) == 'tumvi' else '--cam-calib /root/.local/etc/basalt/euroc_ds_calib.json --config-path /root/.local/etc/basalt/euroc_config.json ') +
           '--show-gui 0 --use-imu 1 --save-trajectory tum > log.txt 2>&1; chmod -R a+rw /out']
    t0 = time.time(); docker_run(cmd, 600)
    est = out / 'trajectory.txt'
    return est if est.exists() else None, time.time() - t0

import threading
GPU_LOCK = threading.Lock()
def run_dpvo(seq, mav0, out):
    out.mkdir(parents=True, exist_ok=True)
    cal = 'tumvi_cam0.txt' if ds(seq) == 'tumvi' else 'euroc_cam0.txt'
    cmd = ['docker', 'run', '--rm', '--gpus', 'all', '--shm-size=4g', '-v', f'{mav0}:/data/mav0:ro', '-v', f'{ANCHOR}:{ANCHOR}:ro',
           '-v', f'{CALIB}:/calib:ro', '-v', f'{out}:/out', 'codecslam/dpvo2', 'bash', '-c',
           f'cd /opt/DPVO && python dpvo_run.py /data/mav0 /out/traj.txt 1 /calib/{cal} > /out/log.txt 2>&1; chmod -R a+rw /out']
    with GPU_LOCK:
        t0 = time.time(); docker_run(cmd, 1800)
    est = out / 'traj.txt'
    return est if est.exists() else None, time.time() - t0

SYSTEMS = {'orbslam3': run_orbslam3, 'basalt': run_basalt, 'dpvo': run_dpvo}
REPS = {'orbslam3': 3, 'basalt': 1, 'dpvo': 1}   # Basalt is deterministic (pilot: identical ATE over 3 reps)
SIM3 = {'dpvo'}                                   # monocular -> scale-aligned ATE

def load_est(p):
    a = np.loadtxt(p, comments='#')
    if a.ndim == 1: a = a[None]
    if a[0, 0] > 1e12: a[:, 0] *= 1e-9   # ns -> s
    return a[:, :8]

def ate(seq, est_path, sim3=False):
    from evo.core import sync, metrics
    from evo.core.trajectory import PoseTrajectory3D
    gt = np.loadtxt(ANCHOR / seq / 'gt_tum.txt'); est = load_est(est_path)
    def traj(a): return PoseTrajectory3D(a[:, 1:4], a[:, [7, 4, 5, 6]], a[:, 0])  # wxyz
    tg, te = traj(gt), traj(est)
    tg, te = sync.associate_trajectories(tg, te, max_diff=0.02)
    te.align(tg, correct_scale=sim3)
    ape = metrics.APE(metrics.PoseRelation.translation_part); ape.process_data((tg, te))
    rpe = metrics.RPE(metrics.PoseRelation.translation_part, delta=1.0, delta_unit=metrics.Unit.meters, all_pairs=False)
    try:
        rpe.process_data((tg, te)); rpe_rmse = rpe.get_statistic(metrics.StatisticsType.rmse)
    except Exception:
        rpe_rmse = np.nan
    cover = (te.timestamps[-1] - te.timestamps[0]) / (gt[-1, 0] - gt[0, 0])
    return ape.get_statistic(metrics.StatisticsType.rmse), rpe_rmse, cover, len(te.timestamps)

# ---------------------------------------------------------------- driver
def task(seq, codec, q, systems, reps, lock):
    tag = f'{seq}_{codec}_q{q}'
    _cur.fps, _cur.gop = fps_of(seq), gop_of(seq)
    work = TMP / tag
    if work.exists(): shutil.rmtree(work)
    work.mkdir(parents=True)
    try:
        if codec == 'raw':
            info, mav0 = anchor_info(seq), ANCHOR / seq / 'mav0'
        else:
            info, mav0 = encode_decode(seq, codec, q, work), work / 'mav0'
        rows = []
        for sysname in systems:
            for r in range(min(reps, REPS[sysname])):
                est, wall = SYSTEMS[sysname](seq, mav0, work / f'{sysname}_{r}')
                row = dict(seq=seq, codec=codec, q=q, system=sysname, rep=r, wall_s=round(wall, 1),
                           kbps_total=info['kbps_total'], psnr_y=info['psnr_y'], ssim_y=info['ssim_y'])
                try:
                    a, rp, cov, npose = ate(seq, est, sysname in SIM3) if est else (np.nan, np.nan, 0.0, 0)
                except Exception as e:
                    a, rp, cov, npose = np.nan, np.nan, 0.0, 0
                row.update(ate_m=round(float(a), 4), rpe1m_m=round(float(rp), 4), coverage=round(float(cov), 3), n_poses=npose)
                rows.append(row)
                keep = RES / 'traj' / tag; keep.mkdir(parents=True, exist_ok=True)  # trajectories + logs (small)
                if est: shutil.copy(est, keep / f'{sysname}_{r}.txt')
                lg = work / f'{sysname}_{r}' / 'log.txt'
                if lg.exists(): shutil.copy(lg, keep / f'{sysname}_{r}.log')
        with lock:
            append_csv(RES / 'codec.csv', [info]); append_csv(RES / 'runs.csv', rows)
        return tag, 'ok'
    except Exception as e:
        return tag, f'ERR {type(e).__name__}: {e}'
    finally:
        shutil.rmtree(work, ignore_errors=True)

CODEC_KEYS = ['seq', 'codec', 'q', 'kbps_total', 'bpp', 'psnr_y', 'ssim_y', 'frames', 'enc_s_cam0', 'enc_s_cam1']

def append_csv(p, rows):
    keys = CODEC_KEYS if 'frames' in rows[0] else list(rows[0].keys()); new = not p.exists()
    with open(p, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        if new: w.writeheader()
        w.writerows(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seqs', default='room1,room2,room3,room4,room5,room6')
    ap.add_argument('--codecs', default='raw,x264,x265,svtav1,vvc,nvh264,nvhevc,mjpeg')
    ap.add_argument('--systems', default='orbslam3,basalt')
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--rates', default='', help='override: comma list applied to all codecs (smoke test)')
    a = ap.parse_args()
    import threading; lock = threading.Lock()
    RES.mkdir(exist_ok=True); TMP.mkdir(exist_ok=True)
    seqs = a.seqs.split(',')
    for s in seqs: make_anchor(s); print('anchor', s, flush=True)
    done = set()
    if (RES / 'codec.csv').exists():
        for r in csv.DictReader(open(RES / 'codec.csv')): done.add((r['seq'], r['codec'], int(r['q'])))
    tasks = []
    for s in seqs:
        for c in a.codecs.split(','):
            sub = json.loads(os.environ.get('QSUB', '{}'))
            qs = [0] if c == 'raw' else (sub[c] if c in sub else ([int(x) for x in a.rates.split(',')] if a.rates else CODECS[c]['rates']))
            for q in qs:
                if (s, c, q) not in done: tasks.append((s, c, q))
    print(f'{len(tasks)} tasks', flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(task, s, c, q, a.systems.split(','), a.reps, lock) for s, c, q in tasks]
        for i, f in enumerate(as_completed(futs)):
            print(i + 1, *f.result(), flush=True)

if __name__ == '__main__':
    main()
