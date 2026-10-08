#!/usr/bin/env python3
"""Selective download of KITTI odometry sequences from the official zip via HTTP range requests,
then build 8-bit anchors (sequences 04-12 are 1226x370) in the benchmark layout."""
import sys, os, io, struct, zlib, zipfile, urllib.request, time, json, shutil, subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
URL = 'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_gray.zip'
POSES = 'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_poses.zip'
ROOT = Path('/home/dmitriy/work/codec-slam'); RAW = ROOT / 'data/kitti_raw'; ANCHOR = ROOT / 'data/tumvi_anchor8'
FF = ROOT / 'tools/ffmpeg-btbn/bin/ffmpeg'
SEQS = sys.argv[1].split(',') if len(sys.argv) > 1 else ['05', '06', '07', '10']

def get(url, a, b, tries=8):
    for t in range(tries):
        try:
            req = urllib.request.Request(url, headers={'Range': f'bytes={a}-{b}'})
            with urllib.request.urlopen(req, timeout=120) as r: d = r.read()
            if len(d) == b - a + 1: return d
        except Exception as e: err = e
        time.sleep(2 + 3 * t)
    raise RuntimeError(f'range {a}-{b} failed')

class HTTPFile(io.RawIOBase):
    def __init__(s, url):
        s.url = url; s.pos = 0; s.buf = b''; s.boff = -1
        with urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=60) as r: s.size = int(r.headers['Content-Length'])
    def seekable(s): return True
    def readable(s): return True
    def tell(s): return s.pos
    def seek(s, off, wh=0):
        s.pos = off if wh == 0 else (s.pos + off if wh == 1 else s.size + off); return s.pos
    def read(s, n=-1):
        if n < 0: n = s.size - s.pos
        n = min(n, s.size - s.pos)
        if n <= 0: return b''
        if not (s.boff <= s.pos and s.pos + n <= s.boff + len(s.buf)):
            s.boff = s.pos; s.buf = get(s.url, s.pos, min(s.size, s.pos + max(n, 8 << 20)) - 1)
        o = s.pos - s.boff; d = s.buf[o:o + n]; s.pos += len(d); return d
    def readinto(s, b):
        d = s.read(len(b)); b[:len(d)] = d; return len(d)

def fetch_span(members):
    a = members[0].header_offset; b = members[-1].header_offset + 30 + len(members[-1].filename.encode()) + 256 + members[-1].compress_size
    buf = get(URL, a, b - 1)
    for m in members:
        o = m.header_offset - a
        sig, = struct.unpack('<I', buf[o:o + 4]); assert sig == 0x04034b50
        nl, el = struct.unpack('<HH', buf[o + 26:o + 30]); s0 = o + 30 + nl + el
        data = buf[s0:s0 + m.compress_size]
        data = zlib.decompress(data, -15) if m.compress_type == 8 else data
        assert len(data) == m.file_size
        p = RAW / m.filename; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(data)
    return len(members)

def download():
    zf = zipfile.ZipFile(HTTPFile(URL))
    want = [m for m in zf.infolist() if not m.is_dir() and any(m.filename.startswith(f'dataset/sequences/{s}/') for s in SEQS)]
    want.sort(key=lambda m: m.header_offset)
    spans, cur, cur_b = [], [], 0
    for m in want:
        if cur and (m.header_offset - cur[0].header_offset > (48 << 20) or m.header_offset - cur_b > (1 << 20)):
            spans.append(cur); cur = []
        cur.append(m); cur_b = m.header_offset + m.compress_size + 512
    if cur: spans.append(cur)
    tot = sum(m.compress_size for m in want); print(len(want), 'members', round(tot / 1e9, 2), 'GB', len(spans), 'spans', flush=True)
    t0 = time.time(); done = 0
    with ThreadPoolExecutor(8) as ex:
        for i, n in enumerate(ex.map(fetch_span, spans)):
            done += n
            if i % 20 == 0: print('span', i, done, round(time.time() - t0), 's', flush=True)
    # poses
    d = urllib.request.urlopen(POSES, timeout=300).read()
    z = zipfile.ZipFile(io.BytesIO(d)); z.extractall(RAW)

def quat(Rm):
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(Rm).as_quat()  # x y z w

def anchor(s):
    seq = 'K' + s; src = RAW / 'dataset/sequences' / s; dst = ANCHOR / seq / 'mav0'
    if (dst / 'DONE').exists(): return
    shutil.rmtree(ANCHOR / seq, ignore_errors=True)
    times = np.loadtxt(src / 'times.txt'); n = len(times)
    for cam, img in (('cam0', 'image_0'), ('cam1', 'image_1')):
        dd = dst / cam / 'data'; dd.mkdir(parents=True, exist_ok=True)
        subprocess.run([str(FF), '-v', 'error', '-y', '-start_number', '0', '-i', str(src / img / '%06d.png'),
                        '-pix_fmt', 'gray', '-start_number', '0', str(dd / '%06d.png')], check=True)
        assert len(list(dd.glob('*.png'))) == n
        with open(dst / cam / 'data.csv', 'w') as f:
            f.write('#timestamp [ns],filename\n')
            for i, t in enumerate(times): f.write(f'{int(round(t * 1e9))},{i:06d}.png\n')
    shutil.copy(src / 'times.txt', ANCHOR / seq / 'times.txt'); shutil.copy(src / 'calib.txt', ANCHOR / seq / 'calib.txt')
    P = np.loadtxt(RAW / 'dataset/poses' / f'{s}.txt').reshape(-1, 3, 4); assert len(P) == n
    with open(ANCHOR / seq / 'gt_tum.txt', 'w') as f:
        for t, M in zip(times, P):
            q = quat(M[:, :3]); f.write(f'{t:.6f} {M[0,3]:.6f} {M[1,3]:.6f} {M[2,3]:.6f} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}\n')
    # Basalt pinhole stereo calibration (same as basalt_convert_kitti_calib.py, resolution 1240x376)
    L = open(src / 'calib.txt').read().split()
    P0 = np.array([float(x) for x in L[1:13]]).reshape(3, 4); P1 = np.array([float(x) for x in L[14:26]]).reshape(3, 4)
    tx = -P1[0, 3] / P1[0, 0]
    pin = lambda P: dict(camera_type='pinhole', intrinsics=dict(fx=P[0, 0], fy=P[1, 1], cx=P[0, 2], cy=P[1, 2]))
    I = dict(px=0.0, py=0.0, pz=0.0, qx=0.0, qy=0.0, qz=0.0, qw=1.0)
    cal = {'value0': {'T_imu_cam': [I, {**I, 'px': tx}], 'intrinsics': [pin(P0), pin(P1)], 'resolution': [[1226, 370], [1226, 370]],
           'vignette': [], 'calib_accel_bias': [0.0] * 9, 'calib_gyro_bias': [0.0] * 12, 'imu_update_rate': 0.0,
           'accel_noise_std': [0.0] * 3, 'gyro_noise_std': [0.0] * 3, 'accel_bias_std': [0.0] * 3, 'gyro_bias_std': [0.0] * 3, 'cam_time_offset_ns': 0}}
    json.dump(cal, open(ANCHOR / seq / 'basalt_calib.json', 'w'), indent=2)
    (dst / 'DONE').write_text('ok'); print('anchor', seq, n, flush=True)

if __name__ == '__main__':
    if 'skipdl' not in sys.argv: download()
    for s in SEQS: anchor(s)
    shutil.rmtree(RAW / 'dataset/sequences', ignore_errors=True)
    print('done', flush=True)
