"""Run DPVO (monocular, cam0) on a EuRoC-layout sequence; write TUM trajectory.
Usage (inside codecslam/dpvo): python dpvo_run.py <mav0> <out.txt> [stride]
Undistorts with the dataset's own cam0/sensor.yaml (equidistant fisheye or radtan).
"""
import sys, re
from pathlib import Path
import numpy as np, cv2, torch
from dpvo.config import cfg
from dpvo.dpvo import DPVO

mav0, out = Path(sys.argv[1]), sys.argv[2]
stride = int(sys.argv[3]) if len(sys.argv) > 3 else 1

def read_yaml_list(txt, key):
    m = re.search(key + r'\s*:\s*\[([^\]]*)\]', txt)
    return [float(x) for x in m.group(1).replace('\n', ' ').split(',')]

if len(sys.argv) > 4:   # simple calib file: "<model> fu fv cu cv d0 d1 d2 d3"
    tok = Path(sys.argv[4]).read_text().split()
    model, (fu, fv, cu, cv), D = tok[0], map(float, tok[1:5]), np.array([float(x) for x in tok[5:9]])
else:
    y = (mav0 / 'cam0/sensor.yaml').read_text()
    fu, fv, cu, cv = read_yaml_list(y, 'intrinsics')
    D = np.array(read_yaml_list(y, 'distortion_coefficients'))
    model = re.search(r'distortion_model\s*:\s*(\w+)', y).group(1)
K = np.array([[fu, 0, cu], [0, fv, cv], [0, 0, 1]])
files = sorted((mav0 / 'cam0/data').glob('*.png'))[::stride]
img0 = cv2.imread(str(files[0]), cv2.IMREAD_GRAYSCALE)
h, w = img0.shape
if model.startswith('equi'):
    Knew = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (w, h), np.eye(3), balance=0.0)
    m1, m2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), Knew, (w, h), cv2.CV_16SC2)
else:
    Knew, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 0)
    m1, m2 = cv2.initUndistortRectifyMap(K, D, None, Knew, (w, h), cv2.CV_16SC2)
H, W = h - h % 16, w - w % 16
intr = torch.tensor([Knew[0, 0], Knew[1, 1], Knew[0, 2], Knew[1, 2]], dtype=torch.float32).cuda()

cfg.merge_from_file('/opt/DPVO/config/default.yaml')
slam = None
with torch.no_grad():
    for i, f in enumerate(files):
        im = cv2.remap(cv2.imread(str(f), cv2.IMREAD_GRAYSCALE), m1, m2, cv2.INTER_LINEAR)[:H, :W]
        im = torch.from_numpy(np.repeat(im[..., None], 3, axis=2)).permute(2, 0, 1).cuda()
        if slam is None:
            slam = DPVO(cfg, '/opt/DPVO/dpvo.pth', ht=H, wd=W, viz=False)
        slam(i, im, intr)
    res = slam.terminate()
poses, tstamps = res[0], res[1]
ts = np.array([int(files[int(t)].stem) * 1e-9 for t in tstamps])
with open(out, 'w') as fo:
    for t, p in zip(ts, np.asarray(poses)):
        fo.write(f'{t:.9f} ' + ' '.join(f'{v:.7f}' for v in p[:7]) + '\n')
print('frames', len(files), 'poses', len(ts))
