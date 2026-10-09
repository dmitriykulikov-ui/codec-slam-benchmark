# Result tables

All rates are stereo totals (both cameras) in kbit/s; errors in meters.

`runs_orbslam3_basalt.csv`, `runs_dpvo.csv` — one row per localization run:
seq, codec, q (rate parameter: CRF, QP or JPEG q; 0 = uncompressed), system, rep (repetition), wall_s (wall-clock time of the run, s),
kbps_total (stereo bitrate, kbit/s), psnr_y / ssim_y (luma PSNR in dB / SSIM of the decoded images, mean over frames and cameras),
ate_m (ATE RMSE, m; empty if no trajectory), rpe1m_m (RPE over 1 m segments, m), coverage (absolute fraction of the ground-truth duration spanned by the estimate), n_poses.
`runs_orbslam3_basalt.csv` has 3,174 rows: 3,042 on the paper grid plus 132 pilot runs at rate parameters outside the grid
(x264 CRF 42/51, x265 47/51, SVT-AV1 38/45/52/58), which `analysis/build_assets.py:on_grid` removes. With the 84 DPVO runs the paper counts 3,126 runs.

Derived per run (analysis.py): rel = ate_m / median ate_m of the same system on the uncompressed reference of the same sequence;
cov_rel = coverage / median reference coverage; a run fails if no trajectory exists or cov_rel < 0.8.

`encodings.csv` — one row per encoded stereo sequence: kbps_total, bpp, psnr_y, ssim_y, frames, enc_s_cam0/enc_s_cam1 (encoding wall time per camera, s; shared workstation).

`summary_per_sequence.csv` — per sequence, system and configuration: R@1.25 and R@2.0 (kbit/s; empty = not reached), cens_R@* (left-censored at the lowest tested rate), fail_kbps, bd_vs_x264 (BD-rate_loc, %).

`bootstrap_ci.csv` — 95% percentile intervals of the medians from the two-level bootstrap (seed 1, 1,000 resamples).

`bsq_per_sequence.csv` — BSQ-rate vs x264 (%) per sequence: bsq_loc (on relative ATE, per system) and bsq_psnr (on luma PSNR).

`reruns_basalt_kitti.csv` — three Basalt reruns each (revision, round 2) of KITTI 06/07 at x264 CRF 40 and HEVC-HW QP 40 on re-encoded streams (identical bitrate and PSNR). The original K07/x264 and K06/HEVC-HW runs had terminated with numerical failures that did not reproduce; the median repetition replaces each of them in `runs_orbslam3_basalt.csv`. The K07/HEVC-HW divergence (ATE 24.4 m) reproduced in all three reruns.

`x264_scenecut_audit.csv` — left camera of every sequence re-encoded with x264 at the six grid CRFs with default scene-cut detection and with `-sc_threshold 0`: frames, keyframes, bytes, bytes in keyframes, luma PSNR.
