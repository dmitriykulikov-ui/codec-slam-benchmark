# codec-slam-benchmark

Rate–accuracy benchmark of video codecs for visual and visual–inertial localization (VO/VIO/SLAM).

This repository contains the code and the per-run results behind the paper

> D. Kulikov and A. Bigler, "How Many Bits Does Visual–Inertial Localization Need? A Rate–Accuracy Benchmark of Software, Hardware, and Intra-Only Video Codecs for SLAM," submitted to *IEEE Access*, 2026.

Stereo camera streams of public datasets (TUM VI room1–6, EuRoC Machine Hall MH_01–MH_05, KITTI odometry 05/06/07/10) are encoded with eight encoder configurations at six rate points each, decoded, and processed by ORB-SLAM3, Basalt and DPVO. Localization accuracy (ATE/RPE/coverage) is compared to the uncompressed 8-bit reference, and codecs are compared with two task-based measures:

* **R_τ** — the lowest stereo bitrate at which the trajectory error stays within a factor τ of the uncompressed result (τ = 1.25 and 2);
* **BD-rate_loc** — a Bjøntegaard-style average bitrate difference at equal localization error (relative ATE 1.05–3), with explicit handling of tracking failures.

## Encoder configurations

| Label | Encoder | Settings | Rate parameter |
|---|---|---|---|
| H.264 | x264 (FFmpeg) | medium, no B-frames, GOP 2 s | CRF 23–47 |
| HEVC | x265 (FFmpeg) | medium, no B-frames, GOP 2 s | CRF 23–50 |
| AV1 | SVT-AV1 (FFmpeg) | preset 8, low delay (pred-struct=1), GOP 2 s | CRF 30–63 |
| VVC-RA | VVenC via FFmpeg libvvenc | preset faster, random access, intra 2 s | QP 27–51 |
| VVC-LD | VVenC 1.15.0-dev (`vvencFFapp`) | `lowdelay_faster.cfg`, GOP 8, no reordering, intra 40 (24 on KITTI) | QP 27–51 |
| H.264-HW | NVENC (FFmpeg 8.1) | p5, tune ll, constqp, no B, GOP 2 s | QP 25–50 |
| HEVC-HW | NVENC (FFmpeg 8.1) | p5, tune ll, constqp, no B, GOP 2 s | QP 25–50 |
| MJPEG | FFmpeg mjpeg | intra only | q 2–31 |

Exact command lines are in `bench/run_bench.py` (`CODECS` and `vvenc_ld`).

## Repository layout

```
bench/      run_bench.py      grid runner: anchor creation, encode/decode, PSNR/SSIM, estimator runs, ATE/RPE (evo)
            euroc_extract.py  extraction of EuRoC sequences from the Machine Hall archive
            kitti_prep.py     selective download of KITTI odometry sequences (HTTP range requests) and anchors
docker/     Dockerfiles for ORB-SLAM3 (+KITTI variant), Basalt, DPVO, VVenC; dpvo_run.py
calib/      DPVO pinhole calibrations, ORB-SLAM3 KITTI settings
scripts/    the shell scripts used to run each phase of the grid on the workstation
analysis/   analysis.py       loading, failure criterion, isotonic curves, R_tau, BD-rate_loc
            bootstrap_ci.py   hierarchical bootstrap confidence intervals
            make_results.py   labels, colors, formatting
            build_assets.py   figures, tables and LaTeX macros of the paper
            final_pipeline.py end-to-end generation of all paper assets
results/    runs_orbslam3_basalt.csv  one row per ORB-SLAM3/Basalt run (seq, codec, q, system, rep, ATE, RPE, coverage, bitrate, PSNR, SSIM)
            runs_dpvo.csv             one row per DPVO run (TUM VI room1-4 subset)
            encodings.csv             one row per encoded stereo sequence (bitrate, bpp, PSNR, SSIM, encoding time)
            summary_per_sequence.csv  R_1.25, R_2 and BD-rate_loc per sequence, estimator and configuration
            bootstrap_ci.csv          95% hierarchical-bootstrap intervals of the medians (1,000 resamples)
```

## Reproducing

1. Build the images: `docker build -t codecslam/orbslam3 -f docker/Dockerfile.orbslam3 docker` (similarly `basalt`, `dpvo`, `dpvo2`, `vvenc`, `orbslam3k`). The Dockerfiles use a Yandex Ubuntu mirror; replace it with your local mirror if needed.
2. Obtain the datasets from their authors (TUM VI `dataset-room*_512_16`, EuRoC `machine_hall.zip`, KITTI `data_odometry_gray.zip` + `data_odometry_poses.zip`) and adjust `ROOT`, `DATA` and `FF` (FFmpeg build with libx264, libx265, libsvtav1, libvvenc) in `bench/run_bench.py`.
3. Run the grid, e.g. `RESDIR=results_main python bench/run_bench.py --seqs room1,room2 --systems orbslam3,basalt --workers 6`.
4. Regenerate the figures and tables from the CSV files in `results/`: in Python, `exec(open('analysis/build_assets.py').read()); exec(open('analysis/final_pipeline.py').read()); build_all(runs, runs_dpvo, encodings, n_boot=1000)` (run from a directory containing `paper/figures/`).

Software versions: ORB-SLAM3 commit 4452a3c, Basalt official release binary, DPVO commit 0ac95b6, VVenC commit de2443e, FFmpeg master N-127233 (software encoders) and FFmpeg 8.1 (NVENC), evo for trajectory evaluation.

## License

Code: MIT (see `LICENSE`). The datasets are subject to the licenses of their authors and are not redistributed here; the result tables contain only derived measurements.
