# codec-slam-benchmark

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23278262.svg)](https://doi.org/10.5281/zenodo.23278262)

Rate–accuracy benchmark of video codecs for visual and visual–inertial localization (VO/VIO/SLAM).

This repository contains the code and the per-run results behind the paper

> D. Kulikov and A. Bigler, "How Many Bits Does Visual–Inertial Localization Need? A Rate–Accuracy Benchmark of Software, Hardware, and Intra-Only Video Codecs for SLAM," submitted to *IEEE Access*, 2026.

Stereo camera streams of public datasets (TUM VI room1–6, EuRoC Machine Hall MH_01–MH_05, KITTI odometry 05/06/07/10) are encoded with eight encoder configurations at six rate points each, decoded, and processed by ORB-SLAM3, Basalt and DPVO. Localization accuracy (ATE/RPE/coverage) is compared to the uncompressed 8-bit reference, and codecs are compared with two task-based measures:

* **R_τ** — the lowest stereo bitrate at which the trajectory error stays within a factor τ of the uncompressed result (τ = 1.25 and 2);
* **BD-rate_loc** — a Bjøntegaard-style average bitrate difference at equal localization error (relative ATE 1.05–3), with explicit handling of tracking failures;
* **BSQ-rate** (Zvezdakova et al., Proc. ISP RAS 32(1), 2020) on PSNR and on localization error, as an image-fidelity comparison and as a second averaging scheme (`analysis/bsq.py`).

Medians over sequences are censoring-aware: a sequence on which a tolerance is not reached within the tested range enters the median as +∞, and the number k/n of sequences on which a measure is defined is reported.

## Encoder configurations

| Label | Encoder | Settings | Rate parameter |
|---|---|---|---|
| H.264 | x264 (FFmpeg) | medium, no B-frames, GOP 2 s | CRF 23–47 |
| HEVC | x265 (FFmpeg) | medium, no B-frames, GOP 2 s | CRF 23–50 |
| AV1 | SVT-AV1 (FFmpeg) | preset 8, low delay (pred-struct=1), GOP 2 s | CRF 30–63 |
| VVC-RA | VVenC via FFmpeg libvvenc | preset faster, random access (GOP 32), intra 2 s, 10-bit input | QP 27–51 |
| VVC-LD | VVenC 1.15.0-dev (`vvencFFapp`) | `lowdelay_faster.cfg`, GOP 8, no reordering, intra 40 (16 on KITTI) | QP 27–51 |
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

### Paper assets from the released tables (minutes, no GPU)

```
pip install -r requirements.txt
python analysis/reproduce.py --n-boot 1000 --seed 1 --out paper
```

This regenerates `paper/generated_numbers.tex`, all `paper/table_*.tex`, `paper/figures/*.pdf` and `paper/bsq_per_sequence.csv`, and checks the per-sequence R_τ and BD-rate_loc against `results/summary_per_sequence.csv` (the released values are reproduced to floating-point precision). The bootstrap uses seed 1 and 1,000 resamples (about 5 min on one core). The figures use matplotlib defaults unless a figure-style function `apply_figure_style()` is defined; numbers and tables do not depend on it.

### Encoding and estimator runs

1. Build the images, e.g. `docker build -t codecslam/orbslam3 -f docker/Dockerfile.orbslam3 docker` (similarly `orbslam3k`, `basalt`, `dpvo`, `dpvo2`, `vvenc`). ORB-SLAM3 (`4452a3c`), DPVO (`0ac95b6`) and VVenC (`de2443e`) are pinned by commit. Basalt is pinned to prebuilt release 0.1.7 via the official install script (the latest release when the paper image was built on 2026-10-07). For DPVO pass `--build-arg TORCH_CUDA_ARCH_LIST=<your GPU arch>` (the paper used 12.0). The Dockerfiles use a Yandex Ubuntu mirror; replace it with your local mirror if needed.
2. Obtain the datasets from their authors (TUM VI `dataset-room*_512_16`, EuRoC `machine_hall.zip`, KITTI odometry gray images and poses; KITTI is licensed CC BY-NC-SA 3.0, register at the KITTI website before downloading).
3. Set `CODECSLAM_ROOT` (working directory with `data/`, `tools/`, results), `CODECSLAM_FFMPEG` (FFmpeg with libx264, libx265, libsvtav1, libvvenc; the paper used the BtbN static build `N-127233-g452820cba6-20261007`) and `CODECSLAM_FFMPEG_NVENC` (FFmpeg 8.1 with NVENC).
4. Run the grid, e.g. `RESDIR=results_main python bench/run_bench.py --seqs room1,room2 --systems orbslam3,basalt --workers 6`; `scripts/` contains the scripts used for each phase.

### Modifications to upstream code

* ORB-SLAM3: C++14 instead of C++11, OpenCV 4.2 accepted instead of 4.4 (Ubuntu 20.04), viewer disabled in all examples; the KITTI stereo example saves the timestamped trajectory of the largest map (`Dockerfile.orbslam3k`).
* DPVO: `.type()` replaced by `.scalar_type()` for current PyTorch; Eigen 3.4.0 added; a runner `docker/dpvo_run.py` for EuRoC/TUM-style image folders.

Software versions: ORB-SLAM3 commit 4452a3c, Basalt prebuilt release 0.1.7, DPVO commit 0ac95b6, VVenC commit de2443e (1.15.0-dev), FFmpeg BtbN master N-127233 (software encoders) and FFmpeg 8.1 (NVENC), evo for trajectory evaluation.

## Citation

Code and data: D. Kulikov and A. Bigler, *codec-slam-benchmark*, Zenodo, 2026, doi:10.5281/zenodo.23278262 (all versions; v1.1.1: doi:10.5281/zenodo.23278263).

## License

Code: MIT (see `LICENSE`). The datasets are subject to the licenses of their authors and are not redistributed here; the result tables contain only derived measurements.
