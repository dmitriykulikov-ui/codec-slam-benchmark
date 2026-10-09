set +e -u
JOB=$PWD; R=${CODECSLAM_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}; cp run_bench.py $R/; mkdir -p $R/calib; cp tumvi_cam0.txt euroc_cam0.txt $R/calib/; cd $R
mkdir -p results_main
if [ ! -f results_main/codec.csv ]; then   # reuse pilot points that coincide with the main grid
  cp results_pilot/pilot_codec.csv results_main/codec.csv; cp results_pilot/pilot_runs.csv results_main/runs.csv; cp -r results_pilot/traj results_main/
fi
RESDIR=results_main venv/bin/python run_bench.py --systems orbslam3,basalt --workers 8 > logs/main_cpu.log 2>&1
tail -3 logs/main_cpu.log; grep -c " ok" logs/main_cpu.log; grep ERR logs/main_cpu.log | head
mkdir -p $JOB/out && cp results_main/*.csv logs/main_cpu.log $JOB/out/
df -h / | tail -1
