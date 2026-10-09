set +e -u
JOB=$PWD; R=${CODECSLAM_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}; cd $R; mkdir -p $JOB/out
nvidia-smi --query-gpu=driver_version,memory.used --format=csv
cp $JOB/run_bench.py run_bench_p7.py
sed -i "s/REPS = {'orbslam3': 3, 'basalt': 1, 'dpvo': 1}/REPS = {'orbslam3': 3, 'basalt': 3, 'dpvo': 1}/" run_bench_p7.py
RESDIR=results_rerun_r2b QSUB='{"nvhevc":[40]}' venv/bin/python run_bench_p7.py --seqs K06,K07 --codecs nvhevc --systems basalt --reps 3 --workers 2 > logs/rerun_r2b.log 2>&1
tail -4 logs/rerun_r2b.log
cat results_rerun_r2b/runs.csv; cat results_rerun_r2b/codec.csv
cp results_rerun_r2b/runs.csv $JOB/out/rerun_nvhevc_runs.csv; cp results_rerun_r2b/codec.csv $JOB/out/rerun_nvhevc_codec.csv
for t in K06_nvhevc_q40 K07_nvhevc_q40; do for r in 0 1 2; do echo "--- $t $r"; tail -n 2 results_rerun_r2b/traj/$t/basalt_$r.log 2>/dev/null; done; done
