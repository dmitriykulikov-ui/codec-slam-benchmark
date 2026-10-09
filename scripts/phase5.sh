set +e -u
JOB=$PWD; R=${CODECSLAM_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}; cp run_bench.py $R/run_bench_p5.py; cd $R
while pgrep -f run_bench_p4.py >/dev/null; do sleep 30; done; echo "start $(date)"
ALL=room1,room2,room3,room4,room5,room6,MH_01_easy,MH_02_easy,MH_03_medium,MH_04_difficult,MH_05_difficult
RESDIR=results_vvcld venv/bin/python run_bench_p5.py --seqs $ALL --codecs vvc_ld --systems orbslam3,basalt --workers 7 > logs/vvcld_cpu.log 2>&1
echo "cpu done $(date)"; grep -c " ok" logs/vvcld_cpu.log; grep ERR logs/vvcld_cpu.log | head -5
mkdir -p $JOB/out && cp results_vvcld/codec.csv $JOB/out/codec_cpu.csv && cp results_vvcld/runs.csv $JOB/out/runs_cpu.csv
export QSUB='{"vvc_ld":[32,37,42,47]}'
RESDIR=results_dpvo_ld venv/bin/python run_bench_p5.py --seqs room1,room2,room3,room4 --codecs vvc_ld --systems dpvo --workers 2 >> logs/vvcld_dpvo.log 2>&1
echo "dpvo done $(date)"; grep -c " ok" logs/vvcld_dpvo.log; grep ERR logs/vvcld_dpvo.log | head -5
cp results_dpvo_ld/codec.csv $JOB/out/codec_dpvo.csv; cp results_dpvo_ld/runs.csv $JOB/out/runs_dpvo.csv
