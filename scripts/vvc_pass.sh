set +e -u
JOB=$PWD; R=/home/dmitriy/work/codec-slam; cp run_bench.py $R/run_bench_vvc.py; cd $R
RESDIR=results_vvc venv/bin/python run_bench_vvc.py --codecs vvc --systems orbslam3,basalt --workers 3 > logs/vvc_pass.log 2>&1
tail -3 logs/vvc_pass.log; grep -c " ok" logs/vvc_pass.log; grep ERR logs/vvc_pass.log | head
mkdir -p $JOB/out && cp results_vvc/*.csv logs/vvc_pass.log $JOB/out/
