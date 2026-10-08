set +e -u
JOB=$PWD; R=/home/dmitriy/work/codec-slam; cp run_bench.py $R/run_bench_dpvo.py; cd $R
systemctl --user stop qwen-llama-server.service
export QSUB='{"x264":[30,36,40,44],"x265":[30,36,42,46],"svtav1":[40,48,55,60],"vvc":[32,37,42,47]}'
RESDIR=results_dpvo venv/bin/python run_bench_dpvo.py --seqs room1,room2,room3,room4 --codecs raw,x264,x265,svtav1,vvc --systems dpvo --workers 2 > logs/dpvo_pass.log 2>&1
tail -3 logs/dpvo_pass.log; grep -c " ok" logs/dpvo_pass.log; grep ERR logs/dpvo_pass.log | head
mkdir -p $JOB/out && cp results_dpvo/*.csv logs/dpvo_pass.log $JOB/out/
systemctl --user start qwen-llama-server.service; sleep 5; systemctl --user is-active qwen-llama-server.service
