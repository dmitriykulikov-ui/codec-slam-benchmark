set +e -u
JOB=$PWD; R=/home/dmitriy/work/codec-slam; cp run_bench.py $R/run_bench_p6.py; cd $R
while pgrep -f "vvc_ld --systems orbslam3,basalt" >/dev/null; do sleep 30; done; echo "start $(date)"
RESDIR=results_kitti venv/bin/python run_bench_p6.py --seqs K05,K06,K07,K10 --codecs raw,x264,x265,svtav1,vvc,vvc_ld,nvh264,nvhevc,mjpeg --systems orbslam3,basalt --workers 7 > logs/kitti.log 2>&1
echo "done $(date)"; grep -c " ok" logs/kitti.log; grep ERR logs/kitti.log | head -5
mkdir -p $JOB/out && cp results_kitti/*.csv logs/kitti.log $JOB/out/
