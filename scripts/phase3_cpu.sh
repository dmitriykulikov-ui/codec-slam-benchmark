set +e -u
JOB=$PWD; R=/home/dmitriy/work/codec-slam; cd $R
cp $JOB/euroc_extract.py data/euroc/; ( cd data/euroc && python3 euroc_extract.py MH_02_easy )
while pgrep -f "run_bench_p2.py" > /dev/null; do sleep 60; done
echo "phase2 finished $(date)"; cp $JOB/run_bench.py $R/run_bench_p3.py
RESDIR=results_euroc venv/bin/python run_bench_p3.py --seqs MH_02_easy --systems orbslam3,basalt --workers 8 > logs/euroc_p3.log 2>&1
echo "p3 done $(date)"; grep -c " ok" logs/euroc_p3.log; grep ERR logs/euroc_p3.log | head -5
mkdir -p $JOB/out && cp results_euroc/*.csv $JOB/out/
