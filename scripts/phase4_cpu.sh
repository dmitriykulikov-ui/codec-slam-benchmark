set +e -u
JOB=$PWD; R=${CODECSLAM_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}; cd $R
cp $JOB/euroc_extract.py data/euroc/; ( cd data/euroc && python3 euroc_extract.py MH_01_easy MH_03_medium ); ls data/euroc
cp $JOB/run_bench.py $R/run_bench_p4.py
RESDIR=results_euroc venv/bin/python run_bench_p4.py --seqs MH_01_easy,MH_03_medium --systems orbslam3,basalt --workers 7 > logs/euroc_p4.log 2>&1
echo "p4 done $(date)"; grep -c " ok" logs/euroc_p4.log; grep ERR logs/euroc_p4.log | head -5
mkdir -p $JOB/out && cp results_euroc/*.csv $JOB/out/
