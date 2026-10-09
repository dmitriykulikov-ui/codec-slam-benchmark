set +e -u
JOB=$PWD; R=${CODECSLAM_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}; cd $R; mkdir -p $JOB/out
FF=tools/ffmpeg-btbn/bin/ffmpeg; FP=tools/ffmpeg-btbn/bin/ffprobe; A=data/tumvi_anchor8
echo "== original Basalt logs"
for t in K06_nvhevc_q40 K07_x264_q40 K07_nvhevc_q40; do echo "--- $t"; ls results_kitti/traj/$t/ 2>/dev/null; tail -n 15 results_kitti/traj/$t/basalt_0.log 2>/dev/null; done
grep -E "^K0[67],(nvhevc|x264),40,basalt" results_kitti/runs.csv
echo "== Basalt reruns (3 reps each, re-encoded)"
sed "s/REPS = {'orbslam3': 3, 'basalt': 1, 'dpvo': 1}/REPS = {'orbslam3': 3, 'basalt': 3, 'dpvo': 1}/" $JOB/run_bench.py > run_bench_p7.py
grep -n "^REPS" run_bench_p7.py
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
RESDIR=results_rerun_r2 QSUB='{"nvhevc":[40],"x264":[40]}' venv/bin/python run_bench_p7.py --seqs K06,K07 --codecs nvhevc,x264 --systems basalt --reps 3 --workers 4 > logs/rerun_r2.log 2>&1
cat logs/rerun_r2.log | tail -8; cp results_rerun_r2/runs.csv $JOB/out/rerun_runs.csv; cp results_rerun_r2/codec.csv $JOB/out/rerun_codec.csv
for t in K06_nvhevc_q40 K07_nvhevc_q40 K07_x264_q40 K06_x264_q40; do echo "--- rerun $t"; for r in 0 1 2; do tail -n 3 results_rerun_r2/traj/$t/basalt_$r.log 2>/dev/null; done; done
echo "== x264 scene-cut audit (cam0, all sequences, grid CRFs)"
python3 - <<'PY' > $JOB/out/x264_scenecut.csv
import subprocess, glob, os, re, sys
FF='tools/ffmpeg-btbn/bin/ffmpeg'; FP='tools/ffmpeg-btbn/bin/ffprobe'; A='data/tumvi_anchor8'
seqs=['room1','room2','room3','room4','room5','room6','MH_01_easy','MH_02_easy','MH_03_medium','MH_04_difficult','MH_05_difficult','K05','K06','K07','K10']
print('seq,crf,mode,frames,keyframes,bytes,key_bytes,psnr')
W='tmp_scenecut'; os.makedirs(W, exist_ok=True)
for s in seqs:
    d=f'{A}/{s}/mav0/cam0/data'
    if not os.path.isdir(d): print(f'{s},NA,missing,,,,,', flush=True); continue
    fps=10 if s.startswith('K') else 20; g=2*fps
    for crf in [23,30,36,40,44,47]:
        for mode,extra in [('default',[]),('nosc',['-sc_threshold','0','-keyint_min',str(g)])]:
            out=f'{W}/{s}_{crf}_{mode}.mkv'
            subprocess.run([FF,'-v','error','-y','-framerate',str(fps),'-pattern_type','glob','-i',d+'/*.png','-c:v','libx264','-preset','medium','-bf','0','-g',str(g),*extra,'-crf',str(crf),'-pix_fmt','yuv420p',out],check=False)
            p=subprocess.run([FP,'-v','error','-select_streams','v:0','-show_entries','packet=flags,size','-of','csv=p=0',out],capture_output=True,text=True).stdout.split()
            n=k=b=kb=0
            for line in p:
                sz,fl=line.split(',')[:2]; n+=1; b+=int(sz)
                if 'K' in fl: k+=1; kb+=int(sz)
            r=subprocess.run([FF,'-v','info','-i',out,'-framerate',str(fps),'-pattern_type','glob','-i',d+'/*.png','-lavfi','[0:v]format=gray[a];[1:v]format=gray[b];[a][b]psnr','-f','null','-'],capture_output=True,text=True).stderr
            m=re.search(r'average:([\d.inf]+)',r)
            print(f'{s},{crf},{mode},{n},{k},{b},{kb},{m.group(1) if m else ""}', flush=True)
            os.remove(out)
PY
rm -rf tmp_scenecut; wc -l $JOB/out/x264_scenecut.csv; echo done
