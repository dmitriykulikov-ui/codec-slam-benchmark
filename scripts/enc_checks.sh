set +e
R=${CODECSLAM_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}; cd $R; FF=tools/ffmpeg-btbn/bin/ffmpeg; FP=tools/ffmpeg-btbn/bin/ffprobe
W=$(mktemp -d -p $R); A=data/tumvi_anchor8/room1/mav0/cam0/data
echo "== range test"
$FF -v error -framerate 20 -pattern_type glob -i "$A/*.png" -frames:v 100 -pix_fmt yuv420p -c:v rawvideo -f nut $W/lim.nut
$FF -v error -i $W/lim.nut -pix_fmt gray -f rawvideo $W/lim_back.gray
$FF -v error -framerate 20 -pattern_type glob -i "$A/*.png" -frames:v 100 -pix_fmt gray -f rawvideo $W/ref.gray
$FF -v info -f rawvideo -pix_fmt gray -s 512x512 -i $W/lim_back.gray -f rawvideo -pix_fmt gray -s 512x512 -i $W/ref.gray -lavfi psnr -f null - 2>&1 | grep -o 'average:[0-9.inf]*'
$FF -v error -i $W/lim.nut -frames:v 1 -vf "format=gray,signalstats,metadata=print" -f null - 2>&1 | grep -E 'YMIN|YMAX' | head -2
$FF -v error -framerate 20 -pattern_type glob -i "$A/*.png" -frames:v 1 -vf "format=gray,signalstats,metadata=print" -f null - 2>&1 | grep -E 'YMIN|YMAX' | head -2
echo "== FFmpeg encoder versions"; $FF -hide_banner -h encoder=libsvtav1 2>&1 | head -1; strings $FF | grep -m1 -E 'x264 - core [0-9]+'; strings $FF | grep -m1 -iE 'x265 \(build|HEVC encoder version'; strings $FF | grep -m1 -E 'SVT-AV1 Encoder Lib v|SVT \[version\]|v[0-9]+\.[0-9]+\.[0-9]+-[0-9a-f]+'
echo "== GOP structure (all 2821 frames of room1 cam0)"
$FF -v error -framerate 20 -pattern_type glob -i "$A/*.png" -c:v libvvenc -preset faster -qp 37 -period 2 -pix_fmt yuv420p10le $W/vvc.mkv
$FF -v error -framerate 20 -pattern_type glob -i "$A/*.png" -c:v libx264 -preset medium -bf 0 -g 40 -crf 36 -pix_fmt yuv420p $W/x264.mkv
for f in vvc x264; do echo "$f keyframes:"; $FP -v error -select_streams v:0 -show_entries packet=flags,size -of csv=p=0 $W/$f.mkv | awk -F, '{n++; s+=$1+0; if($2 ~ /K/) k++} END{print "packets",n,"key",k,"sum_bytes",s}'; ls -l $W/$f.mkv | awk '{print "file_bytes",$5}'; done
$FP -v error -select_streams v:0 -show_entries packet=pts_time,flags -of csv=p=0 $W/vvc.mkv | awk -F, '$2 ~ /K/{print $1}' | head -8 | tr '\n' ' '; echo
$FP -v error -select_streams v:0 -show_entries packet=pts_time,flags -of csv=p=0 $W/x264.mkv | awk -F, '$2 ~ /K/{print $1}' | head -12 | tr '\n' ' '; echo
echo "== Basalt version"; docker run --rm codecslam/basalt bash -c 'cat /root/.local/etc/basalt/VERSION 2>/dev/null; ls /root/.local/share 2>/dev/null | head'
echo "== commits"; docker run --rm codecslam/orbslam3 cat /opt/ORB_SLAM3/COMMIT; docker run --rm codecslam/dpvo2 cat /opt/DPVO/COMMIT; docker run --rm codecslam/vvenc cat /opt/vvenc/COMMIT
rm -rf $W
