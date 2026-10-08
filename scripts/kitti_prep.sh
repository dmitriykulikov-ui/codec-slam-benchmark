set -e
R=/home/dmitriy/work/codec-slam; cp kitti_prep.py $R/; cd $R
python3 -c "import struct;d=open('data/kitti_raw/dataset/sequences/05/image_0/000000.png','rb').read(24);print(struct.unpack('>II',d[16:24]))"
venv/bin/python kitti_prep.py 05,06,07,10 skipdl
du -sh data/tumvi_anchor8/K* ; df -h / | tail -1
