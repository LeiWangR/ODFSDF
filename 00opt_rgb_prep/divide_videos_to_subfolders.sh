#! /bin/bash
VIDEODIR=/flush2/wan305/nturgb+d_rgb
i = 0;
for f in `ls -S ${VIDEODIR}/*.avi | tac`
do
d=folderid_$(printf %03d $((i/3792+1)));
mkdir -p $d;
mv "$f" $d;
i=$((i+1))
done
