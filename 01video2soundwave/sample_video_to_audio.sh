#!/bin/sh

# follow the params settings in NIPS2016 paper
# -i is the path to the input file
# -f specifies the output format is mp3
# -ac is the audio channel (single channel)
# -ar is the sample rate (22 kHz)
# vn means no video

DATASET="moments_test/*"
AUDIOSET="moments_test_audio/"
for f in $DATASET
do
	actionfolder=$(basename -- $f)
	newfolder="$AUDIOSET$actionfolder"
	mkdir $newfolder
	if [ -d "$f" ]
	then
		for ff in $f/*
		do
			videoname=$(basename -- $ff)
			# echo $videoname
			videonameonly=${videoname%.*}	
			# echo $videonameonly
			newfilename="$newfolder/$videonameonly.wav"
			# newfilename="$newfolder/$videonameonly.mp3"
			# echo $newfilename
			echo "processing $newfilename"
			ffmpeg -i $ff -ac 1 -ar 44100 -vn $newfilename
		done
		
	fi
done




