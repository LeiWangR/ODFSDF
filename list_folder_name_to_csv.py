import h5py
import cv2
import os
import argparse
import csv

# Note that this code is used only when you have many frame images of each video

# example usage

# python list_folder_name_to_csv.py -csv_name 'sample_output.csv' -rgb_dir 'sample_mpii/rgb'

# Note that in 'sample_mpii/rgb' should have sub-folders, 
# each sub-folder is a class / action

# ***************SAMPLE VIDEO FRAME IMAGES************************************
# -sample_mpii
# 	-rgb
# 		-video1 (inside it would be many frame images)
# 		-video2 (inside it would be many frame images)
# 		-...
# 	-u
# 		-video1 (inside it would be many frame images)
# 		-video2 (inside it would be many frame images)
# 		-...
# 	-v
# 		-video1 (inside it would be many frame images)
# 		-video2 (inside it would be many frame images)
# 		-...
# ***********************************************************************

#-----------------------------------#

parser = argparse.ArgumentParser(description="list all folder names to a csv file")
parser.add_argument("-csv_name","--csv_name",type=str,default='sample_output.csv')
parser.add_argument("-rgb_dir","--rgb_dir",type=str,default='sample_mpii/rgb')

args = parser.parse_args()

csv_name = args.csv_name
rgb_path = args.rgb_dir

for r, d, f in os.walk(rgb_path, followlinks = False):
    # print('root: ', r)
    # print('dir: ', d)
    # print('files: ', f)
    if len(d) != 0:
        csv_file = open(csv_name,'w')
        wr = csv.writer(csv_file)
        for row in d:
            wr.writerow([row])
            
print('Done!')





        
