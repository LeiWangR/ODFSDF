import numpy as np
import os
import cv2

try:
    import urllib2 as urllib
except ImportError:
    import urllib.request as urllib

import h5py
import json
import imageio
from skimage import feature
import pandas as pd
import argparse

# usage:

# python hog_feat_ext_whole_frame.py -saliency_hdf5 '/home/wan305/research/ongoing/lei/10feature_aggr/sample_info/mpii_sample_saliency_attention_jpeg.hdf5' -info_list '/home/wan305/research/ongoing/lei/10feature_aggr/mpii_sample_info_list.csv' -hdf5_feat 'sample_mpii_frame_saliency.hdf5'

#----------------------------------------------------------------
# Researcher: Lei Wang
# This codes are used to extract the inception resnet v2 feature (pre-trained on imagenet)
# for the whole video frame images (no bb applied)
# The inputs are: hdf5 file contains saliency frame images, and the information list (info list, which provides the video names)

# The outputs is one single hdf5 file that is structured in the following way:
# rgb/videoname/count (frame counts for a given video)
# rgb/videoname/frame00000*** (always starts from index 0 and in total 8 digits)
# 	dictionary format:
# 		- 'sal_frame_feature': store the extracted features in the dimension of (1, 144) per frame

'''
import h5py
import json
import numpy as np

# give a sample video name
videoname = 's20-d07-cam-002_05491_052'
# load saved hdf5 file
filename = 'sample_mpii_frame_saliency.hdf5'
with h5py.File(filename, 'r') as f:
    # using [()] for value
    count = f['rgb/' + videoname + '/count'][()]
    print('count: ', count)
    for ii in range(count):
        dic_info = json.loads(f['rgb/' + videoname + '/frame' + str(ii).zfill(8)][()])
        sal_frame_feature = np.asarray(dic_info['sal_frame_feature'])
        print('feat: ', sal_frame_feature, ' | shape: ', sal_frame_feature.shape)

'''

#----------------------------------------------------------------

parser = argparse.ArgumentParser(description="whole frame feature extraction using HoG")
parser.add_argument("-saliency_hdf5","--saliency_hdf5",type=str,default='/home/wan305/research/ongoing/lei/10feature_aggr/sample_info/mpii_sample_saliency_attention_jpeg.hdf5')
parser.add_argument("-info_list","--info_list",type=str,default='/home/wan305/research/ongoing/lei/10feature_aggr/mpii_sample_info_list.csv')
parser.add_argument("-hdf5_feat","--hdf5_feat",type=str,default='sample_mpii_frame_saliency.hdf5')

args = parser.parse_args()

# saliency maps hdf5 file
saliency_hdf5 = args.saliency_hdf5
# sample csv info list
info_list = args.info_list
# for saving the extracted features
hdf5_feat = args.hdf5_feat

df_info = pd.read_csv(info_list, header = None)
print(df_info.shape)

with h5py.File(saliency_hdf5, 'r') as f_data:
    with h5py.File(hdf5_feat, 'w') as fo:
        for idx in range(df_info.shape[0]):
            # given a video name
            videoname = df_info.iloc[idx, 0]
            print('--------- | processing ', videoname)
            count = f_data['rgb/' + videoname + '/count'][()]
            # print(count)
            f_nameonly = 'rgb/' + videoname + '/count'
            fo.create_dataset(f_nameonly, data = count)
            for ii in range(count):
                infoDic = {}
                # load the rgb frame image
                frame = f_data['rgb/' + videoname + '/frame' + str(ii).zfill(8) + '/saliency'][:]
                imgnameonly = 'rgb/' + videoname + '/frame' + str(ii).zfill(8)
                # print(imgnameonly)
                im = cv2.imdecode(frame, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(im, (32, 32))
                H = feature.hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(4, 4), transform_sqrt=True, block_norm="L1")
                infoDic['sal_frame_feature'] = H.astype(np.float16).tolist()
                
                fo.create_dataset(imgnameonly, data = json.dumps(infoDic))
                print('-------- frame: ', ii, ' | HoG feat. dim.: ', H.shape, ' -------------- ')


