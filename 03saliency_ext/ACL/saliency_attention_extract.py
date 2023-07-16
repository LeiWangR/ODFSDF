from __future__ import division
from keras.layers import Input
from keras.models import Model
import os
import numpy as np
from config import *
from utilities import preprocess_images, postprocess_predictions
from models import acl_vgg
from scipy.misc import imread, imsave
from math import ceil
import json
import argparse
import h5py

# Researcher: Lei Wang
# 2 October 2019
# Codes modified from the original ACL codes
# we just use the pre-trained saliency model for saliency and attention tensors extraction
# and save them into one single hdf5 file

# python saliency_attention_extract.py -rgb_dir '/home/wan305/research/ongoing/lei/00opt_rgb_prep/sample_mpii/rgb/' -hdf5_dir 'saliency_attention_sample.hdf5' 

# ***************SAMPLE VIDEO FRAME IMAGES************************************
# -sample_mpii
#       -rgb
#               -video1 (inside it would be many frame images)
#               -video2 (inside it would be many frame images)
# Output: single hdf5 file
# To load the saliency and attention tensors, using the following codes:

'''
import h5py
import json
import numpy as np

# give a sample video name
videoname = 's20-d07-cam-002_05496_038'
# load saved hdf5 file
filename = 'saliency_attention_sample.hdf5'
with h5py.File(filename, 'r') as f:
    dic_info = json.loads(f['rgb/' + videoname][()])
    saliency = np.asarray(dic_info['saliency'])
    attention = np.asarray(dic_info['attention'])
    print('saliency: ', saliency.shape, ' | attention: ', attention.shape)
    print('------------')
    print(saliency)
    print('------------')
    print(attention)
'''

parser = argparse.ArgumentParser(description="video saliency and attention tensors extraction")
parser.add_argument("-rgb_dir","--rgb_dir",type=str,default='/home/wan305/research/ongoing/lei/00opt_rgb_prep/sample_mpii/rgb')
parser.add_argument("-hdf5_dir","--hdf5_dir",type=str,default='saliency_attention_sample.hdf5')

args = parser.parse_args()
hdf5_name = args.hdf5_dir
rgb_path = args.rgb_dir

# remove frames_path
frames_path = '/'

def get_test(video_test_path):
    images = [video_test_path + frames_path + f for f in os.listdir(video_test_path + frames_path) if
              f.endswith(('.jpg', '.jpeg', '.png'))]
    images.sort()
    start = 0
    while True:
        Xims = np.zeros((1, num_frames, shape_r, shape_c, 3))
        X = preprocess_images(images[start:min(start + num_frames, len(images))], shape_r, shape_c)
        Xims[0, 0:min(len(images)-start, num_frames), :] = np.copy(X)
        yield Xims  #
        start = min(start + num_frames, len(images))

phase = 'test'
if phase == 'train':
    x = Input(batch_shape=(None, None, shape_r, shape_c, 3))
    stateful = False
else:
    x = Input(batch_shape=(1, None, shape_r, shape_c, 3))
    stateful = True

if phase == "test":
    videos_test_path = rgb_path
    videos = [videos_test_path + f for f in os.listdir(videos_test_path) if os.path.isdir(videos_test_path + f)]
    videos.sort()
    nb_videos_test = len(videos)

    m = Model(inputs=x, outputs=acl_vgg(x, stateful))
    print("Loading ACL weights")

    m.load_weights('ACL.h5')

    with h5py.File(hdf5_name, 'w') as fo:

        for i in range(nb_videos_test):
            infoDic = {}
            images_names = [f for f in os.listdir(videos[i] + frames_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            images_names.sort()
            if images_names == []:
                print('empty!')
                continue;

            # print("Predicting saliency maps for " + videos[i])
            prediction = m.predict_generator(get_test(video_test_path=videos[i]), max(ceil(len(images_names)/num_frames),2))
            predictions = np.squeeze(prediction[0])
            # print(predictions.shape, ' --- saliency info')
            # print(predictions)
            attentions = np.squeeze(prediction[3])
            # print(attentions.shape, ' xxx attention info')
            # print(attentions)
            infoDic['saliency'] = predictions.tolist()
            infoDic['attention'] = attentions.tolist()

            new_video_name = videos[i].replace(rgb_path, 'rgb/')
            print('No.: ', i, " | saliency maps for " + new_video_name)
            
            fo.create_dataset(new_video_name, data = json.dumps(infoDic))
            m.reset_states()
        
else:
    raise NotImplementedError
