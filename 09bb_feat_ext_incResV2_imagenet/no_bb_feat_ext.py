import numpy as np
import os
import tensorflow as tf
import cv2

try:
    import urllib2 as urllib
except ImportError:
    import urllib.request as urllib

from datasets import imagenet
from nets import inception
from nets import inception_resnet_v2
from preprocessing import inception_preprocessing

from tensorflow.contrib import slim
import h5py
import json
import imageio
import pandas as pd
import argparse

# usage:

# python no_bb_feat_ext.py -dataset_hdf5 'mpii_sample.hdf5' -info_list 'mpii_sample_list.csv' -hdf5_feat 'sample_mpii_frame_no_bb_feat.hdf5'

#----------------------------------------------------------------
# Researcher: Lei Wang
# This codes are used to extract the inception resnet v2 feature (pre-trained on imagenet)
# for the whole video frame images (no bb applied)
# The inputs are: hdf5 file contains rgb frame images, and the information list (info list, which provides the video names)

# The outputs is one single hdf5 file that is structured in the following way:
# rgb/videoname/count (frame counts for a given video)
# rgb/videoname/frame00000*** (always starts from index 0 and in total 8 digits)
# 	dictionary format:
# 		- 'no_bb_feature': store the extracted features in the dimension of (1, 1001) per frame

'''
import h5py
import json
import numpy as np

# give a sample video name
videoname = 's20-d07-cam-002_05491_052'
# load saved hdf5 file
filename = 'sample_mpii_frame_no_bb_feat.hdf5'
with h5py.File(filename, 'r') as f:
    # using [()] for value
    count = f['rgb/' + videoname + '/count'][()]
    print('count: ', count)
    for ii in range(count):
        dic_info = json.loads(f['rgb/' + videoname + '/frame' + str(ii).zfill(8)][()])
        no_bb_features = np.asarray(dic_info['no_bb_feature'])
        print('feat: ', no_bb_features, ' | shape: ', no_bb_features.shape)

'''

#----------------------------------------------------------------

parser = argparse.ArgumentParser(description="whole frame feature extraction using inception resnet v2")
parser.add_argument("-dataset_hdf5","--dataset_hdf5",type=str,default='mpii_sample.hdf5')
parser.add_argument("-info_list","--info_list",type=str,default='mpii_sample_list.csv')
parser.add_argument("-hdf5_feat","--hdf5_feat",type=str,default='sample_mpii_frame_no_bb_feat.hdf5')

args = parser.parse_args()

# dataset hdf5
dataset_hdf5 = args.dataset_hdf5
# sample csv info list
info_list = args.info_list
# for saving the extracted features
hdf5_feat = args.hdf5_feat

df_info = pd.read_csv(info_list, header = None)
print(df_info.shape)

image_size = inception.inception_resnet_v2.default_image_size
print(image_size, ' --- default image size')
with tf.Graph().as_default():
    image = tf.placeholder(tf.uint8, (None, None, 3))
    processed_image = inception_preprocessing.preprocess_image(image, image_size, image_size, is_training=False)
    processed_images  = tf.expand_dims(processed_image, 0)

    # Create the model, use the default arg scope to configure the batch norm parameters.
    with slim.arg_scope(inception.inception_resnet_v2_arg_scope()):
        logits, _ = inception.inception_resnet_v2(processed_images, num_classes=1001, is_training=False)
    probabilities = tf.nn.softmax(logits)

    init_fn = slim.assign_from_checkpoint_fn('inception_resnet_v2_2016_08_30.ckpt', slim.get_model_variables('InceptionResnetV2'))

    with tf.Session() as sess:
        init_fn(sess)
        # load the hdf5 file
        with h5py.File(dataset_hdf5, 'r') as f_data:
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
                        frame = f_data['rgb/' + videoname + '/frame' + str(ii).zfill(8)][:]
                        imgnameonly = 'rgb/' + videoname + '/frame' + str(ii).zfill(8)
                        # print(imgnameonly)
                        im = cv2.imdecode(frame, cv2.IMREAD_COLOR)
                        # print('im: ', im.shape)
                        _, probability = sess.run([image, probabilities], feed_dict={image:im})
                        infoDic['no_bb_feature'] = probability.astype(np.float16).tolist()
                        fo.create_dataset(imgnameonly, data = json.dumps(infoDic))
                        print('-------- frame: ', ii, ' | pred. per frame: ', probability.shape, ' -------------- ')
                        
                        # print(np.sum(probability, axis = 1)) # sum up to 1, and the number of bbs

