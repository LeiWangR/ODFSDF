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

# python bb_feat_ext.py -dataset_hdf5 'mpii_sample.hdf5' -object_det_hdf5 'object_detect_vf_ava_resnet101_feat_sample.hdf5' -info_list 'mpii_sample_list.csv' -hdf5_bb_feat 'sample_mpii_per_bb_feat.hdf5'

#----------------------------------------------------------------
# Researcher: Lei Wang
# This codes are used to extract the inception resnet v2 feature (pre-trained on imagenet)
# for per bounding box (obtained from object detection model)
# The inputs are: hdf5 file contains rgb frame images, hdf5 file contains the extracted bounding boxes (bb) and the information list (info list, which provides the video names)

# The outputs is one single hdf5 file that is structured in the following way:
# rgb/videoname/count (frame counts for a given video)
# rgb/videoname/frame00000*** (always starts from index 0 and in total 8 digits)
# 	dictionary format:
# 		- 'num_detections': store the total number of detections, can be 0
# 		- 'bb_feature': store the extracted features in the dimension of (num_detections, 1001)
# 			Note that is num_detections = 0, then we do not store the corresponding 'bb_feature' as its empty.

'''
import h5py
import json
import numpy as np

# give a sample video name
videoname = 's20-d07-cam-002_05491_052'
# load saved hdf5 file
filename = 'sample_mpii_per_bb_feat.hdf5'
with h5py.File(filename, 'r') as f:
    # using [()] for value
    count = f['rgb/' + videoname + '/count'][()]
    print('count: ', count)
    for ii in range(count):
        dic_info = json.loads(f['rgb/' + videoname + '/frame' + str(ii).zfill(8)][()])
        num = dic_info['num_detections']
        print('num: ', num)
        if num == 0:
            print('no bb feature stored!')
        else:       
            bb_features = np.asarray(dic_info['bb_feature'])
            print('feat: ', bb_features, ' | shape: ', bb_features.shape)

'''

#----------------------------------------------------------------

parser = argparse.ArgumentParser(description="per bb feature extraction using inception resnet v2")
parser.add_argument("-dataset_hdf5","--dataset_hdf5",type=str,default='mpii_sample.hdf5')
parser.add_argument("-object_det_hdf5","--object_det_hdf5",type=str,default='object_detect_vf_ava_resnet101_feat_sample.hdf5')
parser.add_argument("-info_list","--info_list",type=str,default='mpii_sample_list.csv')
parser.add_argument("-hdf5_bb_feat","--hdf5_bb_feat",type=str,default='sample_mpii_per_bb_feat.hdf5')

args = parser.parse_args()

# dataset hdf5
dataset_hdf5 = args.dataset_hdf5
# object det. info. hdf5
object_det_hdf5 = args.object_det_hdf5
# sample csv info list
info_list = args.info_list
# for saving the extracted features
hdf5_bb_feat = args.hdf5_bb_feat

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
            with h5py.File(object_det_hdf5, 'r') as f_det:
                with h5py.File(hdf5_bb_feat, 'w') as fo:
                    for idx in range(df_info.shape[0]):
                        # given a video name
                        videoname = df_info.iloc[idx, 0]
                        print('--------- | processing ', videoname)
                        count = f_det['rgb/' + videoname + '/count'][()]
                        height = f_det['rgb/' + videoname + '/height'][()]
                        width = f_det['rgb/' + videoname + '/width'][()]
                        # print(count, height, width)
                        f_nameonly = 'rgb/' + videoname + '/count'
                        fo.create_dataset(f_nameonly, data = count)
                        for ii in range(count):
                            infoDic = {}
                            # load the rgb frame image
                            frame = f_data['rgb/' + videoname + '/frame' + str(ii).zfill(8)][:]
                            im = cv2.imdecode(frame, cv2.IMREAD_COLOR)
                            # print('im: ', im.shape)
                            # load the detection information
                            dic_info = json.loads(f_det['rgb/' + videoname + '/frame' + str(ii).zfill(8)][()])
                            num = dic_info['num_detections']
                            boxes = np.asarray(dic_info['detection_boxes'])
                            imgnameonly = 'rgb/' + videoname + '/frame' + str(ii).zfill(8)
                            # print(boxes.shape, num)
                            if num == 0:
                                print('0!!!!!!')
                                infoDic['num_detections'] = num
                                fo.create_dataset(imgnameonly, data = json.dumps(infoDic))
                                continue;
                            else:
                                # num != 0
                                # define an array to store the prediction score of per bb mini image
                                pred_scores = np.zeros((num, 1001))
                                for jj in range(num):
                                    box = boxes[jj, :]
                                    # print('one bb: ', box.shape)
                                    scale_box = [int(box[0] * height), int(box[1] * width), int(box[2] * height), int(box[3] * width)]
                                    # print('scaled: ', scale_box)
                                    mini_bb_im = im[scale_box[0]:scale_box[2], scale_box[1]:scale_box[3], :]
                                    # print('mini bb image size: ', mini_bb_im.shape)

                                    # for checking whether the bb is applied correctly (saved to test folder)
                                    # imageio.imwrite('test/' + str(ii) + '-' + str(jj) + '.jpg', mini_bb_im)

                                    _, probability = sess.run([image, probabilities], feed_dict={image:mini_bb_im})
                                    pred_scores[jj, :] = probability
                                    # print('-------- ', jj, ' | prob.: ', probability.shape)
                                    # print(probability)
                                    '''
                                    # The rest codes are only for class prediction output
                                    prob = probability[0, 0:]
                                    sorted_inds = [i[0] for i in sorted(enumerate(-prob), key=lambda x:x[1])]

                                    names = imagenet.create_readable_names_for_imagenet_labels()
                                    for i in range(5):
                                        index = sorted_inds[i]
                                        print('Probability %0.2f%% => [%s]' % (prob[index] * 100, names[index]))
                                    '''
                                # print(pred_scores.astype(np.float16))
                                infoDic['num_detections'] = num
                                infoDic['bb_feature'] = pred_scores.astype(np.float16).tolist()
                                fo.create_dataset(imgnameonly, data = json.dumps(infoDic))
                                print('-------- frame: ', ii, ' | pred. per frame: ', pred_scores.shape, ' -------------- ')
                                # print(pred_scores)
                                # print(np.sum(pred_scores, axis = 1), np.sum(pred_scores, axis = 1).shape) # sum up to 1, and the number of bbs

