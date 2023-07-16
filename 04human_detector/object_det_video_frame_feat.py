import numpy as np
import tensorflow as tf
import cv2
import time
from DetectorAPI import DetectorAPI
import h5py
import os
from skimage import io as ioo
import json
import argparse

# Researcher: Lei Wang
# 1 October 2019

# This code are used to generate the video descriptor from object detection pipeline
# Input:
#   - rgb video frame images dir (for videos, please refer to video processing code)
#   - hdf5 file name to store the per-frame information (#detection, detection_classes, confidence_scores, bounding_boxes (normalized), and region proposal average pooling features/300)
#   - model path: the pre-trained model for object detection
# ***************SAMPLE VIDEO FRAME IMAGES************************************
# -sample_mpii
#       -rgb
#               -video1 (inside it would be many frame images)
#               -video2 (inside it would be many frame images)
# Output: single hdf5 file

# you need to enter Lei's running environment in conda (myenv)

# python object_det_video_frame_feat.py -hdf5_dir 'object_detect_video_frame_feat_sample.hdf5' -rgb_dir '/home/wan305/research/ongoing/lei/00opt_rgb_prep/sample_mpii/rgb' -model_path '/home/wan305/research/ongoing/lei/04human_detector/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/frozen_inference_graph.pb'

# for video-level, we store rgb/class_folder(action_folder)/video_name(no extension)
#       -/count
#       -/height
#       -/width
# for per-frame level information, we store:
#       -/frame000000** (index from 0 for all videos / video frame images)
#           -information stored in dictionary format (converted to string while saved in hdf5):
#               'num_detections'
#               'detection_classes'
#               'detection_scores'
#               'detection_boxes'
#               'feature_avg'

# To load the dictionary information, you need to use the following codes:
'''
import h5py
import json
import numpy as np

# give a sample video name
videoname = 's20-d07-cam-002_05496_038'
# load saved hdf5 file
filename = 'object_detect_video_frame_feat_sample.hdf5'
with h5py.File(filename, 'r') as f:
    # using [()] for value
    count = f['rgb/' + videoname + '/count'][()]
    print('count: ', count)
    for ii in range(count):
        dic_info = json.loads(f['rgb/' + videoname + '/frame' + str(ii).zfill(8)][()])
        features = np.asarray(dic_info['feature_avg'])
        boxes = np.asarray(dic_info['detection_boxes'])
        classes = np.asarray(dic_info['detection_classes'])
        scores = np.asarray(dic_info['detection_scores'])
        num = dic_info['num_detections']
        print('num: ', num, ' | classes: ', classes, ' | scores: ', scores)
        print('boxes: ', boxes, ' | shape: ', boxes.shape)
        print('feat: ', features, ' | shape: ', features.shape)

'''

parser = argparse.ArgumentParser(description="object detector for action recognition")
parser.add_argument("-model_path","--model_path",type=str,default='/home/wan305/research/ongoing/lei/04human_detector/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/frozen_inference_graph.pb')
# model path options:
# '/home/wan305/research/ongoing/lei/04human_detector/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb'
# '/home/wan305/research/ongoing/lei/04human_detector/faster_rcnn_nas_coco_2018_01_28/frozen_inference_graph.pb'
parser.add_argument("-rgb_dir","--rgb_dir",type=str,default='/home/wan305/research/ongoing/lei/00opt_rgb_prep/sample_mpii/rgb')
parser.add_argument("-hdf5_dir","--hdf5_dir",type=str,default='object_detect_video_frame_feat_sample.hdf5')

args = parser.parse_args()

model_path = args.model_path
hdf5_name = args.hdf5_dir
rgb_path = args.rgb_dir

odapi = DetectorAPI(path_to_ckpt=model_path)

with h5py.File(hdf5_name, 'w') as fo:
    for r, d, f in os.walk(rgb_path, followlinks = False):
        # print('root: ', r)
        # print('dir: ', d)
        # print('files: ', f)
        f.sort()
        if len(f) == 0:
            print('empty!')
            continue;

        rgb_rep = r.replace(rgb_path, 'rgb')
        # print('rgb_rep: ', rgb_rep)
        # print(len(r), len(d), len(f))
        
        length = len(f)
        videoname = rgb_rep
        # print('rgb---------------- ', videoname)
        f_nameonly = videoname + '/count'
        h_nameonly = videoname + '/height'
        w_nameonly = videoname + '/width'
        test = ioo.imread(os.path.join(r, f[0]))
        height = test.shape[0]
        width = test.shape[1]
        # print('********', f_nameonly)
        # print('********', h_nameonly)
        # print('********', w_nameonly)
        # print(videoname, ' | frame count: ', length, ' height: ', height, ' width: ', width)
        fo.create_dataset(f_nameonly, data = length)
        fo.create_dataset(h_nameonly, data = height)
        fo.create_dataset(w_nameonly, data = width)
        for idx in range(len(f)):
            imgname = os.path.join(r, f[idx])
            imgnameonly = videoname + '/frame' + str(idx).zfill(8)
            # imgnameonly = os.path.join(videoname, f[idx][:-4])
            img = ioo.imread(imgname)
            infoDic = {}
            num, boxes, scores, classes, feat = odapi.processFrame(img)
            infoDic['num_detections'] = num
            infoDic['detection_classes'] = classes[0:num, ].tolist()
            infoDic['detection_scores'] = scores[0:num, ].tolist()
            infoDic['detection_boxes'] = boxes[0:num, ].tolist()
            infoDic['feature_avg'] = feat.tolist()
            # print(infoDic['detection_classes'])
            # print(idx, '--- scores: ', scores.shape, ' | classes: ', classes.shape, ' | num: ', num, ' | boxes: ', boxes.shape, ' | feat: ', feat.shape)
            fo.create_dataset(imgnameonly, data = json.dumps(infoDic))
            print(' | video name: ', videoname, ' - ', '/frame' + str(idx).zfill(8), ' old name: ', f[idx][:-4], ' | #detect.: ', num, ' feat. dim.: ', feat.shape)
        # break;

    print('Done!')
