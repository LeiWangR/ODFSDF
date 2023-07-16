import numpy as np
import tensorflow as tf
import cv2
import time
from DetectorAPI import DetectorAPI
import h5py
import os
import json
import argparse

# Researcher: Lei Wang
# 1 October 2019

# This code are used to generate the video descriptor from object detection pipeline
# Input:
#   - rgb video dir (for video frame images, please refer to video frame processing code)
#   - hdf5 file name to store the per-frame information (#detection, detection_classes, confidence_scores, bounding_boxes (normalized), and region proposal average pooling features/300)
#   - model path: the pre-trained model for object detection
# ***************SAMPLE RGB VIDEOS************************************
# -moments_in_time
#       -diving
#               -video1.mp4
#               -video2.mp4
#               -...
#       -shooting
#               -...
# Output: single hdf5 file

# you need to enter Lei's running environment in conda (myenv)

# python object_det_video_feat.py -hdf5_dir 'object_detect_video_feat_sample.hdf5' -rgb_dir '/home/wan305/research/ongoing/lei/00opt_rgb_prep/moments_in_time' -model_path '/home/wan305/research/ongoing/lei/04human_detector/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/frozen_inference_graph.pb'

# for video-level, we store rgb/class_folder(action_folder)/video_name(no extension)
#       -/count
#       -/height
#       -/width
#       -/fps
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
videoname = 'writing/Z3CYTr-mXWk_35'
# load saved hdf5 file
filename = 'object_detect_video_feat_sample.hdf5'
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
parser.add_argument("-rgb_dir","--rgb_dir",type=str,default='/home/wan305/research/ongoing/lei/00opt_rgb_prep/moments_in_time')
parser.add_argument("-hdf5_dir","--hdf5_dir",type=str,default='object_detect_video_feat_sample.hdf5')

args = parser.parse_args()

model_path = args.model_path
hdf5_name = args.hdf5_dir
rgb_path = args.rgb_dir

odapi = DetectorAPI(path_to_ckpt=model_path)

with h5py.File(hdf5_name, 'w') as fo:
    for r, d, f in os.walk(rgb_path, followlinks = False):
        rgb_rep = r.replace(rgb_path, 'rgb')
        for idx in range(len(f)):
            videoname = os.path.join(r, f[idx])
            nameonly = os.path.join(rgb_rep, f[idx][:-4])
            # print(videoname, ' rgb---')
            # print('********rgb | ', nameonly)
            f_nameonly = nameonly + '/count'
            h_nameonly = nameonly + '/height'
            w_nameonly = nameonly + '/width'
            fps_nameonly = nameonly + '/fps'

            cap = cv2.VideoCapture(videoname)
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps    = cap.get(cv2.CAP_PROP_FPS)

            fo.create_dataset(f_nameonly, data = length)
            fo.create_dataset(h_nameonly, data = height)
            fo.create_dataset(w_nameonly, data = width)
            fo.create_dataset(fps_nameonly, data = fps)

            for ii in range(length):
                infoDic = {}
                frame_name = nameonly + '/frame' + str(ii).zfill(8)
                # print(frame_name)
                ret, frame = cap.read()
                num, boxes, scores, classes, feat = odapi.processFrame(frame)
                infoDic['num_detections'] = num
                infoDic['detection_classes'] = classes[0:num, ].tolist()
                infoDic['detection_scores'] = scores[0:num, ].tolist()
                infoDic['detection_boxes'] = boxes[0:num, ].tolist()
                infoDic['feature_avg'] = feat.tolist()

                # print(ii, '--- scores: ', scores.shape, ' | classes: ', classes.shape, ' | num: ', num, ' | boxes: ', boxes.shape, ' | feat: ', feat.shape)
                fo.create_dataset(frame_name, data = json.dumps(infoDic))
                print(idx, ' | video name: ', nameonly, ' -- frame ', ii, ' | #detect.: ', num, ' feat. dim.: ', feat.shape)

    print('Done!')
