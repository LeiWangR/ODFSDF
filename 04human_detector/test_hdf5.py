import h5py
import json
import numpy as np

# give a sample video name
videoname = 's20-d07-cam-002_05496_038'
# load saved hdf5 file
filename = 'object_detect_vf_ava_resnet101_feat_sample.hdf5'
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

