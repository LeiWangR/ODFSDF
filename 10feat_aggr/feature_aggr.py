import h5py
import os
import pandas as pd
import numpy as np
import cv2
import json
import argparse

import time
import imageio

from skimage import feature

# python feature_aggr.py -dataset_name 'mpii_sample' -hdf5_info_dir '/home/wan305/research/ongoing/lei/10feature_aggr/sample_info' -info_list_dir '/home/wan305/research/ongoing/lei/10feature_aggr' -hog_im_size 32 -sal_sqim_size 16

parser = argparse.ArgumentParser(description="feature aggregation aka information fusion")
parser.add_argument("-dataset_name","--dataset_name",type=str,default='mpii_sample')
parser.add_argument("-hdf5_info_dir","--hdf5_info_dir",type=str,default='/home/wan305/research/ongoing/lei/10feature_aggr/sample_info')
parser.add_argument("-info_list_dir","--info_list_dir",type=str,default='/home/wan305/research/ongoing/lei/10feature_aggr')
parser.add_argument("-hog_im_size","-hog_im_size",type=int,default=32)
parser.add_argument("-sal_sqim_size","-sal_sqim_size",type=int,default=16)

args = parser.parse_args()

dataset_name = args.dataset_name
hdf5_info_dir = args.hdf5_info_dir
info_list_dir = args.info_list_dir
hog_im_size = args.hog_im_size
sal_sqim_size = args.sal_sqim_size

info_list = os.path.join(info_list_dir, dataset_name + '_info_list.csv')
df_info = pd.read_csv(info_list, header = None)
print(info_list, ' | ', df_info.shape)

# saliency
f_sal01 = h5py.File(os.path.join(hdf5_info_dir, dataset_name + '_saliency_attention_jpeg.hdf5'), 'r')
f_sal02 = h5py.File(os.path.join(hdf5_info_dir, dataset_name + '_video_saliency_jing.hdf5'), 'r')
# object detector
f_det_incv2 = h5py.File(os.path.join(hdf5_info_dir, dataset_name + '_inception_v2_object_det_info.hdf5'), 'r')
f_det_incresv2 = h5py.File(os.path.join(hdf5_info_dir, dataset_name + '_inception_resnet_v2_object_det_info.hdf5'), 'r')
f_det_res101 = h5py.File(os.path.join(hdf5_info_dir, dataset_name + '_rcnn_resnet101_ava_object_det_info.hdf5'), 'r')
# f_det_nas = h5py.File(os.path.join(hdf5_info_dir, dataset_name + '_faster_rcnn_nas_object_det_info.hdf5'), 'r')
# imagenet feat.
f_img_incv2 = h5py.File(os.path.join(hdf5_info_dir, dataset_name + '_per_bb_feat_incv2.hdf5'), 'r')
f_img_incresv2 = h5py.File(os.path.join(hdf5_info_dir, dataset_name + '_per_bb_feat_incresnetv2.hdf5'), 'r')
f_img_res101 = h5py.File(os.path.join(hdf5_info_dir, dataset_name + '_per_bb_feat_res101_ava.hdf5'), 'r')
# f_img_nas = h5py.File(os.path.join(hdf5_info_dir, dataset_name + '_per_bb_feat_rcnn_nas.hdf5'), 'r')

def hog_saliency(img_saliency, det_num, boxes, sal_sqim_size):
    height = img_saliency.shape[0]
    width = img_saliency.shape[1]
    # for storing the hog feature
    HoG = np.zeros((det_num, int(9*4*4*(hog_im_size/8/4) * (hog_im_size/8/4))))
    SQIM = np.zeros((det_num, sal_sqim_size * sal_sqim_size))
    for ii in range(det_num):
        box = boxes[ii, :]
        scale_box = [int(box[0] * height), int(box[1] * width), int(box[2] * height), int(box[3] * width)]
        # print(scale_box)
        mini_bb_im = img_saliency[scale_box[0]:scale_box[2], scale_box[1]:scale_box[3]]
        # print(mini_bb_im)
        img = cv2.resize(mini_bb_im, (hog_im_size, hog_im_size))
        # print(img)
        sqim = cv2.resize(mini_bb_im, (sal_sqim_size, sal_sqim_size))
        H = feature.hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(4, 4), transform_sqrt=True, block_norm="L1")
        HoG[ii, :] = H
        SQIM[ii, :] = np.reshape(sqim, (1, -1))
        # print(H.shape)
        # print(H)
        # for checking whether the bb is applied correctly (saved to test folder)
        # timestr = time.strftime("%Y%m%d-%H%M%S")
        # imageio.imwrite('test/' + str(ii) + '-' + timestr + '-' + str(img.shape[0]) + '-' + str(img.shape[1]) + str(scale_box) + '.jpg', img)
    return HoG, SQIM

for ii in range(df_info.shape[0]):
    videoname = df_info.iloc[ii, 0]
    print(videoname, ' --- ')
    count = f_det_incv2['rgb/' + videoname + '/count'][()]
    height = f_det_incv2['rgb/' + videoname + '/height'][()]
    width = f_det_incv2['rgb/' + videoname + '/width'][()]
    for jj in range(count):
        # ------------ Saliency (TPAMI2019 & CVPR2018)
        frame_sal01 = f_sal01['rgb/' + videoname + '/frame' + str(jj).zfill(8) + '/saliency'][:]
        frame_sal02 = f_sal02['rgb/' + videoname + '/frame' + str(jj).zfill(8) + '/mini_saliency'][:]

        im_sal01 = cv2.imdecode(frame_sal01, cv2.IMREAD_GRAYSCALE)
        im_sal02 = cv2.imdecode(frame_sal02, cv2.IMREAD_GRAYSCALE)

        print('============ sal. PAMI2019: ', im_sal01.shape)
        print('============ sal. CVPR2018: ', im_sal02.shape)
        
        # ------------ Object detection information
        dic_info_incv2 = json.loads(f_det_incv2['rgb/' + videoname + '/frame' + str(jj).zfill(8)][()])
        dic_info_incresv2 = json.loads(f_det_incresv2['rgb/' + videoname + '/frame' + str(jj).zfill(8)][()])
        dic_info_res101 = json.loads(f_det_res101['rgb/' + videoname + '/frame' + str(jj).zfill(8)][()])
        # dic_info_nas = json.loads(f_det_nas['rgb/' + videoname + '/frame' + str(jj).zfill(8)][()])

        # ------------ Per bb feature pre-trained on ImageNet
        dic_info_img_incv2 = json.loads(f_img_incv2['rgb/' + videoname + '/frame' + str(jj).zfill(8)][()])
        dic_info_img_incresv2 = json.loads(f_img_incresv2['rgb/' + videoname + '/frame' + str(jj).zfill(8)][()])
        dic_info_img_res101 = json.loads(f_img_res101['rgb/' + videoname + '/frame' + str(jj).zfill(8)][()])
        # dic_info_img_nas = json.loads(f_img_nas['rgb/' + videoname + '/frame' + str(jj).zfill(8)][()])

        num_incv2 = dic_info_img_incv2['num_detections']
        if num_incv2 == 0:
            print('incv2 --- no bb feature!')
        else:       
            bb_features_incv2 = np.asarray(dic_info_img_incv2['bb_feature'])
            print('incv2 bb img feat: ', bb_features_incv2.shape)
            boxes_incv2 = np.asarray(dic_info_incv2['detection_boxes'])
            # hog feature extraction per bb applied saliency (one frame only)
            sal_feat01, sqim01 = hog_saliency(im_sal01, num_incv2, boxes_incv2, sal_sqim_size)
            sal_feat02, sqim02 = hog_saliency(im_sal02, num_incv2, boxes_incv2, sal_sqim_size)

            classes_incv2 = np.asarray(dic_info_incv2['detection_classes'])
            scores_incv2 = np.asarray(dic_info_incv2['detection_scores'])
            print('*** incv2 im reshape: ', sqim01.shape, ' | ', sqim02.shape)
            print(' | incv2 BB ', boxes_incv2.shape, ' | incv2 HoG sal01: ', sal_feat01.shape, ' | incv2 HoG sal02: ', sal_feat02.shape)

        print('-----------------------------------------------------')

        num_incresv2 = dic_info_img_incresv2['num_detections']
        if num_incresv2 == 0:
            print('incresv2 --- no bb feature!')
        else:
            bb_features_incresv2 = np.asarray(dic_info_img_incresv2['bb_feature'])
            print('incresv2 bb img feat: ', bb_features_incresv2.shape)
            boxes_incresv2 = np.asarray(dic_info_incresv2['detection_boxes'])
            # hog feature extraction per bb applied saliency (one frame only)
            sal_feat01, sqim01 = hog_saliency(im_sal01, num_incresv2, boxes_incresv2, sal_sqim_size)
            sal_feat02, sqim02 = hog_saliency(im_sal02, num_incresv2, boxes_incresv2, sal_sqim_size)

            classes_incresv2 = np.asarray(dic_info_incresv2['detection_classes'])
            scores_incresv2 = np.asarray(dic_info_incresv2['detection_scores'])
            print('*** incresv2 im reshape: ', sqim01.shape, ' | ', sqim02.shape)
            print(' | incresv2 BB ', boxes_incresv2.shape, ' | incresv2 HoG sal01: ', sal_feat01.shape, ' | incresv2 HoG sal02: ', sal_feat02.shape)

        print('-----------------------------------------------------')

        num_res101 = dic_info_img_res101['num_detections']
        if num_res101 == 0:
            print('res101 --- no bb feature!')
        else:
            bb_features_res101 = np.asarray(dic_info_img_res101['bb_feature'])
            print('res101 bb img feat: ', bb_features_res101.shape)
            boxes_res101 = np.asarray(dic_info_res101['detection_boxes'])
            # hog feature extraction per bb applied saliency (one frame only)
            sal_feat01, sqim01 = hog_saliency(im_sal01, num_res101, boxes_res101, sal_sqim_size)
            sal_feat02, sqim02 = hog_saliency(im_sal02, num_res101, boxes_res101, sal_sqim_size)

            classes_res101 = np.asarray(dic_info_res101['detection_classes'])
            scores_res101 = np.asarray(dic_info_res101['detection_scores'])
            print('*** res101 im reshape: ', sqim01.shape, ' | ', sqim02.shape)
            print(' | res101 BB ', boxes_res101.shape, ' | res101 HoG sal01: ', sal_feat01.shape, ' | res101 HoG sal02: ', sal_feat02.shape)
 
        print('-----------------------------------------------------')
        
        '''
        num_nas = dic_info_img_nas['num_detections']
        if num_nas == 0:
            print('nas --- no bb feature!')
        else:
            bb_features_nas = np.asarray(dic_info_img_nas['bb_feature'])
            print('nas bb img feat: ', bb_features_nas.shape)
            boxes_nas = np.asarray(dic_info_nas['detection_boxes'])
            # hog feature extraction per bb applied saliency (one frame only)
            sal_feat01, sqim01 = hog_saliency(im_sal01, num_nas, boxes_nas, sal_sqim_size)
            sal_feat02, sqim02 = hog_saliency(im_sal02, num_nas, boxes_nas, sal_sqim_size)

            classes_nas = np.asarray(dic_info_nas['detection_classes'])
            scores_nas = np.asarray(dic_info_nas['detection_scores'])
            print('*** nas im reshape: ', sqim01.shape, ' | ', sqim02.shape)
            print(' | nas BB ', boxes_nas.shape, ' | nas HoG sal01: ', sal_feat01.shape, ' | nas HoG sal02: ', sal_feat02.shape)
        '''
        print('-----------------------------------------------------')






