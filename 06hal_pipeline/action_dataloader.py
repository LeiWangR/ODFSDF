import random
import torch
from torch.utils import data
import pandas as pd
import os
import glob
import math
import numpy as np
from skimage import io, transform, img_as_float
import numpy as np
from torchvision import transforms, utils
from skimage.transform import resize

import h5py
import cv2

#-------------------------------------------------------------------------
'''
Researcher: Lei Wang

We resize the video to 256x256
 For training, we random crop to get a square video 224x224, also with horizontal flip
 For testing, we do center crop of the video only

'''
class Rescale(object):
    """Rescale the image in a sample to a given size.
    # Rescale here may be not necessary as each frame has been resized 
    # when loading each image 

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size=(256, 320)):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):

        for key in sample:
            if key == 'rgb' or key == 'opt' or key == 'depth':
                video = sample[key]
                h, w = video.shape[1],video.shape[2]
                desired_frames = video.shape[0]
                channels = video.shape[3]

                if isinstance(self.output_size, int):
                    if h > w:
                        new_h, new_w = self.output_size * h / w, self.output_size
                    else:
                        new_h, new_w = self.output_size, self.output_size * w / h
                else:
                    new_h, new_w = self.output_size
                new_h, new_w = int(new_h), int(new_w)

                new_video=np.zeros((desired_frames,new_h,new_w,channels))
                for i in range(desired_frames):
                    image=video[i,:,:,:]
                    img = transform.resize(image, (new_h, new_w), mode='constant')
                    new_video[i,:,:,:]=img
                sample[key] = new_video
        return sample

class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size=(224,224)):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        for key in sample:
            if key == 'rgb' or key == 'opt' or key == 'depth':
                video = sample[key]
                h, w = video.shape[1],video.shape[2]
                desired_frames = video.shape[0]
                channels = video.shape[3]
                new_h, new_w = self.output_size

                top = np.random.randint(0, h - new_h)
                left = np.random.randint(0, w - new_w)

                new_video=np.zeros((desired_frames,new_h,new_w,channels))
                for i in range(desired_frames):
                    image=video[i,:,:,:]
                    image = image[top: top + new_h,left: left + new_w]
                    new_video[i,:,:,:]=image
                sample[key] = new_video
        return sample

class CenterCrop(object):
    """Crop the given video at the center
    This function is used for testing

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size=(224,224)):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        for key in sample:
            if key == 'rgb' or key == 'opt' or key == 'depth':
                video = sample[key]
                h, w = video.shape[1],video.shape[2]
                desired_frames = video.shape[0]
                channels = video.shape[3]
                new_h, new_w = self.output_size

                top = int(np.round((h - new_h) / 2.))
                left = int(np.round((w - new_w) / 2.))

                new_video=np.zeros((desired_frames,new_h,new_w,channels))
                for i in range(desired_frames):
                    image=video[i,:,:,:]
                    image = image[top: top + new_h,left: left + new_w]
                    new_video[i,:,:,:]=image
                sample[key] = new_video
        return sample

class ToTensor(object):
    def __call__(self, sample):
        for key in sample:
            if key == 'rgb' or key == 'opt' or key == 'depth':
                values = sample[key]
                sample[key] = torch.from_numpy(values.transpose((3, 0, 1, 2)))
            elif key == 'label':
                values = sample[key]
                sample[key] = torch.from_numpy(np.array(values))
            else:
                # features in torch tensor format
                pass
        return sample

class ActionDataset(data.Dataset):
    def __init__(self, hdf5_dir, info_list, frame, tr_te_flag, option_flag, bb_flag, feat_flag, feature_dir, transforms = None):
        # a single hdf5 file contains rgb, opt video frames (may have depth sequences as well)
        self.hdf5_dir = hdf5_dir
        self.bin_video = h5py.File(hdf5_dir, 'r')
        # info_list contains the video name and its corresponding labels (start from 0)
        self.info_list = pd.read_csv(info_list, header = None)
        self.frame = frame
        self.tr_te_flag = tr_te_flag
        # for selecting video modality (rgb, opt, depth)
        self.option_flag = option_flag
        # if bb is applied on top (boundary box of human subject only)
        # however, in any case, Lei suggests to set this flag to True
        self.bb_flag = bb_flag
        # a single hdf5 file contains features like fv, bow, i3d_opt, soundnet8, saliencyACL, stgcn, etc
        self.feature_dir = feature_dir
        # e.g., feat_flag = [] --- not select any features
        # feat_flag = ['i3d-opt', 'soundnet8', 'saliencyACL', 'st-gcn'] means
        # select opt, sound, saliency and gcn features
        self.feat_flag = feat_flag
        if len(self.feat_flag) != 0:
            self.bin_feat = h5py.File(feature_dir, 'r')
        self.transforms = transforms

    def __len__(self):
        return len(self.info_list)

    def __getitem__(self, idx):
        # the first column of the infomation list is the video name
        # sample video name such as 'write/jumping' or just 'jumping' depends on the datasets
        video_name = self.info_list.iloc[idx, 0]
        # the class label should be (0, C-1), where C is the total number of classes
        video_label = self.info_list.iloc[idx, 1]
        # choose which video modality
        # return video
        video = self.get_video(self.bin_video, self.frame, video_name, self.option_flag, self.bb_flag)
        
        # if it is in train mode, do horizontal flip for the video
        # otherwise (in test mode), do not flip
        if self.tr_te_flag == 'train':
            # horizontal flip
            if random.randint(1, 10) % 2 == 0:
                video = self.left_right_flip(video)

        # create a dictionary to store video, label and features (if applicable)
        sample = {self.option_flag: video, 'label': video_label}
        if len(self.feat_flag) != 0:
            num_feat = len(self.feat_flag)
            for tt in range(num_feat):
                feat_tag = self.feat_flag[tt]
                # function to get the features
                feature = self.get_feature(self.bin_feat, video_name, feat_tag)
                sample.update({feat_tag : feature})
        if self.transforms:
            sample = self.transforms(sample)
        return sample

    def get_video(self, bin_video, frame, video_name, option_flag, bb_flag = True):
        # please use true here
        if bb_flag:
            # needs to rescale the video (as bb applied video has various height & width)
            height = 256
            width = 256
        else:
            height = bin_video[option_flag + '/' + video_name + '/height'][()]
            width = bin_video[option_flag + '/' + video_name + '/width'][()]

        # generate a list of index that needs to pick up
        if option_flag == 'rgb':
            actual_frames = bin_video[option_flag + '/' + video_name + '/count'][()]
        else:
            actual_frames = bin_video['u/' + video_name + '/count'][()]
        if actual_frames >= frame:
            # print('frames satisfied')
            evensList = [x for x in range(actual_frames) if x % (math.floor(actual_frames/frame)) == 0]
        else:
            # print('frame less than desired')
            added_num = frame - actual_frames
            oldList = list(range(0, actual_frames, 1))
            addedArray = np.random.choice(oldList, added_num)
            addedList = addedArray.tolist()
            evensList = oldList + addedList
            evensList.sort()
        # print(evensList)

        # For Charades:
        # 	rgb/videofilename (no extension)/frame00000*** (no extension, total 8 digits and *** start from 0)
        # 	u or v/videofilename (no extension)/frame00000*** (no extension, total 8 digits and *** start from 0)
        # For MPII:
        # 	rgb/videofilename (no extension)/frame00000*** (no extension, total 8 digits and *** start from 0)
        # 	u or v/videofilename (no extension)/frame00000*** (no extension, total 8 digits and *** start from 1)
        # For YUP++ dataset:
        # 	rgb/videofilename (no extension)/frame000*** (no extension, total 6 digits and ** start from 1)
        # 	u or v/videofilename (no extension)/frame000*** (no extension, total 6 digits and ** start from 2)
        # For HMDB51 dataset:
        # 	rgb/videofilename (no extension)/frame000*** (no extension, total 6 digits and ** start from 1)
        # 	u or v/videofilename (no extension)/frame000*** (no extension, total 6 digits and ** start from 1)
        # Therefore, to load the hdf5 file for different datasets, you just need to change 'str(ii).zfill(8)' to 'str(ii + 1).zfill(8)' for optical flow components of Charades and MPII, to 'str(ii + 2).zfill(6)' for YUP++ (for YUP++ rgb, you need to change to 'str(ii + 1).zfill(6)')

        # use hmdb51 for example, either rgb and opt are all fine
        if option_flag == 'rgb':
            video = np.zeros((frame, height, width, 3))
            for index in range(frame):
                # get the index from the number list
                im_idx = evensList[index]
                im_frame = bin_video[option_flag + '/' + video_name + '/frame' + str(im_idx + 1).zfill(6)][:]
                im = cv2.imdecode(im_frame, cv2.IMREAD_COLOR)
                # scale pixel values in range(-1, 1)
                tmp_image = (im/255.)*2 - 1
                tmp_image = resize(tmp_image, (height, width), mode='constant')
                video[index, :, :, :] = tmp_image
        elif option_flag == 'opt':
            video = np.zeros((frame, height, width, 2))
            for index in range(frame):
                # get the index from the number list
                im_idx = evensList[index]
                one_frame = np.zeros((height, width, 2))
                im_frame_u = bin_video['u/' + video_name + '/frame' + str(im_idx + 1).zfill(6)][:]
                im_frame_v = bin_video['v/' + video_name + '/frame' + str(im_idx + 1).zfill(6)][:]
                im_u = cv2.imdecode(im_frame_u, cv2.IMREAD_GRAYSCALE)
                im_v = cv2.imdecode(im_frame_v, cv2.IMREAD_GRAYSCALE)
                # scale pixel values in range(-1, 1)
                tmp_u_image = (im_u/255.)*2 - 1
                tmp_v_image = (im_v/255.)*2 - 1

                tmp_u_image = resize(tmp_u_image, (height, width), mode='constant')
                tmp_v_image = resize(tmp_v_image, (height, width), mode='constant')

                one_frame[:, :, 0] = tmp_u_image
                one_frame[:, :, 1] = tmp_v_image

                video[index, :, :, :] = one_frame
        else:
            video = np.zeros((frame, height, width, 3))
            for index in range(frame):
                # get the index from the number list
                im_idx = evensList[index]
                # depth image (hmdb51 no depth image)
                im_frame = bin_video[option_flag + '/' + video_name + '/frame' + str(im_idx + 1).zfill(6)][:]
                # encode depth image is:
                # encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 5]
                # im_buffer = cv2.imencode('.png', im, encode_param)[1]
                # and decode from binary is:
                im = cv2.imdecode(im_frame, cv2.IMREAD_ANYDEPTH)
                # TODO
        return video    

    def left_right_flip(self, video):
        # horizontal flip of the video
        # RGB video left-right flip for each channel
        # Optical flow video left-right flip for each channel, but
        # also reverse the x components
        frames = video.shape[0]
        height, width = video.shape[1], video.shape[2]
        channels = video.shape[3]
        video_flipped = np.zeros((frames, height, width, channels))
        for fi in range(frames):
            # print(fi)
            channel_im = np.zeros((height, width, channels))
            for ci in range(channels):
                flip_c = video[fi, :, :, ci]
                temp_flip_c = np.flip(flip_c, 1)
                if channels == 2 and ci == 0:
                    temp_flip_c = -1 * temp_flip_c
                channel_im[:, :, ci] = temp_flip_c
            video_flipped[fi, :, :, :] = channel_im
        return(video_flipped)

    def get_feature(self, bin_feat, video_name, feat_tag):
        # one single feat_tag
        # one video name
        # bin_feat dictionary contains all the features
        
        feat = bin_feat[feat_tag + '/' + video_name]
        return feat
#-------------------------------------------------------------------------




