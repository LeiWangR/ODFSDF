import torch
import torch.nn.functional as F
import numpy as np
import pdb, os, argparse
import imageio
from model.vgg_models import Back_VGG
from model.ResNet_models import Back_ResNet
from data import test_dataset
import h5py
import cv2

# Researcher: Lei Wang
# 3 October 2019
# Codes modified from the original unsupervised_sal codes
# we just use the pre-trained VGG-based saliency model for video saliency extraction
# and save them into one single hdf5 file

# python video_saliency_to_hdf5_lei.py -rgb_dir '/home/wan305/research/ongoing/lei/00opt_rgb_prep/sample_mpii/rgb/' -hdf5_dir 'video_saliency_sample_jing.hdf5' 

# ***************SAMPLE VIDEO FRAME IMAGES************************************
# -sample_mpii
#       -rgb
#               -video1 (inside it would be many frame images)
#               -video2 (inside it would be many frame images)
# Output: single hdf5 file
# different to the codes of saving dictionary for saliency and attention tensors (for memory constrains)
# the hdf5 saved structure is given below:
#   - rgb/video_name/count (the total number of frames)
#   - rgb/video_name/frame0000000* (8 digits and always starts from 0)/mini_saliency(for small saliency image per frame, always 352x352)
#   - rgb/video_name/frame0000000* (8 digits and always starts from 0)/saliency (for saliency image per frame, original input resolution)
# To load the mini-saliency and saliency (original input resolution), using the following codes:

'''
import h5py
import cv2

videoname = 's20-d07-cam-002_05490_038'
# videoname = 's20-d07-cam-002_05494_039'

filename = 'charades_saliency_attention_sample.hdf5'
f = h5py.File(filename, 'r')
count = f['rgb/' + videoname + '/count'][()]
print(count)

for ii in range(count):
    frame_mini_sal = f['rgb/' + videoname + '/frame' + str(ii).zfill(8) + '/mini_saliency'][:]
    frame_sal = f['rgb/' + videoname + '/frame' + str(ii).zfill(8) + '/saliency'][:]
    # grayscale for attention image
    im_mini_sal = cv2.imdecode(frame_mini_sal, cv2.IMREAD_GRAYSCALE)
    im_sal = cv2.imdecode(frame_sal, cv2.IMREAD_GRAYSCALE)
    print(ii, '---', im_mini_sal.shape, 'xxx', im_sal.shape)
f.close()

'''

parser = argparse.ArgumentParser(description="video mini-saliency and saliency extraction")
parser.add_argument("-rgb_dir","--rgb_dir",type=str,default='/home/wan305/research/ongoing/lei/00opt_rgb_prep/sample_mpii/rgb/')
parser.add_argument("-hdf5_dir","--hdf5_dir",type=str,default='video_saliency_sample_jing.hdf5')
args = parser.parse_args()
hdf5_name = args.hdf5_dir
videos_test_path = args.rgb_dir

testsize = 352 # provided by the original authors
is_ResNet = False # we just use the VGG model for video saliency extraction

if is_ResNet:
    model = Back_ResNet()
    model.load_state_dict(torch.load('CPD-R.pth'))
else:
    model = Back_VGG()
    model.load_state_dict(torch.load('./models/vgg_pce_99.pth', map_location=torch.device('cpu')))

model.eval()

videos = [videos_test_path + f for f in os.listdir(videos_test_path) if os.path.isdir(videos_test_path + f)]
videos.sort()
nb_videos_test = len(videos)

with h5py.File(hdf5_name, 'w') as fo:
    for ii in range(nb_videos_test):
        images_names = [f for f in os.listdir(videos[ii] + '/') if f.endswith(('.jpg', '.jpeg', '.png'))]
        images_names.sort()
        if images_names == []:
            print('empty!')
            continue;
        dataset = videos[ii]
        # print(videos[ii])
        test_loader = test_dataset(dataset + '/', testsize)
        # print(len(images_names))
        new_video_name = dataset.replace(videos_test_path, 'rgb/')
        fo.create_dataset(new_video_name + '/count', data = len(images_names))
        print('No.: ', ii, " | saliency maps for " + new_video_name)
        for i in range(len(images_names)):
            # print(i)
            # always load the ordered/sorted images
            image, HH,WW,name = test_loader.load_data()
            # print(name)
            mini_sal_frame_name = new_video_name + '/frame' + str(i).zfill(8) + '/mini_saliency'
            sal_frame_name = new_video_name + '/frame' + str(i).zfill(8) + '/saliency'
            res = model(image)
            
            mini_sal = res.sigmoid().data.cpu().numpy().squeeze()
            mini_sal = (mini_sal - mini_sal.min()) / (mini_sal.max() - mini_sal.min() + 1e-8)
            mini_sal = (mini_sal*255).astype(np.uint8)
            # print(mini_sal.shape)
            # print(mini_sal)
            # imageio.imwrite('frame' + str(i).zfill(8) + '.png', mini_sal)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
            from_buffer_mini_sal = cv2.imencode('.jpg', mini_sal, encode_param)[1]
            fo.create_dataset(mini_sal_frame_name, data = from_buffer_mini_sal)

            res = F.upsample(res, size=[WW,HH], mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            res = (res*255).astype(np.uint8)
            from_buffer_sal = cv2.imencode('.jpg', res, encode_param)[1]
            fo.create_dataset(sal_frame_name, data = from_buffer_sal)

        


