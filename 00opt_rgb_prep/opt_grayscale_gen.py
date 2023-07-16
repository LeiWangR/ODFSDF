import imageio
from skimage import io
import os
import os.path
import cv2
import sys
import numpy as np
import argparse

#-----------------------------------#
# optical flow grayscale video generation
# you need to provide the path to rgb video directory
# you need to give a new path where to save the generated optical flow
# saved into u and v folders separately
# keep the original fps (frame per second)
# keep the original resolution
# values are normalized into the range (0, 255)

# example usage

# python opt_grayscale_gen.py -rgb_dir 'moments_in_time' -opt_u_dir 'flow_images/u' -opt_v_dir 'flow_images/v'

# Note that in 'moments_in_time/' should have sub-folders, each sub-folder is a class / action
# under each class / action folder, it should be videos

# ***************SAMPLE RGB VIDEOS************************************
# -moments_in_time
# 	-diving
# 		-video1.mp4
# 		-video2.mp4
# 		-...
# 	-shooting
# 		-...
# ***************SAMPLE GENERATED OPT GRAYSCALE VIDEOS******************
# -flow_images
# 	-u
# 		-diving
# 			-video1.mp4
# 			-...
# 		-shooting
# 			-...
# 	-v
# 		-diving
# 			-video1.mp4
# 			-...
# 		-shooting
# 			-...
# ***********************************************************************
# Lei Wang (updated & finished 9 Sep 2019)
#-----------------------------------#
parser = argparse.ArgumentParser(description="optical flow grayscale video generation")

parser.add_argument("-rgb_dir","--rgb_dir",type=str,default='moments_in_time')
parser.add_argument("-opt_u_dir","--opt_u_dir",type=str,default='flow_images/u')
parser.add_argument("-opt_v_dir","--opt_v_dir",type=str,default='flow_images/v')

args = parser.parse_args()

path = args.rgb_dir
opt_u = args.opt_u_dir
opt_v = args.opt_v_dir

for r, d, f in os.walk(path, followlinks = False):
    # sys.stdout.write('xxx')
    # sys.stdout.flush()
    # print(r.split('/')[-1])
    # print('***')
    # print(d)
    # print(f)
    # skip its original folder
    if r == path:
        pass
    else:
        videoname_u = os.path.join(opt_u, r.split('/')[-1])
        videoname_v = os.path.join(opt_v, r.split('/')[-1]) 
        # videoname_u = opt_u + '/' + r.split('/')[-1]
        # videoname_v = opt_v + '/' + r.split('/')[-1]
        # print('1')
        # print(videoname_u)
        # print(videoname_v)
        if not os.path.exists(videoname_u):
            os.makedirs(videoname_u)
            os.makedirs(videoname_v)
        for idx in range(len(f)):
            # videoname = r + '/' + f[idx]
            videoname = os.path.join(r, f[idx])
            # action folder / class name
            # print(r.split('/')[-1])
            print(videoname, '----------')
            # added by Lei to check if the flow components have been computed
            # if exists, skip; otherwise compute optical flow.
            if os.path.exists(os.path.join(videoname_u, f[idx])) and os.path.exists(os.path.join(videoname_v, f[idx])):
                print('exists!')
                continue;

            cap = cv2.VideoCapture(videoname)
            fps = cap.get(cv2.CAP_PROP_FPS)
            ret, frame1 = cap.read()
            prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
            i = 0
            L_u = []
            L_v = []
            # ret is just a boolean type checking frame exists
            while(ret):
                ret, frame2 = cap.read()
                # if no frame
                if not ret:
                    break;
                else:
                    next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
                    optical_flow = cv2.DualTVL1OpticalFlow_create()
                    flow = optical_flow.calc(prvs, next, None)
                    # Change here
                    horz = cv2.normalize(flow[...,0], None, 0, 255, cv2.NORM_MINMAX)
                    vert = cv2.normalize(flow[...,1], None, 0, 255, cv2.NORM_MINMAX)
                    horz = horz.astype('uint8')
                    vert = vert.astype('uint8')

                    # print(vert)

                    L_u.append(horz)
                    L_v.append(vert)

                    # print('horz: ', horz.shape, ' | vert: ', vert.shape)

                    i = i + 1

                    # print('computing opt. between frame No. ', i - 1, ' and frame No. ', i)

                    prvs = next

            cap.release()

            X_u = np.asarray(L_u)
            X_v = np.asarray(L_v)
            kargs = {'fps': fps, 'macro_block_size': None}
            # imageio.mimwrite(videoname_u + '/' + f[idx], X_u, **kargs)
            # imageio.mimwrite(videoname_v + '/' + f[idx], X_v, **kargs)
            imageio.mimwrite(os.path.join(videoname_u, f[idx]), X_u, **kargs)
            imageio.mimwrite(os.path.join(videoname_v, f[idx]), X_v, **kargs)


