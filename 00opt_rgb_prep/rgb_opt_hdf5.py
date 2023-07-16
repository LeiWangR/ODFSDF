import h5py
import cv2
import os
import argparse

#-----------------------------------#
# compress all rgb and optical flow information into a single hdf5 file
# you need to provide the path to rgb video, optical flow u and v directories
# you need to give a new path where to store the hdf5 file and the name of the file
# optical flow is compressed using grayscale image stream
# for rgb video, we store rgb/class_folder(action_folder)/video_name(no extension)
# 	-/count
#	-/height
# 	-/width
# 	-/fps
# 	-/frame000000**
# for opt videos, we store u/class_folder(action_folder)/video_name(no extension)
# 				or
# 				v/class_folder(action_folder)/video_name(no extension)
# 	-/count (for u components only as u and v both have the same frame counts)
# 	-/frame000000**

# example usage

# python rgb_opt_hdf5.py -hdf5_dir 'sample.hdf5' -rgb_dir 'moments_in_time' -opt_u_dir 'flow_images/u' -opt_v_dir 'flow_images/v'

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
# ***************SAMPLE GENERATED OPT VIDEOS******************
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

# to load the hdf5 file:
'''
import h5py
import cv2
# give a sample video name
videoname = 'writing/Z3CYTr-mXWk_35'
# load saved hdf5 file
filename = 'sample.hdf5'
f = h5py.File(filename, 'r')

# using [()] for value
count = f['rgb/' + videoname + '/count'][()]
fps = f['rgb/' + videoname + '/fps'][()]

print('count: ', count, ' | fps: ', fps)

for ii in range(count):
    # using [:] for image stream
    frame = f['rgb/' + videoname + '/frame' + str(ii).zfill(8)][:]
    # print(frame.shape)
    # for RGB video
    im = cv2.imdecode(frame, cv2.IMREAD_COLOR)
    # ------ for optical flow components ------
    # im = cv2.imdecode(frame, cv2.IMREAD_GRAYSCALE)
    # -----------------------------------------
    print(im.shape)
    print('--------')
    # print(im)
f.close()
'''
#-----------------------------------#

parser = argparse.ArgumentParser(description="put rgb and opt components into one hdf5 file")
parser.add_argument("-hdf5_dir","--hdf5_dir",type=str,default='sample.hdf5')
parser.add_argument("-rgb_dir","--rgb_dir",type=str,default='moments_in_time')
parser.add_argument("-opt_u_dir","--opt_u_dir",type=str,default='flow_images/u')
parser.add_argument("-opt_v_dir","--opt_v_dir",type=str,default='flow_images/v')

args = parser.parse_args()

hdf5_name = args.hdf5_dir
rgb_path = args.rgb_dir
opt_u_path = args.opt_u_dir
opt_v_path = args.opt_v_dir

with h5py.File(hdf5_name, 'w') as fo:
    for r, d, f in os.walk(rgb_path, followlinks = False):
        # print('root: ', r)
        # print('dir: ', d)
        # print('files: ', f)
        rgb_rep = r.replace(rgb_path, 'rgb')
        for idx in range(len(f)):
            videoname = os.path.join(r, f[idx])
            nameonly = os.path.join(rgb_rep, f[idx][:-4])
            # print(videoname, ' rgb---')
            print('********rgb | ', nameonly)
            f_nameonly = nameonly + '/count'
            h_nameonly = nameonly + '/height'
            w_nameonly = nameonly + '/width'
            fps_nameonly = nameonly + '/fps'
            # print('********', f_nameonly)
            # print('********', h_nameonly)
            # print('********', w_nameonly)
            # print('********', fps_nameonly)
            cap = cv2.VideoCapture(videoname)
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps    = cap.get(cv2.CAP_PROP_FPS)
            # print(nameonly, ' | frame count: ', length, ' height: ', height, ' width: ', width, ' fps: ', fps)
            fo.create_dataset(f_nameonly, data = length)
            fo.create_dataset(h_nameonly, data = height)
            fo.create_dataset(w_nameonly, data = width)
            fo.create_dataset(fps_nameonly, data = fps)
            for ii in range(length):
                frame_name = nameonly + '/frame' + str(ii).zfill(8)
                # print(frame_name)
                ret, frame = cap.read()
                # print('frame ', ii, ' | shape:', frame.shape)
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
                from_buffer = cv2.imencode('.jpg', frame, encode_param)[1]
                # print('done!')
                fo.create_dataset(frame_name, data = from_buffer)        
    print('rgb done!')

    for r, d, f in os.walk(opt_u_path, followlinks = False):
        # print('root: ', r)
        # print('dir: ', d)
        opt_u_rep = r.replace(opt_u_path, 'u')
        for idx in range(len(f)):
            videoname = os.path.join(r, f[idx])
            nameonly = os.path.join(opt_u_rep, f[idx][:-4])
            # print(videoname, ' opt u---')
            print('opt u | ********', nameonly)
            f_nameonly = nameonly + '/count'
            # h_nameonly = nameonly + '/height'
            # w_nameonly = nameonly + '/width'
            # fps_nameonly = nameonly + '/fps'
            cap = cv2.VideoCapture(videoname)
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # fps    = cap.get(cv2.CAP_PROP_FPS)
            # print(nameonly, ' | frame count: ', length, ' height: ', height, ' width: ', width, ' fps: ', fps)
            fo.create_dataset(f_nameonly, data = length)
            # fo.create_dataset(h_nameonly, data = height)
            # fo.create_dataset(w_nameonly, data = width)
            # fo.create_dataset(fps_nameonly, data = fps)
            for ii in range(length):
                frame_name = nameonly + '/frame' + str(ii).zfill(8)
                # print(frame_name)
                ret, frame = cap.read()
                # convert to grayscale
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # print('frame ', ii, ' | shape:', frame.shape)
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
                from_buffer = cv2.imencode('.jpg', frame, encode_param)[1]
                # print('done!')
                fo.create_dataset(frame_name, data = from_buffer)
    print('opt u done!')

    for r, d, f in os.walk(opt_v_path, followlinks = False):
        # print('root: ', r)
        # print('dir: ', d)
        opt_v_rep = r.replace(opt_v_path, 'v')
        for idx in range(len(f)):
            videoname = os.path.join(r, f[idx])
            nameonly = os.path.join(opt_v_rep, f[idx][:-4])
            # print(videoname, ' opt v---')
            print('opt v | ********', nameonly)
            # f_nameonly = nameonly + '/count'
            # h_nameonly = nameonly + '/height'
            # w_nameonly = nameonly + '/width'
            # fps_nameonly = nameonly + '/fps'
            cap = cv2.VideoCapture(videoname)
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # fps    = cap.get(cv2.CAP_PROP_FPS)
            # print(nameonly, ' | frame count: ', length, ' height: ', height, ' width: ', width, ' fps: ', fps)
            # fo.create_dataset(f_nameonly, data = length)
            # fo.create_dataset(h_nameonly, data = height)
            # fo.create_dataset(w_nameonly, data = width)
            # fo.create_dataset(fps_nameonly, data = fps)
            for ii in range(length):
                frame_name = nameonly + '/frame' + str(ii).zfill(8)
                # print(frame_name)
                ret, frame = cap.read()
                # convert to grayscale
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # print('frame ', ii, ' | shape:', frame.shape)
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
                from_buffer = cv2.imencode('.jpg', frame, encode_param)[1]
                # print('done!')
                fo.create_dataset(frame_name, data = from_buffer)
    print('opt v done!')
# fo.close()





        
