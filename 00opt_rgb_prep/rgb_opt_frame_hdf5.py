import h5py
import cv2
import os
import argparse
import glob
from skimage import io as ioo


# Note that this code is used only when you have many frame images of each video

#-----------------------------------#
# compress all rgb and optical flow information into a single hdf5 file
# you need to provide the path to rgb video, optical flow u and v directories
# you need to give a new path where to store the hdf5 file and the name of the file
# optical flow is compressed using grayscale image stream
# for rgb video, we store rgb/class_folder(action_folder)/video_name(no extension)
# 	-/count
#	-/height
# 	-/width
# 	-/fps (no fps for videos stored in frame images format)
# 	-/frame000000**
# for opt videos, we store u/class_folder(action_folder)/video_name(no extension)
# 				or
# 				v/class_folder(action_folder)/video_name(no extension)
# 	-/count (for u components only as u and v both have the same frame counts)
#	-/height (for u components only)
# 	-/width (for u components only)
# 	-/fps (no fps for videos stored in frame images format)
# 	-/frame000000**

# example usage

# python rgb_opt_frame_hdf5.py -hdf5_dir 'mpii_sample.hdf5' -rgb_dir 'sample_mpii/rgb' -opt_u_dir 'sample_mpii/u' -opt_v_dir 'sample_mpii/v'

# Note that in 'sample_mpii/rgb' or 'sample_mpii/u' or 'sample_mpii/v' should have sub-folders, 
# each sub-folder is a class / action
# under each class / action folder, it should be video frame images (if it is video, refer to Lei's code: rgb_opt_hdf5.py)

# ***************SAMPLE VIDEO FRAME IMAGES************************************
# -sample_mpii
# 	-rgb
# 		-video1 (inside it would be many frame images)
# 		-video2 (inside it would be many frame images)
# 		-...
# 	-u
# 		-video1 (inside it would be many frame images)
# 		-video2 (inside it would be many frame images)
# 		-...
# 	-v
# 		-video1 (inside it would be many frame images)
# 		-video2 (inside it would be many frame images)
# 		-...
# ***********************************************************************
# Lei Wang (updated & finished 12 Sep 2019)

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
height = f['rgb/' + videoname + '/height'][()]
width = f['rgb/' + videoname + '/width'][()]

print('count: ', count, ' | height: ', height, ' | width: ', width)

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

### Important Notes for video frame images 
###(for pure video please note that we all use standard 8 digits 
### and *** all start from 0 for rgb, u and v, refer to rgb_opt_hdf5.py codes):

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

#-----------------------------------#

parser = argparse.ArgumentParser(description="put rgb and opt components (frame images) into one hdf5 file")
parser.add_argument("-hdf5_dir","--hdf5_dir",type=str,default='mpii_sample.hdf5')
parser.add_argument("-rgb_dir","--rgb_dir",type=str,default='sample_mpii/rgb')
parser.add_argument("-opt_u_dir","--opt_u_dir",type=str,default='sample_mpii/u')
parser.add_argument("-opt_v_dir","--opt_v_dir",type=str,default='sample_mpii/v')

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
        if len(f) == 0:
            print('empty!')
            continue;

        rgb_rep = r.replace(rgb_path, 'rgb')
        # print('rgb_rep: ', rgb_rep)
        # print(len(r), len(d), len(f))
        
        length = len(f)
        videoname = rgb_rep
        print('rgb---------------- ', videoname)
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
            # print('idx: ', idx)
            # print(r)
            imgname = os.path.join(r, f[idx])
            # print('********rgb | ', imgname)
            imgnameonly = os.path.join(videoname, f[idx][:-4])
            # print(imgnameonly, ' rgb---')
            
            tmp_image = ioo.imread(imgname)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
            from_buffer = cv2.imencode('.jpg', tmp_image, encode_param)[1]
            # print('done!')
            fo.create_dataset(imgnameonly, data = from_buffer)
    print('rgb done!')


    for r, d, f in os.walk(opt_u_path, followlinks = False):
        # print('root: ', r)
        # print('dir: ', d)
        # print('files: ', f)
        if len(f) == 0:
            print('empty!')
            continue;

        opt_u_rep = r.replace(opt_u_path, 'u')
        # print(len(r), len(d), len(f))
        
        length = len(f)
        videoname = opt_u_rep
        print('opt u---------------- ', videoname)
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
            # print('idx: ', idx)
            # print(r)
            imgname = os.path.join(r, f[idx])
            # print('********opt | ', imgname)
            imgnameonly = os.path.join(videoname, f[idx][:-4])
            # print(imgnameonly, ' opt---')
            
            tmp_image = ioo.imread(imgname)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
            from_buffer = cv2.imencode('.jpg', tmp_image, encode_param)[1]
            # print('done!')
            fo.create_dataset(imgnameonly, data = from_buffer)
    print('opt u done!')

    for r, d, f in os.walk(opt_v_path, followlinks = False):
        # print('root: ', r)
        # print('dir: ', d)
        # print('files: ', f)
        if len(f) == 0:
            print('empty!')
            continue;

        opt_v_rep = r.replace(opt_v_path, 'v')
        # print(len(r), len(d), len(f))
        
        videoname = opt_v_rep
        print('opt v---------------- ', videoname)
        for idx in range(len(f)):
            # print('idx: ', idx)
            # print(r)
            imgname = os.path.join(r, f[idx])
            # print('********opt | ', imgname)
            imgnameonly = os.path.join(videoname, f[idx][:-4])
            # print(imgnameonly, ' opt---')
            
            tmp_image = ioo.imread(imgname)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
            from_buffer = cv2.imencode('.jpg', tmp_image, encode_param)[1]
            # print('done!')
            fo.create_dataset(imgnameonly, data = from_buffer)
    print('opt v done!')
# fo.close()





        
