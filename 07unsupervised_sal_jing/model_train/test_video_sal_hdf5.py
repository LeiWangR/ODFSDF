import h5py
import cv2

videoname = 's20-d07-cam-002_05490_038'
# videoname = 's20-d07-cam-002_05494_039'

filename = 'video_saliency_sample_jing.hdf5'
f = h5py.File(filename, 'r')
count = f['rgb/' + videoname + '/count'][()]
print(count)

for ii in range(count):
    frame_mini_sal = f['rgb/' + videoname + '/frame' + str(ii).zfill(8) + '/mini_saliency'][:]
    frame_sal = f['rgb/' + videoname + '/frame' + str(ii).zfill(8) + '/saliency'][:]
    # grayscale for mini-sal and saliency image
    im_mini_sal = cv2.imdecode(frame_mini_sal, cv2.IMREAD_GRAYSCALE)
    im_sal = cv2.imdecode(frame_sal, cv2.IMREAD_GRAYSCALE)
    print(ii, '---', im_mini_sal.shape, 'xxx', im_sal.shape)
f.close()

