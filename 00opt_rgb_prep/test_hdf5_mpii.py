import h5py
import cv2
# give a sample video name
videoname = 's20-d07-cam-002_05495_042'
# load saved hdf5 file
filename ='mpii_sample_bb.hdf5'
f = h5py.File(filename, 'r')

# using [()] for value
count = f['rgb/' + videoname + '/count'][()]
# height = f['u/' + videoname + '/height'][()]
# width = f['u/' + videoname + '/width'][()]


# print('count: ', count, ' | height: ', height, ' width: ', width)

for ii in range(count):
    # using [:] for image stream
    frame = f['rgb/' + videoname + '/frame' + str(ii).zfill(8)][:]
    # print(frame.shape)
    # for RGB video
    # im = cv2.imdecode(frame, cv2.IMREAD_COLOR)
    # ------ for optical flow components ------
    im = cv2.imdecode(frame, cv2.IMREAD_GRAYSCALE)
    # -----------------------------------------
    print(im.shape)
    print('--------', ii)
    # print(im)
f.close()

