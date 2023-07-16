import h5py
import json
import numpy as np

# give a sample video name
videoname = 's20-d07-cam-002_05491_052'
# load saved hdf5 file
filename = 'sample_mpii_per_bb_feat.hdf5'
with h5py.File(filename, 'r') as f:
    # using [()] for value
    count = f['rgb/' + videoname + '/count'][()]
    print('count: ', count)
    for ii in range(count):
        dic_info = json.loads(f['rgb/' + videoname + '/frame' + str(ii).zfill(8)][()])
        num = dic_info['num_detections']
        print('num: ', num)
        if num == 0:
            print('no bb feature stored!')
        else:       
            bb_features = np.asarray(dic_info['bb_feature'])
            print('feat: ', bb_features, ' | shape: ', bb_features.shape)
        
        
