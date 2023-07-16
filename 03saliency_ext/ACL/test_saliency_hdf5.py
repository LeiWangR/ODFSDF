import h5py
import json
import numpy as np

# give a sample video name
videoname = 's20-d07-cam-002_05496_038'
# load saved hdf5 file
filename = 'saliency_attention_sample.hdf5'
with h5py.File(filename, 'r') as f:
    dic_info = json.loads(f['rgb/' + videoname][()])
    saliency = np.asarray(dic_info['saliency'])
    attention = np.asarray(dic_info['attention'])
    print('saliency: ', saliency.shape, ' | attention: ', attention.shape)
    print('------------')
    print(saliency)
    print('------------')
    print(attention)
