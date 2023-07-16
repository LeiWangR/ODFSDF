from soundnet8 import SoundNet8_pytorch
import librosa
import torch
import numpy as np
import argparse
import pandas as pd
import os.path
import pickle

# python main.py -trteset 'training'
# python main.py -trteset 'validation'

parser = argparse.ArgumentParser(description="soundnet8 pool5 feat. ext.")
parser.add_argument("-trteset","--trteset",type=str,default='training')
# parser.add_argument("-batchsize","--batchsize",type=int,default=1)

args = parser.parse_args()

sounddir = '/media/wan305/eebb5ee7-5e62-3982-bd79-312a3ff3b76f/research/Moments_in_Time_256x256_30fps/'
sepfolder = sounddir + args.trteset + '_audio/'

audiofeat_save = '/media/wan305/eebb5ee7-5e62-3982-bd79-312a3ff3b76f/research/' + 'moments_in_time_soundnet8_pool5_feat_' + args.trteset + '.pickle'

infolist = '/media/wan305/eebb5ee7-5e62-3982-bd79-312a3ff3b76f/research/Moments_in_Time_256x256_30fps/' + args.trteset + 'Set.csv'

dfinfo = pd.read_csv(infolist, header = None)
# pretrained model
pytorch_param_path = './sound8.pth'
model = SoundNet8_pytorch()
model.load_state_dict(torch.load(pytorch_param_path))

# NOTE: Load an audio as the same format in soundnet
# 1. Keep original sample rate (which conflicts their own paper)
# 2. Use first channel in multiple channels
# 3. Keep range in [-256, 256]

def load_audio(audio_path, sr=None):
    # By default, librosa will resample the signal to 22050Hz(sr=None). And range in (-1., 1.)
    sound_sample, sr = librosa.load(audio_path, sr=sr, mono=False)

    return sound_sample * 256, sr

# create a dictionary to store (video_name, sound_feat / False) -- False for no audio info.
sound_feat_dic = {}

for ii in range(dfinfo.shape[0]):
    actiondir = dfinfo.iloc[ii, 0]
    print(ii, '-- | ', actiondir[:-4])
    soundpath = sepfolder + actiondir[:-4] + '.wav'
    # print('-- ', soundpath)
    if os.path.isfile(soundpath):
        soundinfo, sr = load_audio(soundpath)
        if np.sum(soundinfo) == 0.0 and np.max(soundinfo) == 0.0 and np.min(soundinfo) == 0.0:
            # print('------------------------- no sound info.')
            sound_feat_dic[actiondir] = False
        else:
            # print('shape: ', soundinfo.shape, ' | max: ', np.max(soundinfo), ' | min: ', np.min(soundinfo), ' | mean: ', np.mean(soundinfo))
            # input dim.
            x = torch.from_numpy(soundinfo).view(1,1,-1,1)
            feat = model.extract_pool5(x)
            # torch.squeeze() function remove all extra 1 dim.
            x = torch.squeeze(feat)
            # print(x.shape)
            sound_feat_dic[actiondir] = x.data.cpu().numpy()
    else:
        # print('xxx file not exists')
        sound_feat_dic[actiondir] = False

    # if ii == 1000:
    #     break;

# print(sound_feat_dic)
# save dictionary
with open(audiofeat_save, 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(sound_feat_dic, f, pickle.HIGHEST_PROTOCOL)

'''
# to load the pickle file using the following codes
dicfeat = pickle.load(open(audiofeat_save, "rb"))
print(dicfeat)
print('-------------------------')
print(dicfeat['swinging/giphy-LrPz8BdPDraso_1.mp4'])
print('xxxxxxxxxx')
print(dicfeat['howling/vine-A-howling-puppy-is-a-happy-puppy-MegVUezi732_1.mp4'])
'''
