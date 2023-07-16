import pickle
import glob

count = len(glob.glob('./*.pickle'))

# print(count)
train_dict_feat={}
for ii in range(count):
    picklename = './moments_in_time_soundnet8_pool5_feat_training_p' + str(ii + 1) + '.pickle'
    dicfeat = pickle.load(open(picklename, "rb"))
    # print(len(dicfeat))
    # print('xxxxxx')
    # if ii == 0:
    #     print(dicfeat)
    train_dict_feat = {**train_dict_feat, **dicfeat}

print(len(train_dict_feat))

