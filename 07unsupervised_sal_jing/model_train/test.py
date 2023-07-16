import torch
import torch.nn.functional as F

import numpy as np
import pdb, os, argparse
# from scipy import misc
import imageio
from model.vgg_models import Back_VGG
from model.ResNet_models import Back_ResNet
from data import test_dataset
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--is_ResNet', type=bool, default=False, help='VGG or ResNet backbone')
opt = parser.parse_args()

dataset_path = '/home/wan305/research/ongoing/resources/unsupervised_sal/model_train/'
#dataset_path = '/home/jing-zhang/jing_file/RGB_sal_dataset/train/'
#gt_path = '/home/jing-zhang/jing_file/RGB_sal_dataset/test/gt/'

if opt.is_ResNet:
    model = Back_ResNet()
    model.load_state_dict(torch.load('CPD-R.pth'))
else:
    model = Back_VGG()
    print(1)
    model.load_state_dict(torch.load('./models/vgg_pce_99.pth', map_location=torch.device('cpu')))
    print(2)
# model.cuda()
model.eval()

test_datasets = ['Diving-Side-005']
#test_datasets = ['SOD','ECSSD','DUT','DUTS_Test','PASCAL','THUR']

for dataset in test_datasets:
    if opt.is_ResNet:
        save_path = './results/ResNet50/' + dataset + '/'
    else:
        save_path = './results/VGG16/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = dataset_path + dataset + '/'
    #gt_root = gt_path + dataset + '/'
    test_loader = test_dataset(image_root, opt.testsize)
    for i in range(test_loader.size):
        print(i)
        image, HH,WW,name = test_loader.load_data()
        # image = image.cuda()
        res = model(image)
        # print(res.shape, '--- before apply upsampling')
        lei_test = np.squeeze(res)
        print(lei_test.shape, '*** what Lei needs')
        # print(lei_test)
        # print('H: ', HH, ' | W: ', WW)
        res = F.upsample(res, size=[WW,HH], mode='bilinear', align_corners=False)
        # print(res.shape, '--- before sigmoid')
        res = res.sigmoid().data.cpu().numpy().squeeze()
        # print(res.shape, '--- after sigmoid & squeeze')
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        # misc.imsave(save_path+name, res)
        res = (res*255).astype(np.uint8)
        # print('max: ', np.max(res), 'min: ', np.min(res))
        imageio.imwrite(save_path+name, res)
