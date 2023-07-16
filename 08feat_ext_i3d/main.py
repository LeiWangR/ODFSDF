import torch
import torch.nn as nn
import torch.nn.functional as F
from action_dataloader import ActionDataset, ToTensor, CenterCrop
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.autograd import Variable
from pytorch_i3d import InceptionI3d, Unit3D
from collections import OrderedDict
import math
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3'

# dataset_name is used for storing the feature extraction
# option is used to choose the video data modality
# exp_num is used only as an extra term for running different experiments so that the saved models can have different names
# split_num is used when a dataset has many train-test splits
# layers is used to indicate how many layers of weights need to be frozen, the recommendation is 19 for feature extraction
# batchN is the batch size
# num_class is the total number of video classes in the dataset
# desired_frames is the input temporal dimension
# tr_split indicates the location of the training split csv file
# te_split indicates the location of the test split csv file
# video_hdf5 is the path to the video hdf5 file
# pretrained_rgb and pretrained_opt specify the dir of the pre-trained I3D-rgb and I3D-opt models
# lr_rate is the learning rate (1e-3, 1e-4) --- not used in the feature extraction

parser = argparse.ArgumentParser(description="feature extraction")
parser.add_argument("-dataset_name","--dataset_name",type=str,default='hmdb51')
parser.add_argument("-option","--option",type=str,default='opt')
parser.add_argument("-exp_num","--exp_num",type=int,default=0)
parser.add_argument("-split_num","--split_num",type=int,default=1)
parser.add_argument("-layers","--layers",type=int,default=19)
parser.add_argument("-batchN","--batchN",type=int,default=32)
parser.add_argument("-num_class","--num_class",type=int,default=51)
parser.add_argument("-desired_frame","--desired_frame",type=int,default=64)
parser.add_argument("-tr_split","--tr_split",type=str,default='tr.csv')
parser.add_argument("-te_split","--te_split",type=str,default='te.csv')
parser.add_argument("-video_hdf5","--video_hdf5",type=str,default='.hdf5')
parser.add_argument("-pretrained_rgb","--pretrained_rgb",type=str,default='pretrained_I3D/rgb_imagenet.pt')
parser.add_argument("-pretrained_opt","--pretrained_opt",type=str,default='pretrained_I3D/flow_imagenet.pt')
parser.add_argument("-feature_hdf5","--feature_hdf5",type=str,default='.hdf5')
parser.add_argument("-lr_rate","--lr_rate",type=float,default=1e-3)

args = parser.parse_args()

dataset_name = args.dataset_name
option = args.option
exp_num = args.exp_num
split_num = args.split_num
layers = args.layers
batchN = args.batchN
num_class = args.num_class
desired_frame = args.desired_frame
tr_info_list = args.tr_split
te_info_list = args.te_split
video_hdf5 = args.video_hdf5
pretrained_rgb = args.pretrained_rgb
pretrained_opt = args.pretrained_opt
feature_hdf5 = args.feature_hdf5
lr_rate = args.lr_rate

tr_feature_ext = dataset_name + '_' + option + '_F' + str(desired_frame) + '_split' + str(split_num).zfill(2) + '_tr.pth'
te_feature_ext = dataset_name + '_' + option + '_F' + str(desired_frame) + '_split' + str(split_num).zfill(2) + '_te.pth'

dropout_keep_prob=0.5
# epochs
num_epochs = 1

print('frozen layers: ', layers, ' | dataset: ', dataset_name, ' | split num: ', split_num, ' | frames: ', desired_frame, ' | batch size: ', batchN, ' | option: ', option)

class TwoStreamNet(nn.Module):
    def __init__(self):
        super(TwoStreamNet, self).__init__()
        self.i3d_rgb = InceptionI3d(400, in_channels = 3)
        self.i3d_opt = InceptionI3d(400, in_channels = 2)

    def forward(self, x, option):
        if option == 'rgb':
            x = self.i3d_rgb(x)
        else:
            x = self.i3d_opt(x)
        # print(x_rgb.shape)
        # print(x_opt.shape)

        return x

#-------------------------------------------
# weights initialization
def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
        if m.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(m.bias, -bound, bound)
    elif isinstance(m, nn.Conv1d):
        nn.init.normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)

#----------------------------------------------
# dataset
tr_data = ActionDataset(video_hdf5, tr_info_list, desired_frame, tr_te_flag = 'train', option_flag = option, bb_flag = True, feat_flag = [], feature_dir = feature_hdf5, transforms=transforms.Compose([CenterCrop(), ToTensor()]))
te_data = ActionDataset(video_hdf5, te_info_list, desired_frame, tr_te_flag = 'test',  option_flag = option, bb_flag = True, feat_flag = [], feature_dir = feature_hdf5, transforms=transforms.Compose([CenterCrop(), ToTensor()]))

train_dataloader=DataLoader(tr_data, batch_size=batchN, shuffle=False, drop_last = False)
test_dataloader=DataLoader(te_data,  batch_size=batchN, shuffle=False, drop_last = False)
#---------------------------------------------
net = TwoStreamNet()
net.apply(weights_init)

if option == 'rgb':
    net.i3d_rgb.load_state_dict(torch.load(pretrained_rgb))
    for params in net.i3d_rgb.parameters():
        params.requires_grad = False
else:
    net.i3d_opt.load_state_dict(torch.load(pretrained_opt))
    for params in net.i3d_opt.parameters():
        params.requires_grad = False

net.cuda()
net = nn.DataParallel(net)
#---------------------------------------------

for i in range(num_epochs):
    # Testing---------------------------------  
    net.eval()
    with torch.no_grad():
        x_tensor = torch.cuda.FloatTensor(1, 400).fill_(0)
        for t, sample_batched in enumerate(train_dataloader):  
            seq_video = Variable(sample_batched[option], requires_grad = False).float().cuda()
            label = Variable(sample_batched['label'], requires_grad = False).cuda()
            x_seq = net(seq_video, option)
            x_tensor = torch.cat((x_tensor, x_seq), 0)                
            print(t, '-----------------')
        x_tensor = x_tensor[1:, :]
        print('x_tensor: ', x_tensor.shape)
        torch.save(x_tensor, tr_feature_ext)

    with torch.no_grad():
        x_tensor = torch.cuda.FloatTensor(1, 400).fill_(0)
        for t, sample_batched in enumerate(test_dataloader):  
            seq_video = Variable(sample_batched[option], requires_grad = False).float().cuda()
            label = Variable(sample_batched['label'], requires_grad = False).cuda()
            x_seq = net(seq_video, option)
            x_tensor = torch.cat((x_tensor, x_seq), 0)                
            print('-----------------', t)
        x_tensor = x_tensor[1:, :]
        print('x_tensor: ', x_tensor.shape)
        torch.save(x_tensor, te_feature_ext)

