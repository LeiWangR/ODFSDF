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

# dataset_name is used for storing the trained model
# option is used to choose the video data modality
# exp_num is used only as an extra term for running different experiments so that the saved models can have different names
# split_num is used when a dataset has many train-test splits
# layers is used to indicate how many layers of weights need to be frozen, the recommendations are 9 and 15?
# batchN is the batch size
# num_class is the total number of video classes in the dataset
# desired_frames is the input temporal dimension
# tr_split indicates the location of the training split csv file
# te_split indicates the location of the test split csv file
# video_hdf5 is the path to the video hdf5 file
# pretrained_rgb and pretrained_opt specify the dir of the pre-trained I3D-rgb and I3D-opt models
# lr_rate is the learning rate (1e-3, 1e-4)

# python3 main.py -exp_num 1 split_num 1 -layers 15 -batchN 32 -num_class 51 -desired_frame 64 -lr_rate 1e-3

parser = argparse.ArgumentParser(description="model codes")
parser.add_argument("-dataset_name","--dataset_name",type=str,default='hmdb51')
parser.add_argument("-option","--option",type=str,default='opt')
parser.add_argument("-exp_num","--exp_num",type=int,default=0)
parser.add_argument("-split_num","--split_num",type=int,default=1)
parser.add_argument("-layers","--layers",type=int,default=15)
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

dropout_keep_prob=0.5
# epochs
num_epochs = 1000000

trained_model = dataset_name + '_' + option + '_layers' + str(layers).zfill(3) + '_F' + str(desired_frame) + '_split' + str(split_num).zfill(2) + '_exp' + str(exp_num).zfill(3) + '.pt'

print('frozen layers: ', layers, ' | dataset: ', dataset_name, ' | split num: ', split_num, ' | frames: ', desired_frame, ' | batch size: ', batchN, ' | option: ', option)

# ----------------------------------------------------------------------------------

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
#--------------------------------------------
# I3D backbone
class I3D_backbone(nn.Module):
    def __init__(self):
        super(I3D_backbone, self).__init__()
        self.i3d_rgb = InceptionI3d(400, in_channels = 3)
        self.i3d_opt = InceptionI3d(400, in_channels = 2)
        self.i3d_rgb = self.i3d_rgb.net_rebuilt()
        self.i3d_opt = self.i3d_opt.net_rebuilt()
        self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7],
                                     stride=(1, 1, 1))
        self.dropout = nn.Dropout(dropout_keep_prob)
        self.logits = Unit3D(in_channels=384+384+128+128, output_channels=num_class,
                             kernel_shape=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits') 
    def forward(self, x, option = option):
        if option == 'rgb' or option == 'depth':
            x = self.i3d_rgb(x)
        else:
            x = self.i3d_opt(x)
        # output dimension of above is 1024 x 8 x 7 x 7
        x = self.avg_pool(x)
        x = self.dropout(x)
        x = self.logits(x)
        x = x.mean(2)
        # print(x.shape)
        x = x.view(-1, num_class)
        # print(x.shape)
        return x

i3d_backbone = I3D_backbone().cuda()
i3d_backbone.apply(weights_init)
if option == 'rgb' or option == 'depth':
    i3d_backbone.i3d_rgb.load_state_dict(torch.load(pretrained_rgb), strict=False)
    ct = 0
    for child in i3d_backbone.i3d_rgb.children():
        ct += 1
        # print(ct)
        # print(child)
        # print('---------')
        if ct < layers:
            # print(ct)
            # print(child)
            for param in child.parameters():
                param.requires_grad = False
else:
    # optical flow
    i3d_backbone.i3d_opt.load_state_dict(torch.load(pretrained_opt), strict=False)
    ct = 0
    for child in i3d_backbone.i3d_opt.children():
        ct += 1
        if ct < layers:
            # print(ct)
            # print(child)
            for param in child.parameters():
                param.requires_grad = False

'''
# original saved file with DataParallel
state_dict = torch.load(trained_model)
# create new OrderedDict that does not contain `module.`
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
# load params
i3d_backbone.load_state_dict(new_state_dict)
'''

i3d_backbone = nn.DataParallel(i3d_backbone)

#----------------------------------------------
# dataset
tr_data = ActionDataset(video_hdf5, tr_info_list, desired_frame, tr_te_flag = 'train', option_flag = option, bb_flag = True, feat_flag = [], feature_dir = feature_hdf5, transforms=transforms.Compose([CenterCrop(), ToTensor()]))
te_data = ActionDataset(video_hdf5, te_info_list, desired_frame, tr_te_flag = 'test',  option_flag = option, bb_flag = True, feat_flag = [], feature_dir = feature_hdf5, transforms=transforms.Compose([CenterCrop(), ToTensor()]))

train_dataloader=DataLoader(tr_data, batch_size=batchN, shuffle=True, drop_last = False)
test_dataloader=DataLoader(te_data,  batch_size=batchN, shuffle=True, drop_last = False)
#---------------------------------------------

pred_criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad,i3d_backbone.parameters()), lr = lr_rate, momentum = 0.9, weight_decay = 1e-7)

temp_acc = 0
# training and testing stages -------------------------
for i in range(num_epochs):
    i3d_backbone.train()
    print('training ---')
    for t, batch_data in enumerate(train_dataloader):
        with torch.no_grad():
            video_seq = Variable(batch_data[option], requires_grad = False).float().cuda()
            label = Variable(batch_data['label'], requires_grad = False).cuda()
        
        N = label.shape[0]
        
        pred = i3d_backbone(video_seq, option)
        optimizer.zero_grad()
        pred_loss = pred_criterion(pred, label.long())
        pred_loss.backward()
        optimizer.step()
        # if t == 2:
        #    break; 
        print('epoch: ', i, 'bat. id: ', t, 'pred- %.3f'%pred_loss.item())

    print('testing ---')
    i3d_backbone.eval()
    
    correct = 0
    total = 0
    with torch.no_grad():
        for t, batch_data in enumerate(test_dataloader):
            video_seq = Variable(batch_data[option], requires_grad = False).float().cuda()
            label = Variable(batch_data['label'], requires_grad = False).cuda()

            N = label.shape[0]

            pred = i3d_backbone(video_seq, option)

            pred_label = torch.max(pred, 1)[1].cuda()

            total += label.size(0)
            correct += (pred_label == label).sum().item()
            
            acc = 100*correct/total
            print('epoch: ', i,  'bat. id: ', t, ' | classifi. acc.: %.3f'%acc)
            # if t == 2:
            #     break;
        if temp_acc < acc:
            torch.save(i3d_backbone.state_dict(), trained_model)
            temp_acc = acc
        
        print('======== Best Acc.: ', temp_acc)

