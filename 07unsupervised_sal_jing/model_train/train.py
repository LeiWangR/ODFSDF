import torch
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import pdb, os, argparse
from datetime import datetime

from model.vgg_models import Back_VGG
from model.ResNet_models import Back_ResNet
from data import get_loader
from utils import clip_gradient, adjust_lr
import pytorch_ssim
import pytorch_iou
import smoothness
import os
from scipy import misc
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=100, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batchsize', type=int, default=20, help='training batch size')
parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--is_ResNet', type=bool, default=False, help='VGG or ResNet backbone')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
opt = parser.parse_args()

print('Learning Rate: {} ResNet: {}'.format(opt.lr, opt.is_ResNet))
# build models
if opt.is_ResNet:
    model = Back_ResNet()
else:
    model = Back_VGG()

model.cuda()
params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)

image_root = '/home/jing-zhang/jing_file/RGB_sal_dataset/train/DUTS/img/'
gt_root = '/home/jing-zhang/jing_file/CVPR2020/single_noise/data/DUTS/normed/RBD/'
train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
total_step = len(train_loader)

CE = torch.nn.BCEWithLogitsLoss()
ssim_loss = pytorch_ssim.SSIM(window_size=11,size_average=True)
iou_loss = pytorch_iou.IOU(size_average=True)
smooth_loss = smoothness.smoothness_loss(size_average=True)

def visualize_bnd(pred):
    edge_x = gradient_x(pred)
    edge_y = gradient_y(pred)
    edge_xy = torch.tanh(torch.pow(torch.pow(edge_x, 2) + torch.pow(edge_y, 2) + 0.00001, 0.5))

    for kk in range(edge_xy.shape[0]):
        pred_edge_kk = edge_xy[kk, :, :, :]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk = (pred_edge_kk - pred_edge_kk.min()) / (pred_edge_kk.max() - pred_edge_kk.min() + 1e-8)
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}.png'.format(kk)
        misc.imsave(save_path + name, pred_edge_kk)

def generate_gaussian_mask(batch_cur):

    gaussian_map = Variable(torch.rand(batch_cur,1,opt.trainsize,opt.trainsize))
    gaussian_map = torch.gt(gaussian_map,0.5)
    gaussian_map = gaussian_map.float()
    return gaussian_map

def train(train_loader, model, optimizer, epoch):
    model.train()
    for i, pack in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        images, gts = pack
        images = Variable(images)
        gts = Variable(gts)
        images = images.cuda()
        gts = gts.cuda()


        dets = model(images)
        dets_prob = torch.sigmoid(dets)

        ## generate gaussian matrix
        mask_cur = generate_gaussian_mask(dets_prob.shape[0])
        mask_cur = mask_cur.cuda()

        intensity_img = torch.mean(images, 1, keepdim=True)

        img_size = images.size(2) * images.size(3) * images.size(0)
        ratio = img_size / torch.sum(mask_cur)


        dets_prob = torch.sigmoid(dets) * mask_cur
        bce = ratio * CE(dets, gts)
        smoothLoss_cur = smooth_loss(torch.sigmoid(dets), torch.sigmoid(intensity_img))
        # ssim2 = 1 - ssim_loss(torch.sigmoid(dets), gts)
        # iou2 = iou_loss(torch.sigmoid(dets), gts)
        loss = bce+0.1*smoothLoss_cur
        loss.backward()

        clip_gradient(optimizer, opt.clip)
        optimizer.step()

        if i % 10 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:0.4f}'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step, loss.data))

    if opt.is_ResNet:
        save_path = 'models/Resnet/'
    else:
        save_path = 'models/VGG/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if (epoch+1) % 5 == 0:
        torch.save(model.state_dict(), save_path + 'vgg_pce' + '_%d'  % epoch  + '.pth')

print("Let's go!")
for epoch in range(1, opt.epoch):
    adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
    train(train_loader, model, optimizer, epoch)
