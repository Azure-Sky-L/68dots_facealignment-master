from __future__ import print_function
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, utils, models
from torch.autograd import Variable
from torch.nn import DataParallel
import numpy as np

import time
import random
import cv2

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default= 80, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed) #set cpu random seed for weight initial
if args.cuda:
    torch.cuda.manual_seed(args.seed)#set gpu random seed, if you use more than one gpu, please use torch.cuda.manual_seed_all() to set random seed for all gpus

import  data_loader as loader
print('|----------------loadering data-------------------|')
records = loader.getDataFromList('256_1k_train_data/data_tag.list')
#test_data = loader.getDataFromList('Data_test/data_tag.list')
train_dataset = loader.DataLoader(records, 256, 68)
#test_dataset = loader.DataLoader(test_data, 128, 68)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 16, shuffle=True,num_workers=8,pin_memory=True)
#test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1,shuffle=True,num_workers=8,pin_memory=True)

model_save = 'model'
if not os.path.exists(model_save):
    os.makedirs(model_save)
import hg_net as net
#import fy_net as net
model = net.FAN(2)
print('model infos    ....    ', model)
model.cuda()# make model on gpu

optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)
# fy_net
#scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [2, 7, 8],gamma=0.1)
# hg_net
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [20, 40, 60],gamma=0.1)
mse_criterion = nn.MSELoss().cuda()
sl1_criterion = nn.SmoothL1Loss().cuda()

def train(epoch):
    model.train()
    ok = True
    for batch_idx, data in enumerate(train_loader):
        if args.cuda:
            img, tag, dots = data
        optimizer.zero_grad()
        img, tag, dots = Variable(img).cuda(async=True), Variable(tag).cuda(async=True), Variable(dots).cuda(async=True)
        outs, reg_outs = model(img)
        losses = []
        Loss = []

        for out in outs:
            out = F.sigmoid(out)
            loss = mse_criterion(out, tag)
            losses.append(loss)

        for reg in reg_outs:
            loss = sl1_criterion(reg, dots)
            losses.append(loss)
            loss = mse_criterion(reg, dots) / 2.0
            losses.append(loss)
        losses[0] *= 5.0
        loss = (losses[0]).cuda()
        for li in range(1, len(outs)):
            loss = loss + 5.0 * (losses[li]).cuda()
        for li in range(len(outs), len(losses)):
            loss = loss + 1.0 * (losses[li]).cuda()
 
        loss.backward()
        optimizer.step()
        # print(batch_idx)       
        if (batch_idx) % 3388 == 0:
            print('Train Epoch: {} lr: {}  Loss: {:.6f}'.format(epoch ,scheduler.get_lr()[0],loss.data[0]))
            for lsi, aloss in enumerate(losses):
                print('\t\t\t\t{:.1f} part loss: {:.6f}'.format(lsi, aloss.data[0]))
            print('saving model model/model_' + str(epoch) + '_' + str(batch_idx) + '.pkl')
            torch.save(model.state_dict(),'model/model_' + str(epoch) + '_' + str(batch_idx) + '.pkl')
test_imgs = './test_imgs/'
def test(epoch):
    model.eval()
    ans = 0
    for batch_idx, data in enumerate(test_loader):
        if args.cuda:
            img, tag, dots = data
        optimizer.zero_grad()
        img, tag = Variable(img).cuda(async=True), Variable(tag).cuda(async=True)
        outs, reg_outs = model(img)
        show_img = np.zeros((64,64,3), np.float32)
        for k, out in enumerate(outs):
            out = F.sigmoid(out)
            out = out.cpu().data
            for i in range(out.shape[1]):
                show_img[:,:,k] += out[0,i,:,:]

            show_img[:,:,k] = show_img[:,:,k] / np.max(show_img[:,:,k]) * 255.0
        
        show_img = show_img.astype(np.uint8)

        dots *= 4.0
        dots += 32.0
        dots = np.reshape(dots, (-1,2))
        for i in range(0, dots.shape[0]):
            cv2.rectangle(show_img, (int(dots[i,0]) -1, int(dots[i,1]) -1), (int(dots[i,0]) + 1, int(dots[i,1]) + 1), (0,0,255))
        
        colors = [(0,255,255), (255,255,0)]
        for k,reg in enumerate(reg_outs):
            reg = reg.cpu().data
            reg = np.array(reg)
            reg *= 4.0
            reg += 32.0
            reg = np.reshape(reg, (-1,2))
            for i in range(0, reg.shape[0]): 
                cv2.rectangle(show_img, (int(reg[i,0]) -1, int(reg[i,1]) -1), (int(reg[i,0]) + 1, int(reg[i,1]) + 1), colors[k])

        ans = ans + 1
        cv2.imwrite(test_imgs + str(ans) + '.jpg', show_img)    

for epoch in range(1, args.epochs + 1):
    scheduler.step()
    train(epoch)
    if epoch>80:
        test(epoch)
