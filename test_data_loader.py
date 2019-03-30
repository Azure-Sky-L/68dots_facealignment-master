import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
import numpy as np
import cv2

import copy
import random
import os
import math

def getDataFromList(data_list_path):
    records = []
    num = 0
    with open(data_list_path) as f:
        for line in f:
            num += 1
            sline = line.strip('\n').split(' ')
            img_name = sline[0]
            if not os.path.exists(img_name):
                print(img_name)
                continue
            dots = []
            for i in range(1, len(sline)):
                dots.append(float(sline[i]))
            dots = np.array(dots)
            dots = np.reshape(dots,(-1,2))
            records.append((img_name, dots))
    assert len(records) > 0
    return records

def _gaussian(
        size=3, sigma=0.25, amplitude=1, normalize=False, width=None,
        height=None, sigma_horz=None, sigma_vert=None, mean_horz=0.5,
        mean_vert=0.5):
    # handle some defaults
    if width is None:
        width = size
    if height is None:
        height = size
    if sigma_horz is None:
        sigma_horz = sigma
    if sigma_vert is None:
        sigma_vert = sigma
    center_x = mean_horz * width + 0.5
    center_y = mean_vert * height + 0.5
    gauss = np.empty((height, width), dtype=np.float32)
    # generate kernel
    for i in range(height):
        for j in range(width):
            gauss[i][j] = amplitude * math.exp(-(math.pow((j + 1 - center_x) / (
                sigma_horz * width), 2) / 2.0 + math.pow((i + 1 - center_y) / (sigma_vert * height), 2) / 2.0))
    if normalize:
        gauss = gauss / np.sum(gauss)
    return gauss


def draw_gaussian(image, point, sigma):
    # Check if the gaussian is inside
    ul = [math.floor(point[0] - 3 * sigma), math.floor(point[1] - 3 * sigma)]
    br = [math.floor(point[0] + 3 * sigma), math.floor(point[1] + 3 * sigma)]
    if (ul[0] > image.shape[1] or ul[1] > image.shape[0] or br[0] < 1 or br[1] < 1):
        return image
    size = 6 * sigma + 1
    g = _gaussian(size)
    g_x = [int(max(1, -ul[0])), int(min(br[0], image.shape[1])) - int(max(1, ul[0])) + int(max(1, -ul[0]))]
    g_y = [int(max(1, -ul[1])), int(min(br[1], image.shape[0])) - int(max(1, ul[1])) + int(max(1, -ul[1]))]
    img_x = [int(max(1, ul[0])), int(min(br[0], image.shape[1]))]
    img_y = [int(max(1, ul[1])), int(min(br[1], image.shape[0]))]
    assert (g_x[0] > 0 and g_y[1] > 0)
    try:
        image[img_y[0] - 1:img_y[1], img_x[0] - 1:img_x[1]
          ] = image[img_y[0] - 1:img_y[1], img_x[0] - 1:img_x[1]] + g[g_y[0] - 1:g_y[1], g_x[0] - 1:g_x[1]]
    except:
        print('|-----------|')
    image[image > 1] = 1
    return image

class DataLoader(torch.utils.data.Dataset):  #just for train

    def __init__(self,records_data, input_size, output_num):
        super(DataLoader, self).__init__()
        self.data_info = records_data
        self.data_size = len(records_data)
        self.target_size = input_size
        self.output_num = output_num

    def __getitem__(self, index):
        #if index  == 0:
        #    random.shuffle(self.data_info)

        img_name, dots = self.data_info[index]

        im = cv2.imread(img_name)
        assert (im is not None)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        #assert (im.shape[0] == self.target_size and im.shape[1] == self.target_size)
        

        im = im.astype(np.float32)
        im /= 256.0
        input_img = np.reshape(im, (1, im.shape[0], im.shape[1]))
        

        #show_img = np.zeros((self.target_size/4, self.target_size/4), np.float32)
        reg_dots = dots
        dots = copy.deepcopy(dots) / 4.0
        heat_maps = np.zeros((self.output_num, self.target_size/4, self.target_size/4), np.float32)
        for i in range(0,self.output_num):
            heat_maps[i,:,:] = draw_gaussian(heat_maps[i,:,:],(dots[i][0], dots[i][1]), 1)
            
            #show_img += heat_maps[i,:,:]
        
        #show_img = show_img / np.max(show_img) * 255.0
        #show_img = show_img.astype(np.uint8)
        #cv2.imshow('show_img', show_img)
        #cv2.waitKey()
        # 128
        # dots -= 16.0
        # dots /= 2.0
        # 256
        dots -= 32.0
        dots /= 4.0
        dots = dots.astype(np.float32)
        reg_dots = np.reshape(dots, (136, ))
        # train
        # return input_img, heat_maps, reg_dots
        # test
        return input_img, heat_maps, reg_dots, img_name


    def __len__(self):
        return self.data_size

