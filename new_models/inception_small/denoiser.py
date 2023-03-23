import foolbox
import numpy as np
from tqdm import tqdm
import warnings
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch
import torch.optim as optim
import random
import time
import os
import sys
import argparse

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import math


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        # nn.init.uniform(m.weight.data, 1.0, 0.02)
        m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
        nn.init.constant(m.bias.data, 0.0)

def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    return (PSNR/Img.shape[0])

def data_augmentation(image, mode):
    out = np.transpose(image, (1,2,0))
    if mode == 0:
        # original
        out = out
    elif mode == 1:
        # flip up and down
        out = np.flipud(out)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(out)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(out)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(out, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(out, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(out, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(out, k=3)
        out = np.flipud(out)
    return np.transpose(out, (2,0,1))


class DnCNN(nn.Module):
    def __init__(self, channels, num_of_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
    def forward(self, x):
        out = self.dncnn(x)
        return out


class DUNET(nn.Module):
    def __init__(self):
        super(DUNET, self).__init__()
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

        DIM = 4

        forward_block1 = nn.Sequential(
            nn.Conv2d(3, 8 * DIM, 3, stride=1, padding=1),
            nn.BatchNorm2d(8 * DIM),
            nn.ReLU(True),
            nn.Conv2d(8 * DIM, 16 * DIM, 3, stride=1, padding=1),
            nn.BatchNorm2d(16 * DIM),
            nn.ReLU(True),
        )

        forward_block2 = nn.Sequential(
            nn.Conv2d(16 * DIM, 21 * DIM, 3, stride=2, padding=1),
            nn.BatchNorm2d(21 * DIM),
            nn.ReLU(True),
            nn.Conv2d(21 * DIM, 26 * DIM, 3, stride=1, padding=1),
            nn.BatchNorm2d(26 * DIM),
            nn.ReLU(True),
            nn.Conv2d(26 * DIM, 32 * DIM, 3, stride=1, padding=1),
            nn.BatchNorm2d(32 * DIM),
            nn.ReLU(True),
        )

        forward_block3 = nn.Sequential(
            nn.Conv2d(32 * DIM, 42 * DIM, 3, stride=2, padding=1),
            nn.BatchNorm2d(42 * DIM),
            nn.ReLU(True),
            nn.Conv2d(42 * DIM, 52 * DIM, 3, stride=1, padding=1),
            nn.BatchNorm2d(52 * DIM),
            nn.ReLU(True),
            nn.Conv2d(52 * DIM, 64 * DIM, 3, stride=1, padding=1),
            nn.BatchNorm2d(64 * DIM),
            nn.ReLU(True),
        )

        forward_block4 = nn.Sequential(
            nn.Conv2d(64 * DIM, 64 * DIM, 3, stride=2, padding=1),
            nn.BatchNorm2d(64 * DIM),
            nn.ReLU(True),
            nn.Conv2d(64 * DIM, 64 * DIM, 3, stride=1, padding=1),
            nn.BatchNorm2d(64 * DIM),
            nn.ReLU(True),
            nn.Conv2d(64 * DIM, 64 * DIM, 3, stride=1, padding=1),
            nn.BatchNorm2d(64 * DIM),
            nn.ReLU(True),
        )

        forward_block5 = nn.Sequential(
            nn.Conv2d(64 * DIM, 64 * DIM, 3, stride=2, padding=1),
            nn.BatchNorm2d(64 * DIM),
            nn.ReLU(True),
            nn.Conv2d(64 * DIM, 64 * DIM, 3, stride=1, padding=1),
            nn.BatchNorm2d(64 * DIM),
            nn.ReLU(True),
            nn.Conv2d(64 * DIM, 64 * DIM, 3, stride=1, padding=1),
            nn.BatchNorm2d(64 * DIM),
            nn.ReLU(True),
        )

        backward_block1 = nn.Sequential(
            nn.Conv2d(128 * DIM, 108 * DIM, 3, stride=1, padding=1),
            nn.BatchNorm2d(108 * DIM),
            nn.ReLU(True),
            nn.Conv2d(108 * DIM, 88 * DIM, 3, stride=1, padding=1),
            nn.BatchNorm2d(88 * DIM),
            nn.ReLU(True),
            nn.Conv2d(88 * DIM, 64 * DIM, 3, stride=1, padding=1),
            nn.BatchNorm2d(64 * DIM),
            nn.ReLU(True),
        )

        backward_block2 = nn.Sequential(
            nn.Conv2d(128 * DIM, 108 * DIM, 3, stride=1, padding=1),
            nn.BatchNorm2d(108 * DIM),
            nn.ReLU(True),
            nn.Conv2d(108 * DIM, 88 * DIM, 3, stride=1, padding=1),
            nn.BatchNorm2d(88 * DIM),
            nn.ReLU(True),
            nn.Conv2d(88 * DIM, 64 * DIM, 3, stride=1, padding=1),
            nn.BatchNorm2d(64 * DIM),
            nn.ReLU(True),
        )

        backward_block3 = nn.Sequential(
            nn.Conv2d(96 * DIM, 76 * DIM, 3, stride=1, padding=1),
            nn.BatchNorm2d(76 * DIM),
            nn.ReLU(True),
            nn.Conv2d(76 * DIM, 56 * DIM, 3, stride=1, padding=1),
            nn.BatchNorm2d(56 * DIM),
            nn.ReLU(True),
            nn.Conv2d(56 * DIM, 32 * DIM, 3, stride=1, padding=1),
            nn.BatchNorm2d(32 * DIM),
            nn.ReLU(True),
        )

        backward_block4 = nn.Sequential(
            nn.Conv2d(48 * DIM, 38 * DIM, 3, stride=1, padding=1),
            nn.BatchNorm2d(38 * DIM),
            nn.ReLU(True),
            nn.Conv2d(38 * DIM, 28 * DIM, 3, stride=1, padding=1),
            nn.BatchNorm2d(28 * DIM),
            nn.ReLU(True),
            nn.Conv2d(28 * DIM, 16 * DIM, 3, stride=1, padding=1),
            nn.BatchNorm2d(16 * DIM),
            nn.ReLU(True),
        )

        self.forward_block1 = forward_block1
        self.forward_block2 = forward_block2
        self.forward_block3 = forward_block3
        self.forward_block4 = forward_block4
        self.forward_block5 = forward_block5
        self.backward_block1= backward_block1
        self.backward_block2= backward_block2
        self.backward_block3= backward_block3
        self.backward_block4= backward_block4

        self.resize_layer_1 = nn.AdaptiveAvgPool2d((2*DIM, 2*DIM))
        self.resize_layer_2 = nn.AdaptiveAvgPool2d((4*DIM, 4*DIM))
        self.resize_layer_3 = nn.AdaptiveAvgPool2d((8*DIM, 8*DIM))
        self.resize_layer_4 = nn.AdaptiveAvgPool2d((16*DIM, 16*DIM))

        self.last_layer = nn.Conv2d(16 * DIM, 3, 1, stride=1)

    def forward(self, x):
        x=  x.cuda()
        
        

        x_1 = self.forward_block1(x)
        x_2 = self.forward_block2(x_1)
        x_3 = self.forward_block3(x_2)
        x_4 = self.forward_block4(x_3)
        x_5 = self.forward_block5(x_4)


        #backward
        
        x_r_1_temp_1 = self.resize_layer_1(x_5)
        x_r_1_temp_2 = torch.cat((x_4, x_r_1_temp_1), 1)
        x_r_1 = self.backward_block1(x_r_1_temp_2)

        # print("x_r_1", x_r_1[0][0])

        x_r_2_temp_1 = self.resize_layer_2(x_r_1)
        x_r_2_temp_2 = torch.cat((x_3, x_r_2_temp_1), 1)
        x_r_2 = self.backward_block2(x_r_2_temp_2)

        # print("x_r_2", x_r_2[0][0])

        x_r_3_temp_1 = self.resize_layer_3(x_r_2)
        x_r_3_temp_2 = torch.cat((x_2, x_r_3_temp_1), 1)
        x_r_3 = self.backward_block3(x_r_3_temp_2)

        # print("x_r_3", x_r_3[0][0])

        x_r_4_temp_1 = self.resize_layer_4(x_r_3)
        x_r_4_temp_2 = torch.cat((x_1, x_r_4_temp_1), 1)
        x_r_4 = self.backward_block4(x_r_4_temp_2)

        # print("x_r_4", x_r_4[0][0])

        x_final = self.last_layer(x_r_4)

        return x_final*100