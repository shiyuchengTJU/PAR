import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from random import randint
from torch.autograd import Variable

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        self.conv2d_Conv2D = self.__conv(2, name='conv2d/Conv2D', in_channels=3,
                                         out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=None)
        self.batch_normalization_FusedBatchNorm = self.__batch_normalization(
            2, 'batch_normalization/FusedBatchNorm', num_features=64, eps=1.00099996416e-05, momentum=0.0)
        self.conv2d_1_Conv2D = self.__conv(2, name='conv2d_1/Conv2D', in_channels=64,
                                           out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=None)
        self.conv2d_2_Conv2D = self.__conv(2, name='conv2d_2/Conv2D', in_channels=64,
                                           out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=None)
        self.batch_normalization_1_FusedBatchNorm = self.__batch_normalization(
            2, 'batch_normalization_1/FusedBatchNorm', num_features=64, eps=1.00099996416e-05, momentum=0.0)
        self.conv2d_3_Conv2D = self.__conv(2, name='conv2d_3/Conv2D', in_channels=64,
                                           out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=None)
        self.batch_normalization_2_FusedBatchNorm = self.__batch_normalization(
            2, 'batch_normalization_2/FusedBatchNorm', num_features=64, eps=1.00099996416e-05, momentum=0.0)
        self.conv2d_4_Conv2D = self.__conv(2, name='conv2d_4/Conv2D', in_channels=64,
                                           out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=None)
        self.batch_normalization_3_FusedBatchNorm = self.__batch_normalization(
            2, 'batch_normalization_3/FusedBatchNorm', num_features=64, eps=1.00099996416e-05, momentum=0.0)
        self.conv2d_5_Conv2D = self.__conv(2, name='conv2d_5/Conv2D', in_channels=64,
                                           out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=None)
        self.batch_normalization_4_FusedBatchNorm = self.__batch_normalization(
            2, 'batch_normalization_4/FusedBatchNorm', num_features=64, eps=1.00099996416e-05, momentum=0.0)
        self.conv2d_6_Conv2D = self.__conv(2, name='conv2d_6/Conv2D', in_channels=64,
                                           out_channels=128, kernel_size=(1, 1), stride=(2, 2), groups=1, bias=None)
        self.conv2d_7_Conv2D = self.__conv(2, name='conv2d_7/Conv2D', in_channels=64,
                                           out_channels=128, kernel_size=(3, 3), stride=(2, 2), groups=1, bias=None)
        self.batch_normalization_5_FusedBatchNorm = self.__batch_normalization(
            2, 'batch_normalization_5/FusedBatchNorm', num_features=128, eps=1.00099996416e-05, momentum=0.0)
        self.conv2d_8_Conv2D = self.__conv(2, name='conv2d_8/Conv2D', in_channels=128,
                                           out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=None)
        self.batch_normalization_6_FusedBatchNorm = self.__batch_normalization(
            2, 'batch_normalization_6/FusedBatchNorm', num_features=128, eps=1.00099996416e-05, momentum=0.0)
        self.conv2d_9_Conv2D = self.__conv(2, name='conv2d_9/Conv2D', in_channels=128,
                                           out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=None)
        self.batch_normalization_7_FusedBatchNorm = self.__batch_normalization(
            2, 'batch_normalization_7/FusedBatchNorm', num_features=128, eps=1.00099996416e-05, momentum=0.0)
        self.conv2d_10_Conv2D = self.__conv(2, name='conv2d_10/Conv2D', in_channels=128,
                                            out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=None)
        self.batch_normalization_8_FusedBatchNorm = self.__batch_normalization(
            2, 'batch_normalization_8/FusedBatchNorm', num_features=128, eps=1.00099996416e-05, momentum=0.0)
        self.conv2d_11_Conv2D = self.__conv(2, name='conv2d_11/Conv2D', in_channels=128,
                                            out_channels=256, kernel_size=(1, 1), stride=(2, 2), groups=1, bias=None)
        self.conv2d_12_Conv2D = self.__conv(2, name='conv2d_12/Conv2D', in_channels=128,
                                            out_channels=256, kernel_size=(3, 3), stride=(2, 2), groups=1, bias=None)
        self.batch_normalization_9_FusedBatchNorm = self.__batch_normalization(
            2, 'batch_normalization_9/FusedBatchNorm', num_features=256, eps=1.00099996416e-05, momentum=0.0)
        self.conv2d_13_Conv2D = self.__conv(2, name='conv2d_13/Conv2D', in_channels=256,
                                            out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=None)
        self.batch_normalization_10_FusedBatchNorm = self.__batch_normalization(
            2, 'batch_normalization_10/FusedBatchNorm', num_features=256, eps=1.00099996416e-05, momentum=0.0)
        self.conv2d_14_Conv2D = self.__conv(2, name='conv2d_14/Conv2D', in_channels=256,
                                            out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=None)
        self.batch_normalization_11_FusedBatchNorm = self.__batch_normalization(
            2, 'batch_normalization_11/FusedBatchNorm', num_features=256, eps=1.00099996416e-05, momentum=0.0)
        self.conv2d_15_Conv2D = self.__conv(2, name='conv2d_15/Conv2D', in_channels=256,
                                            out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=None)
        self.batch_normalization_12_FusedBatchNorm = self.__batch_normalization(
            2, 'batch_normalization_12/FusedBatchNorm', num_features=256, eps=1.00099996416e-05, momentum=0.0)
        self.conv2d_16_Conv2D = self.__conv(2, name='conv2d_16/Conv2D', in_channels=256,
                                            out_channels=512, kernel_size=(1, 1), stride=(2, 2), groups=1, bias=None)
        self.conv2d_17_Conv2D = self.__conv(2, name='conv2d_17/Conv2D', in_channels=256,
                                            out_channels=512, kernel_size=(3, 3), stride=(2, 2), groups=1, bias=None)
        self.batch_normalization_13_FusedBatchNorm = self.__batch_normalization(
            2, 'batch_normalization_13/FusedBatchNorm', num_features=512, eps=1.00099996416e-05, momentum=0.0)
        self.conv2d_18_Conv2D = self.__conv(2, name='conv2d_18/Conv2D', in_channels=512,
                                            out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=None)
        self.batch_normalization_14_FusedBatchNorm = self.__batch_normalization(
            2, 'batch_normalization_14/FusedBatchNorm', num_features=512, eps=1.00099996416e-05, momentum=0.0)
        self.conv2d_19_Conv2D = self.__conv(2, name='conv2d_19/Conv2D', in_channels=512,
                                            out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=None)
        self.batch_normalization_15_FusedBatchNorm = self.__batch_normalization(
            2, 'batch_normalization_15/FusedBatchNorm', num_features=512, eps=1.00099996416e-05, momentum=0.0)
        self.conv2d_20_Conv2D = self.__conv(2, name='conv2d_20/Conv2D', in_channels=512,
                                            out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=None)
        self.batch_normalization_16_FusedBatchNorm = self.__batch_normalization(
            2, 'batch_normalization_16/FusedBatchNorm', num_features=512, eps=1.00099996416e-05, momentum=0.0)
        self.readout_layer_MatMul = self.__dense(
            name='readout_layer/MatMul', in_features=512, out_features=200, bias=True)

        self.rand_size = randint(64, 70)
        self.resize_layer_1 = nn.AdaptiveAvgPool2d((self.rand_size, self.rand_size))
        self.resize_layer_2 = nn.AdaptiveAvgPool2d((64, 64))


    def resize2d(self, img, size):
        return F.adaptive_avg_pool2d(Variable(img, requires_grad=True), size)


    def scatter_patch(self, x, patch_num, width, height):
        #  input image, how many patches to output, width and height of patches
        
        output = torch.randn(patch_num, x.shape[1], x.shape[2], x.shape[3]).cuda()
        for crop_iter in range(patch_num):
            crop_origin_x = randint(0, 64 - width)
            crop_origin_y = randint(0, 64 - height)
            new_image = x[0, :, crop_origin_x:crop_origin_x+width, crop_origin_y:crop_origin_y+height]
            output[crop_iter] = self.resize2d(new_image, (x.shape[2], x.shape[3]))

        return output


    def randomization(self, x):
        rand_size = randint(64, 70)
        output = self.resize2d(x, (rand_size, rand_size))
        output = self.resize2d(output, (64, 64))

        return output




    def forward(self, x):
        # # print("before", x.shape)
        # x = self.scatter_patch(x, 30, 60, 60)
        # # print("after", x.shape)
        
        # print(x.requires_grad)
        # print(x)
        # x = self.randomization(x)
        # print(x)
        # print(x.requires_grad)
        
        
        x = self.resize_layer_1(x)
        x = self.resize_layer_2(x)

        x = x.clone()
        x[:, 0] = x[:, 0] - 123.68
        x[:, 1] = x[:, 1] - 116.78
        x[:, 2] = x[:, 2] - 103.94

        conv2d_Conv2D_pad = F.pad(x, (1, 1, 1, 1))
        conv2d_Conv2D = self.conv2d_Conv2D(conv2d_Conv2D_pad)
        batch_normalization_FusedBatchNorm = self.batch_normalization_FusedBatchNorm(
            conv2d_Conv2D)
        Relu = F.relu(batch_normalization_FusedBatchNorm)
        conv2d_1_Conv2D = self.conv2d_1_Conv2D(Relu)
        conv2d_2_Conv2D_pad = F.pad(Relu, (1, 1, 1, 1))
        conv2d_2_Conv2D = self.conv2d_2_Conv2D(conv2d_2_Conv2D_pad)
        batch_normalization_1_FusedBatchNorm = self.batch_normalization_1_FusedBatchNorm(
            conv2d_2_Conv2D)
        Relu_1 = F.relu(batch_normalization_1_FusedBatchNorm)
        conv2d_3_Conv2D_pad = F.pad(Relu_1, (1, 1, 1, 1))
        conv2d_3_Conv2D = self.conv2d_3_Conv2D(conv2d_3_Conv2D_pad)
        add = conv2d_3_Conv2D + conv2d_1_Conv2D
        batch_normalization_2_FusedBatchNorm = self.batch_normalization_2_FusedBatchNorm(
            add)
        Relu_2 = F.relu(batch_normalization_2_FusedBatchNorm)
        conv2d_4_Conv2D_pad = F.pad(Relu_2, (1, 1, 1, 1))
        conv2d_4_Conv2D = self.conv2d_4_Conv2D(conv2d_4_Conv2D_pad)
        batch_normalization_3_FusedBatchNorm = self.batch_normalization_3_FusedBatchNorm(
            conv2d_4_Conv2D)
        Relu_3 = F.relu(batch_normalization_3_FusedBatchNorm)
        conv2d_5_Conv2D_pad = F.pad(Relu_3, (1, 1, 1, 1))
        conv2d_5_Conv2D = self.conv2d_5_Conv2D(conv2d_5_Conv2D_pad)
        add_1 = conv2d_5_Conv2D + add
        batch_normalization_4_FusedBatchNorm = self.batch_normalization_4_FusedBatchNorm(
            add_1)
        Relu_4 = F.relu(batch_normalization_4_FusedBatchNorm)
        Pad = F.pad(Relu_4, (0, 0, 0, 0), mode='constant', value=0)
        Pad_1 = F.pad(Relu_4, (1, 1, 1, 1), mode='constant', value=0)
        conv2d_6_Conv2D = self.conv2d_6_Conv2D(Pad)
        conv2d_7_Conv2D = self.conv2d_7_Conv2D(Pad_1)
        batch_normalization_5_FusedBatchNorm = self.batch_normalization_5_FusedBatchNorm(
            conv2d_7_Conv2D)
        Relu_5 = F.relu(batch_normalization_5_FusedBatchNorm)
        conv2d_8_Conv2D_pad = F.pad(Relu_5, (1, 1, 1, 1))
        conv2d_8_Conv2D = self.conv2d_8_Conv2D(conv2d_8_Conv2D_pad)
        add_2 = conv2d_8_Conv2D + conv2d_6_Conv2D
        batch_normalization_6_FusedBatchNorm = self.batch_normalization_6_FusedBatchNorm(
            add_2)
        Relu_6 = F.relu(batch_normalization_6_FusedBatchNorm)
        conv2d_9_Conv2D_pad = F.pad(Relu_6, (1, 1, 1, 1))
        conv2d_9_Conv2D = self.conv2d_9_Conv2D(conv2d_9_Conv2D_pad)
        batch_normalization_7_FusedBatchNorm = self.batch_normalization_7_FusedBatchNorm(
            conv2d_9_Conv2D)
        Relu_7 = F.relu(batch_normalization_7_FusedBatchNorm)
        conv2d_10_Conv2D_pad = F.pad(Relu_7, (1, 1, 1, 1))
        conv2d_10_Conv2D = self.conv2d_10_Conv2D(conv2d_10_Conv2D_pad)
        add_3           = conv2d_10_Conv2D + add_2
        batch_normalization_8_FusedBatchNorm = self.batch_normalization_8_FusedBatchNorm(add_3)
        Relu_8          = F.relu(batch_normalization_8_FusedBatchNorm)
        Pad_2           = F.pad(Relu_8, (0, 0, 0, 0), mode = 'constant', value = 0)
        Pad_3           = F.pad(Relu_8, (1, 1, 1, 1), mode = 'constant', value = 0)
        conv2d_11_Conv2D = self.conv2d_11_Conv2D(Pad_2)
        conv2d_12_Conv2D = self.conv2d_12_Conv2D(Pad_3)
        batch_normalization_9_FusedBatchNorm = self.batch_normalization_9_FusedBatchNorm(conv2d_12_Conv2D)
        Relu_9          = F.relu(batch_normalization_9_FusedBatchNorm)
        conv2d_13_Conv2D_pad = F.pad(Relu_9, (1, 1, 1, 1))
        conv2d_13_Conv2D = self.conv2d_13_Conv2D(conv2d_13_Conv2D_pad)
        add_4           = conv2d_13_Conv2D + conv2d_11_Conv2D
        batch_normalization_10_FusedBatchNorm = self.batch_normalization_10_FusedBatchNorm(add_4)
        Relu_10         = F.relu(batch_normalization_10_FusedBatchNorm)
        conv2d_14_Conv2D_pad = F.pad(Relu_10, (1, 1, 1, 1))
        conv2d_14_Conv2D = self.conv2d_14_Conv2D(conv2d_14_Conv2D_pad)
        batch_normalization_11_FusedBatchNorm = self.batch_normalization_11_FusedBatchNorm(conv2d_14_Conv2D)
        Relu_11         = F.relu(batch_normalization_11_FusedBatchNorm)
        conv2d_15_Conv2D_pad = F.pad(Relu_11, (1, 1, 1, 1))
        conv2d_15_Conv2D = self.conv2d_15_Conv2D(conv2d_15_Conv2D_pad)
        add_5           = conv2d_15_Conv2D + add_4
        batch_normalization_12_FusedBatchNorm = self.batch_normalization_12_FusedBatchNorm(add_5)
        Relu_12         = F.relu(batch_normalization_12_FusedBatchNorm)
        Pad_4           = F.pad(Relu_12, (0, 0, 0, 0), mode = 'constant', value = 0)
        Pad_5           = F.pad(Relu_12, (1, 1, 1, 1), mode = 'constant', value = 0)
        conv2d_16_Conv2D = self.conv2d_16_Conv2D(Pad_4)
        conv2d_17_Conv2D = self.conv2d_17_Conv2D(Pad_5)
        batch_normalization_13_FusedBatchNorm = self.batch_normalization_13_FusedBatchNorm(conv2d_17_Conv2D)
        Relu_13         = F.relu(batch_normalization_13_FusedBatchNorm)
        conv2d_18_Conv2D_pad = F.pad(Relu_13, (1, 1, 1, 1))
        conv2d_18_Conv2D = self.conv2d_18_Conv2D(conv2d_18_Conv2D_pad)
        add_6           = conv2d_18_Conv2D + conv2d_16_Conv2D
        batch_normalization_14_FusedBatchNorm = self.batch_normalization_14_FusedBatchNorm(add_6)
        Relu_14         = F.relu(batch_normalization_14_FusedBatchNorm)
        conv2d_19_Conv2D_pad = F.pad(Relu_14, (1, 1, 1, 1))
        conv2d_19_Conv2D = self.conv2d_19_Conv2D(conv2d_19_Conv2D_pad)
        batch_normalization_15_FusedBatchNorm = self.batch_normalization_15_FusedBatchNorm(conv2d_19_Conv2D)
        Relu_15         = F.relu(batch_normalization_15_FusedBatchNorm)
        conv2d_20_Conv2D_pad = F.pad(Relu_15, (1, 1, 1, 1))
        conv2d_20_Conv2D = self.conv2d_20_Conv2D(conv2d_20_Conv2D_pad)
        add_7           = conv2d_20_Conv2D + add_6
        batch_normalization_16_FusedBatchNorm = self.batch_normalization_16_FusedBatchNorm(add_7)
        Relu_16         = F.relu(batch_normalization_16_FusedBatchNorm)
        Mean            = torch.mean(Relu_16, 3, True)
        Mean            = torch.mean(Mean, 2, True)
        Reshape         = Mean.view(-1, 512)
        readout_layer_MatMul = self.readout_layer_MatMul(Reshape)


        # # print(readout_layer_MatMul.shape)
        # readout_layer_MatMul = torch.mean(readout_layer_MatMul, 0, True)
        # print("result", torch.max(readout_layer_MatMul, 1)[1])
        # # print(readout_layer_MatMul.shape)

        return readout_layer_MatMul


    @staticmethod
    def __batch_normalization(dim, name, **kwargs):
        if   dim == 1:  layer = nn.BatchNorm1d(**kwargs)
        elif dim == 2:  layer = nn.BatchNorm2d(**kwargs)
        elif dim == 3:  layer = nn.BatchNorm3d(**kwargs)
        else:           raise NotImplementedError()

        return layer

    @staticmethod
    def __conv(dim, name, **kwargs):
        if   dim == 1:  layer = nn.Conv1d(**kwargs)
        elif dim == 2:  layer = nn.Conv2d(**kwargs)
        elif dim == 3:  layer = nn.Conv3d(**kwargs)
        else:           raise NotImplementedError()

        return layer

    @staticmethod
    def __dense(name, **kwargs):
        layer = nn.Linear(**kwargs)

        return layer
