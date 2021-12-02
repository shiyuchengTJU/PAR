from .vgg.vgg import vgg19_bn
from .densenet.densenet import densenet161
from .resnet.resnet import resnet101
from .senet.senet import senet154

#transformer
import timm


from torch import nn
import torch
import os


class EnsembleNet_resnet(nn.Module):
    def __init__(self):
        super(EnsembleNet_resnet, self).__init__()
        self.model_3 = resnet101(pretrained=True).cuda().eval()

    def forward(self, x):
        
        output_3 = self.model_3(x)
        return output_3


class EnsembleNet_vgg(nn.Module):
    def __init__(self):
        super(EnsembleNet_vgg, self).__init__()
        self.model_3 = vgg19_bn(pretrained=True).cuda().eval()

    def forward(self, x):
        output_3 = self.model_3(x)
        return output_3


class EnsembleNet_densenet(nn.Module):
    def __init__(self):
        super(EnsembleNet_densenet, self).__init__()
        self.model_3 = densenet161(pretrained=True).cuda().eval()

    def forward(self, x):
        output_3 = self.model_3(x)
        return output_3


class EnsembleNet_senet(nn.Module):
    def __init__(self):
        super(EnsembleNet_senet, self).__init__()
        self.model_3 = senet154(num_classes=1000, pretrained='imagenet').cuda().eval()

    def forward(self, x):
        output_3 = self.model_3(x)
        return output_3


class EnsembleNet_r26_s32(nn.Module):
    def __init__(self):
        super(EnsembleNet_r26_s32, self).__init__()
        self.model_1 = timm.create_model('vit_small_r26_s32_224', pretrained=False)
        self.model_1.load_pretrained('./models/R26_S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz')
        self.model_1 = self.model_1.eval()

    def forward(self, x):
        x = x/255.0
        output_1 = self.model_1(x)
        return output_1


class EnsembleNet_vit_s16(nn.Module):
    def __init__(self):
        super(EnsembleNet_vit_s16, self).__init__()
        self.model_1 = timm.create_model('vit_small_patch16_224', pretrained=False)
        self.model_1.load_pretrained('./models/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz')
        self.model_1 = self.model_1.eval()

    def forward(self, x):
        x = x/255.0
        output_1 = self.model_1(x)
        return output_1


class EnsembleNet_ti_s16(nn.Module):
    def __init__(self):
        super(EnsembleNet_ti_s16, self).__init__()
        self.model_1 = timm.create_model('vit_tiny_patch16_224', pretrained=False)
        self.model_1.load_pretrained('./models/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz')
        self.model_1 = self.model_1.eval()

    def forward(self, x):
        x = x/255.0
        # print("x", torch.max(x), torch.min(x))
        output_1 = self.model_1(x)
        # print("output_1.shape", output_1.shape)
        # exit()
        return output_1



class EnsembleNet_L_16(nn.Module):
    def __init__(self):
        super(EnsembleNet_L_16, self).__init__()
        self.model_1 = timm.create_model('vit_large_patch16_224', pretrained=False)
        self.model_1.load_pretrained('./models/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_224.npz')
        self.model_1 = self.model_1.eval()

    def forward(self, x):
        x = x/255.0
        output_1 = self.model_1(x)

        return output_1


class EnsembleNet_r26_s32_21k(nn.Module):
    def __init__(self):
        super(EnsembleNet_r26_s32_21k, self).__init__()
        self.model_1 = timm.create_model('vit_small_r26_s32_224_in21k', pretrained=False)
        self.model_1.load_pretrained('./models/R26_S_32-i21k-300ep-lr_0.001-aug_medium2-wd_0.03-do_0.0-sd_0.0.npz')
        self.model_1 = self.model_1.eval()

    def forward(self, x):
        x = x/255.0
        output_1 = self.model_1(x)
        return output_1


class EnsembleNet_vit_s16_21k(nn.Module):
    def __init__(self):
        super(EnsembleNet_vit_s16_21k, self).__init__()
        self.model_1 = timm.create_model('vit_small_patch16_224_in21k', pretrained=False)
        self.model_1.load_pretrained('./models/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz')
        self.model_1 = self.model_1.eval()

    def forward(self, x):
        x = x/255.0
        output_1 = self.model_1(x)
        return output_1


class EnsembleNet_ti_s16_21k(nn.Module):
    def __init__(self):
        super(EnsembleNet_ti_s16_21k, self).__init__()
        self.model_1 = timm.create_model('vit_tiny_patch16_224_in21k', pretrained=False)
        self.model_1.load_pretrained('./models/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0.npz')
        self.model_1 = self.model_1.eval()

    def forward(self, x):
        x = x/255.0
        # print("x", torch.max(x), torch.min(x))
        output_1 = self.model_1(x)
        # print("output_1.shape", output_1.shape)
        # exit()
        return output_1


class EnsembleNet_L_16_21k(nn.Module):
    def __init__(self):
        super(EnsembleNet_L_16_21k, self).__init__()
        self.model_1 = timm.create_model('vit_large_patch16_224_in21k', pretrained=False)
        self.model_1.load_pretrained('./models/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1.npz')
        self.model_1 = self.model_1.eval()

    def forward(self, x):
        x = x/255.0
        output_1 = self.model_1(x)

        return output_1

