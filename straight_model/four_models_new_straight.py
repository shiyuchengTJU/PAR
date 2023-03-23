#coding=utf-8
#imagenet 模型直通

import sys
import os

from foolbox.models import PyTorchModel

from models.ensemble import EnsembleNet_resnet, EnsembleNet_vgg, EnsembleNet_densenet, EnsembleNet_senet
import torch
import numpy as np



def create_fmodel(model_type):

    if model_type == "resnet":
        model = EnsembleNet_resnet()
    elif model_type == "densenet":
        model = EnsembleNet_densenet()
    elif model_type == "vgg":
        model = EnsembleNet_vgg()
    elif model_type == "senet":
        model = EnsembleNet_senet()

    model.eval()

    def preprocessing(x):
        assert x.ndim in [3, 4]
        if x.ndim == 3:
            x = np.transpose(x, axes=(2, 0, 1))
        elif x.ndim == 4:
            x = np.transpose(x, axes=(0, 3, 1, 2))
        def grad(dmdp):
            assert dmdp.ndim == 3
            dmdx = np.transpose(dmdp, axes=(1, 2, 0))
            return dmdx
        return x, grad

    fmodel = PyTorchModel(model, bounds=(0,255), num_classes=1000, channel_axis=3, preprocessing=preprocessing)
    return fmodel


if __name__ == '__main__':
    # executable for debuggin and testing
    print(create_fmodel())
