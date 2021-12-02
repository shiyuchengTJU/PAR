import sys
import os

from foolbox.models import PyTorchModel

from models.ensemble import EnsembleNet_resnet, EnsembleNet_vgg, EnsembleNet_densenet, EnsembleNet_senet, EnsembleNet_vit_s16, EnsembleNet_ti_s16, EnsembleNet_r26_s32, EnsembleNet_r26_s32_21k, EnsembleNet_vit_s16_21k, EnsembleNet_ti_s16_21k, EnsembleNet_L_16, EnsembleNet_L_16_21k
import torch
import numpy as np

from new_foolbox_attacks.foolbox_new import PyTorchModel as new_PytorchModel



def create_fmodel(model_type):

    if model_type == "resnet":
        model = EnsembleNet_resnet()
    elif model_type == "densenet":
        model = EnsembleNet_densenet()
    elif model_type == "vgg":
        model = EnsembleNet_vgg()
    elif model_type == "senet":
        model = EnsembleNet_senet()
    elif model_type == "r26_s32":
        model = EnsembleNet_r26_s32()
    elif model_type == "ti_s16":
        model = EnsembleNet_ti_s16()
    elif model_type == "vit_s16":
        model = EnsembleNet_vit_s16()
    elif model_type == "ti_l16":
        model = EnsembleNet_L_16()
    elif model_type == "r26_s32_21k":
        model = EnsembleNet_r26_s32_21k()
    elif model_type == "ti_s16_21k":
        model = EnsembleNet_ti_s16_21k()
    elif model_type == "vit_s16_21k":
        model = EnsembleNet_vit_s16_21k()
    elif model_type == "ti_l16_21k":
        model = EnsembleNet_L_16_21k()

    model.eval()

    # def preprocessing(x):
    #     assert x.ndim in [3, 4]
    #     if x.ndim == 3:
    #         x = np.transpose(x, axes=(2, 0, 1))
    #     elif x.ndim == 4:
    #         x = np.transpose(x, axes=(0, 3, 1, 2))
    #     def grad(dmdp):
    #         assert dmdp.ndim == 3
    #         dmdx = np.transpose(dmdp, axes=(1, 2, 0))
    #         return dmdx
    #     return x, grad

    def preprocessing(x):
        # print("x shape", x.shape)
        assert x.ndim in [3, 4]
        if x.ndim == 3:
            x = np.transpose(x, axes=(2, 0, 1))
        elif x.ndim == 4:
            x = np.transpose(x, axes=(0, 3, 1, 2))
        def grad(dmdp):
            #FIXME
            # print("dmdp.ndim", dmdp.ndim)
            # print("dmdp shape", dmdp.shape)
            # assert dmdp.ndim == 3

            if dmdp.ndim == 3:
                dmdx = np.transpose(dmdp, axes=(1, 2, 0))
            elif dmdp.ndim == 4:
                dmdx = np.transpose(dmdp, axes=(0, 2, 3, 1))
            return dmdx
        return x, grad

    if model_type == "r26_s32_21k" or model_type == "ti_s16_21k" or model_type == "vit_s16_21k" or model_type == "ti_l16_21k":
        fmodel = PyTorchModel(model, bounds=(0,255), num_classes=21843, channel_axis=3, preprocessing=preprocessing)
    else:
        fmodel = PyTorchModel(model, bounds=(0,255), num_classes=1000, channel_axis=3, preprocessing=preprocessing)

    new_preprocessing = dict(axis = -3)

    new_foolbox_model = new_PytorchModel(model, bounds=(0,255), preprocessing=new_preprocessing)
    return fmodel, new_foolbox_model


if __name__ == '__main__':
    # executable for debuggin and testing
    print(create_fmodel())
