import sys
import os

from foolbox.models import PyTorchModel

from new_models.ensemble import EnsembleNet_resnet, EnsembleNet_inception, EnsembleNet_nasnet, EnsembleNet_inception_random, EnsembleNet_inception_v4, EnsembleNet_inception_small, EnsembleNet_inception_resnet, EnsembleNet_inception_pnasnet, EnsembleNet_densenet_adv
from new_models.ensemble import EnsembleNet_inception_v4_adv, EnsembleNet_vgg19_adv, EnsembleNet_three
import torch
import numpy as np

from new_foolbox_attacks.foolbox_new import PyTorchModel as new_PytorchModel



def create_fmodel(model_type):

    if model_type == "resnet":
        model = EnsembleNet_resnet()
    elif model_type == "inception":
        model = EnsembleNet_inception()
    elif model_type == "nasnet":
        model = EnsembleNet_nasnet()
    elif model_type == "inception_random":
        model = EnsembleNet_inception_random()
    elif model_type == "inception_v4":
        model = EnsembleNet_inception_v4()
    elif model_type == "inception_small":
        model = EnsembleNet_inception_small()
    elif model_type == "inception_resnet":
        model = EnsembleNet_inception_resnet()
    elif model_type == "pnasnet":
        model = EnsembleNet_inception_pnasnet()
    elif model_type == "densenet_adv":
        model = EnsembleNet_densenet_adv()
    elif model_type == "inception_v4_adv":
        model = EnsembleNet_densenet_adv()
    elif model_type == "vgg19_adv":
        model = EnsembleNet_vgg19_adv()
    elif model_type == "ensemble_three":
        model = EnsembleNet_three()

    model.eval()

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

    fmodel = PyTorchModel(model, bounds=(0,255), num_classes=200, channel_axis=3, preprocessing=preprocessing)
    new_preprocessing = dict(axis = -3)

    new_foolbox_model = new_PytorchModel(model, bounds=(0,255), preprocessing=new_preprocessing)
    return fmodel, new_foolbox_model


if __name__ == '__main__':
    # executable for debuggin and testing
    print(create_fmodel())
