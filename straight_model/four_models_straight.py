#coding=utf-8
#tiny 模型直通

import sys
import os

from foolbox.models import PyTorchModel

from new_models.ensemble import EnsembleNet_resnet, EnsembleNet_inception, EnsembleNet_nasnet, EnsembleNet_inception_random, EnsembleNet_inception_v4, EnsembleNet_inception_small, EnsembleNet_inception_resnet, EnsembleNet_inception_pnasnet, EnsembleNet_densenet_adv
from new_models.ensemble import EnsembleNet_inception_v4_adv, EnsembleNet_vgg19_adv, EnsembleNet_three
import torch
import numpy as np



def create_fmodel_straight(model_type):

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

    # #FIXME
    # print("model dict", model.__dict__)

    return model

