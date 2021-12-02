import sys
import os

from foolbox.models import PyTorchModel

from models.ensemble import EnsembleNet_resnet, EnsembleNet_vgg, EnsembleNet_densenet, EnsembleNet_senet
import torch
import numpy as np

import argparse
import pathlib
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import tqdm

import torch.utils.model_zoo as model_zoo

import copy

from fvcore.common.checkpoint import Checkpointer
from model_train import pytorch_image_classification

from new_foolbox_attacks.foolbox_new import PyTorchModel as new_PytorchModel

from model_train.pytorch_image_classification import (
    apply_data_parallel_wrapper,
    create_dataloader,
    create_loss,
    create_model,
    get_default_config,
    update_config,
)
from model_train.pytorch_image_classification.utils import (
    AverageMeter,
    create_logger,
    get_rank,
)

# sys.path.append("/home/syc/adv/tpami/model_train/pytorch_image_classification/models/cifar/")
# sys.path.append("/home/syc/adv/tpami/model_train")



def load_config(config_path):
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--config', type=str, required=True)
    # parser.add_argument('options', default=None, nargs=argparse.REMAINDER)
    # args = parser.parse_args()

    config = get_default_config()
    config.merge_from_file(config_path)
    # config.merge_from_list(args.options)
    update_config(config)
    config.freeze()
    return config





def create_fmodel(model_type):
    
    model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
    }
    
    if model_type == "vgg16":
        config = load_config("./model_train/configs/cifar/vgg.yaml")
        ckpt_dir = "./model_train/experiments/cifar10/vgg/exp00/checkpoint_00160.pth"
    elif model_type == "resnet":
        config = load_config("./model_train/configs/cifar/resnet.yaml")
        ckpt_dir = "./model_train/experiments/cifar10/resnet/exp00/checkpoint_00100.pth"

    # if config.test.output_dir is None:
    #     output_dir = pathlib.Path(config.test.checkpoint).parent
    # else:
    #     output_dir = pathlib.Path(config.test.output_dir)
    #     output_dir.mkdir(exist_ok=True, parents=True)

    logger = create_logger(name=__name__, distributed_rank=get_rank())

    model = create_model(config)
    # model = apply_data_parallel_wrapper(config, model)


    # model.load_state_dict(model_zoo.load_url(model_urls['vgg19_bn'], model_dir=ckpt_dir))
    # print("ckpt", torch.load(ckpt_dir))
    
    # FIXME
    # print("model", type(model))
    # print("model", model._modules)
    # exit()

    # model.load_state_dict(torch.load(ckpt_dir)[model])

    # #FIXME
    # print("pwd", sys.argv[0])

    checkpoint = torch.load(ckpt_dir)
    if isinstance(model,
                    (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        model.module.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint['model'])


    model.eval()


    def preprocessing(x):
        assert x.ndim in [3, 4]
        if x.ndim == 3:
            x = np.transpose(x, axes=(2, 0, 1))
        elif x.ndim == 4:
            x = np.transpose(x, axes=(0, 3, 1, 2))
        x = copy.deepcopy(x/255.0)
        x[:,0] = (x[:,0]-0.4914)/0.2470
        x[:,1] = (x[:,1]-0.4822)/0.2435
        x[:,2] = (x[:,2]-0.4465)/0.2616
        def grad(dmdp):
            # assert dmdp.ndim == 3
            # dmdx = np.transpose(dmdp, axes=(1, 2, 0))

            if dmdp.ndim == 3:
                dmdx = np.transpose(dmdp, axes=(1, 2, 0))
            elif dmdp.ndim == 4:
                dmdx = np.transpose(dmdp, axes=(0, 2, 3, 1))

            return dmdx
        return x, grad

    fmodel = PyTorchModel(model, bounds=(0,255), num_classes=10, channel_axis=3, preprocessing=preprocessing)
    new_preprocessing = dict(axis = -3)

    new_foolbox_model = new_PytorchModel(model, bounds=(0,255), preprocessing=new_preprocessing)
    return fmodel, new_foolbox_model



if __name__ == '__main__':
    # executable for debuggin and testing
    print(create_fmodel('vgg16'))
