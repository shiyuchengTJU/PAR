import logging
import numpy as np
import torch
import copy
import random
import eagerpy as ep
import math
import torch_dct
from .surfree import SurFree


import json
import torch
import os
import foolbox as fb
import numpy as np
import itertools
import copy
import pickle
import torchvision.models as models
import argparse
import time
import requests

from datetime import datetime
from PIL import Image
from .foolbox_new.utils import samples
from .foolbox_new.distances import l2
from .foolbox_new.attacks.blended_noise import LinearSearchBlendedUniformNoiseAttack
from .foolbox_new import PyTorchModel
from .foolbox_new.criteria import Misclassification

def sf_refinement(image, temp_adv_img, model, label, total_access):
    criterion = Misclassification(torch.tensor(label).unsqueeze(0).cuda())
    attacker = SurFree(steps=total_access, max_queries=total_access)

    new_image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).cuda()
    new_starting_points = torch.tensor(temp_adv_img).permute(2, 0, 1).unsqueeze(0).cuda()

    # print("new_image shape", new_image.shape)
    # print("new_starting_points shape", new_starting_points.shape)
    
    config = json.load(open("./new_foolbox_attacks/config_example.json", "r"))
    temp_result = attacker(model, new_image, criterion,  starting_points=new_starting_points, **config["run"])

    shape1 = len(temp_result[0])
    shape11 = temp_result[0][0].shape
    shape2 = len(temp_result[1])
    shape22 = temp_result[1][0].shape
    shape3 = temp_result[2].shape
    # print("three shape", shape1, shape11, shape2, shape22, shape3)

    return temp_result