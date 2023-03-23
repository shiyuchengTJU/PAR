#coding=utf-8

from __future__ import print_function
from struct import unpack

from foolbox.criteria import Misclassification
from straight_model.four_models_straight import create_fmodel_straight
import foolbox
from boundary.evolutionary_attack import EvolutionaryAttack
from boundary.evolutionary_attack_sample_test import EvolutionaryAttack as EvolutionaryAttack_sample_test
from boundary.bapp import BoundaryAttackPlusPlus as bapp
from boundary.qeba import BAPP_custom as qeba
from boundary.score_attack import ScoreAttack
from boundary.patch_attack import PatchAttack
from new_foolbox_attacks.surfree_refinement import sf_refinement
from cw_new import CarliniL2

from boundary.sampling.sample_generator import SampleGenerator

from boundary.perlin import BoundaryAttack as perlin_boundary

from adversarial_vision_challenge import store_adversarial
import sys
import os

from new_composite_model import CompositeModel
from model_train import pytorch_image_classification

import copy
import numpy as np
import torch

import time
import argparse

global marginal_doc 
global doc_len
marginal_doc = np.zeros(301)
doc_len = 0

criterion = foolbox.criteria.Misclassification()


def l2_distance(a, b):
    return (np.sum((a/255.0 - b/255.0) ** 2))**0.5


def hsja_refinement(model, image, label, hsja_max_query, hsja_starting_point):
    #ddn攻击
    """
        ddn_steps  : ddn总的搜索步数
    """
    attack = foolbox.attacks.HopSkipJumpAttack(model)



    return attack(image, np.array(label), unpack=False, max_num_evals=1, iterations=int(hsja_max_query/26.0), initial_num_evals=1, starting_point=hsja_starting_point, log_every_n_steps=9999999, )


def run_additive(model, image, label, epsilons):
    criterion = foolbox.criteria.Misclassification()
    attack = foolbox.attacks.AdditiveGaussianNoiseAttack(model, criterion)
    return attack(image, label, epsilons=epsilons, unpack=False)



#不同的噪声压缩方法
def whey_refinement(image, temp_adv_img, model, label, total_access, first_access, doc_or_not=False, mode='untargeted'):   
    #后两个参数表示总的查询次数和第一阶段的查询次数
    
    ori_dist = (np.sum((temp_adv_img/255.0 - image/255.0) ** 2))**0.5
    best_dis = ori_dist
    # evolutionary_doc = np.zeros(total_access)

    # print("ori dist of this step before whey:", ori_dist)

    access = 0
    noise = temp_adv_img - image
    for e in range(10):
        for i in range(256, 0, -1):
            noise_temp = copy.deepcopy(noise)
            noise_temp[(noise_temp >= i) & (noise_temp < i+1)] /= 2.0
            noise_temp[(noise_temp > 0) & (noise_temp < 0.5)] = 0

            l2_ori = np.linalg.norm(image/255.0 - (noise+image)/255.0)
            l2_new = np.linalg.norm(image/255.0 - (noise_temp+image)/255.0)
            if l2_ori - l2_new >= 0.0:
                if (noise != noise_temp).any():
                    access += 1
                    # evolutionary_doc[access-1] = best_dis

                    if mode == 'untargeted':
                        if np.argmax(model.forward_one(np.round(noise_temp + image))) != label:
                            l2 = np.linalg.norm(image/255.0 - (noise_temp+image)/255.0)
                            if l2 < best_dis:
                                best_dis = l2
                            #print(l2)
                            noise = copy.deepcopy(noise_temp)
                    elif mode == 'targeted':
                        if np.argmax(model.forward_one(np.round(noise_temp + image))) == label:
                            l2 = np.linalg.norm(image/255.0 - (noise_temp+image)/255.0)
                            if l2 < best_dis:
                                best_dis = l2
                            #print(l2)
                            noise = copy.deepcopy(noise_temp)

            noise_temp = copy.deepcopy(noise)
            noise_temp[(noise_temp >= -i-1) & (noise_temp < -i)] /= 2.0
            noise_temp[(noise_temp > -0.5) & (noise_temp < 0)] = 0
            l2_ori = np.linalg.norm(image/255.0 - (noise+image)/255.0)
            l2_new = np.linalg.norm(image/255.0 - (noise_temp+image)/255.0)
            if l2_ori - l2_new >= 0.0:
                if (noise != noise_temp).any():
                    access += 1
                    # evolutionary_doc[access-1] = best_dis
                    if mode == 'untargeted':
                        if np.argmax(model.forward_one(np.round(noise_temp + image))) != label:
                            l2 = np.linalg.norm(image/255.0 - (noise_temp+image)/255.0)
                            if l2 < best_dis:
                                best_dis = l2
                            #print(l2)
                            noise = copy.deepcopy(noise_temp)
                    elif mode == 'targeted':
                        if np.argmax(model.forward_one(np.round(noise_temp + image))) == label:
                            l2 = np.linalg.norm(image/255.0 - (noise_temp+image)/255.0)
                            if l2 < best_dis:
                                best_dis = l2
                            #print(l2)
                            noise = copy.deepcopy(noise_temp)

            if access > first_access:
                break
            l2 = np.linalg.norm(image/255.0 - (noise+image)/255.0)

    l2 = np.linalg.norm(image/255.0 - (noise+image)/255.0)
    #print(l2, access)

    while access < total_access:
        # evolutionary_doc[access-1] = best_dis
        i, j = int(np.random.random()*60), int(np.random.random()*60)
        noise_temp = copy.deepcopy(noise)
        noise_temp[i:i+3, j:j+3, :] = 0
        l2_ori = np.linalg.norm(image/255.0 - (noise+image)/255.0)
        l2_new = np.linalg.norm(image/255.0 - (noise_temp+image)/255.0)
        if l2_ori-l2_new >= 0.0:
            access += 1
            if mode == 'untargeted':
                if np.argmax(model.forward_one(np.round(noise_temp + image))) != label:
                    l2 = np.linalg.norm(image/255.0 - (noise_temp+image)/255.0)
                    if l2 < best_dis:
                        best_dis = l2
                    #print(l2)
                    noise = copy.deepcopy(noise_temp)

                elif mode == 'targeted':
                    l2 = np.linalg.norm(image/255.0 - (noise_temp+image)/255.0)
                    if l2 < best_dis:
                        best_dis = l2
                    #print(l2)
                    noise = copy.deepcopy(noise_temp)

        l2 = np.linalg.norm(image/255.0 - (noise+image)/255.0)
    
    perturbed_image = noise + image
    l2 = np.linalg.norm(image/255.0 - (noise+image)/255.0)


    return perturbed_image


def boundary_refinement(image, temp_adv_img, model, label, total_access, rescale_or_not, source_step=3e-3, spherical_step=1e-1, rate = 0.2, big_size=64, center_size=40, mode='untargeted', mask=None):
    # print("ori dist of this step before boundary:", (np.sum((temp_adv_img/255.0 - image/255.0) ** 2))**0.5)

    initial_time = time.time()
    attacker = EvolutionaryAttack(model)
    # attacker = evolution_estimator(model)   #相关性实验

    temp_result= attacker.attack(image, label, temp_adv_img, initial_time, time_limit=99999999, 
                  iterations=total_access, source_step=source_step, spherical_step=spherical_step, rescale_or_not=rescale_or_not, rate = rate, big_size=big_size, center_size=center_size, mode=mode, mask=mask)
    return temp_result

def score_refinement(image, temp_adv_img, model, label, total_access, val_samples, mode='untargeted', strategy=0):
    attacker = ScoreAttack(model)
    temp_result = attacker.attack(image, label, temp_adv_img, total_access, val_samples, mode=mode, strategy=strategy)
    return temp_result

def patch_refinement(image, temp_adv_img, model, label, total_access, mode='untargeted'):
    attacker = PatchAttack(model)
    temp_result = attacker.attack(image, label, temp_adv_img, total_access, mode=mode)
    return temp_result

def boundary_refinement_sample_test(image, temp_adv_img, model, label, total_access, rescale_or_not, source_step=3e-3, spherical_step=1e-1, rate=0.2):
    # print("ori dist of this step before boundary:", (np.sum((temp_adv_img/255.0 - image/255.0) ** 2))**0.5)

    initial_time = time.time()
    attacker = EvolutionaryAttack_sample_test(model)
    # attacker = evolution_estimator(model)   #相关性实验

    temp_result = attacker.attack(image, label, temp_adv_img, initial_time, time_limit=99999999,
                                  iterations=total_access, source_step=source_step, spherical_step=spherical_step, rescale_or_not=rescale_or_not, rate=rate)
    return temp_result


def perlin_refinement(image, temp_adv_img, model, label, total_access, source_step=3e-3, spherical_step=1e-1, pixels=64):
    attacker = perlin_boundary(model)
    random_generator = SampleGenerator(shape = image.shape, pixels=pixels)
    # attacker = evolution_estimator(model)   #相关性实验

    temp_result = attacker(image, label, starting_point=temp_adv_img, 
                  iterations=total_access, source_step=source_step, spherical_step=spherical_step, sample_gen=random_generator)

    return temp_result


def bapp_refinement(image, temp_adv_img, model, label, initial_num_evals=10, iterations=1, max_num_evals=300):
    criterion = foolbox.criteria.Misclassification()
    attack = bapp(model, criterion)
    # attacker = evolution_estimator(model)   #相关性实验

    temp_result = attack(image, label, starting_point=temp_adv_img,
                         initial_num_evals=initial_num_evals, max_num_evals=max_num_evals, iterations=iterations)

    return temp_result


def qeba_refinement(image, temp_adv_img, model, label, max_num_evals):
    criterion = foolbox.criteria.Misclassification()
    attack = qeba(model, criterion)

    temp_result = attack(image, label, starting_point=temp_adv_img, max_num_evals=max_num_evals, iterations=1)

    return temp_result

    

def adversarial_ori_check(adversarial_ori, image, used_iterations, total_access):
    #用来判断adversarial_ori是否正常，下一步该怎么做（是否还需要决策攻击）
    """
        adversarial_ori    : 初始对抗样本
        image              : 原始图像
        used_iterations    : 已经迭代过的次数
        total_access       : 总查询次数
    """
    if adversarial_ori is None:   #本身就不存在，说明攻击失败
        return False, 200
    else:   #说明攻击成功了，需要计算噪声幅度
        temp_dist_ori = l2_distance(adversarial_ori, image)
        if temp_dist_ori > 0:   #说明不是直接成功的
            if total_access > used_iterations:  #次数还没有用完，说明可以继续进行运算
                return True, total_access - used_iterations
            else:   #次数用完了，直接返回当前噪声幅度
                return False, temp_dist_ori

        else:  #没有攻击就成功了
            return False, 0

def adversarial_patch_check(remain_access):
    #  判断patch是否把次数用完
    if remain_access == 0:   #说明次数已经用完
        return False
    else:
        return True


def main(arvg):
    global marginal_doc
    global doc_len

    #输入参数设置
    parser = argparse.ArgumentParser(description='pami')

    parser.add_argument('--dataset', type=str, required=True)   #使用什么数据集

    parser.add_argument('--TAP_or_not', type=str, default=0)   #原本设定是False
    parser.add_argument('--serial_num', type=int, required=True)  #实验编号
    parser.add_argument('--sub_model_num', type=int, default=1, required=True)
    parser.add_argument('--target_model_num', type=int, default=1, required=True)
    parser.add_argument('--attack_method_num', type=int)  #无效参数
    parser.add_argument('--total_capacity', type=int, required=True)  #实验中比较的数量
    parser.add_argument('--all_access', type=int, required=True, default=1000)
    parser.add_argument('--whey_or_not', type=int, default=1)   #是否使用whey，无效参数
    parser.add_argument('--total_whey_access', type=int, default=300)  #whey中查询次数，无效参数
    parser.add_argument('--first_whey_access', type=int, default=150)  #whey中第一步查询次数，无效参数
    parser.add_argument('--boundary_or_not', type=int, default=0)   #是否使用boundary，无效参数
    parser.add_argument('--total_boundary_access', type=int, default=1000)  #boundary总查询次数，无效参数
    parser.add_argument('--boundary_rescale_or_not', type=int, default=0)  #boundary是否放缩，无效参数
    parser.add_argument('--attention_or_not', type=int, default=0)   #boundary中是否使用attention，无效参数
    parser.add_argument('--total_attention_access', type=int, default=300)  #boundary中attention总查询次数，无效参数
    parser.add_argument('--temp_counter', type=int, default=-1)  #一个计数器，不用管
    parser.add_argument('--targeted_mode', type=int, default=0)   #是否为针对性错分，默认为否
    parser.add_argument('--save_curve_doc', type=int, default=0)   #是否将攻击结果保存，用于绘制曲线，默认为否

    parser.add_argument('--IFGSM_stepsize', type=float, default=0.002)   #IFGSM的步长
    parser.add_argument('--IFGSM_return_early', type=int, default=0)   #IFGSM是否return early
    parser.add_argument('--IFGSM_iterations', type=int, default=15)   #IFGSM迭代步数
    parser.add_argument('--IFGSM_binary_search', type=int, default=20)   #IFGSM二分搜索次数

    parser.add_argument('--Curls_vr_or_not', type=int, default=1)
    parser.add_argument('--Curls_scale', type=float, default=1.0)
    parser.add_argument('--Curls_m', type=int, default=2)   #Curls中vr-IGSM总共求导几次
    parser.add_argument('--Curls_worthless', type=int, default=1)   #是否进行步数判断
    parser.add_argument('--Curls_binary', type=int, default=0)      #是否进行二分价值判断
    parser.add_argument('--Curls_RC', type=int, default=1)      #是否进行上下山

    parser.add_argument('--source_step', type=float, default=3e-3)      #Boundary径向步长
    parser.add_argument('--spherical_step', type=float, default=1e-1)    #Boundary法向步长
    parser.add_argument('--rate', type=float, default=0.2)    #Boundary, cab 保留比例
    parser.add_argument('--big_size', type=int, default=64)      #图像整体尺寸
    parser.add_argument('--center_size', type=int, default=40)      #evo所需的中心尺寸
    parser.add_argument('--num_labels', type=int, default=200)      #类别总数

    parser.add_argument('--init_attack_num', type=int, default=0)      #初始对抗噪声编号
    parser.add_argument('--transformer_patch_size', type=int, default=16)      #transformer patch尺寸





    args = parser.parse_args()

#---------------------------------

    if args.dataset == 'TinyImagenet':
        model_dict = {1:"resnet", 2:"inception_small", 3:"inception_resnet", 4:"nasnet", 5:"densenet_adv", 6:"inception_v4_adv", 7:"vgg19_adv", 8:"ensemble_three",}
        from four_models import create_fmodel
        from utils import store_adversarial, compute_MAD, read_images
        from straight_model.four_models_straight import create_fmodel_straight
        
    elif args.dataset == 'Imagenet':
        model_dict = {1:"resnet", 2:"densenet", 3:"vgg", 4:"senet", 5:"r26_s32", 6:"vit_s16", 7:"ti_s16", 8:"ti_l16"}
        from four_models_new import create_fmodel
        from utils_imagenet import store_adversarial, compute_MAD, read_images

    elif args.dataset == 'Imagenet_21k':
        model_dict = {1:"r26_s32_21k", 2:"ti_s16_21k", 3:"vit_s16_21k", 4:"ti_l16_21k"}
        from four_models_new import create_fmodel
        from utils_imagenet import store_adversarial, compute_MAD, read_images

    elif args.dataset == 'CIFAR':
        model_dict = {1:"vgg16", 2:"resnet"}
        from four_models_cifar import create_fmodel
        from utils_cifar import store_adversarial, compute_MAD, read_images

    elif args.dataset == 'MNIST':
        model_dict = {1:"vgg16", 2:"resnet"}
        from four_models_mnist import create_fmodel
        from utils_mnist import store_adversarial, compute_MAD, read_images


    attack_method_dict = {1:run_attack_fgsm, 
                          2:run_attack_ifgsm, 
                          3:run_attack_mifgsm, 
                          4:run_attack_vr_mifgsm, 
                          5:run_additive,
                          8:run_attack_omnipotent_fgsm,
                          14:run_attack_ada_ifgsm,
                          31:run_attack_fgsm_reverse,
                          32:run_attack_ifgsm_reverse,
                          33:run_attack_gaussian_fgsm,
                          34:run_attack_ifgsm_sgd,
                          35:run_attack_cw,
                          36:run_attack_ddn,
                          37:run_attack_deepfool,
                          38:run_attack_ead,
                          40:run_attack_newton,
                          41:run_attack_fmna,
                          }

    #raw model表示还没有封装，可以给新的foolbox用的原始模型
    forward_model, new_foolbox_model_forward = create_fmodel(model_dict[args.target_model_num])
    backward_model, new_foolbox_model_backward = create_fmodel(model_dict[args.sub_model_num])

    model = foolbox.models.CompositeModel(
    forward_model=forward_model,
    backward_model=backward_model)

    # if TAP_or_not:
    #     model = new_composite_model(
    #         forward_model=forward_model,
    #         backward_model=backward_model)
    # else:
    
    # 用新写的composite model，可以用来看替代模型的ce
    # #新composite model
    # model = CompositeModel(
    #     forward_model=forward_model,
    #     backward_model=backward_model)
    
    # adversary = CarliniL2(target.cuda(), model)

    #这里规定三个list包含多少个子列表，其中第二个是噪声压缩比
    aux_dist = []
    aux_percent = []
    curve_doc = []
    #下面是用来暂存决策攻击的中间结果的，方便后面计算噪声幅度
    temp_adv_list = []
    for list_counter in range(args.total_capacity):
        aux_dist.append([]), aux_percent.append([]), curve_doc.append([]), temp_adv_list.append([])



    
    # #用来保存所有的IFGSM初始对抗样本
    # IFGSM_list = []

    #展开实验
    print("serial_num", args.serial_num)
    print("exp_set:", args.sub_model_num, args.target_model_num)





    for (file_name, image, label) in read_images():

        print("---------------------------")
        print(args.temp_counter)

        if args.dataset == 'Imagenet_21k':   #如果使用21k，则使用初始分类结果作为标签
            label = np.argmax(model.forward_one(image))



        args.temp_counter += 1


        if args.init_attack_num == 4:
            adversarial_ori_unpack_1  = attack_method_dict[5](model, image, label, epsilons=int(args.all_access / 10))
            adversarial_ori_1, total_prediction_calls_1 = adversarial_ori_unpack_1._Adversarial__best_adversarial, adversarial_ori_unpack_1._total_prediction_calls

        #######################

        #零号位：原版
        check_0, return_0 = adversarial_ori_check(adversarial_ori_1, image, total_prediction_calls_1, args.all_access)
        if check_0:   #允许攻击
            #######################
            temp_adv_list[0] = adversarial_ori_1
            #######################
            aux_dist[0].append(l2_distance(temp_adv_list[0], image))

        else:
            aux_dist[0].append(return_0)

        check_1, return_1 = adversarial_ori_check(adversarial_ori_1, image, total_prediction_calls_1, args.all_access)
        
        


        patch_used_step = 0
        #patch attack
        if check_1:   #允许攻击
            patch_adversarial_1, patch_used_step = patch_refinement(image, adversarial_ori_1, model, label, int(return_1))
            aux_dist[1].append(l2_distance(patch_adversarial_1, image))
        else:
            aux_dist[1].append(return_1)
        
        patch_dist = aux_dist[1][-1]

        patch_remain_access = int(return_1) - patch_used_step
        check_2 = adversarial_patch_check(int(return_1) - patch_used_step)
        

        #一号位：hsja
        if check_1:   #允许攻击
            #######################
            temp_adv_list[2] = hsja_refinement(model, image, label, int(return_1), adversarial_ori_1)._Adversarial__best_adversarial
            #######################
            aux_dist[2].append(l2_distance(temp_adv_list[2], image))
        else:
            aux_dist[2].append(return_1)


        #  patch + hsja
        if check_1 and check_2:   #允许攻击
            #######################
            temp_adv_list[3] = hsja_refinement(model, image, label, patch_remain_access, patch_adversarial_1)._Adversarial__best_adversarial
            #######################
            aux_dist[3].append(l2_distance(temp_adv_list[3], image))
        else:
            aux_dist[3].append(patch_dist)


        #二号位：BBA
        if check_1:   #允许攻击
            #######################
            temp_adv_list[4] = perlin_refinement(image, adversarial_ori_1, model, label, int(return_1), source_step=args.source_step, spherical_step=args.spherical_step, pixels=args.big_size)
            #######################
            aux_dist[4].append(l2_distance(temp_adv_list[4], image))
        else:
            aux_dist[4].append(return_1)

        #  BBA  + patch
        if check_1 and check_2:   #允许攻击
            #######################
            temp_adv_list[5] = perlin_refinement(image, patch_adversarial_1, model, label, patch_remain_access, source_step=args.source_step, spherical_step=args.spherical_step, pixels=args.big_size)
            #######################
            aux_dist[5].append(l2_distance(temp_adv_list[5], image))
        else:
            aux_dist[5].append(patch_dist)


        #三号位：Evo
        if check_1:   #允许攻击
            #######################
            temp_adv_list[6] = boundary_refinement(image, adversarial_ori_1, model, label, int(return_1), 1, source_step=args.source_step, spherical_step=args.spherical_step, big_size=args.big_size, center_size=args.center_size)
            #######################
            aux_dist[6].append(l2_distance(temp_adv_list[6], image))
        else:
            aux_dist[6].append(return_1)


        # patch + Evo
        if check_1 and check_2:   #允许攻击
            #######################
            temp_adv_list[7] = boundary_refinement(image, patch_adversarial_1, model, label, patch_remain_access, 1, source_step=args.source_step, spherical_step=args.spherical_step, big_size=args.big_size, center_size=args.center_size)
            #######################
            aux_dist[7].append(l2_distance(temp_adv_list[7], image))
        else:
            aux_dist[7].append(patch_dist)



        #四号位: boundary
        if check_1:   #允许攻击
            #######################
            temp_adv_list[8] = boundary_refinement(image, adversarial_ori_1, model, label, int(return_1), 2, source_step=args.source_step, spherical_step=args.spherical_step, rate=args.rate, big_size=args.big_size, center_size=args.center_size)

            #######################
            aux_dist[8].append(l2_distance(temp_adv_list[8], image))
        else:
            aux_dist[8].append(return_1)


        # patch + boundary
        if check_1 and check_2:   #允许攻击
            #######################
            temp_adv_list[9] = boundary_refinement(image, patch_adversarial_1, model, label, patch_remain_access, 2, source_step=args.source_step, spherical_step=args.spherical_step, rate=args.rate, big_size=args.big_size, center_size=args.center_size)

            #######################
            aux_dist[9].append(l2_distance(temp_adv_list[9], image))
        else:
            aux_dist[9].append(patch_dist)

        
        #五号位   surfree 攻击
        if check_1:   #允许攻击
            #######################
            temp_adv_list[10] = sf_refinement(image, adversarial_ori_1, new_foolbox_model_forward, label, int(return_1))[1][0][0].permute(1, 2, 0).cpu().numpy()
            #######################
            aux_dist[10].append(l2_distance(temp_adv_list[10], image))
        else:
            aux_dist[10].append(return_1)
        
        
        #  patch + surfree 攻击
        if check_1 and check_2:   #允许攻击
            #######################
            temp_adv_list[11] = sf_refinement(image, patch_adversarial_1, new_foolbox_model_forward, label, patch_remain_access)[1][0][0].permute(1, 2, 0).cpu().numpy()
            #######################
            aux_dist[11].append(l2_distance(temp_adv_list[11], image))
        else:
            aux_dist[11].append(patch_dist)



        #六号位  cisa
        if check_1:   #允许攻击
            #######################
            temp_adv_list[12] = boundary_refinement(image, adversarial_ori_1, model, label, int(return_1), 39, source_step=args.source_step, spherical_step=args.spherical_step, rate=args.rate, big_size=args.big_size, center_size=args.center_size)
            #######################
            aux_dist[12].append(l2_distance(temp_adv_list[12], image))
        else:
            aux_dist[12].append(return_1)

        # patch + cisa
        if check_1 and check_2:   #允许攻击
            #######################
            temp_adv_list[13] = boundary_refinement(image, patch_adversarial_1, model, label, patch_remain_access, 39, source_step=args.source_step, spherical_step=args.spherical_step, rate=args.rate, big_size=args.big_size, center_size=args.center_size)
            #######################
            aux_dist[13].append(l2_distance(temp_adv_list[13], image))
        else:
            aux_dist[13].append(patch_dist)



        #输出攻击结果
        sys.stdout.write("dist of this step:")
        for stdout_counter in range(args.total_capacity):
            print('%.3f' %aux_dist[stdout_counter][-1], end=', ')
        print(' ')

        sys.stdout.write("median dist:")
        for stdout_counter in range(args.total_capacity):
            print('%.3f' %np.median(aux_dist[stdout_counter]), end=', ')
        print(' ')

        sys.stdout.write("mean dist:")
        for stdout_counter in range(args.total_capacity):
            print('%.3f' %np.mean(aux_dist[stdout_counter]), end=', ')
        print(' ')

        np.save('./experiment_result/'+str(args.serial_num)+'_'+str(args.sub_model_num)+"_"+str(args.target_model_num)+"_"+str(args.init_attack_num)+"_"+'.npy', aux_dist)

    print("serial_num", args.serial_num)
    print("exp_set:", args.sub_model_num, args.target_model_num)




if __name__ == '__main__':
    main(sys.argv)
