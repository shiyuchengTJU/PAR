#coding=utf-8
#基于patch的攻击方法，一个patch一个patch进行攻击

import numpy as np
import time
import copy
from foolbox.utils import crossentropy, softmax
import torch

def to_cuda(x): #将numpy转换为tensor在显卡上计算
    return torch.from_numpy(x).cuda()


def l2_distance(a, b):
    if type(b) != torch.Tensor:
        b = torch.ones_like(a).cuda() * b        

    dist = (torch.sum((torch.round(a)/255.0 - torch.round(b)/255.0) ** 2))**0.5

    return dist


def normalize_noise(direction, distance, original_image):
    norm_direction = direction/l2_distance(direction, 0)   #归一化

    clipped_direction = torch.clip(torch.round(norm_direction*distance + original_image), 0, 255) - original_image

    clipped_dist = l2_distance(clipped_direction, 0)

    return clipped_direction, clipped_dist



def scatter_draw(data):
    save_path = "/home/syc/adversarial_machine_learning/nips18-avc-attack-template__/"
    fig = plt.figure(figsize=(16,9))
    plt.scatter(data[1], data[0], s=1)
    plt.savefig(save_path+"data.png", bbox_inches='tight')



def clip(x, min_x=-1, max_x=1):
    x[x < min_x] = min_x
    x[x > max_x] = max_x
    return x

def value_mask_init(patch_num):    #初始化查询价值mask
    value_mask = torch.ones([patch_num, patch_num]).cuda()
    # value_mask[int(patch_num*0.25):int(patch_num*0.75) , int(patch_num*0.25):int(patch_num*0.75)] = 0.5

    return value_mask

def noise_mask_init(x, image, patch_num, patch_size):    #初始化噪声幅度mask
    noise = x - image
    noise_mask = torch.zeros([patch_num, patch_num]).cuda()
    for row_counter in range(patch_num):
        for col_counter in range(patch_num):
            noise_mask[row_counter][col_counter] = l2_distance(noise[(row_counter*patch_size):(row_counter*patch_size+patch_size) , (col_counter*patch_size):(col_counter*patch_size+patch_size) ], 0)

    return noise_mask


def translate(index, patch_num):  #将价值最高patch的行列输出出来
    best_row = index//patch_num
    best_col = index - patch_num*best_row

    return best_row, best_col




class Attacker:
    def __init__(self, model):
        self.model = model

    def attack(self, inputs):
        return NotImplementedError

    def attack_target(self, inputs, targets):
        return NotImplementedError


class PatchAttack(Attacker):
    def __init__(self, model): 
        self.model = model

    def predictions(self, inputs):
        
        logits = self.model.forward_one(np.round(inputs).astype(np.float32))
        return np.argmax(logits), logits

    def distance(self, input1, input2, min_, max_):
        return np.mean((input1 - input2) ** 2) / ((max_ - min_) ** 2)

    def print_distance(self, distance):
        return np.sqrt(distance * 1*28*28)

    def log_step(self, step, distance, spherical_step, source_step, message=''):
        print('Step {}: {:.5f}, stepsizes = {:.1e}/{:.1e}: {}'.format(
            step,
            self.print_distance(distance),
            spherical_step,
            source_step,
            message))

    def patch_attack(
            self,
            original,    #原始图像
            label,       #原始标签
            starting_point,   #初始对抗样本
            iterations=1000,  #总的查询次数
            min_=0.0,         
            max_=255.0,
            mode='targeted'):

        from numpy.linalg import norm
        from scipy import interpolate
        import collections

        #全部转换为torch来计算
        original = to_cuda(original)
        starting_point = to_cuda(starting_point)
        step = 0

        patch_num = 4   #横纵几等分
        patch_size = int(original.shape[0] / patch_num)


        success_num = 0    #成功和失败的次数
        fail_num = 0

        value_mask = value_mask_init(patch_num)
        noise_mask = noise_mask_init(starting_point, original, patch_num, patch_size)

        best_noise = starting_point - original
        current_min_noise = l2_distance(starting_point, original)

        #FIXME
        evolutionary_doc = np.zeros(iterations)   #记录下当前最小噪声   这个不管了，先不记录了

        while step < iterations:

            if torch.sum(value_mask * noise_mask) == 0:  #当前平分方法下没有可以查询的了
                #FIXME
                print("patch num * 2", step)
                patch_num *= 2

                if patch_num == 64:   #没必要了
                    print("only", step)
                    break

                patch_size = int(original.shape[0] / patch_num)

                value_mask = value_mask_init(patch_num)
                noise_mask = noise_mask_init(best_noise, original, patch_num, patch_size)


            total_mask = value_mask*noise_mask
            best_index = torch.argmax(total_mask)
            best_row, best_col = translate(best_index, patch_num)

            # print("best_row, best_col", best_row.item(), best_col.item())

            temp_noise = copy.deepcopy(best_noise)

            temp_noise[(best_row*patch_size):(best_row*patch_size+patch_size) , (best_col*patch_size):(best_col*patch_size+patch_size) ] = 0

            candidate = torch.clip(torch.round(original + temp_noise), 0, 255)
            
            if l2_distance(candidate, original) >= current_min_noise:
                # print("back")
                
                value_mask[best_row, best_col] = 0
                # print("not worth", torch.sum(value_mask).item())
                
                continue
            
            temp_result, temp_logits = self.predictions((candidate).cpu().numpy())
            

            if mode == 'untargeted':
                is_adversarial = (temp_result != label)
            else:
                is_adversarial = (temp_result == label)


            # #下面更新起点

            if is_adversarial:
                # print(step, current_min_noise.item(), l2_distance(candidate, original).item(), "Success")
                current_min_noise = l2_distance(candidate, original)
                success_num += 1
                best_noise = candidate - original
                noise_mask[best_row, best_col] = l2_distance(best_noise[(best_row*patch_size):(best_row*patch_size+patch_size) , (best_col*patch_size):(best_col*patch_size+patch_size) ], 0)
            else:
                # print("Fail")
                fail_num += 1
                value_mask[best_row, best_col] = 0


            step += 1

            


        final_best_adv_example = best_noise+original
        final_best_adv_example = final_best_adv_example.cpu().numpy().astype(np.float32)

        print("success_num", success_num)

        return final_best_adv_example, step


    def attack(
            self, 
            image,
            label,
            starting_point, 
            iterations=1000,
            val_samples = 1000,
            min_=0.0, 
            max_=255.0,
            mode = 'untargeted',
            strategy = 0):

        if mode == 'untargeted':
            if self.predictions(image)[0] != label:
                return image
            else:
                return self.patch_attack(image, label, starting_point, iterations, min_, max_, mode='untargeted')

        elif mode == 'targeted':
            if self.predictions(image)[0] == label:
                return image
            else:
                return self.patch_attack(image, label, starting_point, iterations, min_, max_, mode='targeted')