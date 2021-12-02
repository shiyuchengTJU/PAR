import numpy as np
import time
import copy
from foolbox.utils import crossentropy, softmax
import torch

def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))

def l2_distance(a, b):
    return (np.sum((a/255.0 - b/255.0) ** 2))**0.5

def ortho(noise):   
    noise_dim=noise.shape

    xr=(np.random.rand(noise_dim[0]))

    xo=xr-(np.sum(noise*xr)/np.sum(noise**2))*noise

    xo -= np.mean(xo)
    xo=np.reshape(xo, (1, 28, 28))

    return xo


def cw_loss_calculator(label, inputs):
    return np.max(inputs) - inputs[label]


class adv_example_with_distance:   
    def __init__(self, adv_example, distance):
        self.adv_example = adv_example
        self.distance = distance


def griddle(noise, rate):  
    noise_temp = np.round(noise)  
    noise_temp = np.abs(noise_temp)
    negative_rate = 1 - rate


    perturbed_num = np.sum(noise_temp != 0)  
    deleted = 0  

    for i in range(1, 256):
        if np.sum(noise_temp == i) != 0:   
            temp_deleted = deleted + np.sum(noise_temp == i)
            if temp_deleted/(perturbed_num * 1.0) >= negative_rate: 
                lottery_rate = (negative_rate*perturbed_num*1.0 - deleted)/(np.sum(noise_temp == i))
                temp_A = copy.deepcopy(noise_temp)
                temp_A[temp_A != i] = 0
                temp_B =  np.random.uniform(0, 1, np.shape(temp_A))
                temp_B[temp_B<lottery_rate] = 0
                temp_B[temp_B>=lottery_rate] = 1

                noise_temp = noise_temp - temp_A + temp_A*temp_B
                break

            else:
                noise_temp[noise_temp == i] = 0
                deleted = temp_deleted


    mask = copy.deepcopy(noise_temp)   
    mask[mask != 0] = 1



    return mask




def clip(x, min_x=-1, max_x=1):
    x[x < min_x] = min_x
    x[x > max_x] = max_x
    return x


def l2_distance(a, b):
    return (np.sum((np.round(a)/255.0 - np.round(b)/255.0) ** 2))**0.5


class Attacker:
    def __init__(self, model):
        self.model = model

    def attack(self, inputs):
        return NotImplementedError

    def attack_target(self, inputs, targets):
        return NotImplementedError


class EvolutionaryAttack(Attacker):
    def __init__(self, model):
        self.model = model

    def ce_and_cw_loss(self, inputs, label):
        logits = self.model.predictions(np.round(inputs).astype(np.float32))
        ce_loss = crossentropy(label, logits)
        cw_loss = cw_loss_calculator(label, logits)

        return ce_loss, cw_loss

    def cw_prob_calculator(self, logits, label):
        predict_label = np.argmax(logits)
        exp_logits = np.exp(logits)
        prob = exp_logits/np.sum(exp_logits)

        if predict_label != label:
            cw_prob = np.max(prob) - prob[label]
        else:
            temp_prob = copy.deepcopy(prob)
            temp_prob[label] = -9999
            near_label = np.argmax(temp_prob)
            cw_prob = prob[near_label] - prob[label]
        return cw_prob

    def predictions(self, inputs):
        logits = self.model.predictions(np.round(inputs).astype(np.float32))
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

    def evolutionary_attack(
            self,
            original,
            label,
            starting_point,
            initial_time,
            time_limit=10,
            iterations=1000,
            spherical_step=3e-2,
            source_step=1e-2,
            min_=0.0,
            max_=255.0,
            mode='targeted',
            rescale_or_not = False,
            rate = 0.2,
            step_decay_factor=0.99):   

        from numpy.linalg import norm
        from scipy import interpolate
        import collections


        resize_factor = 4
        perturbed = starting_point   
        dis = self.distance(perturbed, original, min_, max_)   
        big_size = 64
        center_size = 40    
        shape = [center_size, center_size]
        big_shape = [big_size, big_size, 3]
        decay_factor = 0.99
        init_source_step = copy.deepcopy(source_step)
        init_spherical_step = copy.deepcopy(spherical_step)
        center_shape = [center_size, center_size, 3]

        pert_shape = [int(shape[0]/resize_factor), int(shape[1]/resize_factor), 3]   

        
        c = 0.001                                                  
        stats_step_adversarial = collections.deque(maxlen=20)



        evolutionary_doc = np.zeros(iterations)  
        best_dis = 0   

        dt = np.dtype([('adv_example', np.ndarray),
                       ('distance', np.dtype('float64'))])
        adv_list = np.array([(starting_point, self.distance(original, starting_point, min_, max_))], dtype = dt)

        dist_success_list = []

        for step in range(1, iterations + 1):

            success_num = 0
            

            adv_list = sorted(adv_list, key=lambda x: x[1])
            perturbed = adv_list[0][0]
            
            
            for sample_step in range(100):  
                unnormalized_source_direction = original - perturbed    
                source_norm = norm(unnormalized_source_direction)      

                clipper_counter = 0  

                    
                if rescale_or_not == 2:  
                    perturbation_large = np.random.normal(0, 1, big_shape)




                line_candidate = perturbed + source_step * unnormalized_source_direction   #轴向不变
                candidate = line_candidate + spherical_step * source_norm * perturbation_large / max(norm(perturbation_large), 1e-6)

                candidate = original - (original - candidate) / norm(original - candidate) * norm(original - line_candidate)
                candidate = clip(candidate, min_, max_)

                temp_result, temp_logits = self.predictions(candidate)

                

                if mode == 'untargeted':
                    is_adversarial = (temp_result != label)
                else:
                    is_adversarial = (temp_result == label)
                stats_step_adversarial.appendleft(is_adversarial)


                if is_adversarial:


                    success_num+=1

                    adv_list = np.append(adv_list, np.array([(candidate, self.distance(
                        original, candidate, min_, max_))], dtype = dt))


                else:
                    new_perturbed = None

            print(adv_list[0][1], success_num)
            dist_success_list.append((adv_list[0][1], success_num))
            adv_list = np.delete(adv_list, 0)



        return perturbed.astype(np.float32), dist_success_list

    def attack(
            self, 
            image,
            label,
            starting_point, 
            initial_time,
            time_limit=10,
            iterations=1000, 
            spherical_step=3e-2, 
            source_step=1e-3,
            min_=0.0, 
            max_=255.0,
            rescale_or_not = False,
            rate = 0.2,
            step_decay_factor=0.99):

        if self.predictions(image)[0] != label:
            return image
        else:
            return self.evolutionary_attack(
                image, label, starting_point, initial_time, time_limit,
                iterations, spherical_step, source_step, 
                min_, max_, mode='untargeted', rescale_or_not=rescale_or_not, rate = rate, step_decay_factor=step_decay_factor)
