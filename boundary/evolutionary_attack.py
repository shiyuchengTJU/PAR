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
        logits = self.model.forward_one(np.round(inputs).astype(np.float32))
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
            step_decay_factor=0.99,
            big_size = 64,
            center_size = 40):

        from numpy.linalg import norm
        from scipy import interpolate
        import collections


        resize_factor = 4
        perturbed = starting_point   
        dis = self.distance(perturbed, original, min_, max_)  
        shape = [center_size, center_size]
        big_shape = [big_size, big_size, 3]
        decay_factor = 0.99
        init_source_step = copy.deepcopy(source_step)
        init_spherical_step = copy.deepcopy(spherical_step)

        center_shape = [center_size, center_size, 3]

        pert_shape = [int(shape[0]/resize_factor), int(shape[1]/resize_factor), 3]   


        if rescale_or_not == 1 or rescale_or_not == 5 or rescale_or_not == 55 or rescale_or_not == 29:
            evolution_path = np.zeros(pert_shape , dtype=original.dtype)
            diagonal_covariance = np.ones(pert_shape, dtype=original.dtype)   
        elif rescale_or_not == 4:
            evolution_path = np.zeros(center_shape, dtype=original.dtype)
            diagonal_covariance = np.ones(center_shape, dtype=original.dtype) 
        else:
            evolution_path = np.zeros(big_shape, dtype=original.dtype)
            diagonal_covariance = np.ones(big_shape, dtype=original.dtype) 


        c = 0.001                                                   
        stats_step_adversarial = collections.deque(maxlen=20)

        neg_improve_num = 0

        evolutionary_doc = np.zeros(iterations)   
        best_dis = 0

        success_num = 0

        if rescale_or_not == 15 or rescale_or_not == 22 or rescale_or_not == 39 or rescale_or_not == 23 or rescale_or_not == 29 or rescale_or_not == 31 or rescale_or_not == 33 or rescale_or_not == 34 or rescale_or_not == 35:  # 修正均值,定义是否学习的flag变量
            amend_flag = False
            amend_list = []   

        if rescale_or_not == 16:
            amend_list = []   

        if rescale_or_not == 17:
            amend = 0   

        if rescale_or_not == 18 or rescale_or_not == 19:
            success_list = []
            fail_list = []

        if rescale_or_not == 37:
            last_50_success = 0

        if rescale_or_not == 21 or rescale_or_not == 24 or rescale_or_not == 25 or rescale_or_not == 26 or rescale_or_not == 27 or rescale_or_not == 28:
            success_noise_list = [perturbed - original]
            fail_noise_list = []

        if rescale_or_not == 28:
            temp_result, temp_logits = self.predictions(perturbed)
            success_prob = [self.cw_prob_calculator(temp_logits, label)]

        if rescale_or_not == 30 or rescale_or_not == 31:
            temp_result, temp_logits = self.predictions(perturbed)
            noise_list = [perturbed - original]
            prob_list = [self.cw_prob_calculator(temp_logits, label)]
            prob_saver = []
            sample_num = 10
            backup_perturbation = []   
            backup_prob = []

        if rescale_or_not == 33: 
            prob_est = 0


        for step in range(1, iterations + 1):
            unnormalized_source_direction = original - perturbed   
            source_norm = norm(unnormalized_source_direction)       


            clipper_counter = 0 

            if rescale_or_not == 1: 
                perturbation = np.random.normal(0, 1, pert_shape)     
                perturbation *= np.sqrt(diagonal_covariance)

                x = np.array(range(pert_shape[1]))
                y = np.array(range(pert_shape[2]))

                f1 = interpolate.interp2d(y, x, perturbation[0,:,:], kind='linear')
                newx = np.linspace(0, pert_shape[1], shape[1])  
                newy = np.linspace(0, pert_shape[0], shape[0])
                perturbation_mid = f1(newx, newy).reshape(shape[0], shape[1], 1)
                perturbation_large = np.zeros([big_size, big_size, 1])
                starting_pos = int((big_size - center_size) / 2)
                perturbation_large[starting_pos:(starting_pos+center_size), starting_pos:(starting_pos+center_size), :] = perturbation_mid



                
            elif rescale_or_not == 2:  
                perturbation_large = np.random.normal(0, 1, big_shape)



            elif rescale_or_not == 3:  
                perturbation_large = ortho(np.reshape(unnormalized_source_direction, (-1)))

            elif rescale_or_not == 4:  
                perturbation = np.random.normal(0, 1, center_shape)       
                perturbation *= np.sqrt(diagonal_covariance)

                starting_pos = int((big_size - center_size) / 2)
                perturbation_large = np.zeros([big_size, big_size, 3])
                perturbation_large[starting_pos:(starting_pos+center_size), starting_pos:(starting_pos+center_size), :] = perturbation

            elif rescale_or_not ==5:  
                perturbation = np.random.normal(0, 1, pert_shape)  
                perturbation *= np.sqrt(diagonal_covariance)

                x = np.array(range(pert_shape[1]))
                y = np.array(range(pert_shape[0]))
                f1 = interpolate.interp2d(y, x,perturbation[:,:,0], kind='linear')   
                f2 = interpolate.interp2d(y, x,perturbation[:,:,1], kind='linear')
                f3 = interpolate.interp2d(y, x,perturbation[:,:,2], kind='linear')
                newx = np.linspace(0, pert_shape[1], big_shape[1])  
                newy = np.linspace(0, pert_shape[0], big_shape[0])
                perturbation_large = np.concatenate([f1(newx, newy).reshape(big_shape[0], big_shape[1], 1), 
                                                     f2(newx, newy).reshape(big_shape[0], big_shape[1], 1), 
                                                     f3(newx, newy).reshape(big_shape[0], big_shape[1], 1)], axis=2)

            elif rescale_or_not == 6: 
                perturbation_large = np.random.normal(0, 1, big_shape)
                mask = griddle(perturbed - original, rate)  
                perturbation_large *= mask

            elif rescale_or_not == 7: 

                perturbation_large = np.random.normal(0, 1, big_shape)
                mask = griddle(perturbed - original, rate)  
                perturbation_large *= mask
                perturbation_large *= np.sqrt(diagonal_covariance)

            elif rescale_or_not ==55:  
                perturbation = np.random.normal(0, 1, pert_shape)      

                x = np.array(range(pert_shape[1]))
                y = np.array(range(pert_shape[0]))
                f1 = interpolate.interp2d(y, x,perturbation[:,:,0], kind='linear')
                f2 = interpolate.interp2d(y, x,perturbation[:,:,1], kind='linear')
                f3 = interpolate.interp2d(y, x,perturbation[:,:,2], kind='linear')
                newx = np.linspace(0, pert_shape[1], big_shape[1])  
                newy = np.linspace(0, pert_shape[0], big_shape[0])
                perturbation_large = np.concatenate([f1(newx, newy).reshape(big_shape[0], big_shape[1], 1), 
                                                     f2(newx, newy).reshape(big_shape[0], big_shape[1], 1), 
                                                     f3(newx, newy).reshape(big_shape[0], big_shape[1], 1)], axis=2)

            elif rescale_or_not == 8:  
                perturbation = np.random.normal(0, 1, big_shape)
                

                x = np.array(range(big_shape[1]))
                y = np.array(range(big_shape[0]))
                f1 = interpolate.interp2d(y, x,perturbation[:,:,0], kind='linear')
                f2 = interpolate.interp2d(y, x,perturbation[:,:,1], kind='linear')
                f3 = interpolate.interp2d(y, x,perturbation[:,:,2], kind='linear')

                perturbation_large = np.concatenate([f1(x, y).reshape(big_shape[0], big_shape[1], 1), 
                                                     f2(x, y).reshape(big_shape[0], big_shape[1], 1), 
                                                     f3(x, y).reshape(big_shape[0], big_shape[1], 1)], axis=2)

                mask = griddle(perturbed - original, rate)  
                perturbation_large *= mask


            elif rescale_or_not == 9:
                perturbation = np.random.normal(0, 1, big_shape)
                perturbation *= np.sqrt(diagonal_covariance)

                x = np.array(range(big_shape[1]))
                y = np.array(range(big_shape[0]))
                f1 = interpolate.interp2d(y, x,perturbation[:,:,0], kind='linear')
                f2 = interpolate.interp2d(y, x,perturbation[:,:,1], kind='linear')
                f3 = interpolate.interp2d(y, x,perturbation[:,:,2], kind='linear')

                perturbation_large = np.concatenate([f1(x, y).reshape(big_shape[0], big_shape[1], 1), 
                                                     f2(x, y).reshape(big_shape[0], big_shape[1], 1), 
                                                     f3(x, y).reshape(big_shape[0], big_shape[1], 1)], axis=2)

                mask = griddle(perturbed - original, rate)  
                perturbation_large *= mask

            elif rescale_or_not == 10:  
                perturbation = np.random.normal(0, 1, big_shape)
                

                x = np.array(range(big_shape[1]))
                y = np.array(range(big_shape[0]))
                f1 = interpolate.interp2d(y, x,perturbation[:,:,0], kind='cubic')
                f2 = interpolate.interp2d(y, x,perturbation[:,:,1], kind='cubic')
                f3 = interpolate.interp2d(y, x,perturbation[:,:,2], kind='cubic')

                perturbation_large = np.concatenate([f1(x, y).reshape(big_shape[0], big_shape[1], 1), 
                                                     f2(x, y).reshape(big_shape[0], big_shape[1], 1), 
                                                     f3(x, y).reshape(big_shape[0], big_shape[1], 1)], axis=2)

                mask = griddle(perturbed - original, rate)  
                perturbation_large *= mask

            elif rescale_or_not == 11: 
                perturbation = np.random.normal(0, 1, big_shape)
                

                x = np.array(range(big_shape[1]))
                y = np.array(range(big_shape[0]))
                f1 = interpolate.interp2d(y, x,perturbation[:,:,0], kind='quintic')
                f2 = interpolate.interp2d(y, x,perturbation[:,:,1], kind='quintic')
                f3 = interpolate.interp2d(y, x,perturbation[:,:,2], kind='quintic')

                perturbation_large = np.concatenate([f1(x, y).reshape(big_shape[0], big_shape[1], 1), 
                                                     f2(x, y).reshape(big_shape[0], big_shape[1], 1), 
                                                     f3(x, y).reshape(big_shape[0], big_shape[1], 1)], axis=2)

                mask = griddle(perturbed - original, rate) 
                perturbation_large *= mask


            elif rescale_or_not == 12: 
                var = np.abs(perturbed - original)
                perturbation_large = np.random.normal(0, var, big_shape)

            elif rescale_or_not == 13: 
                while 1:
                    var = np.abs(perturbed - original)
                    perturbation_large = np.random.normal(0, var, big_shape)

                    line_candidate = perturbed + source_step * unnormalized_source_direction
                    candidate = line_candidate + spherical_step * source_norm * perturbation_large / max(norm(perturbation_large), 1e-6)
                    candidate = original - (original - candidate) / norm(original - candidate) * norm(original - line_candidate)
                    candidate = clip(candidate, min_, max_)

                    if l2_distance(original, candidate) < l2_distance(original, perturbed):
                        break 
                    else:
                        clipper_counter += 1


            elif rescale_or_not == 14:
                var = np.abs(perturbed - original)
                # print(np.max(var))
                var+=1
                perturbation_large = np.random.normal(0, var, big_shape)


            elif rescale_or_not == 15: 
                # print(amend_flag)
                if amend_flag == True:  
                    temp_amend_array = np.array(amend_list)
                    mean = -1*np.mean(temp_amend_array, axis=0)
                    # print("mean", np.mean(mean))
                    perturbation_large = np.random.normal(mean, 1, big_shape)
                else:
                    perturbation_large = np.random.normal(0, 1, big_shape)


            elif rescale_or_not == 16: 
                # print(amend_flag)
                if len(amend_list) != 0:  
                    temp_amend_array = np.array(amend_list)
                    mean = -1*np.mean(temp_amend_array, axis=0)
                    # print("mean", np.mean(mean))
                    perturbation_large = np.random.normal(mean, 1, big_shape)
                else:
                    perturbation_large = np.random.normal(0, 1, big_shape)


            elif rescale_or_not == 17: 
                mean = -1*amend
                perturbation_large = np.random.normal(mean, 1, big_shape)

            elif rescale_or_not == 18:  
                
                if len(fail_list) >= 1 and len(success_list) >= 1: 
                    fail_array = np.array(fail_list)
                    success_array = np.array(success_list)
                    while 1:
                        var = np.abs(perturbed - original)
                        perturbation_large = np.random.normal(0, var, big_shape)
                        temp_fail_dist = np.sqrt(np.sum((fail_array - perturbation_large)**2)/len(fail_list))
                        temp_success_dist = np.sqrt(np.sum((success_array - perturbation_large)**2)/len(success_list))

                        
                        if temp_success_dist < 1.1*temp_fail_dist:   

                            break

                else:
                    var = np.abs(perturbed - original)
                    perturbation_large = np.random.normal(0, var, big_shape)


            elif rescale_or_not == 19: 
                
                if len(fail_list) >= 1 and len(success_list) >= 1:  
                    fail_array = np.array(fail_list)
                    success_array = np.array(success_list)
                    while 1:
                        perturbation_large = np.random.normal(0, 1, big_shape)


                        temp_fail_dist = np.sqrt(np.sum((fail_array - perturbation_large)**2)/len(fail_list))
                        temp_success_dist = np.sqrt(np.sum((success_array - perturbation_large)**2)/len(success_list))
                        
                        if temp_success_dist < 1.1*temp_fail_dist:   
                            break

                else:
                    perturbation_large = np.random.normal(0, 1, big_shape)


            elif rescale_or_not == 20:  
                var = np.abs(perturbed - original)
                perturbation_large = np.random.normal(0, var, big_shape)
                mask = griddle(perturbed - original, rate)   
                perturbation_large *= mask


            elif rescale_or_not == 21: 
                if len(fail_noise_list) >= 1 and len(success_noise_list) >= 1: 

                    temp_line_candidate = perturbed + source_step * unnormalized_source_direction
                    while 1:
                        var = np.abs(perturbed - original)
                        perturbation_large = np.random.normal(0, var, big_shape)

                        
                        temp_candidate = temp_line_candidate + spherical_step * source_norm * perturbation_large / max(norm(perturbation_large), 1e-6)
                        temp_candidate = original - (original - temp_candidate) / norm(original - temp_candidate) * norm(original - temp_line_candidate)
                        temp_candidate = clip(temp_candidate, min_, max_)

                        temp_input_noise = temp_candidate - original

                        fail_noise_dist = np.sqrt(np.sum((np.array(fail_noise_list) - temp_input_noise)**2)/len(fail_noise_list))
                        success_noise_dist = np.sqrt(np.sum((np.array(success_noise_list) - temp_input_noise)**2)/len(success_noise_list))
                        print(fail_noise_dist, success_noise_dist)
                        
                        if success_noise_dist < 1.5*fail_noise_dist:   
                            print(len(fail_noise_list), len(success_noise_list))
                            break

                else:
                    var = np.abs(perturbed - original)
                    perturbation_large = np.random.normal(0, var, big_shape)

            elif rescale_or_not == 22 or rescale_or_not == 39: 
                if amend_flag == True:  
                    temp_amend_array = np.array(amend_list)
                    mean = -1*np.mean(temp_amend_array, axis=0)

                    var = np.abs(perturbed - original)
                    perturbation_large = np.random.normal(mean, var, big_shape)
                    mask = griddle(perturbed - original, rate)   
                    perturbation_large *= mask
                else:
                    var = np.abs(perturbed - original)
                    perturbation_large = np.random.normal(0, var, big_shape)
                    mask = griddle(perturbed - original, rate)   
                    perturbation_large *= mask

            elif rescale_or_not == 33:   
                if amend_flag == True:  
                    temp_amend_array = np.array(amend_list)
                    mean = -1*np.mean(temp_amend_array, axis=0)
                    var = np.abs(perturbed - original)
                    perturbation_large = np.random.normal(mean, var, big_shape)
                    mask = griddle(perturbed - original, rate)  
                    perturbation_large *= mask
                else:
                    var = np.abs(perturbed - original)
                    perturbation_large = np.random.normal(0, var, big_shape)
                    mask = griddle(perturbed - original, rate)  
                    perturbation_large *= mask

            elif rescale_or_not == 34: 
                if amend_flag == True: 
                    temp_amend_array = np.array(amend_list)
                    mean = -1*np.mean(temp_amend_array, axis=0)
                    var = np.abs(perturbed - original)
                    perturbation_large = np.random.normal(mean, var, big_shape)
                    mask = griddle(perturbed - original, rate)  
                    perturbation_large *= mask
                else:
                    var = np.abs(perturbed - original)
                    perturbation_large = np.random.normal(0, var, big_shape)
                    mask = griddle(perturbed - original, rate)  
                    perturbation_large *= mask

            elif rescale_or_not == 35: 
                if amend_flag == True:   
                    temp_amend_array = np.array(amend_list)
                    mean = -1*np.mean(temp_amend_array, axis=0)
                    var = np.abs(perturbed - original)
                    perturbation_large = np.random.normal(mean, var, big_shape)
                    mask = griddle(perturbed - original, rate)  
                    perturbation_large *= mask
                else:
                    var = np.abs(perturbed - original)
                    perturbation_large = np.random.normal(0, var, big_shape)
                    mask = var / np.max(var)
                    perturbation_large *= mask


            elif rescale_or_not == 23: 
                if amend_flag == True:   
                    temp_amend_array = np.array(amend_list)
                    mean = -1*np.mean(temp_amend_array, axis=0)
                    # print("mean", np.mean(mean))
                    var = np.abs(perturbed - original)
                    perturbation_large = np.random.normal(mean, var, big_shape)
                else:
                    var = np.abs(perturbed - original)
                    perturbation_large = np.random.normal(0, var, big_shape)

            elif rescale_or_not == 24:  
                success_noise_mean = np.mean(np.array(success_noise_list), axis=0)

                # print(np.shape(success_noise_mean))
                # print(np.max(success_noise_mean), np.min(success_noise_mean), np.mean(success_noise_mean))

                var = np.abs(success_noise_mean)
                perturbation_large = np.random.normal(0, var, big_shape)

            elif rescale_or_not == 25:  
                total_noise_array = np.append(np.array(success_noise_list), -1*np.array(fail_noise_list))
                total_noise_mean = np.mean(total_noise_array, axis=0)


                var = np.abs(total_noise_mean)
                perturbation_large = np.random.normal(0, var, big_shape)

            elif rescale_or_not == 26: 
                success_noise_mean = np.mean(np.array(success_noise_list), axis=0)
                var = np.abs(success_noise_mean)
                perturbation_large = np.random.normal(0, var, big_shape)

            elif rescale_or_not == 27: 
                total_noise_array = np.append(np.array(success_noise_list), np.array(fail_noise_list))
                total_noise_mean = np.mean(total_noise_array, axis=0)
                var = np.abs(total_noise_mean)
                perturbation_large = np.random.normal(0, var, big_shape)

            elif rescale_or_not == 28:  
                if len(success_noise_list)==1: 
                    var = np.abs(perturbed - original)
                    perturbation_large = np.random.normal(0, var, big_shape)
                else:  
                    var_bias = np.array(success_noise_list[-1]) - np.array(success_noise_list[-2])
                    var_bias /= norm(var_bias)
                    var = np.abs(perturbed - original)
                    var /= norm(var)
                    
                    if success_prob[-1] < success_prob[-2]: 
                        # print("down")
                        var -= var_bias
                        var[var < 0] = 0
                        perturbation_large = np.random.normal(0, var, big_shape)
                    else:   
                        var += var_bias
                        var[var < 0] = 0
                        perturbation_large = np.random.normal(0, var, big_shape)
                    # print("mean of var and var_bias", np.mean(np.abs(var)), np.mean(var_bias))
            

            elif rescale_or_not == 29:  
                starting_pos = int((big_size - center_size) / 2)
                if amend_flag == True:  
                    temp_amend_array = np.array(amend_list)
                    mean = -1*np.mean(temp_amend_array, axis=0)
                else:
                    mean = 0

                down_sample_noise = perturbed - original
                down_sample_mid = down_sample_noise[starting_pos:(starting_pos+center_size), starting_pos:(starting_pos+center_size), :]
                center_trans_shape = [1, 3, center_size, center_size]
                down_sampler = torch.nn.AvgPool2d(resize_factor, stride=resize_factor)
                trans_down_sample_mid = torch.from_numpy(np.reshape(np.transpose(down_sample_mid, (2, 0, 1)), center_trans_shape))
                down_sample_small = down_sampler(trans_down_sample_mid).numpy()[0]
                down_sample_small_noise = np.transpose(down_sample_small, (1,2,0))

                perturbation = np.random.normal(mean, np.abs(down_sample_small_noise), pert_shape)       

                x = np.array(range(pert_shape[1]))
                y = np.array(range(pert_shape[0]))
                f1 = interpolate.interp2d(y, x,perturbation[:,:,0], kind='linear')
                f2 = interpolate.interp2d(y, x,perturbation[:,:,1], kind='linear')
                f3 = interpolate.interp2d(y, x,perturbation[:,:,2], kind='linear')
                newx = np.linspace(0, pert_shape[1], shape[1])  
                newy = np.linspace(0, pert_shape[0], shape[0])
                perturbation_mid = np.concatenate([f1(newx, newy).reshape(shape[0], shape[1], 1), 
                                                   f2(newx, newy).reshape(shape[0], shape[1], 1), 
                                                   f3(newx, newy).reshape(shape[0], shape[1], 1)], axis=2)
                perturbation_large = np.zeros([big_size, big_size, 3])
                
                perturbation_large[starting_pos:(starting_pos+center_size), starting_pos:(starting_pos+center_size), :] = perturbation_mid
         
                mask = griddle(perturbed - original, rate)
                perturbation_large *= mask


            if rescale_or_not == 32: 
                perturbation_large = np.random.normal(0, 1, big_shape)
            
            elif rescale_or_not == 30:  
                prob_array = np.array(prob_list)
                noise_array = np.array(noise_list)
                if len(prob_array[prob_array>0]) >= sample_num:
                    if amend_flag == True:   
                        temp_amend_array = np.array(amend_list)
                        mean = -1*np.mean(temp_amend_array, axis=0)
                    else:
                        mean = 0

                    temp_line_candidate = perturbed + source_step * unnormalized_source_direction
                    for temp_counter in range(5):
                        var = np.abs(perturbed - original)
                        perturbation_large = np.random.normal(mean, var, big_shape)


                        temp_candidate = temp_line_candidate + spherical_step * source_norm * perturbation_large / max(norm(perturbation_large), 1e-6)
                        temp_candidate = original - (original - temp_candidate) / norm(original - temp_candidate) * norm(original - temp_line_candidate)
                        temp_candidate = clip(temp_candidate, min_, max_)
                        temp_input_noise = temp_candidate - original

                        
                        dist_array = np.sqrt(np.sum((noise_array/255.0 - temp_input_noise/255.0)**2, axis=(1,2,3)))  #距离数组

                        index = np.argsort(dist_array)
 

                        near_prob = prob_array.take(index[:sample_num])

                        mean_prob = np.mean(near_prob)
                     

                        if mean_prob > np.mean(prob_array) or mean_prob > 0:
                            break

                        else:
                            backup_perturbation.append(perturbation_large)
                            backup_prob.append(mean_prob)
                        

                    
                        if temp_counter==4:  
                            backup_prob_array = np.array(backup_prob)
                            backup_index = np.argsort(backup_prob_array)[-1]  
                            perturbation_large = backup_perturbation[backup_index]

                            backup_prob[backup_index] = -9999


                else:
                    var = np.abs(perturbed - original)
                    perturbation_large = np.random.normal(0, var, big_shape)
                    mask = griddle(perturbed - original, rate)  
                    perturbation_large *= mask
                    

            elif rescale_or_not == 31:   
                prob_array = np.array(prob_list)
                noise_array = np.array(noise_list)
                if len(prob_array[prob_array>0]) >= sample_num:
                    temp_line_candidate = perturbed + source_step * unnormalized_source_direction

                    for temp_counter in range(5):
                        var = np.abs(perturbed - original)
                        perturbation_large = np.random.normal(0, var, big_shape)

                        temp_candidate = temp_line_candidate + spherical_step * source_norm * perturbation_large / max(norm(perturbation_large), 1e-6)
                        temp_candidate = original - (original - temp_candidate) / norm(original - temp_candidate) * norm(original - temp_line_candidate)
                        temp_candidate = clip(temp_candidate, min_, max_)
                        temp_input_noise = temp_candidate - original

                        
                        dist_array = np.sqrt(np.sum((noise_array/255.0 - temp_input_noise/255.0)**2, axis=(1,2,3)))  #距离数组

                        index = np.argsort(dist_array)


                        near_prob = prob_array.take(index[:sample_num])
                        # print("near_prob", near_prob)
                        mean_prob = np.mean(near_prob)
                   

                        if mean_prob > np.mean(prob_array) or mean_prob > 0:
                            break
                        


                        else:
                            backup_perturbation.append(perturbation_large)
                            backup_prob.append(mean_prob)
                        

                    
                        if temp_counter==4:  
                            backup_prob_array = np.array(backup_prob)
                            backup_index = np.argsort(backup_prob_array)[-1]  
                            perturbation_large = backup_perturbation[backup_index]
                            # print("continue", backup_prob_array[backup_index])
                            # print(np.sort(backup_prob_array))
                            backup_prob[backup_index] = -9999
                else:
                    var = np.abs(perturbed - original)
                    perturbation_large = np.random.normal(0, var, big_shape)



            elif rescale_or_not == 36:  
                perturbation_large = np.random.normal(0, 1, big_shape)
                source_step = self.distance(perturbed, original, min_, max_)
                spherical_step = source_step * 30


            elif rescale_or_not == 37:
                perturbation_large = np.random.normal(0, 1, big_shape)
                if step%50 == 0:
                    if last_50_success >=35:
                        source_step *= 2
                        spherical_step = source_step * 30
                    elif last_50_success <= 15:
                        source_step /= 2
                        spherical_step = source_step * 30
                    last_50_success=0

            elif rescale_or_not == 38:  
                perturbation_large = np.random.normal(0, 1, big_shape)



            line_candidate = perturbed + source_step * unnormalized_source_direction   
            
            # print("line_candidate, perturbation_large, rescale_or_not", line_candidate.shape, perturbation_large.shape, rescale_or_not)
            
            candidate = line_candidate + spherical_step * source_norm * perturbation_large / max(norm(perturbation_large), 1e-6)

            candidate = original - (original - candidate) / norm(original - candidate) * norm(original - line_candidate)
            candidate = clip(candidate, min_, max_)

            # #check
            # print(candidate.shape, original.shape)

            temp_result, temp_logits = self.predictions(candidate)

            

            if mode == 'untargeted':
                is_adversarial = (temp_result != label)
            else:
                is_adversarial = (temp_result == label)
            stats_step_adversarial.appendleft(is_adversarial)



            if rescale_or_not == 30 or rescale_or_not == 31: 
                noise_list.append(candidate - original)
                this_prob = self.cw_prob_calculator(temp_logits, label)
                prob_list.append(this_prob)
 


            if is_adversarial:
                

                improvement = l2_distance(original, perturbed) - l2_distance(original, candidate)
                if improvement < 0:
                    neg_improve_num += 1

                if rescale_or_not == 39 and improvement < 0:
                    temp_possibility = np.random.rand(1)[0]
                    if (1.0*step/iterations) < temp_possibility:

                        new_perturbed = None
                    else:
                        success_num += 1
                        new_perturbed = candidate
                        new_dis = self.distance(candidate, starting_point, min_, max_)

                        best_dis = new_dis

                elif rescale_or_not == 38 and improvement < 0:  
                    new_perturbed = None
                else:
                    success_num += 1
                    new_perturbed = candidate
                    new_dis = self.distance(candidate, starting_point, min_, max_)

                    best_dis = new_dis

                if rescale_or_not==1 or rescale_or_not==5 or rescale_or_not==4 or rescale_or_not==55 or rescale_or_not==29:
                    evolution_path = decay_factor * evolution_path + np.sqrt(1 - decay_factor ** 2) * perturbation
                elif rescale_or_not == 7:
                    evolution_path = decay_factor * evolution_path + np.sqrt(1 - decay_factor ** 2) * perturbation_large
                else:
                    evolution_path = decay_factor * evolution_path + np.sqrt(1 - decay_factor ** 2) * perturbation_large
                diagonal_covariance = (1 - c) * diagonal_covariance + c * (evolution_path ** 2)

                if rescale_or_not == 15 or rescale_or_not == 22 or rescale_or_not == 39 or rescale_or_not == 23 or rescale_or_not == 29 or rescale_or_not == 31 or rescale_or_not == 33 or rescale_or_not == 34 or rescale_or_not == 35:
                    if amend_flag == True:   
                        amend_flag = False
                        amend_list=[]

                if rescale_or_not == 18 or rescale_or_not == 19:
                    success_list.append(perturbation_large)

                if rescale_or_not == 21 or rescale_or_not == 24:
                    success_noise_list.append(candidate - original)

                if rescale_or_not == 26 or rescale_or_not == 27:
                    weight = self.cw_prob_calculator(temp_logits, label)
                    success_noise_list.append(weight*(candidate - original))

                if rescale_or_not == 28:
                    success_noise_list.append(candidate - original)
                    success_prob.append(self.cw_prob_calculator(temp_logits, label))

                if rescale_or_not == 30 or rescale_or_not == 31:
                    backup_perturbation = []
                    backup_prob = []

                if rescale_or_not == 32 or rescale_or_not == 22 or rescale_or_not == 35:  #step decay
                    source_step *= step_decay_factor
                    spherical_step *= step_decay_factor

                if rescale_or_not == 34:  
                    source_step = init_source_step
                    spherical_step = init_spherical_step

                if rescale_or_not == 33:  
                    prob_est += 0.01 

                if rescale_or_not == 37:
                    last_50_success += 1


            else:
                new_perturbed = None

                if rescale_or_not == 15 or rescale_or_not == 22 or rescale_or_not == 39 or rescale_or_not == 23 or rescale_or_not == 31:
                    if amend_flag == True:   
                        amend_list.append(perturbation_large)
                    else:  
                        amend_flag = True
                        amend_list.append(perturbation_large)

                if rescale_or_not == 34:  
                    if amend_flag == True:  
                        amend_list.append(perturbation_large)
                        if len(amend_list) == 50:   
                            source_step /= 1.5
                            spherical_step /= 1.5
                            amend_flag = False
                            amend_list=[]
                    else: 
                        amend_flag = True
                        amend_list.append(perturbation_large) 

                if rescale_or_not == 16:
                    amend_list.append(perturbation_large)

                if rescale_or_not == 17:
                    amend = 0.7*amend + 0.3*perturbation_large

                if rescale_or_not == 18 or rescale_or_not == 19:
                    fail_list.append(perturbation_large)

                if rescale_or_not == 21 or rescale_or_not == 24:
                    fail_noise_list.append(candidate - original)

                if rescale_or_not == 26 or rescale_or_not == 27:
                    weight = self.cw_prob_calculator(temp_logits, label)
                    fail_noise_list.append(weight*(candidate - original))

                if rescale_or_not == 29:
                    if amend_flag == True:  
                        amend_list.append(perturbation)
                    else: 
                        amend_flag = True
                        amend_list.append(perturbation)

            if rescale_or_not == 33 and step>0 and step%100 == 0:  
                if prob_est > 0.5:
                    source_step *= 1.5
                    spherical_step *= 1.5

                if prob_est < 0.2:
                    source_step /= 1.5
                    spherical_step /= 1.5

                prob_est = 0


            message = ''
            if new_perturbed is not None:
                abs_improvement = dis - new_dis
                rel_improvement = abs_improvement / dis
                message = 'd. reduced by {:.2f}% ({:.4e})'.format(
                    rel_improvement * 100, abs_improvement)

                perturbed = new_perturbed
                dis = new_dis



            evolutionary_doc[step-1] = l2_distance(original, perturbed)


            if len(stats_step_adversarial) == stats_step_adversarial.maxlen:
                p_step = np.mean(stats_step_adversarial)
                n_step = len(stats_step_adversarial)
                source_step *= np.exp(p_step - 0.2)
                stats_step_adversarial.clear()



        print("success_num, neg_improve_num", success_num, neg_improve_num)
        if rescale_or_not == 13: 
            print("clipper_counter", clipper_counter)

        
        if rescale_or_not == 21:   
            fail_noise_mean = np.mean(np.array(fail_noise_list), axis=0)
            success_noise_mean = np.mean(np.array(success_noise_list), axis=0)

            total_noise_list = fail_noise_list + success_noise_list
            total_noise_array = np.array(total_noise_list)
            total_noise_mean = np.mean(total_noise_array, axis=0)

            fail_noise_dist = np.sqrt(np.sum((np.array(fail_noise_list) - fail_noise_mean)**2)/len(fail_noise_list))
            success_noise_dist = np.sqrt(np.sum((np.array(success_noise_list) - success_noise_mean)**2)/len(success_noise_list))
            total_noise_dist = np.sqrt(np.sum((total_noise_array - total_noise_mean)**2)/len(total_noise_list))
            print("fail_dist", fail_noise_dist)
            print("success_dist", success_noise_dist)
            print("total_dist", total_noise_dist)


        return perturbed.astype(np.float32), evolutionary_doc


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
            step_decay_factor=0.99,
            big_size = 64,
            center_size = 40,
            mode = 'untargeted'):

        if mode == 'untargeted':
            if self.predictions(image)[0] != label:
                return image
            else:
                return self.evolutionary_attack(
                    image, label, starting_point, initial_time, time_limit,
                    iterations, spherical_step, source_step, 
                    min_, max_, mode='untargeted', rescale_or_not=rescale_or_not, rate = rate, step_decay_factor=step_decay_factor, big_size=big_size, center_size=center_size)

        elif mode == 'targeted':
            if self.predictions(image)[0] == label:
                return image
            else:
                return self.evolutionary_attack(
                    image, label, starting_point, initial_time, time_limit,
                    iterations, spherical_step, source_step, 
                    min_, max_, mode='targeted', rescale_or_not=rescale_or_not, rate = rate, step_decay_factor=step_decay_factor, big_size=big_size, center_size=center_size)


        
