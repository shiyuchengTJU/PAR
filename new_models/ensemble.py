from .inception_v4.inception_v4 import inceptionv4
from .inception_small.inception_small import Inception3
from .resnet.resnet import resnet34
from .resnet.resnet_model import Model as resnet18
from .inception_resnet.inception_resnet import inceptionresnetv2
from .densenet.small_densenet import densenet161 as densenet161_small
from .vgg19.vgg_adv import  vgg19_bn as adv_vgg_19
from torch import nn
import torch
import os

from torch.autograd import Variable
from new_models.nasnet.nasnet import nasnetalarge


class EnsembleNet_resnet(nn.Module):
    def __init__(self):
        super(EnsembleNet_resnet, self).__init__()
        #self.model_1 = torch.nn.DataParallel(inception_v3()).cuda().eval()
        #self.model_2 = torch.nn.DataParallel(resnet34()).cuda().eval()
        self.model = resnet18().eval()

        path = os.path.dirname(os.path.abspath(__file__))
        #model_1_path = os.path.join(path, 'inception', 'inception_v3_adv_125.pt')
        #model_2_path = os.path.join(path, 'resnet', 'big_resnet34_ori_7731.pt')
        model_path = os.path.join(path, 'resnet', 'converted_pytorch.pt')

        #self.model_1.load_state_dict(torch.load(model_1_path))
        #self.model_2.load_state_dict(torch.load(model_2_path))
        self.model.load_state_dict(torch.load(model_path))
        self.model = self.model.cuda()

    def forward(self, x):
        #output_1 = self.model_1(x)
        #output_2 = self.model_2(x)
        #x = (output_1 + output_2) / 2
        output = self.model(x)
        return output


class EnsembleNet_inception(nn.Module):
    def __init__(self):
        super(EnsembleNet_inception, self).__init__()
        self.model = torch.nn.DataParallel(inception_v3()).cuda().eval()
        # self.denoiser = torch.nn.DataParallel(DUNET())
        #self.model_2 = torch.nn.DataParallel(resnet34()).cuda().eval()
        # self.model_3 = resnet18().eval()

        path = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(path, 'inception', 'inception_v3_bilinear_48.pt')
        # denoiser_path = os.path.join(path, 'denoiser', 'denoiser_net_15.pth')
        #model_2_path = os.path.join(path, 'resnet', 'big_resnet34_ori_7731.pt')
        # model_3_path = os.path.join(path, 'resnet', 'converted_pytorch.pt')

        self.model.load_state_dict(torch.load(model_path))
        # self.denoiser.load_state_dict(torch.load(denoiser_path)['model'])
        # self.dennoiser = self.denoiser.cuda() 
        #self.model_2.load_state_dict(torch.load(model_2_path))
        # self.model_3.load_state_dict(torch.load(model_3_path))
        self.model = self.model.cuda()

    def forward(self, x):
        #output_1 = self.model_1(x)
        #output_2 = self.model_2(x)
        #x = (output_1 + output_2) / 2
        
        # x = self.denoiser(x)
        output = self.model(x)

        # print(output_1[0])
        # print(output_1.data)
        # print("classification_result", torch.max(output_1.data, 1)[1])
        # print("type of output_1", type(output_1))
        
        return output



class EnsembleNet_nasnet(nn.Module):
    def __init__(self):
        super(EnsembleNet_nasnet, self).__init__()
        self.model_1 = nasnetalarge(num_classes=1000, pretrained=False)

        path = os.path.dirname(os.path.abspath(__file__))
        model_1_path = os.path.join(path, 'nasnet', 'nasnet_target_15.pth')

        self.model_1.load_state_dict(torch.load(model_1_path)['model'])
        self.model_1 = self.model_1.eval().cuda()

    def forward(self, x):
        

        output_1 = self.model_1(x)

        # print("classification_result", torch.max(output_1.data, 1)[1])
        
        return output_1



class EnsembleNet_inception_random(nn.Module):
    def __init__(self):
        super(EnsembleNet_inception_random, self).__init__()
        self.model_1 = torch.nn.DataParallel(inception_v3_random()).cuda().eval()

        path = os.path.dirname(os.path.abspath(__file__))
        model_1_path = os.path.join(path, 'inception', 'inception_v3_adv_125.pt')

        self.model_1.load_state_dict(torch.load(model_1_path))
        self.model_1 = self.model_1.cuda()

    def forward(self, x):
        output_1 = self.model_1(x)    
        return output_1



class EnsembleNet_inception_v4(nn.Module):
    def __init__(self):
        super(EnsembleNet_inception_v4, self).__init__()
        self.model_1 = torch.nn.DataParallel(inceptionv4(num_classes=1000, pretrained=False)).cuda()
        self.model_1.last_linear = nn.Linear(1536, 200).cuda()

        path = os.path.dirname(os.path.abspath(__file__))
        model_1_path = os.path.join(path, 'inception_v4', 'inception_v4_target_adv_3.pt')

        self.model_1.load_state_dict(torch.load(model_1_path))
        self.model_1 = self.model_1.eval().cuda()

    def forward(self, x):
        output_1 = self.model_1(x)[:, :200]
        return output_1


class EnsembleNet_inception_small(nn.Module):
    def __init__(self):
        super(EnsembleNet_inception_small, self).__init__()
        path = os.path.dirname(os.path.abspath(__file__))

        self.model_1 = torch.nn.DataParallel(Inception3().cuda())
     
        model_1_path = os.path.join(path, 'inception_small', 'small_inceptionV3_ori_8047.pt')

        self.model_1.load_state_dict(torch.load(model_1_path))


    def forward(self, x):
        # Tensor = torch.cuda.FloatTensor
        # x = Variable(x.type(Tensor))

        # x = self.denoiser(x)
        # x = torch.clamp(x, max=255, min=0)

        output_1 = self.model_1(x)

        # real_prediction = self.model_1(x)
        # _, ori_labels = torch.max(real_prediction.data, 1)

        return output_1



class EnsembleNet_inception_resnet(nn.Module):
    def __init__(self):
        super(EnsembleNet_inception_resnet, self).__init__()
        self.model_1 = inceptionresnetv2(num_classes=200, pretrained=False).cuda()

        path = os.path.dirname(os.path.abspath(__file__))
        model_1_path = os.path.join(path, 'inception_resnet', 'inception_resnet_ori_12.pt')

        self.model_1.load_state_dict(torch.load(model_1_path))
        self.model_1 = self.model_1.eval()

    def forward(self, x):
        # Tensor = torch.cuda.FloatTensor
        # x = Variable(x.type(Tensor))

        output_1 = self.model_1(x)

        real_prediction = self.model_1(x)
        _, ori_labels = torch.max(real_prediction.data, 1)

        return output_1



class EnsembleNet_inception_pnasnet(nn.Module):
    def __init__(self):
        super(EnsembleNet_inception_pnasnet, self).__init__()
        self.model_1 = pnasnet5large(num_classes=1000, pretrained='imagenet').cuda()
        self.model_1.last_linear = nn.Linear(17280, 200).cuda()

        path = os.path.dirname(os.path.abspath(__file__))
        model_1_path = os.path.join(path, 'pnasnet', 'pnasnet_ori_4.pt')

        self.model_1.load_state_dict(torch.load(model_1_path))
        self.model_1 = self.model_1.eval()

    def forward(self, x):
        # Tensor = torch.cuda.FloatTensor
        # x = Variable(x.type(Tensor))

        output_1 = self.model_1(x)

        real_prediction = self.model_1(x)
        _, ori_labels = torch.max(real_prediction.data, 1)

        return output_1


class EnsembleNet_densenet_adv(nn.Module):
    def __init__(self):
        super(EnsembleNet_densenet_adv, self).__init__()
        self.model_3 = torch.nn.DataParallel(densenet161_small()).cuda().eval()

        path = os.path.dirname(os.path.abspath(__file__))
        model_3_path = os.path.join(path, 'densenet', 'dense_adv_epoch_25.pt')

        self.model_3.load_state_dict(torch.load(model_3_path))
        self.model_3 = self.model_3.cuda()


    def forward(self, x):
        output_3 = self.model_3(x)

        return output_3


class EnsembleNet_inception_v4_adv(nn.Module):
    def __init__(self):
        super(EnsembleNet_resnet, self).__init__()
        self.model_4 = torch.nn.DataParallel(inceptionV4_big(num_classes=1000, pretrained=False)).cuda()
        self.model_4.last_linear = nn.Linear(1536, 200).cuda()

        path = os.path.dirname(os.path.abspath(__file__))
        model_4_path = os.path.join(path, 'inception_v4', 'inc_v4_adv_epoch_5.pt')

        self.model_4.load_state_dict(torch.load(model_4_path))
        self.model_4 = self.model_4.cuda()

    def forward(self, x):
        output_4 = self.model_4(x)[:, :200]

        return output_4


class EnsembleNet_vgg19_adv(nn.Module):
    def __init__(self):
        super(EnsembleNet_vgg19_adv, self).__init__()
        self.model_5 = torch.nn.DataParallel(adv_vgg_19()).cuda().eval()

        path = os.path.dirname(os.path.abspath(__file__))
        model_5_path = os.path.join(path, 'vgg19', 'vgg19_bn_adv.pt')

        self.model_5.load_state_dict(torch.load(model_5_path))
        self.model_5 = self.model_5.cuda()

    def forward(self, x):
        output_5 = self.model_5(x)

        return output_5


class EnsembleNet_three(nn.Module):
    def __init__(self):
        super(EnsembleNet_three, self).__init__()
        self.model_1 = torch.nn.DataParallel(adv_vgg_19()).cuda().eval()
        self.model_2 = nasnetalarge(num_classes=1000, pretrained=False)
        self.model_3 = torch.nn.DataParallel(Inception3().cuda())

        path = os.path.dirname(os.path.abspath(__file__))
        model_1_path = os.path.join(path, 'vgg19', 'vgg19_bn_adv.pt')
        model_2_path = os.path.join(path, 'nasnet', 'nasnet_target_15.pth')
        model_3_path = os.path.join(path, 'inception_small', 'small_inceptionV3_ori_8047.pt')

        self.model_1.load_state_dict(torch.load(model_1_path))
        self.model_1 = self.model_1.cuda()
        self.model_2.load_state_dict(torch.load(model_2_path)['model'])
        self.model_2 = self.model_2.eval().cuda()
        self.model_3.load_state_dict(torch.load(model_3_path))

    def forward(self, x):
        output_1 = self.model_1(x)
        output_2 = self.model_2(x)
        output_3 = self.model_3(x)

        final_output = (output_1 + output_2 + output_3) / 3
        
        return final_output