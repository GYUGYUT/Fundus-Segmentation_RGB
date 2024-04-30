from torchvision import models
import torch.nn as nn
import torch
import numpy as np
from sklearn.decomposition import PCA
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp
import torch.nn.functional as F
def create_feature_map(image_tensor):
    batch_size, num_channels, height, width = image_tensor.size()
    
    # 소프트맥스 적용
    softmax_output = F.softmax(image_tensor, dim=1)

    # 각 픽셀에서 최대 확률을 가진 채널의 인덱스 찾기
    max_indices = torch.argmax(softmax_output, dim=1)

    # feature_map 초기화
    feature_map = torch.zeros_like(image_tensor)

    # 가장 높은 확률을 가지는 픽셀의 인덱스를 1로 설정
    feature_map.scatter_(1, max_indices.unsqueeze(1), 1)
    # print(feature_map)
    return feature_map  
def getModel(backbone,numclass):
    net = None
    if backbone == 'resnet50': 
        net = models.resnet50(weights="IMAGENET1K_V1")
        net.fc = nn.Sequential( nn.Linear(2048, 1024),
                                nn.Dropout(0.5),
                                nn.Linear(1024,numclass) )

    elif backbone == 'resnet101': 
        net = models.resnet101(weights="IMAGENET1K_V1")
        net.fc = nn.Sequential( nn.Linear(2048, 1024),
                                nn.Dropout(0.5),
                                nn.Linear(1024,numclass) )
    elif backbone == 'resnet151': 
        net = models.resnet152(weights="IMAGENET1K_V1")
        net.fc = nn.Sequential( nn.Linear(2048, 1024),
                                nn.Dropout(0.5),
                                nn.Linear(1024,numclass) )
    elif backbone == 'densenet121': 
        net = models.densenet121(weights="IMAGENET1K_V1")
        net.classifier = nn.Sequential( nn.Linear( 1024, 512),
                                nn.Dropout(0.5),
                                nn.Linear(512,numclass) )

    elif backbone == 'densenet169': 
        net = models.densenet169(weights="IMAGENET1K_V1")
        net.classifier = nn.Sequential( nn.Linear(1664, 832),
                                        nn.Dropout(0.5),
                                        nn.Linear(832,numclass) )

    elif backbone == 'vgg16' : 
        net = models.vgg16(weights="IMAGENET1K_V1")
        net.classifier[6] = nn.Sequential( nn.Linear(4096, 2048),
                                        nn.Dropout(0.5),
                                        nn.Linear(2048,numclass) )
    elif backbone == 'vgg19' : 
        net = models.vgg19(weights="IMAGENET1K_V1")
        net.classifier[6] = nn.Sequential( nn.Linear(4096, 2048),
                                        nn.Dropout(0.5),
                                        nn.Linear(2048,numclass) )
    elif backbone == 'alexnet' : 
        net = models.alexnet(weights="IMAGENET1K_V1")
        net.classifier[6] = nn.Sequential( nn.Linear(4096, 2048),
                                        nn.Dropout(0.5),
                                        nn.Linear(2048,numclass) )

    elif backbone == 'resnext50' : 
        net = models.resnext50_32x4d(weights="IMAGENET1K_V1")
        net.fc = nn.Sequential( nn.Linear(2048, 1024),
                                        nn.Dropout(0.5),
                                        nn.Linear(1024,numclass) )
    elif backbone == 'resnext101' : 
        net = models.resnext101_32x8d(weights="IMAGENET1K_V1")
        net.fc = nn.Sequential( nn.Linear(2048, 1024),
                                        nn.Dropout(0.5),
                                        nn.Linear(1024,numclass) )
    elif backbone == 'shufflenet' : 
        net = models.shufflenet_v2_x1_0(weights="IMAGENET1K_V1")
        net.fc = nn.Sequential( nn.Linear( 1024, 512),
                                nn.Dropout(0.5),
                                nn.Linear(512,numclass) )
    elif backbone == 'mobilenet_v2' : 
        net = models.mobilenet_v2(weights="IMAGENET1K_V1")
        net.classifier[1] = nn.Sequential( nn.Linear( 1280, 640),
                                nn.Dropout(0.5),
                                nn.Linear(640,numclass) )
    elif backbone == 'mobilenet_v3' : 
        net = models.MobileNetV3(weights="IMAGENET1K_V1")
        net.classifier[1] = nn.Sequential( nn.Linear( 1280, 640),
                                nn.Dropout(0.5),
                                nn.Linear(640,numclass) )
    elif backbone == 'mnasnet' : 
        net = models.mnasnet1_0(weights="IMAGENET1K_V1")
        net.classifier[1] = nn.Sequential( nn.Linear( 1280, 640),
                                nn.Dropout(0.5),
                                nn.Linear(640,numclass) )
    elif backbone == 'efficientnet_b6' : 
        net = models.efficientnet_b6(weights="IMAGENET1K_V1")
        net.classifier[1] = nn.Sequential( nn.Linear( 2560 , 1280),
                                nn.Dropout(0.5),
                                nn.Linear(1280,numclass) )
    elif backbone == 'efficientnet_b7' : 
        net = models.efficientnet_b7(weights="IMAGENET1K_V1")
        net.classifier[1] = nn.Sequential( nn.Linear( 2560 , 1280),
                                nn.Dropout(0.5),
                                nn.Linear(1280,numclass) )
    elif backbone == 'UNet' :

        net = smp.Unet(
            encoder_name="mobilenet_v2",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=5,     
            activation='sigmoid'
        )
        print(net)
    elif backbone == 'UNet++' :

        net = smp.UnetPlusPlus(
            encoder_name="mobilenet_v2",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=5,      
            activation='sigmoid'
        )
        print(net)
    elif backbone == 'Linknet' :

        net = smp.Linknet(
            encoder_name="mobilenet_v2",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                 # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=5,     
            activation='sigmoid'
        )
        print(net)
    elif backbone == 'MAnet' :

        net = smp.MAnet(
            encoder_name="mobilenet_v2",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                 # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=5,  
            activation='sigmoid'   
        )
        print(net)
    elif backbone == 'FPN' :

        net = smp.FPN(
            encoder_name="mobilenet_v2",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                   # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=5,    
            activation='sigmoid'
        )
        print(net)
    elif backbone == 'PSPNet' :

        net = smp.PSPNet(
            encoder_name="mobilenet_v2",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=5,     
            activation='sigmoid'
        )
        print(net)
    elif backbone == 'PAN' :

        net = smp.PAN(
            encoder_name="mobilenet_v2",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=5,   
            activation='sigmoid'   
        )
        print(net)
    elif backbone == 'DeepLabV3' :

        net = smp.DeepLabV3(
            encoder_name="mobilenet_v2",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=5,      
            activation='sigmoid'
        )
        print(net)
    elif backbone == 'DeepLabV3Plus' :

        net = smp.DeepLabV3Plus(
            encoder_name="mobilenet_v2",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=5,     
            activation='sigmoid'
        )
        print(net)
    # class three_Channel(nn.Module):
    #     def __init__(self,net):
    #         super(three_Channel, self).__init__()
    #         # 입력 이미지 채널을 3채널로 조정하는 부분 수정
            
    #         self.net = net
    #         self.out = create_feature_map
    #     def forward(self, x):
    #         x = self.net(x)
    #         x = self.out(x)
    #         return x
    # net = three_Channel(net)
    return net