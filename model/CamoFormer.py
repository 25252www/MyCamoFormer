import torch
from torch import nn
from torch.utils import model_zoo

from .encoder.swin_encoder import SwinTransformer
from .encoder.pvtv2_encoder import pvt_v2_b4
from .decoder.decoder_p import Decoder

from timm.models import create_model
import collections
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

def weight_init_backbone(module):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear): 
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, (nn.ReLU, nn.Sigmoid, nn.PReLU, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool1d, nn.Sigmoid, nn.Identity)):
            pass
        else:
            m.initialize()

def weight_init(module):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            #nn.init.xavier_normal_(m.weight, gain=1)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear): 
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            #nn.init.xavier_normal_(m.weight, gain=1)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, (nn.ReLU, nn.Sigmoid, nn.PReLU, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool1d, nn.Sigmoid, nn.Identity)):
            pass
        else:
            m.initialize()

class CamoFormer(torch.nn.Module):
    def __init__(self, pretrained=None, snapshot=None):
        super(CamoFormer, self).__init__()
        self.encoder = pvt_v2_b4()
        self.snapshot = snapshot
        if pretrained is not None:
            pretrained_dict = torch.load(pretrained)  
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in self.encoder.state_dict()}
            self.encoder.load_state_dict(pretrained_dict)
            print('Pretrained encoder loaded.')

        self.decoder = Decoder(128)
        self.initialize()

    def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels, input_channel):
        return block(dilation_series, padding_series, NoLabels, input_channel)

    def forward(self, x):

        features = self.encoder(x)
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]

        # 保证输入输出的尺寸一致
        shape = x.size()[2:]

        P5, P4, P3, P2, P1= self.decoder(x1, x2, x3, x4, shape)
        # P1是最细腻的特征图，P5是最粗糙的特征图
        return P5, P4, P3, P2, P1
    
    def initialize(self):
        if self.snapshot is not None:
            self.load_state_dict(torch.load(self.snapshot))
            print('Snapshot loaded: ' + self.snapshot)
        else:
            weight_init(self)


       
    


