import torch
import torch.nn as nn
import torch.nn.functional as F
from block import *

class YOLOv3(nn.Module):

    def __init__(self,
                 in_channels: int,
                 n_channels: int,
                 n_classes: int,
                 block_counts: list = [1,2,8,8,4]
                 ):
        
        super(YOLOv3, self).__init__()
        self.n_classes = n_classes
        self.inconv = TransConv(in_channels, n_channels, kernel_size=3, stride=1, padding=1)

        self.trans1 = TransConv(n_channels, n_channels*2)
        self.stage_1 = nn.Sequential(*[ResConv(n_channels, n_channels*2) for i in range(block_counts[0])])

        self.trans2 = TransConv(n_channels*2, n_channels*4)
        self.stage_2 = nn.Sequential(*[ResConv(n_channels*2, n_channels*4) for i in range(block_counts[1])])
        
        self.trans3 = TransConv(n_channels*4, n_channels*8)
        self.stage_3 = nn.Sequential(*[ResConv(n_channels*4, n_channels*8) for i in range(block_counts[2])])
        
        self.trans4 = TransConv(n_channels*8, n_channels*16)
        self.stage_4 = nn.Sequential(*[ResConv(n_channels*8, n_channels*16) for i in range(block_counts[3])])
        
        self.trans5 = TransConv(n_channels*16, n_channels*32)
        self.stage_5 = nn.Sequential(*[ResConv(n_channels*16, n_channels*32) for i in range(block_counts[4])])
        
        self.pred_1 = ScalePrediction(n_channels*32, n_classes, scale_factor=1)
        self.pred_2 = ScalePrediction(n_channels*32, n_classes, scale_factor=2)
        self.pred_3 = ScalePrediction(n_channels*16, n_classes, scale_factor=2)
        
    def forward(self,x):
        x = self.inconv(x) #32*HW

        x = self.trans1(x) #64*HW/2
        x1 = self.stage_1(x) #64*HW/2

        x2 = self.trans2(x1) #128*HW/4
        x2 = self.stage_2(x2) #128*HW/4

        x3 = self.trans3(x2) #256*HW/8
        x3 = self.stage_3(x3) #256*HW/8
        
        x4 = self.trans4(x3) #512*HW/16
        x4 = self.stage_4(x4) #512*HW/16

        x5 = self.trans5(x4) #1024*HW/32
        x5 = self.stage_5(x5) #1024*HW/32

        pred_1, prev_1 = self.pred_1(x5)  #512*HW/32
        pred_2, prev_2 = self.pred_2(prev_1,x4) #512*HW/16
        pred_3, prev_3 = self.pred_3(prev_2,x3) #512*HW/8

        del prev_1, prev_2, prev_3

        return pred_1, pred_2, pred_3
        
