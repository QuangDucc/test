import torch
import torch.nn as nn
import torch.nn.functional as F
from unet.blocks import *

class Unet(nn.Module):
    def __init__(self, in_channels, hidden_channels, n_classes):
        super(Unet, self).__init__()
        self.down1 = DownConv(in_channels, hidden_channels)
        self.down2 = DownConv(hidden_channels, hidden_channels*2)
        self.down3 = DownConv(hidden_channels*2, hidden_channels*4)
        self.down4 = DownConv(hidden_channels*4, hidden_channels*8)

        self.bottleneck = DownConv(hidden_channels*8, hidden_channels*16)

        self.up1 = UpConv(hidden_channels*16, hidden_channels*8)
        self.up2 = UpConv(hidden_channels*8, hidden_channels*4)
        self.up3 = UpConv(hidden_channels*4, hidden_channels*2)
        self.up4 = UpConv(hidden_channels*2, hidden_channels)

        self.out = OutConv(hidden_channels, n_classes)


    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        x_ = self.bottleneck(x4)

        x_4 = self.up1(x_, x4)
        x_3 = self.up2(x_4, x3)
        x_2 = self.up3(x_3, x2)
        x_1 = self.up4(x_2, x1)
        
        return self.out(x_1)