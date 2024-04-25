import torch
import torch.nn as nn
import torch.nn.functional as F



class DoubleConv(nn.Module):
    def __init__ (self,in_channels,out_channels):
        super(DoubleConv,self).__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1)
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self,x):
        x = F.relu(self.bn(self.conv1(x)))
        x = F.relu(self.bn(self.conv2(x)))
        return x


class DownConv(nn.Module):

    def __init__ (self,in_channels,out_channels):
        super(DownConv,self).__init__()
        self.conv = DoubleConv(in_channels,out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)

    def forward(self,x):
        x = self.conv(x)
        return self.pool(x)
    
class UpConv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(UpConv,self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels,in_channels//2,kernel_size=2,stride=1,padding=0)
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = DoubleConv(in_channels,out_channels)
    
    def forward(self,x1,x2):
        x1 = self.upconv(x1)

        diffX = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1,[diffX//2,diffX-diffX//2,diffY//2,diffY-diffY//2])      
        x = torch.cat([x2,x1],dim=1)
        return self.conv(x)
    
class OutConv(nn.Module):
    def __init__(self,in_channels,n_classes):
        super(OutConv,self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(in_channels,n_classes,kernel_size=1,stride=1,padding=0)
        
    def forward(self,x):
        x = self.up(x)
        x = self.conv(x)
        return x