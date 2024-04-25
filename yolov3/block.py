import torch
import torch.nn as nn
import torch.nn.functional as F

class TransConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2,padding=1):
        super(TransConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,padding)
        self.bn = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        return x
    
class ResConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResConv, self).__init__()
        self.conv1 = nn.Conv2d(out_channels, in_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x1 = F.relu(self.bn(self.conv1(x)))
        x2 = self.conv2(x1)
        return x+x2
    

class ScalePrediction(nn.Module):
    def __init__(self, in_channels, n_classes, scale_factor=1):

        super(ScalePrediction, self).__init__()
        self.num_classes = n_classes
        self.scale_factor = scale_factor
        self.conv = nn.Sequential(*[nn.Conv2d(in_channels, in_channels, kernel_size=1),
                                   nn.BatchNorm2d(in_channels),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels, in_channels//(2*scale_factor), kernel_size=3,stride=1,padding=1),
                                   nn.BatchNorm2d(in_channels//(2*scale_factor)),
                                   ])
        # self.conv0 = nn.Conv2d(in_channels, in_channels//2, kernel_size=1)
        self.pred = nn.Conv2d(in_channels//(2*scale_factor), 3*(n_classes+5),kernel_size=1)
    def forward(self, x1, x2=None):
        if self.scale_factor == 1:
            x_prev = self.conv(x1)
            x_pred = self.pred(x_prev)
            
        else:
            # x1 = self.conv0(x1)
            x1 = F.interpolate(x1, scale_factor=self.scale_factor, mode='bilinear', align_corners=True)
            x = torch.cat([x1,x2],dim=1)
            x_prev = self.conv(x)
            x_pred = self.pred(x_prev)
        
        return x_pred.reshape(x_pred.shape[0],3,self.num_classes + 5, x_pred.shape[2],x1.shape[3]), x_prev
