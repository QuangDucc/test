from model import YOLOv3
from torchsummary import summary
import torch
import torch.nn as nn

def main():
    model = YOLOv3(3, 32, 3)
    x = torch.ones(1,3,224,224)
    o1, o2, o3 = model(x)
    
    

    print(o1.shape)
    print(o2.shape)
    print(o3.shape)
if __name__ == '__main__':
    main()