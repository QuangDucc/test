from unet.model import Unet
from torchsummary import summary
import torch
import torch.nn as nn

def main():
    model = Unet(3, 64, 3)
    x = torch.ones(1,3,224,224)
    output = model(x)
    output = nn.Softmax(dim=1)(output)
    output = output.argmax(0)

    print(output.shape)
if __name__ == '__main__':
    main()