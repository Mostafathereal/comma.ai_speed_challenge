import numpy as np
import torch
import torch.nn
import torchvision as tv
import torchvision.models as models
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

cuda = torch.device("cuda")

x = torch.randn(1, 3, 640, 480).to(cuda)

class InceptionSpeed(nn.Module):
    def __init__(self):
        super(InceptionSpeed, self).__init__()

        self.base = tv.models.inception_v3(pretrained=True)
        self.lin1 = torch.nn.Linear(1000, 100)
        self.lin2 = torch.nn.Linear(100, 10)
        self.lin3 = torch.nn.Linear(10, 1)

    def forward(self, x):
        x = self.base(x)
        x = F.leaky_relu(self.lin1(x))
        x = F.leaky_relu(self.lin2(x))
        x = F.leaky_relu(self.lin3(x))
        return x

# model = InceptionSpeed().to(cuda)
# model.eval()
# outputs = model(x)


# print(len(list(model.parameters()))

# print(model)



# inception = tv.models.inception_v3(pretrained=True)
# inception.eval()
# print(len(inception(x)[0]))

'''
print(tv.__version__)
x = torch.randn(1, 5, 640, 480)


inception = tv.models.inception_v3(pretrained=True)
inception.cuda()


first_conv_layer = [nn.Conv2d(5, 3, kernel_size=(3, 3), stride=(2, 2), bias=True)]


first_conv_layer.extend(list(inception.features))

inception.features= nn.Sequential(*first_conv_layer )

output = inception(x)

print(inception)
'''

# summary(inception, (3, 640, 480

# x = torch.randn(1, 1, 224, 224)
# model = models.vgg16(pretrained=False) # pretrained=False just for debug reasons
# first_conv_layer = [nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)]
# first_conv_layer.extend(list(model.features))  
# model.features= nn.Sequential(*first_conv_layer )  
# output = model(x)