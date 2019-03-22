import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo

import torch
# __all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           # 'resnet152','resnet18_conv1']
__all__ = ['S2DF','S2DF_3dense','S2DF_3dense_nodilation',
           'S2DF_3last','S2DF_2dense', 'BasicBlock']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, dilation = 1, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=int(dilation*(3-1)/2), dilation=dilation, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, dilation = 1, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes,dilation, stride)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        # out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        # out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, dilation = 1, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=int(dilation*(3-1)/2), dilation = dilation, bias=False)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        # self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        # out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        # out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        # out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class S2DF(nn.Module):

    def __init__(self, block, num_blocks,dense = True,dilation=True):
        self.inplanes = 64
        super(S2DF, self).__init__()
        self.dense = dense
        self.num_block = num_blocks
        assert(num_blocks>=1 and num_blocks<=4)
        self.block1 = nn.Sequential(*[
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.ReLU(inplace=True)
        ])

        self.dilation = dilation
        # for i in range(1, num_blocks):
        self.block2 = block(self.inplanes, 64, dilation = 4 if dilation else 1) if num_blocks>=2 else None
        self.block3 = block(self.inplanes, 64, dilation = 8 if dilation else 1) if num_blocks>=3 else None
        self.block4 = block(self.inplanes, 64, dilation = 16 if dilation else 1) if num_blocks>=4 else None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        y = []

        y.append(x) #raw feature
        x = self.block1(x)
        if (self.num_block > 1 and self.dense) or self.num_block == 1:
            y.append(x)

        x = self.block2(x) if self.num_block>=2 else x
        if (self.num_block > 2 and self.dense) or self.num_block == 2:
            y.append(x)

        x = self.block3(x) if self.num_block>=3 else x
        if (self.num_block > 3 and self.dense) or self.num_block == 3:
            y.append(x)

        x = self.block4(x) if self.num_block== 4 else x
        if self.num_block == 4 :
            y.append(x)

        return torch.cat(y,dim=1)


class S2DFsim(nn.Module):

    def __init__(self, block, num_blocks,dense = True,dilation=True):
        self.inplanes = 64
        super(S2DFsim, self).__init__()
        self.dense = dense
        self.num_block = num_blocks
        assert(num_blocks>=1 and num_blocks<=4)
        self.block1 = nn.Sequential(*[
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
        ])

        self.dilation = dilation
        # for i in range(1, num_blocks):
        self.block2 = nn.Sequential(*[
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
        ]) if num_blocks >= 2 else None
        self.block3 =  nn.Sequential(*[
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
        ]) if num_blocks >= 3 else None
        self.block4 = nn.Sequential(*[
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
        ]) if num_blocks >= 4 else None

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def forward(self, x):
        y = []

        y.append(x) #raw feature
        x = self.block1(x)
        if (self.num_block > 1 and self.dense) or self.num_block == 1:
            y.append(x)

        x = self.block2(x) if self.num_block>=2 else x
        if (self.num_block > 2 and self.dense) or self.num_block == 2:
            y.append(x)

        x = self.block3(x) if self.num_block>=3 else x
        if (self.num_block > 3 and self.dense) or self.num_block == 3:
            y.append(x)

        x = self.block4(x) if self.num_block== 4 else x
        if self.num_block == 4 :
            y.append(x)

        return torch.cat(y,dim=1)
def S2DF_3dense_nodilation():
    model = S2DFsim(None,3,dense=True,dilation=False)
    return model
def S2DF_3dense():
    model = S2DF(BasicBlock,3,dense=True)
    return model
def S2DF_3last():
    model = S2DF(BasicBlock,3,dense=False)
    return model
def S2DF_2dense():
    model = S2DF(BasicBlock,2,dense=True)
    return model



from torch.autograd import Variable

if __name__ == '__main__':

    x= Variable(torch.randn(2,3,224,448))
    # model =    S2DF(BasicBlock,3,True)
    # y = model(x)

    model = S2DF(BasicBlock,4,False)
    y = model(x)
    exit(0)
