import torch
import sys
from torch.autograd import Variable
import numpy as np
from .options.train_options import TrainOptions
from .models.models import create_model
__all__ = ['HourGlass']



def HourGlass(pretrained=None):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    opt = TrainOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch
    model = create_model(opt,pretrained)
    #netG is the real nn.Module
    return model.netG
