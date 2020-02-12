# modules/InterpolationLayer.py
from torch.nn import Module
from functions.SeparableConvLayer import SeparableConvLayer

class SeparableConvModule(Module):
    def __init__(self,filtersize):
        super(SeparableConvModule, self).__init__()
        self.f = SeparableConvLayer(filtersize)

    def forward(self, input1, input2, input3):
        return self.f(input1, input2, input3)

    #we actually dont need to write the backward code for a module, since we have 

