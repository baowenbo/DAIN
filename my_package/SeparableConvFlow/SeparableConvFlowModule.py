# modules/InterpolationLayer.py
from torch.nn import Module
from .SeparableConvFlowLayer import SeparableConvFlowLayer
import  torch
class SeparableConvFlowModule(Module):
    def __init__(self,filtersize):
        super(SeparableConvFlowModule, self).__init__()
        self.f = SeparableConvFlowLayer(filtersize)

    def forward(self, input1, input2, input3):
        # temp2 = torch.div(input2, torch.sum(input2,dim=1,keepdim=True))
        return self.f(input1, input2, input3)

    #we actually dont need to write the backward code for a module, since we have 

