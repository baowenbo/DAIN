# modules/InterpolationLayer.py
from torch.nn import Module
from .InterpolationChLayer import InterpolationChLayer

class InterpolationChModule(Module):
    def __init__(self,ch):
        super(InterpolationChModule, self).__init__()
        self.ch = ch
        # self.f = InterpolationChLayer(ch)

    def forward(self, input1, input2):
        return InterpolationChLayer.apply(input1, input2)

    #we actually dont need to write the backward code for a module, since we have 

